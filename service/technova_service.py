from __future__ import annotations

"""
service.technova_service

Service ML central du projet TechNova Partners.

Ce module encapsule la logique "métier ML" côté API :
- charger les artefacts du modèle (pipeline joblib + seuil) au démarrage,
- préparer les données au format attendu par le pipeline,
- produire une prédiction (probabilité + décision binaire),
- interroger la base pour récupérer les features déjà préparées (tables `clean.*`).

Philosophie MLOps
-------------------------------------
1) Pas de preprocessing dans l'API
   L'API ne "nettoie" pas les données à chaque requête.
   Elle consomme des features déjà prêtes en base (ETL séparé).

2) Un service dédié au ML
   L'API appelle ce service, ce qui rend le code :
   - plus testable (on peut mocker le service),
   - plus clair (responsabilités séparées),
   - plus maintenable (V2 plus simple).

Artefacts du modèle
-------------------
Le service cherche d'abord les fichiers en local dans `artifacts/`.
S'ils ne sont pas présents, il les télécharge depuis Hugging Face Hub.

Variables d'environnement utiles
--------------------------------
MODEL_REPO  : dépôt HF contenant les artefacts (repo type = "model")
MODEL_FILE  : nom du fichier joblib du modèle (pipeline)
THRESH_FILE : nom du fichier JSON contenant le seuil

Exemples
--------
- MODE production (recommandé) :
    predict_from_clean(db, employee_id)

- MODE debug / scoring direct :
    predict_from_payload(request)
"""

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
from sqlalchemy import text
from sqlalchemy.orm import Session

from domain.domain import ModelRequest, ModelResponse

MODEL_REPO = os.getenv("MODEL_REPO", "SteGONZALEZ/technova-model")
MODEL_FILE = os.getenv("MODEL_FILE", "modele_classification_technova.joblib")
THRESH_FILE = os.getenv("THRESH_FILE", "threshold.json")

# Note : on pourrait aussi imaginer stocker ces artefacts dans un bucket S3 ou autre, et les télécharger à la volée au démarrage de l'API.
def _get_artifact_path(artifacts_dir: Path, filename: str) -> str:
    """
    Retourne le chemin vers un artefact (modèle ou threshold).

    Stratégie :
    1) si le fichier existe en local dans `artifacts_dir`, on l'utilise,
    2) sinon, on le télécharge depuis Hugging Face Hub.

    Parameters
    ----------
    artifacts_dir : pathlib.Path
        Dossier local contenant les artefacts (ex: /app/artifacts).
    filename : str
        Nom du fichier recherché (ex: "modele_classification_technova.joblib").

    Returns
    -------
    str
        Chemin local utilisable par joblib/open().
        Si téléchargé depuis HF, `hf_hub_download` renvoie un chemin vers le cache local.

    Raises
    ------
    huggingface_hub.utils.HfHubHTTPError
        Si le repo ou le fichier est introuvable ou non accessible.
    """
    local_path = artifacts_dir / filename
    if local_path.exists():
        return str(local_path)

    return hf_hub_download(
        repo_id=MODEL_REPO,
        filename=filename,
        repo_type="model",
    )


# Service encapsulant la logique de chargement du modèle, de préparation des données et de prédiction.
class TechNovaService:
    """
    Service ML utilisé par l'API.

    Responsabilités
    ---------------
    - Charger le modèle (pipeline joblib) et le seuil de décision (threshold).
    - Définir la liste ordonnée des features attendues par le pipeline (`feature_columns`).
    - Récupérer une ligne de features "clean" depuis la base.
    - Convertir ces données en DataFrame compatible avec `predict_proba`.
    - Retourner un `ModelResponse` standard.

    Notes importantes
    -----------------
    - Le modèle est chargé une seule fois au démarrage de l'API (via lifespan).
    - `feature_columns` est l'ordre **exact** attendu par le pipeline.
    - La prédiction binaire `will_leave` dépend du seuil chargé depuis `threshold.json`.
    """

    def __init__(self) -> None:
        """
        Initialise le service et charge les artefacts du modèle.

        Étapes :
        1) Détermine le dossier `artifacts/` du projet.
        2) Charge le modèle joblib (local si présent, sinon HF Hub).
        3) Charge le seuil depuis un JSON (local si présent, sinon HF Hub).
        4) Déclare l'ordre des features attendues.
        5) Prépare une requête SQL pour récupérer la dernière ligne "clean" par employee_id.

        Raises
        ------
        FileNotFoundError / OSError
            Si le fichier threshold local est introuvable et que le téléchargement HF échoue.
        Exception
            Toute erreur de chargement du modèle (joblib) ou de parsing du JSON.
        """
        artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"

        # Load model (local or HF)
        model_path = _get_artifact_path(artifacts_dir, MODEL_FILE)
        self.model = joblib.load(model_path)

        # Load threshold (local or HF)
        thresh_path = _get_artifact_path(artifacts_dir, THRESH_FILE)
        with open(thresh_path, "r", encoding="utf-8") as f:
            self.threshold = float(json.load(f)["threshold"])

        # Ordre EXACT attendu par le pipeline
        self.feature_columns: list[str] = [
            "note_evaluation_precedente",
            "niveau_hierarchique_poste",
            "note_evaluation_actuelle",
            "heures_supplementaires",
            "augmentation_salaire_precedente",
            "age",
            "genre",
            "revenu_mensuel",
            "statut_marital",
            "departement",
            "poste",
            "nombre_experiences_precedentes",
            "annee_experience_totale",
            "annees_dans_l_entreprise",
            "annees_dans_le_poste_actuel",
            "nombre_participation_pee",
            "nb_formations_suivies",
            "distance_domicile_travail",
            "niveau_education",
            "domaine_etude",
            "frequence_deplacement",
            "annees_depuis_la_derniere_promotion",
            "annees_sous_responsable_actuel",
            "satisfaction_moyenne",
            "nonlineaire_participation_pee",
            "ratio_heures_sup_salaire",
            "nonlinaire_charge_contrainte",
            "nonlinaire_surmenage_insatisfaction",
            "jeune_surcharge",
            "anciennete_sans_promotion",
            "mobilite_carriere",
            "risque_global",
        ]

        self._sql_clean_latest = text(
            f"""
            SELECT {", ".join(self.feature_columns)}
            FROM clean.ml_features_employees
            WHERE employee_id = :employee_id
            ORDER BY created_at DESC
            LIMIT 1
            """
        )

    # Récupère la dernière ligne de features "clean" pour un employee_id donné, ou None si pas trouvé
    def fetch_latest_clean_row(self, db: Session, employee_id: int) -> dict[str, Any] | None:
        """
        Récupère la dernière ligne de features "clean" d'un employé.

        La table `clean.ml_features_employees` peut contenir plusieurs versions dans le temps.
        On prend la plus récente (`ORDER BY created_at DESC LIMIT 1`).

        Parameters
        ----------
        db : sqlalchemy.orm.Session
            Session SQLAlchemy ouverte sur la base.
        employee_id : int
            Identifiant interne (clé utilisée dans `clean.ml_features_employees.employee_id`).

        Returns
        -------
        dict[str, Any] | None
            Un dictionnaire {feature: valeur} si trouvé, sinon None.

        Raises
        ------
        sqlalchemy.exc.SQLAlchemyError
            Si la requête échoue (table absente, schéma absent, etc.).
        """
        row = (
            db.execute(self._sql_clean_latest, {"employee_id": employee_id})
            .mappings()
            .first()
        )
        return dict(row) if row else None

    # Transforme une ligne de features "clean" en DataFrame prête à être ingérée par le modèle, en vérifiant que toutes les features attendues sont présentes
    def _row_to_X(self, row: dict[str, Any]) -> pd.DataFrame:
        """
        Convertit une ligne "clean" (dict) en DataFrame modèle-ready.

        Vérifie que toutes les features attendues sont présentes, puis construit un DataFrame
        avec **exactement** les colonnes dans l'ordre `feature_columns`.

        Parameters
        ----------
        row : dict[str, Any]
            Dictionnaire contenant les features.

        Returns
        -------
        pandas.DataFrame
            DataFrame (1 ligne) au format attendu par le pipeline.

        Raises
        ------
        ValueError
            Si des features attendues manquent.
        """
        missing = [c for c in self.feature_columns if c not in row]
        if missing:
            raise ValueError(f"Clean row missing required features: {missing}")

        X = pd.DataFrame([{c: row[c] for c in self.feature_columns}])
        return X[self.feature_columns]

    # Prédit le risque de départ d'un employé à partir de son employee_id, en lisant les features dans clean.ml_features_employees et en appliquant le modèle chargé. Retourne une ModelResponse avec la probabilité et la prédiction binaire (will_leave).
    def predict_from_clean(self, db: Session, employee_id: int) -> ModelResponse:
        """
        Prédit à partir de la base (mode production, recommandé).

        Pipeline :
        1) récupère la dernière ligne "clean" pour `employee_id`,
        2) transforme en DataFrame au bon format,
        3) calcule la probabilité de départ via `predict_proba`,
        4) compare au seuil pour produire `will_leave`.

        Parameters
        ----------
        db : sqlalchemy.orm.Session
            Session SQLAlchemy.
        employee_id : int
            Identifiant interne utilisé dans la table clean.

        Returns
        -------
        ModelResponse
            - employee_id : renseigné
            - turnover_probability : proba [0, 1]
            - will_leave : bool selon le seuil

        Raises
        ------
        ValueError
            Si aucune ligne clean n'existe pour cet employee_id.
        """
        row = self.fetch_latest_clean_row(db, employee_id)
        if row is None:
            raise ValueError(
                f"Aucune ligne dans clean.ml_features_employees pour employee_id={employee_id}. "
                f"As-tu bien exécuté l'ETL vers clean ?"
            )

        X = self._row_to_X(row)
        proba = float(self.model.predict_proba(X)[0, 1])

        return ModelResponse(
            employee_id=employee_id,
            turnover_probability=proba,
            will_leave=proba >= self.threshold,
        )

    # Transforme un ModelRequest (payload de features brutes) en DataFrame prête à être ingérée par le modèle, en vérifiant que toutes les features attendues sont présentes et qu'il n'y a pas de features inattendues.
    def adapt_input(self, request: ModelRequest) -> pd.DataFrame:
        """
        Adapte un payload JSON (ModelRequest) vers un DataFrame modèle-ready.

        Utilisé par l'endpoint `/predict/by-features`.

        Contrôles :
        - toutes les features attendues doivent être présentes,
        - aucune feature inconnue ne doit être envoyée,
        - l'ordre final est imposé par `feature_columns`.

        Parameters
        ----------
        request : ModelRequest
            Payload validé par Pydantic.

        Returns
        -------
        pandas.DataFrame
            DataFrame (1 ligne) au format attendu.

        Raises
        ------
        ValueError
            Si features manquantes ou inattendues.
        """
        data = request.model_dump()
        X = pd.DataFrame([data])

        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        extra = [c for c in X.columns if c not in self.feature_columns]
        if extra:
            raise ValueError(f"Unexpected features: {extra}")

        return X[self.feature_columns]

    # Prédit le risque de départ à partir d'un payload de features brutes, en adaptant le payload pour obtenir les features attendues par le modèle, puis en appliquant le modèle. Retourne une ModelResponse avec la probabilité et la prédiction binaire (will_leave).
    def predict_from_payload(self, request: ModelRequest) -> ModelResponse:
        """
        Prédit à partir d'un payload de features (mode debug / scoring direct).

        Étapes :
        1) transforme le payload en DataFrame via `adapt_input`,
        2) calcule la probabilité via `predict_proba`,
        3) applique le seuil pour obtenir `will_leave`.

        Parameters
        ----------
        request : ModelRequest
            Features prêtes (pas de preprocessing dans l'API).

        Returns
        -------
        ModelResponse
            - employee_id : None (pas d'ID côté payload)
            - turnover_probability : proba [0, 1]
            - will_leave : bool selon le seuil
        """
        X = self.adapt_input(request)
        proba = float(self.model.predict_proba(X)[0, 1])

        return ModelResponse(
            employee_id=None,  # pas d'id côté payload
            turnover_probability=proba,
            will_leave=proba >= self.threshold,
        )
