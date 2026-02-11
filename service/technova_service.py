from __future__ import annotations

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

def _get_artifact_path(artifacts_dir: Path, filename: str) -> str:
    """
    1) Try local artifacts/ (dev)
    2) Fallback to Hugging Face Hub (prod / Space)
    Returns a filesystem path as string.
    """
    local_path = artifacts_dir / filename
    if local_path.exists():
        return str(local_path)

    return hf_hub_download(
        repo_id=MODEL_REPO,
        filename=filename,
        repo_type="model",
    )

class TechNovaService:
    def __init__(self) -> None:
        artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"

        # --- Load model (local or HF) ---
        model_path = _get_artifact_path(artifacts_dir, MODEL_FILE)
        self.model = joblib.load(model_path)

        # --- Load threshold (local or HF) ---
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

    def fetch_latest_clean_row(self, db: Session, employee_id: int) -> dict[str, Any] | None:
        row = (
            db.execute(self._sql_clean_latest, {"employee_id": employee_id})
            .mappings()
            .first()
        )
        return dict(row) if row else None

    def _row_to_X(self, row: dict[str, Any]) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in row]
        if missing:
            raise ValueError(f"Clean row missing required features: {missing}")

        X = pd.DataFrame([{c: row[c] for c in self.feature_columns}])
        return X[self.feature_columns]

    def predict_from_clean(self, db: Session, employee_id: int) -> ModelResponse:
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

    # ---------- Payload ----------
    def adapt_input(self, request: ModelRequest) -> pd.DataFrame:
        data = request.model_dump()
        X = pd.DataFrame([data])

        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        extra = [c for c in X.columns if c not in self.feature_columns]
        if extra:
            raise ValueError(f"Unexpected features: {extra}")

        return X[self.feature_columns]

    def predict_from_payload(self, request: ModelRequest) -> ModelResponse:
        X = self.adapt_input(request)
        proba = float(self.model.predict_proba(X)[0, 1])

        return ModelResponse(
            employee_id=None,  # pas d'id côté payload
            turnover_probability=proba,
            will_leave=proba >= self.threshold,
        )
