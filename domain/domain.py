"""
Schémas Pydantic partagés (API + dashboard) pour le projet TechNova Partners.

Ce module définit les modèles de données utilisés pour :
- valider les entrées de l'API (payload de prédiction),
- standardiser les sorties de l'API (réponse de prédiction),
- partager une "source de vérité" côté types entre FastAPI et Streamlit.

Pourquoi Pydantic ?
- Pydantic vérifie automatiquement les types (int, float, str…)
- Il rejette les payloads invalides (erreurs 422 côté FastAPI)
- Il documente l'API via OpenAPI/Swagger (schéma JSON généré)

Conventions
- Les variables catégorielles binaires sont encodées en 0/1 (ex: genre, heures_supplementaires).
- Certaines variables sont limitées par des `Literal[...]` pour éviter les valeurs hors domaine.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel

# Ce module contient les définitions de classes et constantes partagées entre le dashboard Streamlit et l'API FastAPI, notamment la liste des features utilisées par le modèle de prédiction, ainsi que les modèles de données pour les requêtes et réponses de l'API.

# Modèles de données pour les requêtes et réponses de l'API, utilisés pour la validation des données d'entrée et de sortie.
class ModelRequest(BaseModel):
    """
    Schéma d'entrée pour une prédiction "par features" (mode scoring direct).

    Ce modèle représente **exactement** l'ensemble des features attendues par le modèle ML,
    dans le bon type. Il est utilisé par FastAPI pour valider le JSON reçu sur l'endpoint
    `/predict/by-features`.

    Points importants
    -----------------
    - Aucune étape de preprocessing n'est faite ici.
      Le payload doit déjà être "propre" et au format attendu.
    - Les champs en `Literal[...]` restreignent volontairement les valeurs autorisées
      (ex: 0/1 pour un booléen encodé, ou 0..3 pour une fréquence).

    Effet côté API
    --------------
    - Si un champ manque, ou si un type/valeur est invalide, FastAPI renvoie une erreur 422.
    - Le schéma apparaît automatiquement dans Swagger (/docs).

    Notes
    -----
    - Les variables catégorielles (ex: statut_marital, departement, poste, domaine_etude)
      sont des chaînes. Le pipeline du modèle doit savoir les gérer (encodage/OneHot etc.)
      ou attendre des valeurs déjà compatibles.
    """
    note_evaluation_precedente: int
    niveau_hierarchique_poste: int
    note_evaluation_actuelle: int
    heures_supplementaires: Literal[0, 1]
    augmentation_salaire_precedente: float
    age: int
    genre: Literal[0, 1]
    revenu_mensuel: int
    statut_marital: str
    departement: str
    poste: str
    nombre_experiences_precedentes: int
    annee_experience_totale: int
    annees_dans_l_entreprise: int
    annees_dans_le_poste_actuel: int
    nombre_participation_pee: int
    nb_formations_suivies: int
    distance_domicile_travail: int
    niveau_education: int
    domaine_etude: str
    frequence_deplacement: Literal[0, 1, 2, 3]
    annees_depuis_la_derniere_promotion: int
    annees_sous_responsable_actuel: int
    satisfaction_moyenne: float
    nonlineaire_participation_pee: float
    ratio_heures_sup_salaire: float
    nonlinaire_charge_contrainte: float
    nonlinaire_surmenage_insatisfaction: float
    jeune_surcharge: Literal[0, 1]
    anciennete_sans_promotion: float
    mobilite_carriere: float
    risque_global: float

# Le modèle de réponse inclut la probabilité de départ et la classe prédite (will_leave), ainsi que l'employee_id si disponible.
class ModelResponse(BaseModel):
    """
    Schéma de sortie standard pour une prédiction.

    Ce modèle est renvoyé par les endpoints de prédiction (`/predict/by-id` et `/predict/by-features`).

    Champs
    ------
    employee_id : int | None
        Identifiant de l'employé si la prédiction est faite "par ID".
        `None` si la prédiction est faite "par features" (pas d'employé en base).
    turnover_probability : float
        Probabilité estimée que l'employé quitte l'entreprise (classe positive).
        Valeur attendue entre 0 et 1.
    will_leave : bool
        Décision binaire calculée en comparant `turnover_probability` à un seuil (`threshold`).
        True = départ prédit, False = non-départ prédit.

    Exemple
    -------
    {
      "employee_id": 123,
      "turnover_probability": 0.78,
      "will_leave": true
    }
    """
    employee_id: Optional[int] = None
    turnover_probability: float
    will_leave: bool
