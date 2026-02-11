from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal

@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    dtype: Literal["int", "float", "cat"]
    required: bool = True
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[str]] = None

# DB keys
DB_ID_KEY = "id"  # id technique BIGSERIAL
EMPLOYEE_ID_KEY = "employee_external_id"  # id métier stable (SIRH)
TARGET_KEY = "a_quitte_l_entreprise"

# Liste complète des features utilisées par le modèle, avec leurs types et contraintes,
# qui seront utilisées à la fois pour la validation des inputs dans le dashboard Streamlit,
# et pour construire les payloads de prédiction envoyés à l'API FastAPI. 
# L'ordre des features doit correspondre à celui attendu par le modèle de prédiction.
FEATURES: List[Feature] = [
    Feature("note_evaluation_precedente", "Note d’évaluation précédente", "int", min=1, max=5),
    Feature("note_evaluation_actuelle", "Note d’évaluation actuelle", "int", min=1, max=5),
    Feature("niveau_hierarchique_poste", "Niveau hiérarchique du poste", "int", min=1, max=5),
    Feature("heures_supplementaires", "Heures supplémentaires (0/1)", "int", min=0, max=1),
    Feature("augmentation_salaire_precedente", "Augmentation salariale précédente (%)", "float", min=0),
    Feature("age", "Âge", "int", min=16, max=80),
    Feature("genre", "Genre (0 = F, 1 = M)", "int", min=0, max=1),
    Feature("revenu_mensuel", "Revenu mensuel (€)", "int", min=0),
    Feature("statut_marital", "Statut marital", "cat"),
    Feature("niveau_education", "Niveau d’éducation", "int", min=1, max=5),
    Feature("domaine_etude", "Domaine d’étude", "cat"),
    Feature("departement", "Département", "cat"),
    Feature("poste", "Poste occupé", "cat"),
    Feature("nombre_experiences_precedentes", "Expériences précédentes", "int", min=0),
    Feature("annee_experience_totale", "Années d’expérience totale", "int", min=0),
    Feature("annees_dans_l_entreprise", "Ancienneté dans l’entreprise", "int", min=0),
    Feature("annees_dans_le_poste_actuel", "Ancienneté dans le poste", "int", min=0),
    Feature("nombre_participation_pee", "Participations au PEE", "int", min=0),
    Feature("nb_formations_suivies", "Formations suivies", "int", min=0),
    Feature("frequence_deplacement", "Fréquence de déplacement (0–3)", "int", min=0, max=3),
    Feature("annees_depuis_la_derniere_promotion", "Années depuis la dernière promotion", "int", min=0),
    Feature("annees_sous_responsable_actuel", "Années sous le responsable actuel", "int", min=0),
    Feature("distance_domicile_travail", "Distance domicile–travail (km)", "int", min=0),
    Feature("satisfaction_moyenne", "Satisfaction moyenne", "float", min=0, max=5),
    Feature("nonlineaire_participation_pee", "Participation PEE (non linéaire)", "float"),
    Feature("ratio_heures_sup_salaire", "Ratio heures sup / salaire", "float"),
    Feature("nonlinaire_charge_contrainte", "Charge contrainte (non linéaire)", "float"),
    Feature("nonlinaire_surmenage_insatisfaction", "Surmenage & insatisfaction", "float"),
    Feature("jeune_surcharge", "Jeune avec surcharge (0/1)", "int", min=0, max=1),
    Feature("anciennete_sans_promotion", "Ancienneté sans promotion", "float"),
    Feature("mobilite_carriere", "Mobilité de carrière", "float"),
    Feature("risque_global", "Risque global agrégé", "float")
]

# Ordre des features attendu par le modèle
MODEL_FEATURE_ORDER: List[str] = [f.key for f in FEATURES]

# Colonnes clean utiles en API (sans target)
DEPLOYMENT_COLUMNS: List[str] = [EMPLOYEE_ID_KEY] + MODEL_FEATURE_ORDER

# Colonnes complètes DB clean
DB_COLUMNS: List[str] = [DB_ID_KEY, EMPLOYEE_ID_KEY] + MODEL_FEATURE_ORDER + [TARGET_KEY]
