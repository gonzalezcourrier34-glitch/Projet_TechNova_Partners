from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel

# Ce module contient les définitions de classes et constantes partagées entre le dashboard Streamlit et l'API FastAPI, notamment la liste des features utilisées par le modèle de prédiction, ainsi que les modèles de données pour les requêtes et réponses de l'API.

# Modèles de données pour les requêtes et réponses de l'API, utilisés pour la validation des données d'entrée et de sortie.
class ModelRequest(BaseModel):
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
    employee_id: Optional[int] = None
    turnover_probability: float
    will_leave: bool
