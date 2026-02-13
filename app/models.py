"""
Modèles ORM SQLAlchemy du projet TechNova Partners.

Ce module définit les classes Python qui représentent les tables SQL de la base,
en suivant une organisation en 3 schémas :

- raw.*   : données brutes (ingestion / stockage initial)
- clean.* : données nettoyées + features prêtes pour le modèle ML
- app.*   : logs applicatifs (requêtes de prédiction + prédictions)

Pourquoi c'est utile ?
- On sépare clairement les responsabilités :
  - raw = ce qu'on reçoit
  - clean = ce qu'on donne au modèle
  - app = ce qu'on trace (audit / monitoring)

Compatibilité SQLite / PostgreSQL
- En local dev / prod : PostgreSQL
- En tests : SQLite
SQLite gère moins bien certains types (ex: BigInteger), donc on utilise
`with_variant(Integer, "sqlite")` pour que les mêmes modèles fonctionnent
dans les deux environnements.

Contenu principal
- RawEmployee, RawEmployeeSnapshot, RawSurvey : tables brutes (raw)
- MLFeaturesEmployee : features prêtes (clean)
- PredictionRequest, Prediction : traçabilité des appels (app)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """
    Retourne l'heure actuelle en UTC.

    Cette fonction sert de valeur par défaut pour les champs `created_at`,
    afin que toutes les dates stockées soient cohérentes (UTC) quel que soit
    le serveur ou le pays où l'application tourne.

    Returns
    -------
    datetime
        Date et heure courantes avec timezone UTC.
    """
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """
    Classe de base SQLAlchemy pour tous les modèles ORM.

    Tous les modèles (tables) héritent de `Base`. SQLAlchemy utilise cette base
    pour :
    - enregistrer les classes déclarées,
    - générer le schéma (create_all),
    - mapper les objets Python vers les lignes SQL.
    """
    pass


# Identifiants compatibles Postgres / SQLite
ID_PK = BigInteger().with_variant(Integer, "sqlite")
"""
Type de colonne pour une clé primaire (Primary Key).

- Sur PostgreSQL : BigInteger (supporte de grands ids)
- Sur SQLite : Integer (BigInteger n'est pas géré pareil)

Cela garantit que les mêmes modèles fonctionnent :
- en production (PostgreSQL)
- en tests (SQLite)
"""

ID_FK = BigInteger().with_variant(Integer, "sqlite")
"""
Type de colonne pour une clé étrangère (Foreign Key), même logique que ID_PK.
"""


class RawEmployee(Base):
    """
    Table `raw.employees` : employés (données brutes).

    Rôle
    ----
    Cette table représente l'identité "de base" d'un employé.
    Dans ton projet, elle sert surtout à fournir un identifiant interne (`id`)
    qui peut être référencé par d'autres tables.

    Champs principaux
    -----------------
    id : int
        Clé primaire interne.
    employee_external_id : int
        Identifiant provenant du dataset (id d'origine).

    Index
    -----
    Un index est défini sur `employee_external_id` pour accélérer les recherches
    (utile si on mappe un id externe vers l'id interne).
    """
    __tablename__ = "employees"
    __table_args__ = (
        Index("ix_raw_employees_employee_external_id", "employee_external_id"),
        {"schema": "raw"},
    )

    id: Mapped[int] = mapped_column(ID_PK, primary_key=True, autoincrement=True)
    employee_external_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)


class RawEmployeeSnapshot(Base):
    """
    Table `raw.employee_snapshots` : snapshots bruts (optionnel).

    Idée
    ----
    Un snapshot peut représenter un état de l'employé à un instant T
    (ex: poste, salaire, département à une date).

    Dans V1
    -----------
    le champ `id` pour pouvoir relier des logs de prédiction à un snapshot.
    """
    __tablename__ = "employee_snapshots"
    __table_args__ = ({"schema": "raw"},)

    id: Mapped[int] = mapped_column(ID_PK, primary_key=True, autoincrement=True)


class RawSurvey(Base):
    """
    Table `raw.surveys` : enquêtes / questionnaires bruts (optionnel).

    Idée
    ----
    Certaines features peuvent venir d'enquêtes internes (satisfaction, etc.).
    Cette table permet de stocker ces données "raw" si tu les ajoutes plus tard.

    Dans V1
    -----------
    Seul `id` est nécessaire pour conserver une structure évolutive.
    """
    __tablename__ = "surveys"
    __table_args__ = ({"schema": "raw"},)

    id: Mapped[int] = mapped_column(ID_PK, primary_key=True, autoincrement=True)


class MLFeaturesEmployee(Base):
    """
    Table `clean.ml_features_employees` : features ML prêtes pour l'inférence.

    Rôle
    ---------------------
    Cette table contient les variables déjà nettoyées et transformées
    (feature engineering terminé). Elle est directement compatible avec le
    pipeline du modèle ML.

    MLOps-friendly ?
    -------------------------------
    - L'API ne fait PAS de preprocessing.
    - Elle lit la dernière ligne "clean" et prédit.
    - Le pipeline de préparation tourne séparément (scripts ETL).

    Organisation
    ------------
    - Une ligne = un jeu de features à un instant donné (created_at).
    - Pour un employee_id, on peut avoir plusieurs lignes au fil du temps.
      L'API récupère généralement la plus récente.

    Champs
    ------
    employee_id : int
        Référence vers `raw.employees.id`.
    created_at : datetime
        Date de création du snapshot de features.
    a_quitte_l_entreprise : int
        Target (0/1). Utile pour entraînement / audit.
        En production, peut être null (selon ta stratégie),
        mais ici il est requis.

    Index
    -----
    Plusieurs index pour accélérer :
    - récupération du dernier snapshot par employee_id
    - requêtes d'analyse sur la target
    """
    __tablename__ = "ml_features_employees"
    __table_args__ = (
        Index("idx_clean_employee_id_created_at", "employee_id", "created_at"),
        Index("idx_ml_features_target", "a_quitte_l_entreprise"),
        Index("idx_ml_features_target_created_at", "a_quitte_l_entreprise", "created_at"),
        {"schema": "clean"},
    )

    id: Mapped[int] = mapped_column(ID_PK, primary_key=True, autoincrement=True)

    employee_id: Mapped[int] = mapped_column(
        ID_FK,
        ForeignKey("raw.employees.id", ondelete="CASCADE"),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )

    employee: Mapped["RawEmployee"] = relationship("RawEmployee")

    # (Ensuite: tes features, inchangées)
    note_evaluation_precedente: Mapped[int] = mapped_column(Integer, nullable=False)
    niveau_hierarchique_poste: Mapped[int] = mapped_column(Integer, nullable=False)
    note_evaluation_actuelle: Mapped[int] = mapped_column(Integer, nullable=False)

    heures_supplementaires: Mapped[int] = mapped_column(Integer, nullable=False)
    augmentation_salaire_precedente: Mapped[float] = mapped_column(Float, nullable=False)

    age: Mapped[int] = mapped_column(Integer, nullable=False)
    genre: Mapped[int] = mapped_column(Integer, nullable=False)

    revenu_mensuel: Mapped[int] = mapped_column(Integer, nullable=False)
    statut_marital: Mapped[str] = mapped_column(String(50), nullable=False)
    departement: Mapped[str] = mapped_column(String(120), nullable=False)
    poste: Mapped[str] = mapped_column(String(120), nullable=False)

    nombre_experiences_precedentes: Mapped[int] = mapped_column(Integer, nullable=False)
    annee_experience_totale: Mapped[int] = mapped_column(Integer, nullable=False)
    annees_dans_l_entreprise: Mapped[int] = mapped_column(Integer, nullable=False)
    annees_dans_le_poste_actuel: Mapped[int] = mapped_column(Integer, nullable=False)

    nombre_participation_pee: Mapped[int] = mapped_column(Integer, nullable=False)
    nb_formations_suivies: Mapped[int] = mapped_column(Integer, nullable=False)
    distance_domicile_travail: Mapped[int] = mapped_column(Integer, nullable=False)

    niveau_education: Mapped[int] = mapped_column(Integer, nullable=False)
    domaine_etude: Mapped[str] = mapped_column(String(120), nullable=False)
    frequence_deplacement: Mapped[int] = mapped_column(Integer, nullable=False)

    annees_depuis_la_derniere_promotion: Mapped[int] = mapped_column(Integer, nullable=False)
    annees_sous_responsable_actuel: Mapped[int] = mapped_column(Integer, nullable=False)

    satisfaction_moyenne: Mapped[float] = mapped_column(Float, nullable=False)
    nonlineaire_participation_pee: Mapped[float] = mapped_column(Float, nullable=False)
    ratio_heures_sup_salaire: Mapped[float] = mapped_column(Float, nullable=False)
    nonlinaire_charge_contrainte: Mapped[float] = mapped_column(Float, nullable=False)
    nonlinaire_surmenage_insatisfaction: Mapped[float] = mapped_column(Float, nullable=False)

    jeune_surcharge: Mapped[int] = mapped_column(Integer, nullable=False)
    anciennete_sans_promotion: Mapped[float] = mapped_column(Float, nullable=False)
    mobilite_carriere: Mapped[float] = mapped_column(Float, nullable=False)
    risque_global: Mapped[float] = mapped_column(Float, nullable=False)

    a_quitte_l_entreprise: Mapped[int] = mapped_column(Integer, nullable=False)


class PredictionRequest(Base):
    """
    Table `app.prediction_requests` : journal des requêtes de prédiction.

    Rôle
    ----
    Chaque appel à l'API de prédiction crée une "request" :
    - qui a fait l'appel (via le contexte)
    - pour quel employé (si applicable)
    - et/ou quel payload a été envoyé

    Pourquoi ?
    ----------
    - Traçabilité (audit)
    - Débogage (retrouver quel payload a produit une prédiction)
    - Monitoring (volume d'appels dans le temps)

    Champs importants
    -----------------
    payload_json : dict
        Contenu de la requête (mode, employee_id ou features).
    created_at : datetime
        Timestamp UTC de la requête.

    Relations
    ---------
    predictions : list[Prediction]
        Une request peut avoir une ou plusieurs prédictions associées
        (dans ton cas, en général une seule).
    """
    __tablename__ = "prediction_requests"
    __table_args__ = (
        Index("ix_prediction_requests_created_at", "created_at"),
        {"schema": "app"},
    )

    id: Mapped[int] = mapped_column(ID_PK, primary_key=True, autoincrement=True)

    employee_id: Mapped[Optional[int]] = mapped_column(
        ID_FK,
        ForeignKey("raw.employees.id", ondelete="SET NULL"),
        nullable=True,
    )
    snapshot_id: Mapped[Optional[int]] = mapped_column(
        ID_FK,
        ForeignKey("raw.employee_snapshots.id", ondelete="SET NULL"),
        nullable=True,
    )
    survey_id: Mapped[Optional[int]] = mapped_column(
        ID_FK,
        ForeignKey("raw.surveys.id", ondelete="SET NULL"),
        nullable=True,
    )

    payload_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",
    )

    employee: Mapped[Optional["RawEmployee"]] = relationship("RawEmployee", foreign_keys=[employee_id])
    snapshot: Mapped[Optional["RawEmployeeSnapshot"]] = relationship("RawEmployeeSnapshot", foreign_keys=[snapshot_id])
    survey: Mapped[Optional["RawSurvey"]] = relationship("RawSurvey", foreign_keys=[survey_id])


class Prediction(Base):
    """
    Table `app.predictions` : journal des résultats de prédiction.

    Rôle
    ----
    Stocke le résultat d'une prédiction ML, lié à une `PredictionRequest`.

    Pourquoi ?
    ----------
    - Conserver l'historique de scoring
    - Auditer les décisions (classe / probabilité / seuil)
    - Comparer des versions de modèles (model_version)

    Champs importants
    -----------------
    model_version : str
        Identifiant de version du modèle (ex: "xgb_v1").
    predicted_class : int
        Classe binaire prédite (0/1).
    predicted_proba : float
        Probabilité prédite d'appartenir à la classe positive (turnover=1).
    threshold_used : float
        Seuil appliqué pour transformer proba -> classe.
    latency_ms : int | None
        Temps de réponse de la prédiction côté API (utile monitoring).
    created_at : datetime
        Timestamp UTC de la prédiction.

    Relation
    --------
    request : PredictionRequest
        La requête associée à cette prédiction.
    """
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_request_id_created_at", "request_id", "created_at"),
        Index("ix_predictions_created_at", "created_at"),
        Index("ix_predictions_model_version", "model_version"),
        {"schema": "app"},
    )

    id: Mapped[int] = mapped_column(ID_PK, primary_key=True, autoincrement=True)

    request_id: Mapped[int] = mapped_column(
        ID_FK,
        ForeignKey("app.prediction_requests.id", ondelete="CASCADE"),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_proba: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    request: Mapped["PredictionRequest"] = relationship(back_populates="predictions")
