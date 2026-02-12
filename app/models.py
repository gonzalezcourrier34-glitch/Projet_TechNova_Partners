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

# Fonction utilitaire pour obtenir l'heure actuelle en UTC, utilisée comme valeur par défaut pour les champs created_at
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

# Base de données SQLAlchemy pour les modèles ORM
class Base(DeclarativeBase):
    pass

# Types personnalisés pour les clés primaires et étrangères, avec compatibilité SQLite (qui n'a pas de BigInteger)
ID_PK = BigInteger().with_variant(Integer, "sqlite")
ID_FK = BigInteger().with_variant(Integer, "sqlite")


# Modèles SQLAlchemy représentant les tables de la base de données, organisés par schéma (raw, clean, app)
class RawEmployee(Base):
    __tablename__ = "employees"
    __table_args__ = (
        Index("ix_raw_employees_employee_external_id", "employee_external_id"),
        {"schema": "raw"},
    )

    id: Mapped[int] = mapped_column(ID_PK, primary_key=True, autoincrement=True)
    employee_external_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)


# Les autres champs de la table raw.employees peuvent être ajoutés ici si nécessaire,
# mais ne sont pas indispensables pour les prédictions basées sur employee_id
class RawEmployeeSnapshot(Base):
    __tablename__ = "employee_snapshots"
    __table_args__ = ({"schema": "raw"},)

    id: Mapped[int] = mapped_column(ID_PK, primary_key=True, autoincrement=True)


# Les champs de la table raw.employee_snapshots peuvent être ajoutés ici si nécessaire,
# mais ne sont pas indispensables pour les prédictions basées sur employee_id
class RawSurvey(Base):
    __tablename__ = "surveys"
    __table_args__ = ({"schema": "raw"},)

    id: Mapped[int] = mapped_column(ID_PK, primary_key=True, autoincrement=True)


# Les champs de la table raw.surveys peuvent être ajoutés ici si nécessaire,
# mais ne sont pas indispensables pour les prédictions basées sur employee_id
class MLFeaturesEmployee(Base):
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


# Les champs de la table app.prediction_requests peuvent être ajustés en fonction des besoins,
# mais les champs essentiels pour le suivi des prédictions sont employee_id, payload_json, et created_at
class PredictionRequest(Base):
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

    # Relations optionnelles vers les données brutes associées à la prédiction, si disponibles.
    employee: Mapped[Optional["RawEmployee"]] = relationship("RawEmployee", foreign_keys=[employee_id])
    snapshot: Mapped[Optional["RawEmployeeSnapshot"]] = relationship("RawEmployeeSnapshot", foreign_keys=[snapshot_id])
    survey: Mapped[Optional["RawSurvey"]] = relationship("RawSurvey", foreign_keys=[survey_id])


# Les champs de la table app.predictions peuvent être ajustés en fonction des besoins,
# mais les champs essentiels pour le suivi des prédictions sont request_id, predicted_proba, et created_at
class Prediction(Base):
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
