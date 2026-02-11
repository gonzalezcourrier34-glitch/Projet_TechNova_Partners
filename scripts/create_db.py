import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL manquant dans .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# DDL pour créer les schemas et tables de la base de données
DDL = """
-- SCHEMAS
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS clean;
CREATE SCHEMA IF NOT EXISTS app;

-- RAW TABLES
CREATE TABLE IF NOT EXISTS raw.employees (
  id BIGSERIAL PRIMARY KEY,
  employee_external_id INTEGER NOT NULL UNIQUE,

  age INTEGER,
  genre VARCHAR(20),
  statut_marital VARCHAR(50),
  ayant_enfants VARCHAR(10),
  niveau_education INTEGER,
  domaine_etude VARCHAR(120),
  departement VARCHAR(120),
  poste VARCHAR(120),
  distance_domicile_travail INTEGER,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_raw_employees_employee_external_id
  ON raw.employees(employee_external_id);

CREATE TABLE IF NOT EXISTS raw.employee_snapshots (
  id BIGSERIAL PRIMARY KEY,
  employee_id BIGINT NOT NULL REFERENCES raw.employees(id) ON DELETE CASCADE,

  nombre_experiences_precedentes INTEGER,
  nombre_heures_travaillees INTEGER,
  annee_experience_totale INTEGER,
  annees_dans_l_entreprise INTEGER,
  annees_dans_le_poste_actuel INTEGER,
  annees_sous_responsable_actuel INTEGER,
  niveau_hierarchique_poste INTEGER,
  revenu_mensuel INTEGER,

  augmentation_salaire_precedente VARCHAR(50),
  heures_supplementaires VARCHAR(50),

  nombre_participation_pee INTEGER,
  nb_formations_suivies INTEGER,
  nombre_employee_sous_responsabilite INTEGER,

  frequence_deplacement VARCHAR(50),
  annees_depuis_la_derniere_promotion INTEGER,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_raw_snapshots_employee_id_created_at
  ON raw.employee_snapshots(employee_id, created_at);

CREATE TABLE IF NOT EXISTS raw.surveys (
  id BIGSERIAL PRIMARY KEY,
  employee_id BIGINT NOT NULL REFERENCES raw.employees(id) ON DELETE CASCADE,

  code_sondage INTEGER,
  eval_number VARCHAR(50),

  note_evaluation_precedente INTEGER,
  note_evaluation_actuelle INTEGER,

  satisfaction_employee_environnement INTEGER,
  satisfaction_employee_nature_travail INTEGER,
  satisfaction_employee_equipe INTEGER,
  satisfaction_employee_equilibre_pro_perso INTEGER,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_raw_surveys_employee_id_created_at
  ON raw.surveys(employee_id, created_at);

CREATE TABLE IF NOT EXISTS raw.ground_truth (
  id BIGSERIAL PRIMARY KEY,
  employee_id BIGINT NOT NULL REFERENCES raw.employees(id) ON DELETE CASCADE,

  date_event TIMESTAMPTZ NOT NULL DEFAULT now(),
  a_quitte_l_entreprise INTEGER NOT NULL CHECK (a_quitte_l_entreprise IN (0,1)),

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_ground_truth_employee_id_date_event
  ON raw.ground_truth(employee_id, date_event);

-- APP LOGGING TABLES
CREATE TABLE IF NOT EXISTS app.prediction_requests (
  id BIGSERIAL PRIMARY KEY,

  employee_id BIGINT REFERENCES raw.employees(id) ON DELETE SET NULL,
  snapshot_id BIGINT REFERENCES raw.employee_snapshots(id) ON DELETE SET NULL,
  survey_id BIGINT REFERENCES raw.surveys(id) ON DELETE SET NULL,

  payload_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_prediction_requests_created_at
  ON app.prediction_requests(created_at);

CREATE TABLE IF NOT EXISTS app.predictions (
  id BIGSERIAL PRIMARY KEY,
  request_id BIGINT NOT NULL REFERENCES app.prediction_requests(id) ON DELETE CASCADE,

  model_version VARCHAR(50) NOT NULL,
  predicted_class INTEGER NOT NULL,
  predicted_proba DOUBLE PRECISION NOT NULL,
  threshold_used DOUBLE PRECISION NOT NULL,
  latency_ms INTEGER,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_predictions_request_id_created_at
  ON app.predictions(request_id, created_at);

CREATE INDEX IF NOT EXISTS ix_predictions_created_at
  ON app.predictions(created_at);

CREATE INDEX IF NOT EXISTS ix_predictions_model_version
  ON app.predictions(model_version);
"""

# Fonction utilitaire pour exécuter du DDL de manière robuste
def _exec_ddl(conn, ddl: str) -> None:
    # robuste: exécute statement par statement
    parts = [p.strip() for p in ddl.split(";") if p.strip()]
    for stmt in parts:
        conn.execute(text(stmt))

# Point d'entrée du script
def main():
    with engine.begin() as conn:
        _exec_ddl(conn, DDL)
    print("Schemas + tables créés / vérifiés (raw / clean / app)")

if __name__ == "__main__":
    main()
