import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL manquant dans .env")

# Créer une connexion à la base de données
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# DDL pour créer la table clean.ml_features_employees
DDL = """
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS clean;

CREATE TABLE IF NOT EXISTS clean.ml_features_employees (
  id BIGSERIAL PRIMARY KEY,

  employee_id BIGINT NOT NULL
    REFERENCES raw.employees(id) ON DELETE CASCADE,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  note_evaluation_precedente SMALLINT NOT NULL,
  niveau_hierarchique_poste   SMALLINT NOT NULL,
  note_evaluation_actuelle    SMALLINT NOT NULL,

  heures_supplementaires SMALLINT NOT NULL CHECK (heures_supplementaires IN (0,1)),
  augmentation_salaire_precedente NUMERIC(10,2) NOT NULL CHECK (augmentation_salaire_precedente >= 0),

  age   SMALLINT NOT NULL CHECK (age BETWEEN 16 AND 80),
  genre SMALLINT NOT NULL CHECK (genre IN (0,1)),
  revenu_mensuel INTEGER NOT NULL CHECK (revenu_mensuel >= 0),

  statut_marital VARCHAR(50) NOT NULL,
  departement    VARCHAR(120) NOT NULL,
  poste          VARCHAR(120) NOT NULL,

  nombre_experiences_precedentes SMALLINT NOT NULL CHECK (nombre_experiences_precedentes >= 0),
  annee_experience_totale        SMALLINT NOT NULL CHECK (annee_experience_totale >= 0),
  annees_dans_l_entreprise       SMALLINT NOT NULL CHECK (annees_dans_l_entreprise >= 0),
  annees_dans_le_poste_actuel    SMALLINT NOT NULL CHECK (annees_dans_le_poste_actuel >= 0),

  a_quitte_l_entreprise SMALLINT NOT NULL CHECK (a_quitte_l_entreprise IN (0,1)),

  nombre_participation_pee SMALLINT NOT NULL CHECK (nombre_participation_pee >= 0),
  nb_formations_suivies    SMALLINT NOT NULL CHECK (nb_formations_suivies >= 0),
  distance_domicile_travail SMALLINT NOT NULL CHECK (distance_domicile_travail >= 0),

  niveau_education SMALLINT NOT NULL CHECK (niveau_education BETWEEN 1 AND 5),
  domaine_etude    VARCHAR(120) NOT NULL,
  frequence_deplacement SMALLINT NOT NULL CHECK (frequence_deplacement BETWEEN 0 AND 3),

  annees_depuis_la_derniere_promotion SMALLINT NOT NULL CHECK (annees_depuis_la_derniere_promotion >= 0),
  annees_sous_responsable_actuel      SMALLINT NOT NULL CHECK (annees_sous_responsable_actuel >= 0),

  satisfaction_moyenne NUMERIC(10,4) NOT NULL,
  nonlineaire_participation_pee       NUMERIC(18,16) NOT NULL,
  ratio_heures_sup_salaire            NUMERIC(18,16) NOT NULL,
  nonlinaire_charge_contrainte        NUMERIC(18,16) NOT NULL,
  nonlinaire_surmenage_insatisfaction NUMERIC(18,16) NOT NULL,

  jeune_surcharge SMALLINT NOT NULL CHECK (jeune_surcharge IN (0,1)),
  anciennete_sans_promotion NUMERIC(18,16) NOT NULL,
  mobilite_carriere         NUMERIC(18,16) NOT NULL,
  risque_global             NUMERIC(18,16) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_clean_employee_id_created_at
  ON clean.ml_features_employees(employee_id, created_at);

CREATE INDEX IF NOT EXISTS idx_ml_features_target
  ON clean.ml_features_employees(a_quitte_l_entreprise);

CREATE INDEX IF NOT EXISTS idx_ml_features_target_created_at
  ON clean.ml_features_employees(a_quitte_l_entreprise, created_at);
"""

# Fonction pour exécuter les commandes DDL
def _exec_ddl(conn, ddl: str) -> None:
    parts = [p.strip() for p in ddl.split(";") if p.strip()]
    for stmt in parts:
        conn.execute(text(stmt))

# Point d'entrée du script
def main():
    with engine.begin() as conn:
        _exec_ddl(conn, DDL)
    print("Table clean.ml_features_employees créée / vérifiée (avec employee_id)")

if __name__ == "__main__":
    main()
