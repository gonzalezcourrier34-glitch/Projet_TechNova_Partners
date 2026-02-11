from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv, find_dotenv

# CONFIG & LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("technova.etl")

# SETTINGS
@dataclass(frozen=True)
class Settings:
    # RAW schema tables
    raw_employees: str = "raw.employees"
    raw_snapshots: str = "raw.employee_snapshots"
    raw_surveys: str = "raw.surveys"
    ground_truth: str = "raw.ground_truth"

    # CLEAN destination
    dst_schema: str = "clean"
    dst_name: str = "ml_features_employees"
    dst_qualified: str = "clean.ml_features_employees"

# UTILS
def get_engine():
    load_dotenv(find_dotenv())
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL manquant dans .env")
    return create_engine(db_url, pool_pre_ping=True)

# TRANSFORMS
class Transform:
    @staticmethod
    def clean_raw_inputs(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # % -> float
        if "augmentation_salaire_precedente" in df.columns:
            s = df["augmentation_salaire_precedente"].astype("string")
            s = s.str.replace("%", "", regex=False)
            df["augmentation_salaire_precedente"] = pd.to_numeric(s, errors="coerce")

        # Oui/Non -> 1/0
        if "heures_supplementaires" in df.columns:
            df["heures_supplementaires"] = df["heures_supplementaires"].map(
                {"Oui": 1, "Non": 0, "oui": 1, "non": 0, 1: 1, 0: 0, True: 1, False: 0}
            )

        # Genre -> 1/0
        if "genre" in df.columns:
            df["genre"] = df["genre"].map({"M": 1, "F": 0, 1: 1, 0: 0})

        # Fréquence déplacement -> 0/1/2 (+ éventuellement 3 si tu veux)
        if "frequence_deplacement" in df.columns:
            df["frequence_deplacement"] = df["frequence_deplacement"].map(
                {
                    "Aucun": 0,
                    "Occasionnel": 1,
                    "Frequent": 2,
                    "Fréquent": 2,
                    0: 0, 1: 1, 2: 2, 3: 3,
                }
            )

        numeric_cols = [
            "employee_id",
            "age",
            "revenu_mensuel",
            "niveau_education",
            "distance_domicile_travail",
            "note_evaluation_precedente",
            "note_evaluation_actuelle",
            "niveau_hierarchique_poste",
            "nombre_experiences_precedentes",
            "annee_experience_totale",
            "annees_dans_l_entreprise",
            "annees_dans_le_poste_actuel",
            "annees_depuis_la_derniere_promotion",
            "annees_sous_responsable_actuel",
            "nombre_participation_pee",
            "nb_formations_suivies",
            "a_quitte_l_entreprise"
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    @staticmethod
    def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        sat_cols = [
            "satisfaction_employee_environnement",
            "satisfaction_employee_nature_travail",
            "satisfaction_employee_equipe",
            "satisfaction_employee_equilibre_pro_perso"
        ]
        missing = [c for c in sat_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Colonnes satisfaction manquantes (raw.surveys?) : {missing}")

        # satisfaction moyenne
        df["satisfaction_moyenne"] = df[sat_cols].mean(axis=1)

        # features dérivées
        df["nonlineaire_participation_pee"] = (
            df["nombre_participation_pee"]
            / (df["nombre_participation_pee"] + df["annees_dans_l_entreprise"] + 1)
        )

        df["ratio_heures_sup_salaire"] = (
            df["heures_supplementaires"] / (df["revenu_mensuel"] + 1)
        )

        d = df["distance_domicile_travail"]
        df["nonlinaire_charge_contrainte"] = (
            df["heures_supplementaires"] * d / (d + 10) / (d + 10)
        )

        df["nonlinaire_surmenage_insatisfaction"] = (
            df["heures_supplementaires"] * (1 - df["satisfaction_moyenne"])
        )

        df["jeune_surcharge"] = (
            (df["age"] < 30) & (df["heures_supplementaires"] == 1)
        ).astype(int)

        df["anciennete_sans_promotion"] = (
            (df["annees_dans_l_entreprise"] - df["annees_depuis_la_derniere_promotion"])
            / (df["annees_dans_l_entreprise"] + 1)
        )

        df["mobilite_carriere"] = (
            df["nombre_experiences_precedentes"]
            / (df["annee_experience_totale"] + 1)
        )

        df["risque_global"] = (
            df["ratio_heures_sup_salaire"]
            * df["anciennete_sans_promotion"]
            * (1 - df["satisfaction_moyenne"])
        )

        return df

    @staticmethod
    def suppression_features(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(
            columns=[
                "satisfaction_employee_environnement",
                "satisfaction_employee_nature_travail",
                "satisfaction_employee_equipe",
                "satisfaction_employee_equilibre_pro_perso"
            ],
            errors="ignore",
        )

# DESTINATION COLUMNS
DEST_COLS = [
    "employee_id",
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
    "a_quitte_l_entreprise",
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
    "risque_global"
]


def build_sql_master(s: Settings) -> str:
    return f"""
WITH last_snapshot AS (
  SELECT DISTINCT ON (employee_id)
    employee_id,
    nombre_experiences_precedentes,
    annee_experience_totale,
    annees_dans_l_entreprise,
    annees_dans_le_poste_actuel,
    annees_sous_responsable_actuel,
    niveau_hierarchique_poste,
    revenu_mensuel,
    augmentation_salaire_precedente,
    heures_supplementaires,
    nombre_participation_pee,
    nb_formations_suivies,
    frequence_deplacement,
    annees_depuis_la_derniere_promotion,
    created_at
  FROM {s.raw_snapshots}
  ORDER BY employee_id, created_at DESC
),
last_survey AS (
  SELECT DISTINCT ON (employee_id)
    employee_id,
    note_evaluation_precedente,
    note_evaluation_actuelle,
    satisfaction_employee_environnement,
    satisfaction_employee_nature_travail,
    satisfaction_employee_equipe,
    satisfaction_employee_equilibre_pro_perso,
    created_at
  FROM {s.raw_surveys}
  ORDER BY employee_id, created_at DESC
),
last_truth AS (
  SELECT DISTINCT ON (employee_id)
    employee_id,
    a_quitte_l_entreprise,
    date_event
  FROM {s.ground_truth}
  ORDER BY employee_id, date_event DESC
)
SELECT
  e.id AS employee_id,
  e.age,
  e.genre,
  e.statut_marital,
  e.niveau_education,
  e.domaine_etude,
  e.departement,
  e.poste,
  e.distance_domicile_travail,
  s.nombre_experiences_precedentes,
  s.annee_experience_totale,
  s.annees_dans_l_entreprise,
  s.annees_dans_le_poste_actuel,
  s.annees_sous_responsable_actuel,
  s.niveau_hierarchique_poste,
  s.revenu_mensuel,
  s.augmentation_salaire_precedente,
  s.heures_supplementaires,
  s.nombre_participation_pee,
  s.nb_formations_suivies,
  s.frequence_deplacement,
  s.annees_depuis_la_derniere_promotion,
  sv.note_evaluation_precedente,
  sv.note_evaluation_actuelle,
  sv.satisfaction_employee_environnement,
  sv.satisfaction_employee_nature_travail,
  sv.satisfaction_employee_equipe,
  sv.satisfaction_employee_equilibre_pro_perso,
  COALESCE(gt.a_quitte_l_entreprise, 0) AS a_quitte_l_entreprise

FROM {s.raw_employees} e
LEFT JOIN last_snapshot s ON s.employee_id = e.id
LEFT JOIN last_survey   sv ON sv.employee_id = e.id
LEFT JOIN last_truth    gt ON gt.employee_id = e.id

WHERE s.employee_id IS NOT NULL
  AND sv.employee_id IS NOT NULL
;
"""

# ETL STEPS
def fetch_master_df(engine, sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

def validate_columns(df: pd.DataFrame, dst_qualified: str) -> None:
    missing = [c for c in DEST_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes pour insertion dans {dst_qualified}: {missing}")

def validate_quality(df: pd.DataFrame) -> None:
    critical = ["employee_id", "age", "revenu_mensuel", "heures_supplementaires", "a_quitte_l_entreprise"]
    bad = [c for c in critical if c in df.columns and df[c].isna().mean() > 0.20]
    if bad:
        raise ValueError(f"Trop de NaN sur colonnes critiques (>20%): {bad}")

def truncate_destination(engine, dst_qualified: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {dst_qualified} RESTART IDENTITY;"))

def enforce_not_null_ready(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=DEST_COLS)

    # cast ints propres (pandas peut garder float si NaN existe, mais là on a drop)
    int_like = [
        "employee_id",
        "note_evaluation_precedente",
        "niveau_hierarchique_poste",
        "note_evaluation_actuelle",
        "heures_supplementaires",
        "age",
        "genre",
        "revenu_mensuel",
        "nombre_experiences_precedentes",
        "annee_experience_totale",
        "annees_dans_l_entreprise",
        "annees_dans_le_poste_actuel",
        "a_quitte_l_entreprise",
        "nombre_participation_pee",
        "nb_formations_suivies",
        "distance_domicile_travail",
        "niveau_education",
        "frequence_deplacement",
        "annees_depuis_la_derniere_promotion",
        "annees_sous_responsable_actuel",
        "jeune_surcharge"
    ]
    for c in int_like:
        if c in out.columns:
            out[c] = out[c].astype(int)

    return out

def insert_destination(engine, df: pd.DataFrame, dst_schema: str, dst_name: str, chunk_size: int = 2000) -> int:
    df_out = df[DEST_COLS].copy()
    with engine.begin() as conn:
        df_out.to_sql(
            name=dst_name,
            schema=dst_schema,
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=chunk_size,
        )
    return len(df_out)

# MAIN
def main():
    refresh = "--refresh" in sys.argv
    dry_run = "--dry-run" in sys.argv

    s = Settings()
    engine = get_engine()
    sql_master = build_sql_master(s)

    logger.info("Build master DF depuis raw tables")
    df = fetch_master_df(engine, sql_master)
    logger.info("Master DF: %s lignes | %s colonnes", df.shape[0], df.shape[1])

    logger.info("Nettoyage types / mapping")
    df = Transform.clean_raw_inputs(df)

    logger.info("Feature engineering")
    df = Transform.feature_engineering(df)

    logger.info("Drop colonnes intermédiaires")
    df = Transform.suppression_features(df)

    logger.info("Validation colonnes destination")
    validate_columns(df, s.dst_qualified)

    logger.info("Contrôle qualité (NaN critiques)")
    validate_quality(df)

    logger.info("Préparation NOT NULL (drop rows invalides + cast ints)")
    before = len(df)
    df = enforce_not_null_ready(df)
    after = len(df)
    if after < before:
        logger.warning("Lignes retirées (NULL sur colonnes clean NOT NULL): %s -> %s", before, after)

    if dry_run:
        logger.info("DRY-RUN: aucune écriture en base. Aperçu:")
        logger.info("Colonnes finales: %s", list(df[DEST_COLS].columns))
        logger.info("Head:\n%s", df[DEST_COLS].head(5).to_string(index=False))
        return

    if refresh:
        logger.info("Mode REFRESH: TRUNCATE %s", s.dst_qualified)
        truncate_destination(engine, s.dst_qualified)

    logger.info("Insertion vers %s", s.dst_qualified)
    n = insert_destination(engine, df, s.dst_schema, s.dst_name)
    logger.info("OK: %s lignes insérées dans %s", n, s.dst_qualified)

if __name__ == "__main__":
    main()
