from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine, text, bindparam

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("technova.seed")

# Utilitaire pour convertir des valeurs oui/non en 1/0
def yesno_to_int(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)) and x in (0, 1):
        return int(x)

    s = str(x).strip().lower()
    if s in {"oui", "y", "yes", "1", "true", "vrai"}:
        return 1
    if s in {"non", "n", "no", "0", "false", "faux"}:
        return 0
    return None

# Normalisation basique des colonnes (trim, etc.)
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

# Utilitaire pour pick une valeur parmi plusieurs clés possibles (utile pour gérer les variations de noms de colonnes)
def pick(r: dict, *keys, default=None):
    for k in keys:
        if k in r and r.get(k) is not None:
            return r.get(k)
    return default

# Point d'entrée du script
def main(
    sirh_csv: str = "data/extrait_sirh.csv",
    eval_csv: str = "data/extrait_eval.csv",
    sondage_csv: str = "data/extrait_sondage.csv",
):
    refresh = "--refresh" in sys.argv

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL manquant dans .env")

    engine = create_engine(db_url, pool_pre_ping=True)

    base = Path(__file__).resolve().parents[1]
    sirh_path = (base / sirh_csv).resolve()
    eval_path = (base / eval_csv).resolve()
    sondage_path = (base / sondage_csv).resolve()

    for p in (sirh_path, eval_path, sondage_path):
        if not p.exists():
            raise RuntimeError(f"Fichier introuvable: {p}")

    df_sirh = norm_cols(pd.read_csv(sirh_path))
    df_eval = norm_cols(pd.read_csv(eval_path))
    df_sond = norm_cols(pd.read_csv(sondage_path))

    if "id_employee" not in df_sirh.columns:
        raise RuntimeError("extrait_sirh.csv doit contenir la colonne id_employee")

    if not (len(df_sirh) == len(df_eval) == len(df_sond)):
        raise RuntimeError("Les 3 CSV doivent avoir le même nombre de lignes (jointure par index).")

    df = pd.concat([df_sirh, df_eval, df_sond], axis=1)

    # Remplace NaN -> None
    df = df.where(pd.notnull(df), None)
    # Optionnel mais utile: transforme "" / "   " -> None
    df = df.replace(r"^\s*$", None, regex=True)

    # SQL (SCHEMA raw.*)
    upsert_employees = text(
        """
        INSERT INTO raw.employees (
            employee_external_id, age, genre, statut_marital, ayant_enfants,
            niveau_education, domaine_etude, departement, poste, distance_domicile_travail
        )
        VALUES (
            :employee_external_id, :age, :genre, :statut_marital, :ayant_enfants,
            :niveau_education, :domaine_etude, :departement, :poste, :distance_domicile_travail
        )
        ON CONFLICT (employee_external_id) DO UPDATE SET
            age = EXCLUDED.age,
            genre = EXCLUDED.genre,
            statut_marital = EXCLUDED.statut_marital,
            ayant_enfants = EXCLUDED.ayant_enfants,
            niveau_education = EXCLUDED.niveau_education,
            domaine_etude = EXCLUDED.domaine_etude,
            departement = EXCLUDED.departement,
            poste = EXCLUDED.poste,
            distance_domicile_travail = EXCLUDED.distance_domicile_travail
        """
    )

    select_emp_map = (
        text(
            """
            SELECT id, employee_external_id
            FROM raw.employees
            WHERE employee_external_id IN :ext_ids
            """
        )
        .bindparams(bindparam("ext_ids", expanding=True))
    )

    insert_snapshot = text(
        """
        INSERT INTO raw.employee_snapshots (
            employee_id,
            nombre_experiences_precedentes,
            nombre_heures_travaillees,
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
            nombre_employee_sous_responsabilite,
            frequence_deplacement,
            annees_depuis_la_derniere_promotion
        )
        VALUES (
            :employee_id,
            :nombre_experiences_precedentes,
            :nombre_heures_travaillees,
            :annee_experience_totale,
            :annees_dans_l_entreprise,
            :annees_dans_le_poste_actuel,
            :annees_sous_responsable_actuel,
            :niveau_hierarchique_poste,
            :revenu_mensuel,
            :augmentation_salaire_precedente,
            :heures_supplementaires,
            :nombre_participation_pee,
            :nb_formations_suivies,
            :nombre_employee_sous_responsabilite,
            :frequence_deplacement,
            :annees_depuis_la_derniere_promotion
        )
        """
    )

    insert_survey = text(
        """
        INSERT INTO raw.surveys (
            employee_id,
            code_sondage,
            eval_number,
            note_evaluation_precedente,
            note_evaluation_actuelle,
            satisfaction_employee_environnement,
            satisfaction_employee_nature_travail,
            satisfaction_employee_equipe,
            satisfaction_employee_equilibre_pro_perso
        )
        VALUES (
            :employee_id,
            :code_sondage,
            :eval_number,
            :note_evaluation_precedente,
            :note_evaluation_actuelle,
            :satisfaction_employee_environnement,
            :satisfaction_employee_nature_travail,
            :satisfaction_employee_equipe,
            :satisfaction_employee_equilibre_pro_perso
        )
        """
    )

    insert_gt = text(
        """
        INSERT INTO raw.ground_truth (employee_id, date_event, a_quitte_l_entreprise)
        VALUES (:employee_id, now(), :a_quitte_l_entreprise)
        """
    )

    # Build records
    employees_records = []
    for r in df.to_dict(orient="records"):
        employees_records.append(
            {
                "employee_external_id": int(r["id_employee"]),
                "age": r.get("age"),
                "genre": r.get("genre"),
                "statut_marital": r.get("statut_marital"),
                "ayant_enfants": r.get("ayant_enfants"),
                "niveau_education": r.get("niveau_education"),
                "domaine_etude": r.get("domaine_etude"),
                "departement": r.get("departement"),
                "poste": r.get("poste"),
                "distance_domicile_travail": r.get("distance_domicile_travail")
            }
        )

    ext_ids = df["id_employee"].astype(int).unique().tolist()

    # RUN
    with engine.begin() as conn:
        if refresh:
            logger.info("Mode --refresh: purge raw tables (snapshots/surveys/ground_truth)")
            conn.execute(text("TRUNCATE TABLE raw.employee_snapshots RESTART IDENTITY CASCADE;"))
            conn.execute(text("TRUNCATE TABLE raw.surveys RESTART IDENTITY CASCADE;"))
            conn.execute(text("TRUNCATE TABLE raw.ground_truth RESTART IDENTITY CASCADE;"))
            # raw.employees: on garde l'upsert

        # 1) upsert employees
        conn.execute(upsert_employees, employees_records)

        # 2) map ext -> id
        emp_rows = conn.execute(select_emp_map, {"ext_ids": ext_ids}).mappings().all()
        ext_to_id = {int(r["employee_external_id"]): int(r["id"]) for r in emp_rows}

        snapshots_records = []
        surveys_records = []
        gt_records = []

        for r in df.to_dict(orient="records"):
            ext = int(r["id_employee"])
            employee_id = ext_to_id.get(ext)
            if not employee_id:
                continue

            snapshots_records.append(
                {
                    "employee_id": employee_id,
                    "nombre_experiences_precedentes": pick(r, "nombre_experiences_precedentes"),
                    "nombre_heures_travaillees": pick(r, "nombre_heures_travaillees", "nombre_heures_travailless"),
                    "annee_experience_totale": pick(r, "annee_experience_totale"),
                    "annees_dans_l_entreprise": pick(r, "annees_dans_l_entreprise"),
                    "annees_dans_le_poste_actuel": pick(r, "annees_dans_le_poste_actuel"),
                    "annees_sous_responsable_actuel": pick(r, "annees_sous_responsable_actuel", "annes_sous_responsable_actuel"),
                    "niveau_hierarchique_poste": pick(r, "niveau_hierarchique_poste"),
                    "revenu_mensuel": pick(r, "revenu_mensuel"),
                    "augmentation_salaire_precedente": pick(r, "augmentation_salaire_precedente", "augementation_salaire_precedente"),
                    "heures_supplementaires": pick(r, "heures_supplementaires", "heure_supplementaires"),
                    "nombre_participation_pee": pick(r, "nombre_participation_pee"),
                    "nb_formations_suivies": pick(r, "nb_formations_suivies"),
                    "nombre_employee_sous_responsabilite": pick(r, "nombre_employee_sous_responsabilite"),
                    "frequence_deplacement": pick(r, "frequence_deplacement"),
                    "annees_depuis_la_derniere_promotion": pick(r, "annees_depuis_la_derniere_promotion")
                }
            )

            surveys_records.append(
                {
                    "employee_id": employee_id,
                    "code_sondage": pick(r, "code_sondage"),
                    "eval_number": pick(r, "eval_number"),
                    "note_evaluation_precedente": pick(r, "note_evaluation_precedente"),
                    "note_evaluation_actuelle": pick(r, "note_evaluation_actuelle"),
                    "satisfaction_employee_environnement": pick(r, "satisfaction_employee_environnement"),
                    "satisfaction_employee_nature_travail": pick(r, "satisfaction_employee_nature_travail"),
                    "satisfaction_employee_equipe": pick(r, "satisfaction_employee_equipe"),
                    "satisfaction_employee_equilibre_pro_perso": pick(r, "satisfaction_employee_equilibre_pro_perso")
                }
            )

            gt_records.append(
                {
                    "employee_id": employee_id,
                    "a_quitte_l_entreprise": yesno_to_int(pick(r, "a_quitte_l_entreprise"))
                }
            )

        if snapshots_records:
            conn.execute(insert_snapshot, snapshots_records)
        if surveys_records:
            conn.execute(insert_survey, surveys_records)
        if gt_records:
            conn.execute(insert_gt, gt_records)

    print("Seed terminé depuis les 3 CSV")
    print(f"   employees: {len(employees_records)} (upsert)")
    print(f"   snapshots: {len(snapshots_records)}")
    print(f"   surveys:   {len(surveys_records)}")
    print(f"   ground_truth: {len(gt_records)}")

if __name__ == "__main__":
    main()
