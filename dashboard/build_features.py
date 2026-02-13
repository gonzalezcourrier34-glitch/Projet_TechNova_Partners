from __future__ import annotations

from typing import Any, Dict
import math


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _i(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return int(default)


def compute_engineered(values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reproduit à l'identique Transform.feature_engineering(df) mais sur un dict.

    IMPORTANT
    - Nécessite 4 champs de satisfaction (comme dans l'ETL):
      satisfaction_employee_environnement
      satisfaction_employee_nature_travail
      satisfaction_employee_equipe
      satisfaction_employee_equilibre_pro_perso
    - Le dashboard peut cacher ces champs dans l'UI si tu veux,
      mais ils doivent exister pour un calcul strictement identique.
    """
    d = dict(values)

    sat_env = _f(d.get("satisfaction_employee_environnement"))
    sat_nat = _f(d.get("satisfaction_employee_nature_travail"))
    sat_eqp = _f(d.get("satisfaction_employee_equipe"))
    sat_wlb = _f(d.get("satisfaction_employee_equilibre_pro_perso"))

    d["satisfaction_moyenne"] = (sat_env + sat_nat + sat_eqp + sat_wlb) / 4.0

    pee = _f(d.get("nombre_participation_pee"))
    anc = _f(d.get("annees_dans_l_entreprise"))
    d["nonlineaire_participation_pee"] = pee / (pee + anc + 1.0)

    hs = _f(d.get("heures_supplementaires"))
    salaire = _f(d.get("revenu_mensuel"))
    d["ratio_heures_sup_salaire"] = hs / (salaire + 1.0)

    dist = _f(d.get("distance_domicile_travail"))
    # d/(d+10)/(d+10) = d / (d+10)^2
    denom = (dist + 10.0)
    d["nonlinaire_charge_contrainte"] = hs * dist / (denom * denom)

    d["nonlinaire_surmenage_insatisfaction"] = hs * (1.0 - _f(d["satisfaction_moyenne"]))

    age = _i(d.get("age"))
    d["jeune_surcharge"] = int((age < 30) and (hs == 1.0))

    # (annees_dans_l_entreprise - annees_depuis_la_derniere_promotion)/(annees_dans_l_entreprise + 1)
    adlp = _f(d.get("annees_depuis_la_derniere_promotion"))
    d["anciennete_sans_promotion"] = (anc - adlp) / (anc + 1.0)

    nb_exp = _f(d.get("nombre_experiences_precedentes"))
    tot_exp = _f(d.get("annee_experience_totale"))
    d["mobilite_carriere"] = nb_exp / (tot_exp + 1.0)

    d["risque_global"] = (
        _f(d["ratio_heures_sup_salaire"])
        * _f(d["anciennete_sans_promotion"])
        * (1.0 - _f(d["satisfaction_moyenne"]))
    )

    return d
