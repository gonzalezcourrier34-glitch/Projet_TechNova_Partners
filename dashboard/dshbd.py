import os
import streamlit as st
import requests
from typing import Any, Dict, List, Tuple

# charge .env quand le dashboard est lancé via VS Code / bouton
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

st.set_page_config(page_title="TechNova Dashboard", layout="centered")

# Ce script Streamlit affiche un dashboard simple pour interagir avec l'API FastAPI et visualiser les prédictions.
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
API_PREDICT_BY_ID = f"{API_BASE}/predict/by-id"
API_PREDICT_DEBUG = f"{API_BASE}/predict/debug"
API_LATEST = f"{API_BASE}/predictions/latest"
API_ROOT = f"{API_BASE}/"

# API key (le dashboard est un client HTTP, il envoie seulement le header)
API_KEY = os.getenv("API_KEY")
DEFAULT_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# La fonction safe_request encapsule les appels à l'API avec une gestion d'erreur simple,
# affichant un message d'erreur dans le dashboard en cas de problème de connexion ou de timeout.
from dashboard.feature_schema import FEATURES

def safe_request(method: str, url: str, **kwargs):
    try:
        # injection automatique du header X-API-Key
        headers = kwargs.pop("headers", {}) or {}
        merged_headers = {**DEFAULT_HEADERS, **headers}

        # appel de l'API avec un timeout de 10 secondes
        return requests.request(
            method,
            url,
            timeout=10,
            headers=merged_headers,
            **kwargs
        )
    except requests.RequestException as e:
        st.error(f"Impossible de joindre l'API : {e}")
        return None

# La fonction validate_inputs vérifie que les valeurs saisies par l'utilisateur respectent les contraintes définies dans FEATURES,
# telles que les types de données, les plages de valeurs, et les choix pour les variables catégorielles.
def validate_inputs(values: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    for f in FEATURES:
        v = values.get(f.key)
        if f.required and (v is None or (isinstance(v, str) and v.strip() == "")):
            errors.append(f"{f.label} est requis.")
            continue
        if f.dtype in ("int", "float"):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errors.append(f"{f.label} doit être un nombre.")
                continue
            if f.dtype == "int":
                if isinstance(v, float) and not v.is_integer():
                    errors.append(f"{f.label} doit être un entier.")
                    continue
            if f.min is not None and v < f.min:
                errors.append(f"{f.label} doit être ≥ {f.min}.")
            if f.max is not None and v > f.max:
                errors.append(f"{f.label} doit être ≤ {f.max}.")
        elif f.dtype == "cat":
            if not isinstance(v, str):
                errors.append(f"{f.label} doit être une chaîne.")
                continue
            if f.choices is not None and v not in f.choices:
                errors.append(f"{f.label} doit être dans {f.choices}.")

    return (len(errors) == 0), errors

# Le reste du code Streamlit affiche le dashboard avec deux onglets : un pour faire des prédictions,
# et un pour visualiser l'historique des prédictions enregistrées dans la base de données via l'API.
st.title("TechNova – Dashboard")
st.caption("Interface Streamlit connectée à une API FastAPI et une base")

# Cadre gestion de l'API
with st.expander("Configuration API", expanded=False):
    st.write(f"API utilisée : {API_BASE}")
    st.write(f"API_KEY chargée : {'oui' if API_KEY else 'non'}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Tester l’API"):
            r = safe_request("GET", API_ROOT)
            if r is None:
                st.stop()
            if r.ok:
                st.success("API accessible")
            else:
                st.error(f"Erreur API ({r.status_code})")
                try:
                    st.write(r.json())
                except Exception:
                    st.write(r.text)

    with c2:
        st.write("L’URL peut être modifiée via la variable API_BASE")

tab_predict, tab_history = st.tabs(["Prédire", "Historique"])

# ONGLET PRÉDICTION
with tab_predict:
    st.subheader("Prédiction")
    # 2 modes de validation : par ID employé (recommandé) ou par features (debug)
    mode = st.radio(
        "Mode",
        ["Par ID employé (prod, clean)", "Par features (debug)"],
        horizontal=True,
    )

    # En mode ID, l'utilisateur saisit un employee_external_id,
    # et l'API lit les features correspondantes dans clean.ml_features_employees pour faire la prédiction.
    if mode == "Par ID employé (prod, clean)":
        st.caption("L’API lit les features dans clean.ml_features_employees via employee_external_id.")

        employee_external_id = st.number_input("employee_external_id", min_value=1, step=1, value=1)
        run_pred = st.button("Lancer la prédiction (ID)")

        if run_pred:
            url = f"{API_PREDICT_BY_ID}/{int(employee_external_id)}"
            response = safe_request("POST", url)
            if response is None:
                st.stop()
            if response.ok:
                result = response.json()
                st.success("Prédiction réalisée")
                st.write("Employé :", result.get("employee_id"))
                st.write("Départ prédit :", result.get("will_leave"))
                st.write("Probabilité :", round(result.get("turnover_probability", 0), 4))
            else:
                st.error(f"Erreur API ({response.status_code})")
                try:
                    st.write(response.json())
                except Exception:
                    st.write(response.text)

    else:
        # En mode features, l'utilisateur remplit un formulaire avec toutes les features nécessaires à la prédiction,
        st.caption("Mode debug: envoi d’un payload complet de features à l’API.")

        # 2 options d'affichage permettent de choisir entre un affichage compact (6 champs par ligne)
        # et un affichage avec les noms techniques des features (keys).
        col1, col2 = st.columns(2)
        with col1:
            compact = st.checkbox("Affichage compact", value=True)
        with col2:
            show_keys = st.checkbox("Afficher les noms techniques", value=False)

        values_by_key: Dict[str, Any] = {}

        for idx, f in enumerate(FEATURES):
            label = f.label if not show_keys else f"{f.label} ({f.key})"

            if f.dtype == "int":
                default = int(f.min) if f.min is not None else 0
                v = st.number_input(
                    label,
                    min_value=int(f.min) if f.min is not None else None,
                    max_value=int(f.max) if f.max is not None else None,
                    value=default,
                    step=1,
                    key=f"feat_{f.key}"
                )
                values_by_key[f.key] = int(v)

            elif f.dtype == "float":
                default = float(f.min) if f.min is not None else 0.0
                v = st.number_input(
                    label,
                    min_value=float(f.min) if f.min is not None else None,
                    max_value=float(f.max) if f.max is not None else None,
                    value=default,
                    step=0.1,
                    key=f"feat_{f.key}"
                )
                values_by_key[f.key] = float(v)

            else:
                if f.choices:
                    values_by_key[f.key] = st.selectbox(label, f.choices, key=f"feat_{f.key}")
                else:
                    values_by_key[f.key] = st.text_input(label, key=f"feat_{f.key}").strip()

            if compact and (idx + 1) % 6 == 0:
                st.divider()

        c1, c2 = st.columns(2)
        with c1:
            run_pred = st.button("Lancer la prédiction (features)")
        with c2:
            if st.button("Réinitialiser"):
                st.rerun()

        # Lors du lancement de la prédiction, les valeurs saisies sont d'abord validées via la fonction validate_inputs,
        if run_pred:
            ok, errors = validate_inputs(values_by_key)
            if not ok:
                st.error("Erreurs dans le formulaire")
                for e in errors:
                    st.write(e)
                st.stop()

            # puis envoyées à l'API via la fonction safe_request, et les résultats sont affichés dans le dashboard.
            response = safe_request("POST", API_PREDICT_DEBUG, json=values_by_key)
            if response is None:
                st.stop()
            if response.ok:
                result = response.json()
                st.success("Prédiction réalisée")
                st.write("Départ prédit :", result.get("will_leave"))
                st.write("Probabilité :", round(result.get("turnover_probability", 0), 4))
            else:
                st.error(f"Erreur API ({response.status_code})")
                try:
                    st.write(response.json())
                except Exception:
                    st.write(response.text)

# ONGLET HISTORIQUE
with tab_history:
    st.subheader("Historique des prédictions")

    limit = st.slider("Nombre de lignes", 5, 200, 20)

    if st.button("Rafraîchir"):
        # L'onglet historique affiche les dernières prédictions enregistrées dans la base de données via l'API,
        # avec un slider pour choisir le nombre de lignes à afficher.
        response = safe_request("GET", API_LATEST, params={"limit": limit})
        if response and response.ok:
            rows = response.json()
            if not rows:
                st.info("Aucune prédiction enregistrée.")
            for row in rows:
                title = f'{row.get("created_at", "")} | proba={row.get("predicted_proba", "")}'
                with st.expander(title):
                    st.json(row)
        else:
            st.error("Impossible de récupérer l’historique.")
            if response is not None:
                st.write(response.text)
