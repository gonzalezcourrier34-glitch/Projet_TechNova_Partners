#!/bin/sh
set -e

# 1) Démarre l'API en interne
# IMPORTANT: ton FastAPI doit être "monté" sur /api pour que ça route proprement.
# -> soit tu changes tes routes pour commencer par /api
# -> soit tu ajoutes root_path, ou un include_router(prefix="/api")
poetry run uvicorn app.api:app --host 0.0.0.0 --port 8000 &

# 2) Démarre le dashboard en interne
# Adapte le chemin si ton dashboard est ailleurs.
# Exemple: dashboard/app.py
poetry run streamlit run dashboard/dshbd.py --server.address 0.0.0.0 --server.port 8501 --server.headless true &

# 3) Reverse proxy public sur 7860
nginx -g "daemon off;"
