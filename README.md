---
title: technova-api
sdk: docker
app_port: 7860
---

# TechNova Partners – Déploiement d’un modèle de Machine Learning

Ce projet met en œuvre le **déploiement d’un modèle de Machine Learning** de prédiction du **turnover employé** via une **API FastAPI**, connectée à une **base de données PostgreSQL**, et pilotée par un **dashboard Streamlit**.

L’objectif est de proposer une architecture **production-ready**, respectant les bonnes pratiques **MLOps** :
- séparation ingestion / préparation / inference,
- API sans logique de preprocessing,
- traçabilité des prédictions,
- tests automatisés,
- CI/CD,
- déploiement reproductible.

---
## Live API

Base URL:
https://stegonzalez-technova-api.hf.space

Swagger UI:
https://stegonzalez-technova-api.hf.space/docs

Health checks:
- /health
- /ready

---

## Architecture du projet

Projet_TechNova_Partners/
│
├── app/
│   ├── api.py              # API FastAPI (routes, orchestration)
│   ├── database.py         # Connexion DB (SQLAlchemy)
│   ├── main.py             # Launcher API + Dashboard
│   ├── models.py           # Modèles ORM
│   └── security.py         # Sécurité via API Key
│
├── artifacts/
│   ├── modele_classification_technova.joblib   # Modèle ML entraîné
│   └── threshold.json                           # Seuil de décision
│
├── dashboard/
│   ├── dshbd.py            # Dashboard Streamlit
│   └── feature_schema.py   # Schéma des features (source de vérité UI)
│
├── domain/
│   └── domain.py           # Schémas Pydantic (ModelRequest / ModelResponse)
│
├── my-postgres/
│   └── docker-compose.yml  # PostgreSQL via Docker
│
├── scripts/
│   ├── build_ml_features.py     # Création des tables clean
│   ├── create_db.py             # Création DB + tables (one-shot)
│   ├── seed_from_csv.py         # Seed optionnel depuis CSV
│   ├── seed_ml_features.py      # Nettoyage + feature engineering
│   ├── init_project_technova.py # Lance tous les scripts nécessaires
│   └── generate_docs.py         # Génération de documentation
│
├── service/
│   └── technova_service.py  # Logique ML (chargement modèle + prédiction)
│
├── tests/                   # Tests Pytest
│
├── Dockerfile
├── pyproject.toml
├── .env
├── .env.example
└── README.md

---

---

## Principe de fonctionnement

### Séparation des pipelines
- Les **données brutes** sont stockées dans des tables dédiées.
- Un **pipeline de nettoyage et de transformation** est exécuté via des scripts indépendants.
- Les données transformées sont stockées dans des tables **clean**.
- **L’API consomme uniquement les tables clean**, directement compatibles avec le modèle.

**Aucun preprocessing n’est effectué dans l’API**.

Cette approche améliore :
- la performance,
- la robustesse,
- la reproductibilité,
- la conformité aux standards MLOps.

---

## API – Endpoints

### Sécurité
Tous les endpoints sont protégés par une **API Key**, transmise via le header :
X-API-Key: <API_KEY>

---

### 🔹 Prédiction – mode production (recommandé)
POST /predict/by-id/{employee_id}

- Récupère les features depuis la table **clean**
- Effectue la prédiction
- Enregistre la requête et la réponse en base

---

### 🔹 Prédiction – mode scoring (features prêtes)
POST /predict/by-features

- Attend des **features déjà préparées au format attendu par le modèle**
- Aucun nettoyage ou feature engineering dans l’API
- Utile pour intégration externe, tests ou scoring

---

### 🔹 Logs & monitoring
GET /predictions/latest
Retourne les dernières prédictions stockées en base.
GET /health
GET /ready

- `/health` : API disponible
- `/ready` : API + base de données + modèle opérationnels

---

## Dashboard Streamlit
Le dashboard permet :
- la saisie des features métier,
- l’appel à l’API,
- la visualisation des prédictions,
- la consultation de l’historique des prédictions stockées en base.

---

## Installation & exécution (Quickstart)

### 🔹 Cloner le projet
git clone <repo_url>
cd Projet_TechNova_Partners

### 🔹 Installer les dépendances
poetry install

### 🔹 Configurer l’environnement
cp .env.example .env

### 🔹 Lancer PostgreSQL via Docker
cd my-postgres
docker-compose up -d

### 🔹 Initialiser la base (une seule fois)
poetry run python scripts/init_project_technova.py

### 🔹 Lancer l’API et le dashboard
poetry run technova

### Accès
API : http://127.0.0.1:8000
Swagger : http://127.0.0.1:8000/docs
Dashboard : http://127.0.0.1:8501

---
## Example Request

curl -X POST "https://stegonzalez-technova-api.hf.hf.space/predict/by-features" \
  -H "X-API-Key: technova-secret-2026" \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 1.2, "feature_2": 0.8}'


## Tests & Qualité

## Lancement des tests
poetry run pytest

## Couverture de tests
poetry run pytest --cov=app --cov=service

    . Couverture actuelle : > 90 %
    . Tests exécutés sur SQLite pour rapidité et portabilité
    . PostgreSQL utilisé pour le développement et la production

---

## CI/CD
CI :
    . Tests et coverage exécutés automatiquement à chaque Pull Request
    . Branche main protégée : merge bloqué si la CI échoue
CD :
    . Build de l’image Docker après merge sur main
    . Push automatique de l’image vers GitHub Container Registry (GHCR)

---

## Auteur

Projet réalisé par Stéphane Gonzalez
Formation OpenClassrooms — Data Scientist / AI Engineer
