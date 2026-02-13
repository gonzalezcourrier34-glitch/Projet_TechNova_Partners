"""
TechNova Partners – API FastAPI (Turnover Prediction)

Ce module expose l'API HTTP du projet TechNova Partners. L'API permet :
- de prédire le risque de départ (turnover) d'un employé via un modèle ML,
- de consulter l'état de santé du service (health / ready),
- de journaliser (logguer) les requêtes et prédictions en base de données.

Principes d'architecture (MLOps)
- Le modèle ML est chargé une seule fois au démarrage (lifespan).
- La prédiction "production" se fait via les features déjà préparées en base (tables `clean.*`).
- Les appels de prédiction sont enregistrés (request + prediction) afin d'assurer la traçabilité.

Sécurité
Tous les endpoints (sauf la racine `/`) sont protégés par une API Key via le header :
`X-API-Key: <API_KEY>`

Version du modèle
`MODEL_VERSION` est renvoyé dans `/ready` et stocké dans les logs de prédiction.
"""
from __future__ import annotations
from time import perf_counter

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from domain.domain import ModelRequest, ModelResponse
from service.technova_service import TechNovaService
from app.security import verify_api_key

from contextlib import asynccontextmanager

from .database import get_db
from .models import PredictionRequest, Prediction

MODEL_VERSION = "xgb_v1"

service_singleton: TechNovaService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestion du cycle de vie de l'application (startup / shutdown).

    Au démarrage :
    - Instancie `TechNovaService` une seule fois (singleton)
    - Charge le modèle ML et le seuil de décision (threshold)

    À l'arrêt :
    - Libère la référence du service (bonne pratique pour un clean shutdown)

    Pourquoi ?
    Charger le modèle à chaque requête serait trop coûteux et non scalable.
    """

    global service_singleton
    service_singleton = TechNovaService()   # charge modèle + seuil une seule fois
    yield
    service_singleton = None

app = FastAPI(lifespan=lifespan)

def get_service() -> TechNovaService:
    assert service_singleton is not None, "Service not initialized"
    return service_singleton

#accessible via "/" pour rediriger vers la doc interactive, mais pas dans la doc elle-même
@app.get("/", include_in_schema=False)
def root():
    """
    Route racine.

    Redirige vers la documentation interactive Swagger (`/docs`).
    Cette route n'apparaît pas dans le schéma OpenAPI (`include_in_schema=False`).
    """
    return RedirectResponse(url="/docs")

# endpoint de santé pour monitoring
@app.get("/health", dependencies=[Depends(verify_api_key)])
def health():
    """
    Endpoint de santé (healthcheck).

    Objectif :
    - Vérifier que l'API répond (service up)
    - Utilisé pour monitoring simple

    Sécurité :
    - Protégé par API Key

    Returns
    -------
    dict
        Statut minimal indiquant que l'API est disponible.
    """

    return {"status": "ok"}
from sqlalchemy import text

# endpoint de readiness pour vérifier que le modèle est chargé et que la DB est accessible
@app.get("/ready", dependencies=[Depends(verify_api_key)])
def ready(
    db: Session = Depends(get_db),
    service: TechNovaService = Depends(get_service),
):
    """
    Endpoint de readiness (prêt à servir).

    Vérifie :
    1) que le modèle ML est chargé (service.model non nul)
    2) que la base de données est accessible (SELECT 1)
    3) que la table `clean.ml_features_employees` existe et est requêtable

    Pourquoi ?
    Un service peut être "up" (health ok) sans être "ready" :
    - DB non accessible
    - table clean non présente
    - modèle non chargé

    Parameters
    ----------
    db : sqlalchemy.orm.Session
        Session de base de données injectée par FastAPI.
    service : TechNovaService
        Service ML (singleton) injecté par FastAPI.

    Returns
    -------
    dict
        Informations de readiness + version du modèle.

    Raises
    ------
    fastapi.HTTPException
        503 si DB non prête, table clean absente, ou modèle non chargé.
    """
    if service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB not ready: {e}") from e
    try:
        db.execute(text("SELECT 1 FROM clean.ml_features_employees LIMIT 1"))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Clean table not ready: {e}") from e

    return {"status": "ready", "model_version": MODEL_VERSION}

#prédiction via ID employé — mode recommandé pour la production
@app.post("/predict/by-id/{employee_id}", response_model=ModelResponse, dependencies=[Depends(verify_api_key)])
def predict_by_employee_id(
    employee_id: int,
    db: Session = Depends(get_db),
    service: TechNovaService = Depends(get_service),
) -> ModelResponse:
    """
    Prédiction par identifiant employé (mode production).

    Fonctionnement :
    - L'API récupère les features déjà préparées depuis `clean.ml_features_employees`
    - Le modèle calcule une probabilité de départ
    - La réponse inclut :
      - la probabilité
      - la décision binaire (will_leave) basée sur un seuil `threshold`

    Traçabilité :
    - Enregistre un `PredictionRequest` (contexte de requête)
    - Enregistre un `Prediction` (résultat, proba, seuil, latence, version modèle)

    Parameters
    ----------
    employee_id : int
        Identifiant de l'employé utilisé pour rechercher la ligne "clean" la plus récente.
    db : sqlalchemy.orm.Session
        Session SQLAlchemy.
    service : TechNovaService
        Service ML déjà initialisé (modèle chargé).

    Returns
    -------
    ModelResponse
        Résultat de prédiction.

    Raises
    ------
    fastapi.HTTPException
        - 404 si aucune ligne clean n'existe pour cet employee_id
        - 500 pour toute autre erreur inattendue
    """
    try:
        t0 = perf_counter()
        response = service.predict_from_clean(db=db, employee_id=employee_id)
        latency_ms = int((perf_counter() - t0) * 1000)

        req = PredictionRequest(
            employee_id=employee_id,
            payload_json={"mode": "by_employee_id", "employee_id": employee_id},
        )
        db.add(req)
        db.flush()

        db.add(
            Prediction(
                request_id=req.id,
                model_version=MODEL_VERSION,
                predicted_class=int(response.will_leave),
                predicted_proba=float(response.turnover_probability),
                threshold_used=float(service.threshold),
                latency_ms=latency_ms,
            )
        )
        db.commit()
        return response

    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e)) from e

#endpoint de prédiction via features brutes — utile pour tests et debug, mais pas recommandé pour la production
@app.post("/predict/by-features", response_model=ModelResponse, dependencies=[Depends(verify_api_key)])
def predict_by_features(
    payload: ModelRequest,
    db: Session = Depends(get_db),
    service: TechNovaService = Depends(get_service),
) -> ModelResponse:
    """
    Prédiction par payload de features (mode debug / scoring direct).

    Cas d'usage :
    - Tests automatisés
    - Démonstration
    - Scoring d'un nouvel employé lorsque les features sont déjà au format attendu

    Important :
    - Ce endpoint ne fait pas de preprocessing (les features doivent être prêtes)
    - Il est moins "production-ready" que la prédiction par ID si un pipeline clean existe

    Traçabilité :
    - Enregistre la requête et le résultat en base (comme le mode by-id)

    Parameters
    ----------
    payload : ModelRequest
        Ensemble complet des features attendues par le modèle.
    db : sqlalchemy.orm.Session
        Session SQLAlchemy.
    service : TechNovaService
        Service ML initialisé.

    Returns
    -------
    ModelResponse
        Résultat de prédiction.

    Raises
    ------
    fastapi.HTTPException
        422 si les features sont manquantes / inattendues / invalides.
    """
    try:
        t0 = perf_counter()
        response = service.predict_from_payload(request=payload)
        latency_ms = int((perf_counter() - t0) * 1000)

        req = PredictionRequest(
            payload_json={"mode": "by_features", "payload": payload.model_dump()},
        )
        db.add(req)
        db.flush()

        db.add(
            Prediction(
                request_id=req.id,
                model_version=MODEL_VERSION,
                predicted_class=int(response.will_leave),
                predicted_proba=float(response.turnover_probability),
                threshold_used=float(service.threshold),
                latency_ms=latency_ms,
            )
        )
        db.commit()
        return response

    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(e)) from e

#endpoint pour récupérer les dernières prédictions stockées en base, avec pagination
@app.get("/predictions/latest", dependencies=[Depends(verify_api_key)])
def latest_predictions(
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """
    Retourne les dernières prédictions enregistrées en base.

    L'endpoint récupère :
    - `Prediction` : résultat ML (classe, proba, seuil, latence, version)
    - `PredictionRequest` : contexte / payload de la requête

    Pourquoi ?:
    - audit / traçabilité
    - monitoring simple
    - affichage dans un dashboard

    Parameters
    ----------
    limit : int, default=20
        Nombre maximal de prédictions retournées (tri par date décroissante).
    db : sqlalchemy.orm.Session
        Session SQLAlchemy.

    Returns
    -------
    list[dict]
        Liste des prédictions récentes, avec métadonnées utiles (timestamps, proba, payload, etc.).
    """
    rows = (
        db.query(Prediction, PredictionRequest)
        .join(PredictionRequest, Prediction.request_id == PredictionRequest.id)
        .order_by(desc(Prediction.created_at))
        .limit(limit)
        .all()
    )

    return [
        {
            "prediction_id": p.id,
            "request_id": p.request_id,
            "created_at": p.created_at.isoformat(),
            "predicted_class": int(p.predicted_class),
            "predicted_proba": round(float(p.predicted_proba), 4),
            "threshold_used": float(p.threshold_used),
            "model_version": p.model_version,
            "latency_ms": p.latency_ms,
            "payload": r.payload_json,
        }
        for p, r in rows
    ]
