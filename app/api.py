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
    return RedirectResponse(url="/docs")

# endpoint de santé pour monitoring
@app.get("/health", dependencies=[Depends(verify_api_key)])
def health():
    return {"status": "ok"}
from sqlalchemy import text

@app.get("/ready", dependencies=[Depends(verify_api_key)])
def ready(
    db: Session = Depends(get_db),
    service: TechNovaService = Depends(get_service),
):
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
