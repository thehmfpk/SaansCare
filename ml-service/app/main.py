"""SaansCare ML service (FastAPI).

Run:   uvicorn app.main:app --reload --port 8000        (from the ml-service folder)
Docs:  http://127.0.0.1:8000/docs
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .registry import ModelRegistry, default_models_dir
from .schemas import (AQIForecastRequest, AQIForecastResponse, HealthRiskRequest, HealthRiskResponse, RouteRequest,
                      RouteResponse, VehicleRequest, VehicleResponse)

log = logging.getLogger("saanscare.ml")
logging.basicConfig(level=logging.INFO)

LOG_PATH = Path(os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl"))
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    state["registry"] = ModelRegistry(default_models_dir())
    state["started"] = datetime.now(timezone.utc).isoformat()
    log.info("Models loaded from %s in %.1fs", default_models_dir(), time.time() - t0)
    yield
    state.clear()


app = FastAPI(title="SaansCare ML Service", version="1.0.0",
              description="Prediction API for AQI forecasting, personalized health risk, vehicle emission / Green Score and route health risk.",
              lifespan=lifespan)

origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


def require_key(x_api_key: str | None = Header(default=None)):
    """If ML_API_KEY is set, the Node backend must send the same value in the X-API-Key header."""
    expected = os.environ.get("ML_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def log_prediction(endpoint: str, model_version: str, summary: dict):
    """Append one line per prediction (model version + result) so every output stays traceable. Personal inputs are not stored."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "endpoint": endpoint,
                                "model_version": model_version, **summary}, default=str) + "\n")
    except OSError as e:                                   # logging must never break a prediction
        log.warning("could not write prediction log: %s", e)


def reg() -> ModelRegistry:
    return state["registry"]


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "models_loaded": sorted(reg().describe().keys()), "started": state.get("started")}


@app.get("/models", tags=["system"], dependencies=[Depends(require_key)])
def models():
    """Model names, versions, input features and test metrics (used by the Admin 'model monitor')."""
    return reg().describe()


@app.post("/predict/health-risk", response_model=HealthRiskResponse, tags=["predict"], dependencies=[Depends(require_key)])
def predict_health(req: HealthRiskRequest):
    out = reg().predict_health(req)
    log_prediction("health-risk", out["model_version"], {"aqi": req.aqi, "risk_tier": out["risk_tier"]})
    return out


@app.post("/predict/vehicle", response_model=VehicleResponse, tags=["predict"], dependencies=[Depends(require_key)])
def predict_vehicle(req: VehicleRequest):
    out = reg().predict_vehicle(req)
    log_prediction("vehicle", out["model_version"], {"vehicle_type": req.vehicle_type, "emission_tier": out["emission_tier"],
                                                      "green_score": out["green_score"]})
    return out


@app.post("/predict/route-risk", response_model=RouteResponse, tags=["predict"], dependencies=[Depends(require_key)])
def predict_route(req: RouteRequest):
    out = reg().predict_route(req)
    log_prediction("route-risk", out["model_version"], {"origin": req.origin_zone, "dest": req.dest_zone, "mode": req.travel_mode,
                                                         "route_label": out["route_label"]})
    return out


@app.post("/predict/aqi-forecast", response_model=AQIForecastResponse, tags=["predict"], dependencies=[Depends(require_key)])
def predict_aqi(req: AQIForecastRequest):
    out = reg().forecast_aqi(req)
    log_prediction("aqi-forecast", out["model_version"], {"zone": req.zone, "forecasts": out["forecasts"]})
    return out
