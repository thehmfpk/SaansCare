"""Run from the ml-service folder:  pytest -q      (uses the real models in <repo>/models)"""
import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.pop("ML_API_KEY", None)
from app.main import app  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


HEALTH = {"aqi": 280, "age": 8, "gender": "male", "condition": "asthma", "outdoor_hours": 2, "activity_level": "moderate", "mask_use": "none"}
VEHICLE = {"vehicle_type": "car", "make": "Toyota", "model": "Corolla", "year": 2012, "fuel_type": "petrol", "engine_cc": 1600,
           "weight_kg": 1250, "odometer_km": 165000, "months_since_service": 14, "months_since_smoke_test": 20, "emission_standard": "Euro-2"}
ROUTE = {"hour": 8, "month": 11, "origin_zone": "Johar Town", "dest_zone": "Gulberg", "travel_mode": "bike", "distance_km": 9.5,
         "duration_min": 32, "avg_aqi": 260, "max_aqi": 340, "aqi_std": 40, "pct_segments_aqi_over_150": 95, "traffic_density": 0.8,
         "pct_arterial_roads": 55, "n_signals": 9, "pct_near_industrial": 10, "green_cover_pct": 8, "wind_speed": 1.2,
         "temperature": 14, "humidity": 70, "user_group": "sensitive"}


def test_health_and_catalogue(client):
    r = client.get("/health"); assert r.status_code == 200 and r.json()["status"] == "ok"
    m = client.get("/models").json(); assert set(m) == {"health_risk", "vehicle_emission", "route_risk", "aqi_forecast"}


def test_health_risk_orders_sensibly(client):
    high = client.post("/predict/health-risk", json=HEALTH).json()
    low = client.post("/predict/health-risk", json={**HEALTH, "aqi": 40, "age": 35, "condition": "none", "activity_level": "light"}).json()
    assert high["risk_tier"] in ("High", "Very High") and low["risk_tier"] == "Low"
    assert abs(sum(high["probabilities"].values()) - 1) < 1e-3 and high["precaution"]


def test_vehicle(client):
    r = client.post("/predict/vehicle", json=VEHICLE).json()
    assert r["emission_tier"] in ("Low", "Medium", "High") and 0 <= r["green_score"] <= 100
    clean = client.post("/predict/vehicle", json={**VEHICLE, "make": "Toyota", "model": "Prius", "fuel_type": "hybrid", "year": 2022,
                                                   "engine_cc": 1800, "weight_kg": 1380, "odometer_km": 30000, "months_since_service": 2,
                                                   "months_since_smoke_test": 3, "emission_standard": "Euro-4"}).json()
    assert clean["green_score"] > r["green_score"]


def test_route_winter_vs_summer(client):
    winter = client.post("/predict/route-risk", json=ROUTE).json()
    summer = client.post("/predict/route-risk", json={**ROUTE, "month": 7, "avg_aqi": 70, "max_aqi": 110, "pct_segments_aqi_over_150": 0,
                                                       "travel_mode": "car", "user_group": "general", "temperature": 34}).json()
    assert winter["route_label"] == "Avoid" and summer["route_label"] == "Safe"


def _history(zone="Gulberg", days=20):
    df = pd.read_csv(REPO / "data" / "aqi_forecasting.csv")
    d = df[df.zone == zone].sort_values("date").copy()
    num = d.select_dtypes("number").columns
    d[num] = d[num].interpolate(limit_direction="both")          # the raw data has sensor gaps; the backend must send complete days
    d = d.tail(days)
    cols = ["date", "pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", "wind_speed", "wind_dir", "pressure", "rainfall"]
    return d[cols].to_dict("records")


def test_aqi_forecast(client):
    r = client.post("/predict/aqi-forecast", json={"zone": "Gulberg", "history": _history()}).json()
    assert [f["horizon_days"] for f in r["forecasts"]] == [1, 7]
    assert all(f["pred_pm25"] > 0 and 0 <= f["pred_aqi"] <= 500 for f in r["forecasts"])


def test_validation_errors(client):
    assert client.post("/predict/health-risk", json={**HEALTH, "condition": "flu"}).status_code == 422
    assert client.post("/predict/health-risk", json={**HEALTH, "aqi": 900}).status_code == 422
    assert client.post("/predict/aqi-forecast", json={"zone": "Gulberg", "history": _history(days=10)}).status_code == 422
    h = _history(); del h[5]
    assert client.post("/predict/aqi-forecast", json={"zone": "Gulberg", "history": h}).status_code == 422


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("ML_API_KEY", "secret")
    assert client.post("/predict/health-risk", json=HEALTH).status_code == 401
    assert client.post("/predict/health-risk", json=HEALTH, headers={"X-API-Key": "secret"}).status_code == 200
