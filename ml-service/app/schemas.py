"""Request / response models (input validation). Allowed values match what the models were trained on."""
from datetime import date as Date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ZONES = ["Cantt", "DHA Phase 5", "Data Ganj Bakhsh", "Faisal Town", "Gulberg", "Johar Town", "Model Town",
         "Samanabad", "Shahdara", "Thokar Niaz Baig", "Township", "Wapda Town"]
Zone = Literal[tuple(ZONES)]  # type: ignore[valid-type]


# ----------------------------------------------------------------------------- Model 2: health risk
class HealthRiskRequest(BaseModel):
    aqi: float = Field(..., ge=0, le=500, description="Current AQI (0-500)")
    age: int = Field(..., ge=0, le=110)
    gender: Literal["male", "female"]
    condition: Literal["none", "asthma", "respiratory_disease", "heart_disease", "pregnancy"] = "none"
    outdoor_hours: float = Field(..., ge=0, le=24, description="Hours planned outdoors today")
    activity_level: Literal["sedentary", "light", "moderate", "vigorous"]
    mask_use: Literal["none", "cloth", "surgical", "N95"] = "none"

    model_config = {"json_schema_extra": {"example": {"aqi": 280, "age": 8, "gender": "male", "condition": "asthma",
                                                       "outdoor_hours": 2, "activity_level": "moderate", "mask_use": "none"}}}


class HealthRiskResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    risk_tier: str
    probabilities: Dict[str, float]
    precaution: str
    aqi_category: str
    model_name: str
    model_version: str


# ----------------------------------------------------------------------------- Model 3: vehicle
class VehicleRequest(BaseModel):
    vehicle_type: Literal["motorcycle", "rickshaw", "car", "van_pickup", "bus", "truck"]
    make: str = Field(..., min_length=1, max_length=40)
    model: str = Field(..., min_length=1, max_length=60)
    year: int = Field(..., ge=1980, le=2027)
    fuel_type: Literal["petrol", "diesel", "cng", "lpg", "hybrid"]
    engine_cc: int = Field(..., gt=0, le=20000)
    weight_kg: float = Field(..., gt=0, le=60000)
    odometer_km: float = Field(..., ge=0, le=3_000_000)
    months_since_service: float = Field(..., ge=0, le=240)
    months_since_smoke_test: float = Field(..., ge=0, le=240)
    emission_standard: Literal["pre-Euro", "Euro-2", "Euro-4"] = "Euro-2"

    model_config = {"json_schema_extra": {"example": {
        "vehicle_type": "car", "make": "Toyota", "model": "Corolla", "year": 2012, "fuel_type": "petrol", "engine_cc": 1600,
        "weight_kg": 1250, "odometer_km": 165000, "months_since_service": 14, "months_since_smoke_test": 20,
        "emission_standard": "Euro-2"}}}


class VehicleResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    emission_tier: str
    tier_probabilities: Dict[str, float]
    green_score: float
    green_rating: str
    model_name: str
    model_version: str


# ----------------------------------------------------------------------------- Model 4: route risk
class RouteRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    month: int = Field(..., ge=1, le=12)
    origin_zone: Zone
    dest_zone: Zone
    travel_mode: Literal["car", "bike", "walk"]
    distance_km: float = Field(..., gt=0, le=100)
    duration_min: float = Field(..., gt=0, le=600)
    avg_aqi: float = Field(..., ge=0, le=500)
    max_aqi: float = Field(..., ge=0, le=500)
    aqi_std: float = Field(..., ge=0, le=300)
    pct_segments_aqi_over_150: float = Field(..., ge=0, le=100)
    traffic_density: float = Field(..., ge=0, le=1)
    pct_arterial_roads: float = Field(..., ge=0, le=100)
    n_signals: int = Field(..., ge=0, le=200)
    pct_near_industrial: float = Field(..., ge=0, le=100)
    green_cover_pct: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0, le=40)
    temperature: float = Field(..., ge=-10, le=55)
    humidity: float = Field(..., ge=0, le=100)
    user_group: Literal["general", "sensitive"] = "general"

    model_config = {"json_schema_extra": {"example": {
        "hour": 8, "month": 11, "origin_zone": "Johar Town", "dest_zone": "Gulberg", "travel_mode": "bike", "distance_km": 9.5,
        "duration_min": 32, "avg_aqi": 260, "max_aqi": 340, "aqi_std": 40, "pct_segments_aqi_over_150": 95,
        "traffic_density": 0.8, "pct_arterial_roads": 55, "n_signals": 9, "pct_near_industrial": 10, "green_cover_pct": 8,
        "wind_speed": 1.2, "temperature": 14, "humidity": 70, "user_group": "sensitive"}}}


class RouteResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    route_label: str
    probabilities: Dict[str, float]
    model_name: str
    model_version: str


# ----------------------------------------------------------------------------- Model 1: AQI forecast
class DailyRecord(BaseModel):
    date: Date
    pm25: float = Field(..., ge=0, le=1000)
    pm10: float = Field(..., ge=0, le=2000)
    no2: float = Field(..., ge=0, le=1000)
    so2: float = Field(..., ge=0, le=1000)
    co: float = Field(..., ge=0, le=100)
    o3: float = Field(..., ge=0, le=1000)
    temperature: float = Field(..., ge=-30, le=60)
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0, le=60)
    wind_dir: float = Field(..., ge=0, le=360)
    pressure: float = Field(..., ge=800, le=1100)
    rainfall: float = Field(..., ge=0, le=500)


class AQIForecastRequest(BaseModel):
    zone: Zone
    history: List[DailyRecord] = Field(..., min_length=15, max_length=400,
                                       description="At least 15 consecutive daily records, newest last")
    horizons: List[Literal[1, 7]] = Field(default=[1, 7])

    @field_validator("history")
    @classmethod
    def consecutive_days(cls, v):
        days = sorted(r.date for r in v)
        for a, b in zip(days, days[1:]):
            if (b - a).days != 1:
                raise ValueError("history must contain consecutive days with no gaps or duplicates")
        return v


class AQIForecastItem(BaseModel):
    horizon_days: int
    target_date: Date
    pred_pm25: float
    pred_aqi: int
    category: str


class AQIForecastResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    zone: str
    based_on_date: Date
    forecasts: List[AQIForecastItem]
    model_name: str
    model_version: str
    test_rmse_pm25: Dict[str, Optional[float]]
