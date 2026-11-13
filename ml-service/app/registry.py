"""Loads the four trained models once and turns validated requests into predictions.

The feature code below mirrors the notebooks exactly (same column names, same formulas), so the saved
scikit-learn pipelines receive precisely the inputs they were trained on.
"""
import json
import os
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

AQI_BANDS = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
_BP = [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
       (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)]


def default_models_dir() -> Path:
    env = os.environ.get("MODELS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "models"      # <repo>/models  (app -> ml-service -> repo)


def pm25_to_aqi(pm: float) -> int:
    """US EPA PM2.5 -> AQI (2024 breakpoints), same as the training notebooks."""
    c = np.floor(max(pm, 0.0) * 10) / 10
    if c > 325.4:
        return 500
    for lo, hi, ilo, ihi in _BP:
        if lo <= c <= hi:
            return int(round((ihi - ilo) / (hi - lo) * (c - lo) + ilo))
    return 0


def aqi_category(aqi: float) -> str:
    return AQI_BANDS[int(np.digitize(aqi, [50, 100, 150, 200, 300]))]


def green_rating(score: float) -> str:
    """Presentation label for the certificate (display logic only; not part of the trained model)."""
    return "Excellent" if score >= 75 else "Good" if score >= 50 else "Fair" if score >= 25 else "Poor"


def load_bundle(path: Path) -> dict:
    """Load a model bundle. Bundles saved in the portable format keep XGBoost boosters as JSON text
    (key "xgb_json") and are rebuilt here; older bundles load as plain joblib."""
    b = joblib.load(path)
    for key, text in (b.pop("xgb_json", None) or {}).items():
        bundle_key, step = key.split("::")
        est = b[bundle_key].named_steps[step]
        booster = xgb.Booster()
        booster.load_model(bytearray(text.encode("utf-8")))
        est._Booster = booster
    return b


class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.dir = Path(models_dir)
        self.health = load_bundle(self.dir / "health_risk_model.joblib")
        self.vehicle = load_bundle(self.dir / "vehicle_emission_models.joblib")
        self.route = load_bundle(self.dir / "route_risk_mlp.joblib")
        self.aqi = {h: load_bundle(self.dir / f"aqi_forecast_h{h}.joblib") for h in (1, 7)}

    # ------------------------------------------------------------------ catalogue for /models
    def describe(self):
        def brief(meta, extra=None):
            d = {"model": meta.get("model"), "version": meta.get("version"), "features": meta.get("features") or meta.get("profile_features")}
            d.update(extra or {})
            return d
        return {
            "health_risk": brief(self.health["meta"], {"metrics_test": self.health["meta"].get("metrics_test")}),
            "vehicle_emission": brief(self.vehicle["meta"], {"metrics_tier_test": self.vehicle["meta"].get("metrics_tier_test"),
                                                             "metrics_green_score_test": self.vehicle["meta"].get("metrics_green_score_test")}),
            "route_risk": brief(self.route["meta"], {"metrics_test": self.route["meta"].get("metrics_test_mlp")}),
            "aqi_forecast": {f"h{h}": brief(b["meta"], {"metrics_test": b["meta"].get("metrics_test")}) for h, b in self.aqi.items()},
        }

    # ------------------------------------------------------------------ Model 2
    def predict_health(self, req):
        meta = self.health["meta"]
        row = pd.DataFrame([req.model_dump()])
        row["aqi_band"] = np.digitize(row["aqi"], [50, 100, 150, 200, 300])
        row["is_sensitive"] = ((row["age"] < 18) | (row["age"] >= 60) | (row["condition"] != "none")).astype(int)
        p = self.health["pipeline"].predict_proba(row[meta["features"]])[0]
        labels = meta["labels"]
        tier = labels[int(p.argmax())]
        return {"risk_tier": tier, "probabilities": {l: round(float(x), 4) for l, x in zip(labels, p)},
                "precaution": meta["precautions"][tier], "aqi_category": aqi_category(req.aqi),
                "model_name": meta["model"], "model_version": meta["version"]}

    # ------------------------------------------------------------------ Model 3
    def predict_vehicle(self, req):
        meta = self.vehicle["meta"]
        v = req.model_dump()
        v["age"] = meta["reference_year"] - v["year"]
        v["km_per_year"] = v["odometer_km"] / max(v["age"], 1)
        row = pd.DataFrame([v])[meta["profile_features"]]
        p = self.vehicle["tier_pipeline"].predict_proba(row)[0]
        labels = meta["tier_labels"]
        score = float(np.clip(self.vehicle["green_score_pipeline"].predict(row)[0], 0, 100))
        return {"emission_tier": labels[int(p.argmax())], "tier_probabilities": {l: round(float(x), 4) for l, x in zip(labels, p)},
                "green_score": round(score, 1), "green_rating": green_rating(score),
                "model_name": meta["model"], "model_version": meta["version"]}

    # ------------------------------------------------------------------ Model 4
    def predict_route(self, req):
        meta = self.route["meta"]
        d = pd.DataFrame([req.model_dump()])
        d["avg_speed_kmh"] = d["distance_km"] / (d["duration_min"] / 60)
        d["is_rush_hour"] = d["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
        d["same_zone"] = (d["origin_zone"] == d["dest_zone"]).astype(int)
        d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
        d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
        d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
        d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
        p = self.route["pipeline"].predict_proba(d[meta["features"]])[0]
        labels = meta["labels"]
        return {"route_label": labels[int(p.argmax())], "probabilities": {l: round(float(x), 4) for l, x in zip(labels, p)},
                "model_name": meta["model"], "model_version": meta["version"]}

    # ------------------------------------------------------------------ Model 1
    @staticmethod
    def _aqi_features(req) -> pd.DataFrame:
        d = pd.DataFrame([r.model_dump() for r in req.history]).sort_values("date").reset_index(drop=True)
        d["date"] = pd.to_datetime(d["date"])
        p = d["pm25"]
        for L in (1, 2, 3, 7, 14):
            d[f"pm25_lag{L}"] = p.shift(L)
        for W in (3, 7, 14):
            d[f"pm25_roll{W}"] = p.shift(1).rolling(W).mean()
        d["pm25_std7"] = p.shift(1).rolling(7).std()
        d["pm25_diff1"] = p - p.shift(1)
        d["dow"] = d["date"].dt.dayofweek
        d["is_weekend"] = (d["dow"] >= 5).astype(int)
        d["month_sin"] = np.sin(2 * np.pi * d["date"].dt.month / 12)
        d["month_cos"] = np.cos(2 * np.pi * d["date"].dt.month / 12)
        d["doy_sin"] = np.sin(2 * np.pi * d["date"].dt.day_of_year / 365.25)
        d["doy_cos"] = np.cos(2 * np.pi * d["date"].dt.day_of_year / 365.25)
        d["wd_sin"] = np.sin(np.radians(d["wind_dir"]))
        d["wd_cos"] = np.cos(np.radians(d["wind_dir"]))
        d["zone"] = req.zone
        return d.iloc[[-1]]

    def forecast_aqi(self, req):
        last = self._aqi_features(req)
        based_on = last["date"].iloc[0].date()
        items, rmse, name, version = [], {}, None, None
        for h in sorted(set(req.horizons)):
            b = self.aqi[h]
            meta = b["meta"]
            pm = float(max(b["pipeline"].predict(last[meta["features"]])[0], 0.0))
            aqi = pm25_to_aqi(pm)
            items.append({"horizon_days": h, "target_date": based_on + timedelta(days=h), "pred_pm25": round(pm, 1),
                          "pred_aqi": aqi, "category": aqi_category(aqi)})
            rmse[f"h{h}"] = round(float(meta["metrics_test"]["RMSE"]), 2)
            name, version = meta["model"], meta["version"]
        return {"zone": req.zone, "based_on_date": based_on, "forecasts": items, "model_name": name,
                "model_version": version, "test_rmse_pm25": rmse}
