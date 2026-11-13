# SaansCare ML Service (FastAPI)

Loads the four trained models from `../models/` and exposes them as a JSON API. The Node/Express backend calls this service; the React app never calls it directly.

```
React  ->  Node/Express backend  ->  THIS SERVICE (FastAPI)  ->  models/*.joblib
```

## Folder (put it at the repo root, next to `models/`)
```
saanscare/
  models/            trained .joblib + metadata (already in your repo)
  data/              datasets
  ml/notebooks/      training notebooks
  ml-service/        <- this folder
    app/main.py      endpoints, API-key check, prediction log
    app/schemas.py   input validation (allowed values and ranges)
    app/registry.py  model loading + the same feature code as the notebooks
    tests/test_api.py
    requirements.txt, Dockerfile, .env.example
```

## Run on your laptop (Windows, VS Code terminal)
Use Python 3.10 - 3.13.
```
cd ml-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open http://127.0.0.1:8000/docs - every endpoint has an example you can run with the "Try it out" button.
(macOS/Linux: `source .venv/bin/activate` instead of the activate line.)

**scikit-learn must be exactly 1.6.1** (the version Colab trained the models with); `requirements.txt` already pins it. A different version can fail to load the models.

## Test
```
pip install -r requirements-dev.txt
pytest -q
```
7 tests use the real models and `data/aqi_forecasting.csv`: sensible ordering of predictions, input validation (HTTP 422), and the API key.

## Endpoints
| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Is the service up and which models are loaded |
| GET | `/models` | Names, versions, input features and test metrics (Admin model monitor) |
| POST | `/predict/health-risk` | Model 2: risk tier + probabilities + precautions |
| POST | `/predict/vehicle` | Model 3: emission tier + Green Score (0-100) + rating |
| POST | `/predict/route-risk` | Model 4: Safe / Caution / Avoid + probabilities |
| POST | `/predict/aqi-forecast` | Model 1: next-day and 7-day PM2.5 / AQI for a zone |

`/predict/aqi-forecast` needs the last 15 or more **consecutive daily** records for the zone (newest last, no gaps). The backend should read them from MongoDB and fill short gaps first.
`/predict/route-risk` expects route features. The route optimizer (next phase) will compute them from the road graph and the AQI map, then call this endpoint.

## Example (health risk)
```
curl -X POST http://127.0.0.1:8000/predict/health-risk -H "Content-Type: application/json" -d "{\"aqi\":280,\"age\":8,\"gender\":\"male\",\"condition\":\"asthma\",\"outdoor_hours\":2,\"activity_level\":\"moderate\",\"mask_use\":\"none\"}"
```
Response: `{"risk_tier":"Very High","probabilities":{...},"precaution":"Avoid outdoor activity...","aqi_category":"Very Unhealthy",...}`

## Calling it from Node/Express (next phase)
```js
const r = await fetch(`${process.env.ML_URL}/predict/health-risk`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": process.env.ML_API_KEY },
  body: JSON.stringify(req.body),
});
const result = await r.json();
```

## Settings (environment variables, see `.env.example`)
`ML_API_KEY` (if set, every predict call must send it in `X-API-Key`), `ALLOWED_ORIGINS`, `MODELS_DIR`, `PREDICTION_LOG`.
Every prediction appends one line (time, endpoint, model version, result) to `logs/predictions.jsonl` for traceability. Personal health inputs are not stored.

## Docker (used later for deployment)
From the repo root: `docker build -f ml-service/Dockerfile -t saanscare-ml .` then `docker run -p 8000:8000 saanscare-ml`.
