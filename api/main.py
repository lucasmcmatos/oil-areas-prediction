"""FastAPI service exposing the trained oil probability model.

Loads the RandomForest classifier from models/oil_probability_model.joblib,
serves single-point and batch predictions, and also serves the dashboard
itself as static files from the same origin/port. Serving both from one
process means a phone on the same network can load the whole dashboard
(and reach the API with no CORS/host juggling) by visiting a single URL —
important for field use alongside the satellite payload.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "gradient_boosting_model.joblib"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

HIGH_PROBABILITY_THRESHOLD = 70.0
MODERATE_PROBABILITY_THRESHOLD = 30.0

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Oil Probability API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    latitude: float = Field(..., ge=-34.0, le=5.5)
    longitude: float = Field(..., ge=-50.0, le=-28.0)
    methane_ppm: float = Field(..., gt=0)
    pressure_hpa: float = Field(..., gt=0)


class PredictionOutput(BaseModel):
    oil_probability_percent: float
    classification: str


def classify(probability_percent: float) -> str:
    if probability_percent >= HIGH_PROBABILITY_THRESHOLD:
        return "high"
    if probability_percent >= MODERATE_PROBABILITY_THRESHOLD:
        return "moderate"
    return "low"


def predict_one(point: PredictionInput) -> PredictionOutput:
    features = [[point.methane_ppm, point.pressure_hpa, point.latitude, point.longitude]]
    probability = model.predict_proba(features)[0, 1] * 100
    return PredictionOutput(
        oil_probability_percent=round(probability, 2),
        classification=classify(probability),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(point: PredictionInput) -> PredictionOutput:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predict_one(point)


@app.post("/predict/batch", response_model=list[PredictionOutput])
def predict_batch(points: list[PredictionInput]) -> list[PredictionOutput]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return [predict_one(point) for point in points]


# Mounted last so it only catches paths not matched by the routes above
# (e.g. "/" and "/index.html" serve the dashboard, "/health" still works).
app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
