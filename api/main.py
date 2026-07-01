"""FastAPI service exposing the trained oil probability model.

Loads the RandomForest classifier from models/oil_probability_model.joblib
and serves single-point and batch predictions for the dashboard.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "oil_probability_model.joblib"
FEATURE_COLUMNS = ["methane_ppm", "pressure_hpa", "latitude", "longitude"]

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
    latitude: float = Field(..., ge=-27.5, le=-19.0)
    longitude: float = Field(..., ge=-45.5, le=-37.5)
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
