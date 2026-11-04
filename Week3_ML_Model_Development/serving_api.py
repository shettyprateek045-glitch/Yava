"""
Week 3 (Advanced) - Model Serving API
=========================================
A minimal, production-shaped FastAPI service that wraps the trained churn
pipeline behind a REST endpoint. This turns the project from "a model that
sits in a notebook" into something a frontend, a cron job, or another
service could actually call.

Endpoints:
    GET  /health          -> liveness check
    GET  /model-info       -> which model, when trained, cross-val metrics
    POST /predict           -> single-customer churn probability + risk band
    POST /predict/batch     -> multiple customers at once

Run locally:
    pip install fastapi uvicorn pydantic
    uvicorn serving_api:app --reload --port 8000

Then try:
    curl http://127.0.0.1:8000/health
    open http://127.0.0.1:8000/docs   # interactive Swagger UI, auto-generated

Note: this requires network access to `pip install` fastapi/uvicorn/pydantic,
which the sandbox that generated this repo does not have — the code follows
standard, well-tested FastAPI/pydantic patterns and is meant to be run in
your own environment (or the Docker container in this repo's root).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
MODEL_PATH = BASE / "outputs" / "churn_model.joblib"
METRICS_PATH = BASE / "outputs" / "metrics.json"

app = FastAPI(
    title="Streaming Service Churn Prediction API",
    description="Serves the Week 3 tuned churn-prediction pipeline.",
    version="1.0.0",
)

_model = None
_metrics = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not found — run model_training.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model


def get_metrics():
    global _metrics
    if _metrics is None and METRICS_PATH.exists():
        _metrics = json.loads(METRICS_PATH.read_text())
    return _metrics or {}


class CustomerFeatures(BaseModel):
    age: float = Field(..., ge=13, le=100, example=34)
    region: Literal["North America", "Europe", "Asia-Pacific", "Latin America", "Other"] = "North America"
    tenure_months: int = Field(..., ge=0, le=200, example=14)
    plan_type: Literal["Basic", "Standard", "Premium"] = "Standard"
    contract_type: Literal["Monthly", "Annual"] = "Monthly"
    payment_method: Literal["Credit Card", "PayPal", "Bank Transfer", "Gift Card"] = "Credit Card"
    monthly_charge: float = Field(..., ge=0, example=13.99)
    avg_watch_hours_week: float = Field(..., ge=0, example=4.5)
    num_devices: int = Field(..., ge=1, le=10, example=2)
    support_tickets: int = Field(..., ge=0, example=1)
    days_since_last_login: int = Field(..., ge=0, example=25)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 34, "region": "North America", "tenure_months": 4,
                "plan_type": "Basic", "contract_type": "Monthly",
                "payment_method": "Credit Card", "monthly_charge": 12.99,
                "avg_watch_hours_week": 1.2, "num_devices": 1,
                "support_tickets": 3, "days_since_last_login": 45,
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_band: str
    recommended_threshold: float
    would_flag_as_churn_risk: bool


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["charge_per_watch_hour"] = df["monthly_charge"] / (df["avg_watch_hours_week"] + 0.1)
    df["is_inactive"] = (df["days_since_last_login"] > 30).astype(int)
    return df


def _risk_band(prob: float) -> str:
    if prob < 0.25:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    elif prob < 0.75:
        return "High"
    return "Critical"


# F1-optimal threshold from advanced_interpretability.py; falls back to 0.5.
DECISION_THRESHOLD = 0.5
_interp_path = BASE / "outputs" / "advanced" / "interpretability_summary.json"
if _interp_path.exists():
    DECISION_THRESHOLD = json.loads(_interp_path.read_text())["threshold_optimization"]["optimal_threshold"]


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root requests to interactive Swagger documentation so visitors
    do not encounter a 404 Not Found error."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/model-info")
def model_info():
    metrics = get_metrics()
    return {
        "model": metrics.get("best_model", "unknown"),
        "test_metrics": metrics.get("test_metrics", {}),
        "decision_threshold": DECISION_THRESHOLD,
        "note": "Prototype model trained on synthetic data — not production-validated.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    model = get_model()
    df = _engineer_features(pd.DataFrame([customer.model_dump()]))
    proba = float(model.predict_proba(df)[0, 1])
    return PredictionResponse(
        churn_probability=round(proba, 4),
        risk_band=_risk_band(proba),
        recommended_threshold=DECISION_THRESHOLD,
        would_flag_as_churn_risk=proba >= DECISION_THRESHOLD,
    )


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(customers: list[CustomerFeatures]):
    if not customers:
        raise HTTPException(status_code=400, detail="Provide at least one customer.")
    model = get_model()
    df = _engineer_features(pd.DataFrame([c.model_dump() for c in customers]))
    probas = model.predict_proba(df)[:, 1]
    return [
        PredictionResponse(
            churn_probability=round(float(p), 4),
            risk_band=_risk_band(p),
            recommended_threshold=DECISION_THRESHOLD,
            would_flag_as_churn_risk=p >= DECISION_THRESHOLD,
        )
        for p in probas
    ]
