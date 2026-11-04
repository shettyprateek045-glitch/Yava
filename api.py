"""
Advanced Product API — Churn Risk Control Center
====================================================
Supersedes `Week3_ML_Model_Development/serving_api.py` as the product-facing
backend. `serving_api.py` is kept in place as the Week 3 assignment
deliverable (it's what that week's DOC describes); this file is the
professional-presentation layer built on top of the same trained artifacts.

WHAT THIS FILE DOES, IN PLAIN WORDS
------------------------------------
1. Loads the trained churn model once when the server starts (not on every
   request — that would be slow).
2. Exposes that model over HTTP so anything (the control-center frontend,
   curl, Postman, another program) can ask "how likely is this customer to
   churn?" and get a JSON answer back with explainability & recommendations.
3. Also exposes dataset-level analytics about the training dataset itself
   (GET /portfolio), so the frontend has real numbers for its KPI cards
   without needing to touch the CSV file directly.
4. Provides Explainable AI (XAI): computes parameter risk drivers and prescriptive
   retention recommendations for each subscriber.
5. Provides What-If scenario simulations (POST /simulate) to project risk reduction
   and annual revenue protected before taking action.
6. Stores an audit trail of scored customers in a local SQLite database
   (data/predictions.db) accessible via GET /predictions/history.
7. Supports Batch CSV scoring (POST /predict/csv) with summary metrics.
8. Serves the frontend (the `static/` folder) from the same server, so
   there's only one command to run (`uvicorn api:app`) and one URL to open.

Run locally:
    pip install -r requirements.txt -r requirements-advanced.txt
    uvicorn api:app --reload --port 8000

Then open:
    http://127.0.0.1:8000/           -> control center UI
    http://127.0.0.1:8000/docs       -> interactive Swagger API docs
    http://127.0.0.1:8000/health     -> liveness check
"""

import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# 1) SETUP — logging, file paths, and database initialization
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retention_console")

BASE = Path(__file__).parent
DATA_PATH = BASE / "Week1_Project_Planning" / "data" / "churn_dataset.csv"
MODEL_V2_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "churn_model_v2.joblib"
MODEL_V1_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "churn_model.joblib"
METRICS_V2_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "metrics_v2.json"
METRICS_V1_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "metrics.json"
STATIC_DIR = BASE / "static"
DB_PATH = BASE / "data" / "predictions.db"


def init_db():
    """Initializes the SQLite database for prediction audit logging."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                age REAL,
                region TEXT,
                tenure_months INTEGER,
                plan_type TEXT,
                contract_type TEXT,
                payment_method TEXT,
                monthly_charge REAL,
                avg_watch_hours_week REAL,
                num_devices INTEGER,
                support_tickets INTEGER,
                days_since_last_login INTEGER,
                churn_probability REAL,
                risk_band TEXT,
                would_flag_as_churn_risk INTEGER,
                clv_at_risk REAL,
                top_risk_factor TEXT,
                recommended_action TEXT
            )
            """
        )
        conn.commit()


def log_prediction_to_db(c: dict, prob: float, risk_band: str, flagged: bool, clv: float, top_factor: str, action: str):
    """Safely logs a scored customer to SQLite without throwing exceptions to callers."""
    try:
        init_db()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO predictions_history (
                    timestamp, age, region, tenure_months, plan_type, contract_type,
                    payment_method, monthly_charge, avg_watch_hours_week, num_devices,
                    support_tickets, days_since_last_login, churn_probability,
                    risk_band, would_flag_as_churn_risk, clv_at_risk, top_risk_factor, recommended_action
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    c.get("age"), c.get("region"), c.get("tenure_months"), c.get("plan_type"),
                    c.get("contract_type"), c.get("payment_method"), c.get("monthly_charge"),
                    c.get("avg_watch_hours_week"), c.get("num_devices"), c.get("support_tickets"),
                    c.get("days_since_last_login"), prob, risk_band, 1 if flagged else 0,
                    clv, top_factor, action
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Could not log prediction to SQLite: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts, before it accepts any requests.
    Initializes SQLite database and preloads the ML model."""
    init_db()
    try:
        get_model_bundle()
    except HTTPException as exc:
        logger.error("Startup model preload failed: %s", exc.detail)
    yield


app = FastAPI(
    title="Streaming Service Churn — Control Center API",
    description=(
        "Serves the trained churn-prediction pipeline, prescriptive action engine, "
        "what-if simulations, and audit history behind the advanced control-center UI."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# 2) MIDDLEWARE — Request logging & exception handling
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal error — check the server logs for details."},
        )
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


# =============================================================================
# 3) MODEL + DATA LOADING
# =============================================================================

_model = None
_model_version = None
_metrics = None
_threshold = None
_df = None


def get_model_bundle():
    global _model, _model_version, _metrics, _threshold
    if _model is not None:
        return _model, _model_version, _metrics, _threshold

    if MODEL_V2_PATH.exists() and METRICS_V2_PATH.exists():
        _model = joblib.load(MODEL_V2_PATH)
        _model_version = "v2"
        _metrics = json.loads(METRICS_V2_PATH.read_text())
        _threshold = _metrics["test_metrics_optimal_threshold"]["threshold"]
        logger.info("Loaded model v2 (StackingEnsemble) — decision threshold %.4f", _threshold)
    elif MODEL_V1_PATH.exists():
        _model = joblib.load(MODEL_V1_PATH)
        _model_version = "v1"
        _metrics = json.loads(METRICS_V1_PATH.read_text()) if METRICS_V1_PATH.exists() else {}
        _threshold = 0.5
        logger.warning("v2 model not found — falling back to v1 with default threshold 0.5")
    else:
        logger.error("No trained model found at %s or %s", MODEL_V2_PATH, MODEL_V1_PATH)
        raise HTTPException(
            status_code=503,
            detail="No trained model found — run model_training.py and/or model_training_v2.py first.",
        )
    return _model, _model_version, _metrics, _threshold


def get_dataset() -> pd.DataFrame:
    global _df
    if _df is None:
        if not DATA_PATH.exists():
            raise HTTPException(status_code=503, detail="Dataset not found — run generate_dataset.py first.")
        df = pd.read_csv(DATA_PATH)
        df = df.drop_duplicates(subset=[c for c in df.columns if c != "customer_id"])
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        _df = df
        logger.info("Loaded dataset: %d rows after cleaning", len(df))
    return _df


# =============================================================================
# 4) REQUEST / RESPONSE SHAPES (pydantic models)
# =============================================================================

class CustomerFeatures(BaseModel):
    age: float = Field(..., ge=13, le=100)
    region: Literal["North America", "Europe", "Asia-Pacific", "Latin America", "Other"] = "North America"
    tenure_months: int = Field(..., ge=0, le=200)
    plan_type: Literal["Basic", "Standard", "Premium"] = "Standard"
    contract_type: Literal["Monthly", "Annual"] = "Monthly"
    payment_method: Literal["Credit Card", "PayPal", "Bank Transfer", "Gift Card"] = "Credit Card"
    monthly_charge: float = Field(..., ge=0)
    avg_watch_hours_week: float = Field(..., ge=0)
    num_devices: int = Field(..., ge=1, le=10)
    support_tickets: int = Field(..., ge=0)
    days_since_last_login: int = Field(..., ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 34, "region": "North America", "tenure_months": 4,
                "plan_type": "Basic", "contract_type": "Monthly",
                "payment_method": "Credit Card", "monthly_charge": 12.99,
                "avg_watch_hours_week": 1.2, "num_devices": 1,
                "support_tickets": 3, "days_since_last_login": 45,
            }
        }
    )


class RiskFactor(BaseModel):
    parameter: str
    impact: str
    detail: str
    weight: float


class PrescriptiveAction(BaseModel):
    action: str
    description: str
    expected_risk_reduction: str


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_band: Literal["Low", "Moderate", "High"]
    decision_threshold: float
    would_flag_as_churn_risk: bool
    model_version: str
    customer_lifetime_value_at_risk: float = 0.0
    top_risk_factors: list[RiskFactor] = []
    prescriptive_recommendations: list[PrescriptiveAction] = []


class SimulationRequest(BaseModel):
    customer: CustomerFeatures
    switch_to_annual: bool = True
    reduce_inactivity_to_days: int | None = 2
    boost_watch_hours: float | None = 5.0
    resolve_support_tickets: bool = True


class SimulationResponse(BaseModel):
    baseline_probability: float
    baseline_risk_band: str
    simulated_probability: float
    simulated_risk_band: str
    risk_reduction_pct: float
    annual_revenue_protected: float
    interventions_applied: list[str]


# =============================================================================
# 5) SCORING & EXPLAINABLE PARAMETER ANALYSIS
# =============================================================================

def _engineer_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["charge_per_watch_hour"] = df["monthly_charge"] / (df["avg_watch_hours_week"] + 0.1)
    df["is_inactive"] = (df["days_since_last_login"] > 30).astype(int)
    df["log_days_since_login"] = np.log1p(df["days_since_last_login"])
    df["watch_hours_per_device"] = df["avg_watch_hours_week"] / df["num_devices"]
    df["engagement_score"] = df["avg_watch_hours_week"] / (df["days_since_last_login"] + 1)
    df["support_tenure_ratio"] = df["support_tickets"] / (df["tenure_months"] + 1)
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"], bins=[-1, 6, 12, 24, 999],
        labels=["0-6mo", "7-12mo", "13-24mo", "25mo+"]
    ).astype(str)
    return df


def _risk_band(prob: float, threshold: float) -> str:
    moderate_cutoff = threshold * 0.6
    if prob >= threshold:
        return "High"
    elif prob >= moderate_cutoff:
        return "Moderate"
    return "Low"


def _score(df: pd.DataFrame) -> np.ndarray:
    model, version, _metrics, _threshold = get_model_bundle()
    if version == "v2":
        df = _engineer_features_v2(df)
    return model.predict_proba(df)[:, 1]


def _analyze_parameters(c: dict, prob: float) -> tuple[list[RiskFactor], list[PrescriptiveAction], str, str]:
    """Evaluates all parameters the model depends on to determine risk drivers
    and prescribe operational retention tactics."""
    factors: list[RiskFactor] = []
    actions: list[PrescriptiveAction] = []

    # 1. Inactivity & Login
    days = c.get("days_since_last_login", 0)
    if days >= 60:
        factors.append(RiskFactor(
            parameter="days_since_last_login",
            impact="Critical",
            detail=f"{days} days dormant (benchmark: 72.7% historical churn rate)",
            weight=0.40,
        ))
        actions.append(PrescriptiveAction(
            action="Immediate Winback Campaign",
            description="Send urgent email with personalized top recommendations & 'Free 1-Month Extension' offer.",
            expected_risk_reduction="-35%",
        ))
    elif days >= 30:
        factors.append(RiskFactor(
            parameter="days_since_last_login",
            impact="High Risk",
            detail=f"{days} days inactive (cooling off stage, 46.3% churn rate)",
            weight=0.25,
        ))
        actions.append(PrescriptiveAction(
            action="Re-engagement Push Notification",
            description="Send automated alert highlighting newly added trending series matching watch history.",
            expected_risk_reduction="-20%",
        ))
    elif days <= 7:
        factors.append(RiskFactor(
            parameter="days_since_last_login",
            impact="Low Risk (Protective)",
            detail=f"Active subscriber ({days} days since last login)",
            weight=-0.15,
        ))

    # 2. Contract Type
    contract = c.get("contract_type", "Monthly")
    if contract == "Monthly":
        factors.append(RiskFactor(
            parameter="contract_type",
            impact="High Risk",
            detail="Month-to-Month agreement has 31.3% churn hazard vs 18.0% for Annual",
            weight=0.30,
        ))
        actions.append(PrescriptiveAction(
            action="Annual Contract Upgrade Incentive",
            description="Offer 2 months free (16.7% discount) to convert to an Annual commitment.",
            expected_risk_reduction="-40%",
        ))
    else:
        factors.append(RiskFactor(
            parameter="contract_type",
            impact="Low Risk (Protective)",
            detail="Annual commitment locks in subscriber retention",
            weight=-0.25,
        ))

    # 3. Support Tickets
    tickets = c.get("support_tickets", 0)
    if tickets >= 3:
        factors.append(RiskFactor(
            parameter="support_tickets",
            impact="High Risk",
            detail=f"{tickets} support tickets logged (severe friction / frustration indicator)",
            weight=0.22,
        ))
        actions.append(PrescriptiveAction(
            action="VIP Concierge Escalation",
            description="Assign dedicated Customer Success specialist to personally resolve pending issues within 24h.",
            expected_risk_reduction="-30%",
        ))
    elif tickets >= 1:
        factors.append(RiskFactor(
            parameter="support_tickets",
            impact="Moderate Risk",
            detail=f"{tickets} support ticket logged",
            weight=0.10,
        ))
        actions.append(PrescriptiveAction(
            action="Post-Support Satisfaction Follow-up",
            description="Deliver 1-question CSAT survey and courtesy billing credit if issue took >48h.",
            expected_risk_reduction="-10%",
        ))

    # 4. Tenure
    tenure = c.get("tenure_months", 0)
    if tenure <= 6:
        factors.append(RiskFactor(
            parameter="tenure_months",
            impact="Moderate Risk",
            detail=f"{tenure} months tenure (vulnerable onboarding window; habit not yet formed)",
            weight=0.18,
        ))
        actions.append(PrescriptiveAction(
            action="Onboarding Content Journey",
            description="Trigger personalized walkthrough email series explaining playlist, multi-profile, and download features.",
            expected_risk_reduction="-15%",
        ))
    elif tenure >= 24:
        factors.append(RiskFactor(
            parameter="tenure_months",
            impact="Low Risk (Protective)",
            detail=f"Loyal subscriber with {tenure} months tenure",
            weight=-0.20,
        ))

    # 5. Usage & Perceived Value (charge per watch hour)
    hours = c.get("avg_watch_hours_week", 0.0)
    charge = c.get("monthly_charge", 0.0)
    cost_per_hour = charge / (hours + 0.1)
    if cost_per_hour >= 6.0:
        factors.append(RiskFactor(
            parameter="charge_per_watch_hour",
            impact="Moderate Risk",
            detail=f"${cost_per_hour:.2f}/hr watched indicates perceived price friction",
            weight=0.15,
        ))
        actions.append(PrescriptiveAction(
            action="Right-Sizing Plan Suggestion",
            description="Proactively suggest a lower-tier plan or bundle to prevent outright churn.",
            expected_risk_reduction="-20%",
        ))

    # 6. Payment Method
    payment = c.get("payment_method", "Credit Card")
    if payment in ["PayPal", "Gift Card"]:
        factors.append(RiskFactor(
            parameter="payment_method",
            impact="Low-Moderate Risk",
            detail=f"{payment} has higher involuntary churn/expiration rate (28.5%)",
            weight=0.08,
        ))
        actions.append(PrescriptiveAction(
            action="Payment Auto-Update Prompt",
            description="Offer $5 credit to add an auto-renewing primary credit card.",
            expected_risk_reduction="-8%",
        ))

    factors.sort(key=lambda x: abs(x.weight), reverse=True)
    top_factor = factors[0].parameter + ": " + factors[0].detail if factors else "Normal usage profile"
    primary_action = actions[0].action + " - " + actions[0].description if actions else "Standard engagement monitoring"

    return factors, actions, top_factor, primary_action


# =============================================================================
# 6) ROUTES — Core Endpoints & Advanced Features
# =============================================================================

@app.get("/health", summary="Liveness check")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/model-info", summary="Which model is live, and how good is it")
def model_info():
    _model, version, metrics, threshold = get_model_bundle()
    if version == "v2":
        perf = metrics["test_metrics_optimal_threshold"]
        default_perf = metrics["test_metrics_default_threshold"]
        summary = {
            "model_type": metrics.get("best_model", "unknown"),
            "roc_auc": default_perf.get("roc_auc"),
            "precision_at_threshold": perf.get("precision"),
            "recall_at_threshold": perf.get("recall"),
            "f1_at_threshold": perf.get("f1_score"),
        }
    else:
        perf = metrics.get("test_metrics", {})
        summary = {
            "model_type": metrics.get("best_model", "unknown"),
            "roc_auc": perf.get("roc_auc"),
            "precision_at_threshold": perf.get("precision"),
            "recall_at_threshold": perf.get("recall"),
            "f1_at_threshold": perf.get("f1_score"),
        }
    return {
        "model_version": version,
        "decision_threshold": threshold,
        "performance": summary,
        "trained_on": "synthetic dataset (Week1_Project_Planning/generate_dataset.py)",
        "warning": "Prototype model trained on synthetic data — not production-validated. "
                   "See MODEL_CARD.md for full limitations.",
    }


@app.post("/predict", response_model=PredictionResponse, summary="Score one customer with explainability")
def predict(customer: CustomerFeatures):
    _model, version, _metrics, threshold = get_model_bundle()
    c_dict = customer.model_dump()
    df = pd.DataFrame([c_dict])
    proba = float(_score(df)[0])
    risk_band = _risk_band(proba, threshold)
    flagged = proba >= threshold
    clv = round(customer.monthly_charge * 12, 2)

    factors, actions, top_factor, primary_action = _analyze_parameters(c_dict, proba)
    log_prediction_to_db(c_dict, proba, risk_band, flagged, clv, top_factor, primary_action)

    return PredictionResponse(
        churn_probability=round(proba, 4),
        risk_band=risk_band,
        decision_threshold=round(threshold, 4),
        would_flag_as_churn_risk=flagged,
        model_version=version,
        customer_lifetime_value_at_risk=clv,
        top_risk_factors=factors,
        prescriptive_recommendations=actions,
    )


@app.post("/predict/batch", response_model=list[PredictionResponse], summary="Score many customers at once")
def predict_batch(customers: list[CustomerFeatures]):
    if not customers:
        raise HTTPException(status_code=400, detail="Provide at least one customer.")
    _model, version, _metrics, threshold = get_model_bundle()
    df = pd.DataFrame([c.model_dump() for c in customers])
    probas = _score(df)

    responses = []
    for p, c in zip(probas, customers):
        c_dict = c.model_dump()
        prob_val = round(float(p), 4)
        band = _risk_band(prob_val, threshold)
        flagged = prob_val >= threshold
        clv = round(c.monthly_charge * 12, 2)
        factors, actions, top_factor, primary_action = _analyze_parameters(c_dict, prob_val)
        log_prediction_to_db(c_dict, prob_val, band, flagged, clv, top_factor, primary_action)

        responses.append(
            PredictionResponse(
                churn_probability=prob_val,
                risk_band=band,
                decision_threshold=round(threshold, 4),
                would_flag_as_churn_risk=flagged,
                model_version=version,
                customer_lifetime_value_at_risk=clv,
                top_risk_factors=factors,
                prescriptive_recommendations=actions,
            )
        )
    return responses


@app.post("/simulate", response_model=SimulationResponse, summary="What-if retention scenario simulation")
def simulate(req: SimulationRequest):
    """Simulates the impact of retention interventions (e.g., Annual plan upgrade,
    re-engaging inactivity, resolving support tickets) on churn probability."""
    _model, version, _metrics, threshold = get_model_bundle()
    base_dict = req.customer.model_dump()
    base_df = pd.DataFrame([base_dict])
    base_prob = round(float(_score(base_df)[0]), 4)
    base_band = _risk_band(base_prob, threshold)

    sim_dict = dict(base_dict)
    applied = []

    if req.switch_to_annual and sim_dict.get("contract_type") != "Annual":
        sim_dict["contract_type"] = "Annual"
        applied.append("Switched contract to Annual commitment")

    if req.reduce_inactivity_to_days is not None:
        sim_dict["days_since_last_login"] = req.reduce_inactivity_to_days
        applied.append(f"Reduced inactivity to {req.reduce_inactivity_to_days} days via re-engagement")

    if req.boost_watch_hours is not None:
        sim_dict["avg_watch_hours_week"] += req.boost_watch_hours
        applied.append(f"Increased engagement by +{req.boost_watch_hours} hrs/week")

    if req.resolve_support_tickets and sim_dict.get("support_tickets", 0) > 0:
        sim_dict["support_tickets"] = 0
        applied.append("Resolved all open support tickets")

    sim_df = pd.DataFrame([sim_dict])
    sim_prob = round(float(_score(sim_df)[0]), 4)
    sim_band = _risk_band(sim_prob, threshold)

    reduction = round(((base_prob - sim_prob) / (base_prob + 1e-9)) * 100, 1)
    annual_saved = round(sim_dict.get("monthly_charge", 0.0) * 12, 2)

    return SimulationResponse(
        baseline_probability=base_prob,
        baseline_risk_band=base_band,
        simulated_probability=sim_prob,
        simulated_risk_band=sim_band,
        risk_reduction_pct=max(0.0, reduction),
        annual_revenue_protected=annual_saved,
        interventions_applied=applied,
    )


@app.get("/predictions/history", summary="View SQLite audit trail of scored customers")
def prediction_history(limit: int = Query(50, ge=1, le=500), risk_band: str | None = None):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if risk_band:
            cur.execute(
                "SELECT * FROM predictions_history WHERE risk_band = ? ORDER BY id DESC LIMIT ?",
                (risk_band, limit),
            )
        else:
            cur.execute("SELECT * FROM predictions_history ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cur.fetchall()]
    return {
        "count": len(rows),
        "history": rows,
    }


@app.delete("/predictions/history", summary="Clear SQLite audit trail")
def clear_prediction_history():
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM predictions_history")
        conn.commit()
    return {"status": "cleared", "message": "Prediction audit history cleared."}


@app.post("/predict/csv", summary="Upload CSV file and score all subscribers")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    content = await file.read()
    try:
        df_raw = pd.read_csv(StringIO(content.decode("utf-8", errors="replace")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    required = ["age", "tenure_months", "monthly_charge", "avg_watch_hours_week", "num_devices", "support_tickets", "days_since_last_login"]
    for col in required:
        if col not in df_raw.columns:
            raise HTTPException(status_code=422, detail=f"Missing required column in CSV: {col}")

    if "region" not in df_raw.columns:
        df_raw["region"] = "North America"
    if "plan_type" not in df_raw.columns:
        df_raw["plan_type"] = "Standard"
    if "contract_type" not in df_raw.columns:
        df_raw["contract_type"] = "Monthly"
    if "payment_method" not in df_raw.columns:
        df_raw["payment_method"] = "Credit Card"

    _model, version, _metrics, threshold = get_model_bundle()
    probas = _score(df_raw)

    df_raw["churn_probability"] = [round(float(p), 4) for p in probas]
    df_raw["risk_band"] = [_risk_band(float(p), threshold) for p in probas]
    df_raw["flagged_for_retention"] = [bool(float(p) >= threshold) for p in probas]

    return {
        "total_rows": len(df_raw),
        "high_risk_count": int((df_raw["risk_band"] == "High").sum()),
        "moderate_risk_count": int((df_raw["risk_band"] == "Moderate").sum()),
        "low_risk_count": int((df_raw["risk_band"] == "Low").sum()),
        "flagged_count": int(df_raw["flagged_for_retention"].sum()),
        "revenue_at_risk": round(float(df_raw.loc[df_raw["flagged_for_retention"], "monthly_charge"].sum()), 2),
        "preview": df_raw.head(10).to_dict(orient="records"),
    }


@app.get("/portfolio", summary="Dataset-wide KPIs for the Overview panel")
def portfolio():
    df = get_dataset()
    total_customers = len(df)
    churned = int(df["churned"].sum())
    churn_rate = float(df["churned"].mean())
    revenue_at_risk = float(df.loc[df["churned"] == 1, "monthly_charge"].sum())
    total_monthly_revenue = float(df["monthly_charge"].sum())
    avg_tenure = float(df["tenure_months"].mean())

    by_plan = (
        df.groupby("plan_type")["churned"].mean().mul(100).round(2).sort_values(ascending=False)
    )
    by_contract = (
        df.groupby("contract_type")["churned"].mean().mul(100).round(2).sort_values(ascending=False)
    )
    by_region = (
        df.groupby("region")["churned"].mean().mul(100).round(2).sort_values(ascending=False)
    )
    engagement_bucket = pd.cut(
        df["days_since_last_login"], bins=[-1, 7, 30, 60, 1000],
        labels=["Active", "Cooling off", "At risk", "Dormant"],
    )
    by_engagement = (
        df.groupby(engagement_bucket, observed=True)["churned"].mean().mul(100).round(2)
    )

    return {
        "kpis": {
            "total_customers": total_customers,
            "churned_customers": churned,
            "churn_rate_pct": round(churn_rate * 100, 2),
            "monthly_revenue_at_risk": round(revenue_at_risk, 2),
            "total_monthly_revenue": round(total_monthly_revenue, 2),
            "revenue_at_risk_pct": round(revenue_at_risk / total_monthly_revenue * 100, 2)
            if total_monthly_revenue else 0,
            "avg_tenure_months": round(avg_tenure, 1),
        },
        "churn_rate_by_plan": by_plan.to_dict(),
        "churn_rate_by_contract": by_contract.to_dict(),
        "churn_rate_by_region": by_region.to_dict(),
        "churn_rate_by_engagement": by_engagement.to_dict(),
        "data_source": "synthetic dataset — not real customer records",
    }


# =============================================================================
# 7) STATIC FRONTEND
# =============================================================================
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
