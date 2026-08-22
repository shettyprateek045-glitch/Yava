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
   churn?" and get a JSON answer back.
3. Also exposes some read-only analytics about the training dataset itself
   (GET /portfolio), so the frontend has real numbers for its KPI cards
   without needing to touch the CSV file directly.
4. Serves the frontend (the `static/` folder) from the same server, so
   there's only one command to run (`uvicorn api:app`) and one URL to open.

What changed vs. serving_api.py:
  - Serves the v2 model (stacking ensemble on the engineered feature set)
    by default, with a documented fallback to v1 if v2 is missing, instead
    of always loading v1.
  - Uses ONE decision threshold end to end: the F1-optimal threshold from
    metrics_v2.json (0.206), matching MODEL_CARD.md and the fixed
    Streamlit app. serving_api.py and app.py previously disagreed with
    each other on this number; this file and app.py are now the source of
    truth and read the same metrics file.
  - Adds CORS so the static control-center frontend (or any external
    frontend) can call this API from a different origin/port.
  - Adds GET /portfolio — dataset-level analytics (KPIs, churn-by-segment,
    revenue-at-risk) so the frontend doesn't need direct file access.
  - Mounts the static/ control-center UI, so `uvicorn api:app` alone
    serves both the API and the product frontend.
  - Adds request logging and a catch-all error handler, so failures come
    back as a clean JSON message instead of a raw Python traceback.

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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# 1) SETUP — logging, file paths, and the FastAPI app object itself
# =============================================================================

# Plain console logging so you can see, in the terminal running uvicorn,
# exactly which model loaded and which requests came in. This is the kind
# of visibility a "real" backend has and a bare script usually doesn't.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retention_console")

# All file paths the API needs, defined once, relative to this file's own
# location — so it works no matter what directory you happen to run
# `uvicorn` from.
BASE = Path(__file__).parent
DATA_PATH = BASE / "Week1_Project_Planning" / "data" / "churn_dataset.csv"
MODEL_V2_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "churn_model_v2.joblib"
MODEL_V1_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "churn_model.joblib"
METRICS_V2_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "metrics_v2.json"
METRICS_V1_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "metrics.json"
STATIC_DIR = BASE / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts, before it accepts any requests.
    Loading the model here (instead of waiting for the first request)
    means: (a) the very first API call a user makes is fast, not slow,
    and (b) if the model file is missing, you find out immediately in the
    startup logs — not confusingly, on whichever request happens to
    arrive first."""
    try:
        get_model_bundle()
    except HTTPException as exc:
        logger.error("Startup model preload failed: %s", exc.detail)
    yield
    # (nothing to clean up on shutdown for this app)


app = FastAPI(
    title="Streaming Service Churn — Control Center API",
    description=(
        "Serves the trained churn-prediction pipeline plus portfolio-level "
        "analytics behind the advanced control-center UI. Prototype model "
        "trained on synthetic data — see /model-info for details."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Permissive CORS: this is a portfolio/demo project meant to be called from
# a locally-served static frontend on a different port. Tighten this to a
# specific origin allowlist before any real deployment (see
# ADVANCED_VERSION.md -> "Recommended next stage").
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 2) MIDDLEWARE — runs around every request. Two jobs: log it, and make sure
#    an unexpected crash never leaks a raw Python traceback to the caller.
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs "METHOD /path -> status (Nms)" for every request, and turns any
    unhandled exception into a clean 500 JSON response instead of letting it
    surface as a broken connection or a stack trace. This is what makes it
    easy to see what the frontend is actually doing while you're demoing
    it — just watch the terminal running uvicorn."""
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
# 3) MODEL + DATA LOADING — cached in memory after the first call, so
#    predictions stay fast (loading a .joblib file from disk on every
#    request would be wasteful and slow).
# =============================================================================

_model = None
_model_version = None
_metrics = None
_threshold = None
_df = None


def get_model_bundle():
    """Loads v2 (preferred) or falls back to v1, caching the result so this
    only actually touches disk once per server run. Returns a 4-tuple:
    (model, version_label, metrics_dict, decision_threshold)."""
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
            detail="No trained model found — run model_training.py and/or "
                   "model_training_v2.py first.",
        )
    return _model, _model_version, _metrics, _threshold


def get_dataset() -> pd.DataFrame:
    """Loads and lightly cleans the training dataset once, then reuses the
    cleaned copy for every /portfolio call. The cleaning here (dedupe,
    fill missing values) mirrors what the Week 2/3 scripts already do, so
    the numbers shown on the console match the numbers in the reports."""
    global _df
    if _df is None:
        if not DATA_PATH.exists():
            raise HTTPException(status_code=503, detail="Dataset not found — run generate_dataset.py first.")
        df = pd.read_csv(DATA_PATH)
        df = df.drop_duplicates(subset=[c for c in df.columns if c != "customer_id"])
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        _df = df
        logger.info("Loaded dataset: %d rows after cleaning", len(df))
    return _df


# =============================================================================
# 4) REQUEST / RESPONSE SHAPES (pydantic models)
#    FastAPI uses these to: validate incoming JSON automatically (reject
#    bad requests with a 422 before our code even runs), and to generate
#    the interactive docs at /docs.
# =============================================================================

class CustomerFeatures(BaseModel):
    """One (hypothetical) customer's attributes — exactly the fields the
    trained pipeline was built on in Week 1/3. Every field has a sensible
    valid range (`ge`/`le`) so obviously-wrong input (negative age, a
    plan_type that was never in the dataset) is rejected automatically."""
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


class PredictionResponse(BaseModel):
    """What we send back for one scored customer."""
    churn_probability: float
    risk_band: Literal["Low", "Moderate", "High"]
    decision_threshold: float
    would_flag_as_churn_risk: bool
    model_version: str


# =============================================================================
# 5) SCORING LOGIC — turning raw customer fields into a churn probability
# =============================================================================

def _engineer_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Recreates the extra columns the v2 model was trained on (engagement
    score, tenure bucket, etc.) from the raw fields a caller sends us.

    IMPORTANT: this must exactly match
    Week3_ML_Model_Development/model_training_v2.py's engineer_features(),
    or the v2 pipeline will silently see missing columns filled with NaN
    by the ColumnTransformer and produce a meaningless prediction."""
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
    """Turns a raw probability into a human-friendly Low/Moderate/High
    label. "Moderate" starts at 60% of the way to the model's actual
    decision threshold — not at an arbitrary fixed number — so the bands
    always make sense relative to wherever the threshold is tuned to."""
    moderate_cutoff = threshold * 0.6
    if prob >= threshold:
        return "High"
    elif prob >= moderate_cutoff:
        return "Moderate"
    return "Low"


def _score(df: pd.DataFrame) -> np.ndarray:
    """Runs whichever model is currently loaded (v2 or v1) on a dataframe
    of one-or-more customers and returns an array of churn probabilities."""
    model, version, _metrics, _threshold = get_model_bundle()
    if version == "v2":
        df = _engineer_features_v2(df)
    return model.predict_proba(df)[:, 1]


# =============================================================================
# 6) ROUTES — the actual API endpoints
# =============================================================================

@app.get("/health", summary="Liveness check")
def health():
    """Returns 200 the instant the server is up, regardless of whether the
    model has loaded yet. The frontend polls this to show the green/red
    status dot in the sidebar."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/model-info", summary="Which model is live, and how good is it")
def model_info():
    """Reports which model version is currently serving predictions, its
    decision threshold, and its held-out performance numbers — pulled
    straight from the metrics.json file the training script wrote, so this
    can never drift out of sync with what MODEL_CARD.md documents."""
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


@app.post("/predict", response_model=PredictionResponse, summary="Score one customer")
def predict(customer: CustomerFeatures):
    """The core endpoint: takes one customer's attributes, returns a churn
    probability, a Low/Moderate/High band, and whether they cross the
    model's actual (F1-optimized) decision threshold."""
    _model, version, _metrics, threshold = get_model_bundle()
    df = pd.DataFrame([customer.model_dump()])
    proba = float(_score(df)[0])
    return PredictionResponse(
        churn_probability=round(proba, 4),
        risk_band=_risk_band(proba, threshold),
        decision_threshold=round(threshold, 4),
        would_flag_as_churn_risk=proba >= threshold,
        model_version=version,
    )


@app.post("/predict/batch", response_model=list[PredictionResponse], summary="Score many customers at once")
def predict_batch(customers: list[CustomerFeatures]):
    """Same scoring logic as /predict, but for a whole list of customers in
    one call — e.g. for a CSV upload of an entire retention-outreach list.
    Runs all rows through the model together (one predict_proba call, not
    one per customer), which is both simpler and considerably faster."""
    if not customers:
        raise HTTPException(status_code=400, detail="Provide at least one customer.")
    _model, version, _metrics, threshold = get_model_bundle()
    df = pd.DataFrame([c.model_dump() for c in customers])
    probas = _score(df)
    return [
        PredictionResponse(
            churn_probability=round(float(p), 4),
            risk_band=_risk_band(float(p), threshold),
            decision_threshold=round(threshold, 4),
            would_flag_as_churn_risk=float(p) >= threshold,
            model_version=version,
        )
        for p in probas
    ]


@app.get("/portfolio", summary="Dataset-wide KPIs for the Overview panel")
def portfolio():
    """Portfolio-level analytics for the KPI cards / churn-exposure section
    of the control center. Computed directly from the (synthetic) dataset —
    revenue-at-risk is a simple exposure estimate (churned customers' total
    monthly_charge), not a real revenue model."""
    df = get_dataset()
    total_customers = len(df)
    churned = int(df["churned"].sum())
    churn_rate = float(df["churned"].mean())
    revenue_at_risk = float(df.loc[df["churned"] == 1, "monthly_charge"].sum())
    total_monthly_revenue = float(df["monthly_charge"].sum())
    avg_tenure = float(df["tenure_months"].mean())

    # Churn rate broken down by each segment the frontend lets you tab
    # between (Plan / Contract / Region / Engagement bucket).
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
# 7) STATIC FRONTEND — mounted LAST, so it never shadows the API routes
#    declared above it. html=True makes it serve static/index.html at "/".
# =============================================================================
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
