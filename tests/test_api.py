"""
API-level tests for the Control Center backend (api.py).

Run from the repository root:
    pytest tests/test_api.py -v

These exercise the FastAPI app directly through Starlette's TestClient
(no network/socket needed), against the real trained model artifacts
checked into Week3_ML_Model_Development/outputs/. If those artifacts are
missing, the model-dependent tests are skipped rather than failed, so CI
still tells you *why* rather than producing a confusing error.
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import api  # noqa: E402  (import after sys.path tweak, by design)

client = TestClient(api.app)

MODEL_AVAILABLE = api.MODEL_V2_PATH.exists() or api.MODEL_V1_PATH.exists()
DATA_AVAILABLE = api.DATA_PATH.exists()

requires_model = pytest.mark.skipif(not MODEL_AVAILABLE, reason="no trained model artifact found")
requires_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="churn_dataset.csv not found")

VALID_CUSTOMER = {
    "age": 34, "region": "North America", "tenure_months": 4,
    "plan_type": "Basic", "contract_type": "Monthly",
    "payment_method": "Credit Card", "monthly_charge": 12.99,
    "avg_watch_hours_week": 1.2, "num_devices": 1,
    "support_tickets": 3, "days_since_last_login": 45,
}


# ------------------------------------------------------------------ health
def test_health_ok():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "timestamp" in body


# --------------------------------------------------------------- model-info
@requires_model
def test_model_info_shape():
    res = client.get("/model-info")
    assert res.status_code == 200
    body = res.json()
    for key in ("model_version", "decision_threshold", "performance", "trained_on", "warning"):
        assert key in body
    assert body["model_version"] in ("v1", "v2")
    assert 0.0 <= body["decision_threshold"] <= 1.0
    assert "synthetic" in body["warning"].lower()


@requires_model
def test_model_info_matches_metrics_file():
    """Guards against the exact bug the audit flagged: the threshold served
    by the API must match the one recorded in metrics_v2.json / MODEL_CARD."""
    res = client.get("/model-info")
    body = res.json()
    if body["model_version"] == "v2":
        metrics = json.loads(api.METRICS_V2_PATH.read_text())
        expected = metrics["test_metrics_optimal_threshold"]["threshold"]
        assert body["decision_threshold"] == pytest.approx(expected, abs=1e-9)


# ------------------------------------------------------------------ predict
@requires_model
def test_predict_single_valid_customer():
    res = client.post("/predict", json=VALID_CUSTOMER)
    assert res.status_code == 200
    body = res.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in ("Low", "Moderate", "High")
    assert body["would_flag_as_churn_risk"] == (body["churn_probability"] >= body["decision_threshold"])


@requires_model
def test_predict_is_deterministic():
    r1 = client.post("/predict", json=VALID_CUSTOMER).json()
    r2 = client.post("/predict", json=VALID_CUSTOMER).json()
    assert r1["churn_probability"] == r2["churn_probability"]


@requires_model
def test_predict_risk_band_consistent_with_threshold():
    """A customer scored well above the threshold should be flagged High."""
    high_risk_customer = dict(VALID_CUSTOMER, days_since_last_login=180, support_tickets=9,
                               avg_watch_hours_week=0.0, tenure_months=1)
    res = client.post("/predict", json=high_risk_customer)
    body = res.json()
    if body["churn_probability"] >= body["decision_threshold"]:
        assert body["risk_band"] == "High"
        assert body["would_flag_as_churn_risk"] is True


def test_predict_rejects_missing_required_field():
    bad_payload = dict(VALID_CUSTOMER)
    del bad_payload["age"]
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422


def test_predict_rejects_out_of_range_value():
    bad_payload = dict(VALID_CUSTOMER, age=-5)
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422


def test_predict_rejects_invalid_categorical():
    bad_payload = dict(VALID_CUSTOMER, plan_type="Ultra Deluxe")
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422


# ------------------------------------------------------------ predict/batch
@requires_model
def test_predict_batch_returns_one_result_per_customer():
    res = client.post("/predict/batch", json=[VALID_CUSTOMER, VALID_CUSTOMER, VALID_CUSTOMER])
    assert res.status_code == 200
    body = res.json()
    assert len(body) == 3
    for row in body:
        assert 0.0 <= row["churn_probability"] <= 1.0


def test_predict_batch_rejects_empty_list():
    res = client.post("/predict/batch", json=[])
    assert res.status_code == 400


@requires_model
def test_predict_batch_matches_single_predict():
    single = client.post("/predict", json=VALID_CUSTOMER).json()
    batch = client.post("/predict/batch", json=[VALID_CUSTOMER]).json()
    assert batch[0]["churn_probability"] == single["churn_probability"]


# ---------------------------------------------------------------- portfolio
@requires_data
def test_portfolio_shape_and_ranges():
    res = client.get("/portfolio")
    assert res.status_code == 200
    body = res.json()
    kpis = body["kpis"]
    assert kpis["total_customers"] > 0
    assert 0.0 <= kpis["churn_rate_pct"] <= 100.0
    assert 0.0 <= kpis["revenue_at_risk_pct"] <= 100.0
    assert kpis["monthly_revenue_at_risk"] <= kpis["total_monthly_revenue"]
    assert "synthetic" in body["data_source"].lower()
    for segment_key in ("churn_rate_by_plan", "churn_rate_by_contract",
                         "churn_rate_by_region", "churn_rate_by_engagement"):
        assert segment_key in body
        assert len(body[segment_key]) > 0


# ---------------------------------------------------------------------- CORS
def test_cors_headers_present_for_cross_origin_request():
    res = client.get("/health", headers={"Origin": "http://localhost:5500"})
    assert res.headers.get("access-control-allow-origin") == "*"

# ------------------------------------------------------------- Explainability & CLV
@requires_model
def test_predict_explainability_and_clv():
    res = client.post("/predict", json=VALID_CUSTOMER)
    assert res.status_code == 200
    body = res.json()
    assert "top_risk_factors" in body
    assert "prescriptive_recommendations" in body
    assert "customer_lifetime_value_at_risk" in body
    assert body["customer_lifetime_value_at_risk"] == pytest.approx(VALID_CUSTOMER["monthly_charge"] * 12, rel=1e-2)
    assert len(body["top_risk_factors"]) > 0
    assert len(body["prescriptive_recommendations"]) > 0


# ----------------------------------------------------------------- What-If Simulation
@requires_model
def test_simulation_endpoint():
    high_risk = dict(VALID_CUSTOMER, days_since_last_login=65, support_tickets=4, contract_type="Monthly")
    sim_payload = {
        "customer": high_risk,
        "switch_to_annual": True,
        "reduce_inactivity_to_days": 1,
        "boost_watch_hours": 8.0,
        "resolve_support_tickets": True,
    }
    res = client.post("/simulate", json=sim_payload)
    assert res.status_code == 200
    body = res.json()
    assert body["simulated_probability"] < body["baseline_probability"]
    assert body["risk_reduction_pct"] > 0
    assert len(body["interventions_applied"]) > 0


# ------------------------------------------------------------ SQLite Prediction History
@requires_model
def test_prediction_history_sqlite():
    client.delete("/predictions/history")
    client.post("/predict", json=VALID_CUSTOMER)
    res = client.get("/predictions/history")
    assert res.status_code == 200
    body = res.json()
    assert body["count"] >= 1
    assert "churn_probability" in body["history"][0]
    assert "top_risk_factor" in body["history"][0]


# ---------------------------------------------------------------- Batch CSV Scoring
@requires_model
def test_predict_csv_upload():
    csv_content = (
        "age,tenure_months,monthly_charge,avg_watch_hours_week,num_devices,support_tickets,days_since_last_login,plan_type,contract_type,payment_method,region\n"
        "28,12,14.99,10.5,2,1,5,Standard,Monthly,Credit Card,North America\n"
        "45,2,19.99,1.0,1,4,75,Premium,Monthly,PayPal,Europe\n"
    )
    files = {"file": ("subscribers.csv", csv_content.encode("utf-8"), "text/csv")}
    res = client.post("/predict/csv", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["total_rows"] == 2
    assert "revenue_at_risk" in body
    assert len(body["preview"]) == 2

# ------------------------------------------------------------- Drift Monitoring
@requires_data
def test_monitoring_drift():
    res = client.get("/monitoring/drift")
    assert res.status_code == 200
    body = res.json()
    assert "metrics" in body
    assert "overall_health" in body
    assert len(body["metrics"]) >= 5


# -------------------------------------------------------- Model Comparison (v1 vs v2)
@requires_model
def test_predict_compare_models():
    res = client.post("/predict/compare", json=VALID_CUSTOMER)
    assert res.status_code == 200
    body = res.json()
    assert "v1_baseline" in body
    assert "v2_ensemble" in body
    assert "models_agree_on_risk" in body
    assert "divergence_explanation" in body
    assert 0.0 <= body["v1_baseline"]["churn_probability"] <= 1.0
    assert 0.0 <= body["v2_ensemble"]["churn_probability"] <= 1.0


# --------------------------------------------------------- Retention ROI Matrix
@requires_data
@requires_model
def test_retention_roi_calculation():
    roi_payload = {
        "outreach_cost_per_customer": 15.0,
        "retention_success_rate": 0.40,
        "target_plan": "All"
    }
    res = client.post("/retention/roi", json=roi_payload)
    assert res.status_code == 200
    body = res.json()
    assert body["total_subscribers_flagged"] > 0
    assert body["outreach_campaign_cost"] > 0
    assert body["expected_saved_subscribers"] > 0
    assert body["gross_annual_revenue_saved"] > 0
    assert "recommendation" in body
