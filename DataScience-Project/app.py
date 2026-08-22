"""
Streamlit Live Demo — Churn Risk Predictor
==============================================
A small interactive app that loads the trained model (v2 if available,
falling back to v1) and lets a user enter a hypothetical customer's
attributes to get a live churn-risk score, plus a portfolio-level view
of the underlying dataset.

This is a genuine live demo, not a mockup — it loads the real
`.joblib` pipeline saved by Week 3 and calls `.predict_proba()` on
whatever the user enters.

Run:
    streamlit run app.py
(requires: pip install streamlit pandas numpy scikit-learn joblib plotly)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
DATA_PATH = BASE / "Week1_Project_Planning" / "data" / "churn_dataset.csv"
MODEL_V2_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "churn_model_v2.joblib"
MODEL_V1_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "churn_model.joblib"
METRICS_V2_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "metrics_v2.json"
METRICS_V1_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "metrics.json"

st.set_page_config(page_title="Churn Risk Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    """Loads the trained model plus its *actual* F1-optimized decision
    threshold, so the UI never hardcodes a cutoff that disagrees with
    what MODEL_CARD.md documents."""
    if MODEL_V2_PATH.exists():
        metrics = json.loads(METRICS_V2_PATH.read_text())
        threshold = metrics["test_metrics_optimal_threshold"]["threshold"]
        return joblib.load(MODEL_V2_PATH), "v2 (engineered features + stacking ensemble)", metrics, threshold
    metrics = json.loads(METRICS_V1_PATH.read_text())
    # v1 has no engineered-feature threshold file at this path; fall back to
    # the documented default of 0.5 if advanced_interpretability.py hasn't run.
    interp_path = BASE / "Week3_ML_Model_Development" / "outputs" / "advanced" / "interpretability_summary.json"
    threshold = 0.5
    if interp_path.exists():
        threshold = json.loads(interp_path.read_text())["threshold_optimization"]["optimal_threshold"]
    return joblib.load(MODEL_V1_PATH), "v1 (baseline logistic regression)", metrics, threshold


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "customer_id"])
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def engineer_features_for_v2(row: dict) -> dict:
    row = dict(row)
    row["charge_per_watch_hour"] = row["monthly_charge"] / (row["avg_watch_hours_week"] + 0.1)
    row["is_inactive"] = int(row["days_since_last_login"] > 30)
    row["log_days_since_login"] = np.log1p(row["days_since_last_login"])
    row["watch_hours_per_device"] = row["avg_watch_hours_week"] / row["num_devices"]
    row["engagement_score"] = row["avg_watch_hours_week"] / (row["days_since_last_login"] + 1)
    row["support_tenure_ratio"] = row["support_tickets"] / (row["tenure_months"] + 1)
    tenure = row["tenure_months"]
    row["tenure_bucket"] = (
        "0-6mo" if tenure <= 6 else "7-12mo" if tenure <= 12 else "13-24mo" if tenure <= 24 else "25mo+"
    )
    return row


model, model_label, metrics, DECISION_THRESHOLD = load_model()
df = load_data()
is_v2 = "v2" in model_label

st.title("📉 Streaming Service Churn — Risk Predictor")
st.caption(f"Live demo backed by the trained model: **{model_label}**")

st.warning(
    "⚠️ **Prototype trained on synthetic data.** This model and every number on "
    "this page come from a synthetic dataset generated for this project "
    "(`Week1_Project_Planning/generate_dataset.py`) — no real customer records "
    "were used. Predictions are illustrative only and are **not validated for "
    "production or real-world retention decisions**. See `MODEL_CARD.md` for "
    "full limitations.",
    icon="⚠️",
)
st.info(
    f"Risk bands below are calculated relative to the model's **documented "
    f"F1-optimal decision threshold of {DECISION_THRESHOLD:.3f}** "
    f"(see `MODEL_CARD.md`), not an arbitrary 0.5 cutoff.",
    icon="🎯",
)

tab1, tab2 = st.tabs(["🔮 Predict a Customer", "📊 Portfolio Overview"])

with tab1:
    st.subheader("Enter customer attributes")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 13, 85, 35)
        region = st.selectbox("Region", sorted(df["region"].unique()))
        plan_type = st.selectbox("Plan Type", sorted(df["plan_type"].unique()))
        contract_type = st.selectbox("Contract Type", sorted(df["contract_type"].unique()))

    with col2:
        payment_method = st.selectbox("Payment Method", sorted(df["payment_method"].unique()))
        tenure_months = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charge = st.slider("Monthly Charge ($)", 4.99, 29.99, 13.99)
        num_devices = st.slider("Number of Devices", 1, 4, 2)

    with col3:
        avg_watch_hours_week = st.slider("Avg. Watch Hours / Week", 0.0, 40.0, 8.0)
        support_tickets = st.slider("Support Tickets (recent)", 0, 10, 0)
        days_since_last_login = st.slider("Days Since Last Login", 0, 180, 5)

    if st.button("Predict Churn Risk", type="primary"):
        row = {
            "age": age, "region": region, "tenure_months": tenure_months,
            "plan_type": plan_type, "contract_type": contract_type,
            "payment_method": payment_method, "monthly_charge": monthly_charge,
            "avg_watch_hours_week": avg_watch_hours_week, "num_devices": num_devices,
            "support_tickets": support_tickets, "days_since_last_login": days_since_last_login,
        }
        if is_v2:
            row = engineer_features_for_v2(row)
        input_df = pd.DataFrame([row])
        proba = model.predict_proba(input_df)[0, 1]

        # Risk bands are anchored to the model's actual optimized threshold
        # (DECISION_THRESHOLD), not a hardcoded 0.5/0.3 split, so this stays
        # consistent with MODEL_CARD.md no matter which model version is loaded.
        moderate_cutoff = DECISION_THRESHOLD * 0.6
        flagged = proba >= DECISION_THRESHOLD
        if flagged:
            risk_label = "🔴 High Risk (above optimized threshold)"
        elif proba >= moderate_cutoff:
            risk_label = "🟡 Moderate Risk"
        else:
            risk_label = "🟢 Low Risk"

        st.metric("Predicted Churn Probability", f"{proba:.1%}", risk_label)
        st.progress(min(proba, 1.0))
        st.caption(
            f"Decision threshold: **{DECISION_THRESHOLD:.3f}** · "
            f"Moderate-risk band starts at **{moderate_cutoff:.3f}**"
        )

        if flagged:
            st.warning("This customer crosses the model's optimized churn-risk threshold — "
                       "consider proactive retention outreach.")
        elif proba >= moderate_cutoff:
            st.info("Moderate risk — worth monitoring, especially if engagement drops further.")
        else:
            st.success("Low risk based on current engagement and account profile.")

with tab2:
    st.subheader("Dataset & Model Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churned", f"{int(df['churned'].sum()):,}")
    c3.metric("Churn Rate", f"{df['churned'].mean():.1%}")
    if is_v2:
        roc = metrics["test_metrics_default_threshold"]["roc_auc"]
    else:
        roc = metrics["test_metrics"]["roc_auc"]
    c4.metric("Model ROC-AUC", f"{roc:.3f}")

    st.markdown("#### Churn rate by plan type")
    st.bar_chart(df.groupby("plan_type")["churned"].mean() * 100)

    st.markdown("#### Churn rate by engagement (days since last login)")
    bucketed = pd.cut(df["days_since_last_login"], bins=[-1, 7, 30, 60, 1000],
                       labels=["Active", "Cooling off", "At risk", "Dormant"])
    st.bar_chart(df.groupby(bucketed, observed=True)["churned"].mean() * 100)

    with st.expander("Raw model metrics"):
        st.json(metrics)

st.markdown("---")
st.caption("Synthetic data for demonstration purposes — not real customer records.")
