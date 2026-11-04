"""
Week 4 - Comprehensive Report & Insights Presentation
=========================================================
Project: Predicting Customer Churn for a Subscription-Based Streaming Service

Pulls together the EDA findings (Week 2) and model results (Week 3) into a
single executive-ready markdown report, plus a couple of extra "storytelling"
visuals aimed at a non-technical audience (business-impact framing rather
than raw statistical charts).

Run:
    python generate_report.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

BASE = Path(__file__).parent.parent
DATA_PATH = BASE / "Week1_Project_Planning" / "data" / "churn_dataset.csv"
METRICS_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "metrics.json"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_clean_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "customer_id"])
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def business_impact_chart(df):
    """Translate the churn-by-plan finding into a revenue-at-risk story."""
    revenue_at_risk = (
        df[df["churned"] == 1]
        .groupby("plan_type")["monthly_charge"]
        .sum()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    revenue_at_risk.plot(kind="bar", color="#C44E52", ax=ax)
    ax.set_title("Estimated Monthly Revenue at Risk from Churn, by Plan")
    ax.set_ylabel("Revenue at Risk ($/month)")
    ax.set_xlabel("Plan Type")
    for i, v in enumerate(revenue_at_risk):
        ax.text(i, v + 5, f"${v:,.0f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_revenue_at_risk.png", dpi=120)
    plt.close()
    return revenue_at_risk


def engagement_story_chart(df):
    """Simple, non-technical framing: engagement bucket vs churn rate."""
    bins = [-1, 7, 30, 60, 1000]
    labels = ["Active (<=7 days)", "Cooling off (8-30 days)",
              "At risk (31-60 days)", "Dormant (60+ days)"]
    df = df.copy()
    df["engagement_bucket"] = pd.cut(df["days_since_last_login"], bins=bins, labels=labels)
    churn_by_bucket = df.groupby("engagement_bucket", observed=True)["churned"].mean() * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    churn_by_bucket.plot(kind="bar", color="#4C72B0", ax=ax)
    ax.set_title("Churn Rate by Customer Engagement Level")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    for i, v in enumerate(churn_by_bucket):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_engagement_vs_churn.png", dpi=120)
    plt.close()
    return churn_by_bucket


def main():
    df = load_clean_data()
    metrics = json.loads(METRICS_PATH.read_text())

    revenue_at_risk = business_impact_chart(df)
    engagement_churn = engagement_story_chart(df)

    total_customers = len(df)
    churned_customers = df["churned"].sum()
    churn_rate = df["churned"].mean()
    total_revenue_at_risk = revenue_at_risk.sum()
    test_metrics = metrics["test_metrics"]

    report = f"""# Executive Report — Streaming Service Churn Prediction

## Executive Summary
Out of **{total_customers:,}** customers analyzed, **{churned_customers:,}
({churn_rate:.1%})** churned in the observed period, representing an
estimated **${total_revenue_at_risk:,.0f}/month** in recurring revenue at
risk. A predictive model was developed to flag at-risk customers in
advance, achieving **{test_metrics['roc_auc']:.0%} ROC-AUC** and
**{test_metrics['recall']:.0%} recall** on churners in the held-out test
set — meaning the model correctly identifies roughly {test_metrics['recall']:.0%}
of the observed churn cases, making it useful as a customer-risk
prioritization tool.

## Methodology Recap
1. Assembled and cleaned customer usage, billing, and engagement data.
2. Explored patterns across tenure, plan type, engagement, and support
   history to understand what separates retained customers from churners.
3. Engineered features capturing engagement recency and value-for-money.
4. Trained and compared multiple classification models
   ({', '.join(metrics['cv_roc_auc'].keys())}), selected **{metrics['best_model']}**
   as the best performer, and tuned it via cross-validated grid search.
5. Validated performance on a held-out test set never seen during training.

## Key Insights

### 1. Engagement recency is the strongest churn signal
Customers who haven't logged in recently churn at dramatically higher
rates than active users:

| Engagement Level | Churn Rate |
|---|---|
{chr(10).join(f"| {k} | {v:.1f}% |" for k, v in engagement_churn.items())}

*(See `outputs/02_engagement_vs_churn.png`)*

### 2. Revenue exposure varies significantly by plan
{chr(10).join(f"- **{k}**: ${v:,.0f}/month at risk" for k, v in revenue_at_risk.items())}

*(See `outputs/01_revenue_at_risk.png`)*

### 3. Monthly (vs. annual) contracts and low tenure correlate with churn
Shorter-tenure, monthly-billed customers are meaningfully more likely to
churn than customers on annual plans with longer histories — consistent
with the exploratory analysis in Week 2.

### 4. Model performance
| Metric | Value |
|---|---|
{chr(10).join(f"| {k.replace('_', ' ').title()} | {v:.3f} |" for k, v in test_metrics.items())}

## Presentation Strategy for Non-Technical Stakeholders
- Lead with the **dollar figure**, not the AUC — "$X/month at risk" lands
  better with executives than a statistical metric.
- Use the **engagement-bucket chart** as the single visual anchor of the
  presentation; it maps directly to an actionable retention trigger
  ("re-engage anyone who hasn't logged in for 30 days").
- Frame the model as a **prioritization tool** for the retention team,
  not a black box — "this ranks customers by risk so outreach can focus
  on the top 20%," rather than describing the algorithm.
- Use a before/after narrative: "today we react to cancellations; with
  this model we can act 30+ days earlier."

## Recommendations & Next Steps
1. Pilot a proactive outreach campaign targeting customers in the
   "cooling off" and "at risk" engagement buckets.
2. Investigate whether incentivizing annual contracts (discount, bundled
   perks) reduces churn among monthly subscribers.
3. Collect additional behavioral data (e.g., content genre preferences,
   customer support sentiment) to improve model recall beyond
   {test_metrics['recall']:.0%}.
4. Re-train the model quarterly as usage patterns evolve, and monitor for
   data/model drift.
"""

    report_path = OUT_DIR / "executive_report.md"
    report_path.write_text(report)
    print(f"Executive report written to {report_path}")
    print(f"Churn rate: {churn_rate:.1%} | Revenue at risk: ${total_revenue_at_risk:,.0f}/mo")


if __name__ == "__main__":
    main()
