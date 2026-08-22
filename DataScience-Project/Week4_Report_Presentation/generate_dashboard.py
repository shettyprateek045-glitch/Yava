"""
Week 4 (Advanced) - Interactive Executive Dashboard
=======================================================
Generates a single self-contained HTML file with interactive charts
(Chart.js via CDN) summarizing the churn analysis: KPI cards, revenue at
risk by plan, churn by engagement bucket, churn by contract type, and
model performance. No server, no build step — open the file directly in
a browser, or publish it with GitHub Pages.

Run:
    python generate_dashboard.py
Output:
    outputs/dashboard.html
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def build_dashboard_data(df, metrics):
    total_customers = int(len(df))
    churn_rate = float(df["churned"].mean())
    revenue_at_risk = df[df["churned"] == 1].groupby("plan_type")["monthly_charge"].sum()

    bins = [-1, 7, 30, 60, 1000]
    labels = ["Active (0-7d)", "Cooling off (8-30d)", "At risk (31-60d)", "Dormant (60d+)"]
    engagement = pd.cut(df["days_since_last_login"], bins=bins, labels=labels)
    churn_by_engagement = df.groupby(engagement, observed=True)["churned"].mean() * 100

    churn_by_contract = df.groupby("contract_type")["churned"].mean() * 100
    churn_by_region = df.groupby("region")["churned"].mean() * 100
    plan_counts = df["plan_type"].value_counts()

    return {
        "kpis": {
            "total_customers": total_customers,
            "churn_rate_pct": round(churn_rate * 100, 1),
            "revenue_at_risk": round(float(revenue_at_risk.sum()), 0),
            "model_name": metrics.get("best_model", "N/A"),
            "model_roc_auc": round(metrics.get("test_metrics", {}).get("roc_auc", 0), 3),
            "model_recall": round(metrics.get("test_metrics", {}).get("recall", 0), 3),
        },
        "revenue_at_risk_by_plan": {
            "labels": list(revenue_at_risk.index),
            "values": [round(v, 0) for v in revenue_at_risk.values],
        },
        "churn_by_engagement": {
            "labels": list(churn_by_engagement.index.astype(str)),
            "values": [round(v, 1) for v in churn_by_engagement.values],
        },
        "churn_by_contract": {
            "labels": list(churn_by_contract.index),
            "values": [round(v, 1) for v in churn_by_contract.values],
        },
        "churn_by_region": {
            "labels": list(churn_by_region.index),
            "values": [round(v, 1) for v in churn_by_region.values],
        },
        "plan_distribution": {
            "labels": list(plan_counts.index),
            "values": [int(v) for v in plan_counts.values],
        },
        "cv_model_comparison": {
            "labels": list(metrics.get("cv_roc_auc", {}).keys()),
            "values": [round(v, 3) for v in metrics.get("cv_roc_auc", {}).values()],
        },
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Churn Prediction — Executive Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {
    --blue: #4C72B0; --green: #55A868; --red: #C44E52;
    --orange: #E0922C; --purple: #8172B2; --grey: #8C8C8C;
    --bg: #F7F8FA; --card-bg: #FFFFFF; --text: #22262B; --muted: #6B7280;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text);
  }
  header {
    padding: 28px 32px 18px; background: linear-gradient(135deg, var(--blue), var(--purple));
    color: white;
  }
  header h1 { margin: 0 0 4px; font-size: 22px; }
  header p { margin: 0; opacity: 0.9; font-size: 13px; }
  .container { padding: 24px 32px 48px; max-width: 1200px; margin: 0 auto; }
  .kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; margin-bottom: 28px;
  }
  .kpi-card {
    background: var(--card-bg); border-radius: 12px; padding: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-top: 4px solid var(--blue);
  }
  .kpi-card.churn { border-top-color: var(--red); }
  .kpi-card.revenue { border-top-color: var(--orange); }
  .kpi-card.model { border-top-color: var(--green); }
  .kpi-card .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .kpi-card .value { font-size: 28px; font-weight: 700; margin-top: 4px; }
  .kpi-card .sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .chart-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 20px;
  }
  .chart-card {
    background: var(--card-bg); border-radius: 12px; padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  .chart-card h3 { margin: 0 0 12px; font-size: 14px; color: var(--text); }
  .chart-card .wrap { position: relative; height: 260px; }
  footer { text-align: center; color: var(--muted); font-size: 12px; padding: 20px; }
  .badge {
    display: inline-block; background: rgba(255,255,255,0.15); padding: 3px 10px;
    border-radius: 999px; font-size: 11px; margin-top: 8px;
  }
</style>
</head>
<body>
<header>
  <h1>Streaming Service Churn — Executive Dashboard</h1>
  <p>Synthetic dataset · Weeks 1–4 technical pipeline · Generated automatically from live pipeline outputs</p>
  <span class="badge">Prototype model — not production-validated</span>
</header>

<div class="container">
  <div class="kpi-grid" id="kpi-grid"></div>

  <div class="chart-grid">
    <div class="chart-card">
      <h3>Revenue at Risk by Plan ($/month)</h3>
      <div class="wrap"><canvas id="chartRevenue"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Churn Rate by Engagement Level (%)</h3>
      <div class="wrap"><canvas id="chartEngagement"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Churn Rate: Monthly vs Annual Contract (%)</h3>
      <div class="wrap"><canvas id="chartContract"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Churn Rate by Region (%)</h3>
      <div class="wrap"><canvas id="chartRegion"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Customer Distribution by Plan</h3>
      <div class="wrap"><canvas id="chartPlanDist"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Model Comparison (5-Fold CV ROC-AUC)</h3>
      <div class="wrap"><canvas id="chartModels"></canvas></div>
    </div>
  </div>
</div>

<footer>Generated by generate_dashboard.py — open directly in any browser, no server required.</footer>

<script>
const DATA = __DATA_JSON__;
const PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#E0922C", "#8172B2", "#64B5CD"];

function kpiCard(cls, label, value, sub) {
  const div = document.createElement("div");
  div.className = "kpi-card " + cls;
  div.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub || ""}</div>`;
  return div;
}

const grid = document.getElementById("kpi-grid");
grid.appendChild(kpiCard("", "Total Customers", DATA.kpis.total_customers.toLocaleString(), "in synthetic dataset"));
grid.appendChild(kpiCard("churn", "Churn Rate", DATA.kpis.churn_rate_pct + "%", "of analyzed customers"));
grid.appendChild(kpiCard("revenue", "Revenue at Risk", "$" + DATA.kpis.revenue_at_risk.toLocaleString() + "/mo", "from churned customers"));
grid.appendChild(kpiCard("model", "Model", DATA.kpis.model_name, "ROC-AUC " + DATA.kpis.model_roc_auc + " · Recall " + DATA.kpis.model_recall));

const commonOpts = {
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { display: false } },
};

new Chart(document.getElementById("chartRevenue"), {
  type: "bar",
  data: { labels: DATA.revenue_at_risk_by_plan.labels, datasets: [{ data: DATA.revenue_at_risk_by_plan.values, backgroundColor: PALETTE[2] }] },
  options: commonOpts,
});

new Chart(document.getElementById("chartEngagement"), {
  type: "bar",
  data: { labels: DATA.churn_by_engagement.labels, datasets: [{ data: DATA.churn_by_engagement.values, backgroundColor: PALETTE[0] }] },
  options: commonOpts,
});

new Chart(document.getElementById("chartContract"), {
  type: "bar",
  data: { labels: DATA.churn_by_contract.labels, datasets: [{ data: DATA.churn_by_contract.values, backgroundColor: PALETTE[3] }] },
  options: { ...commonOpts, indexAxis: "y" },
});

new Chart(document.getElementById("chartRegion"), {
  type: "bar",
  data: { labels: DATA.churn_by_region.labels, datasets: [{ data: DATA.churn_by_region.values, backgroundColor: PALETTE[4] }] },
  options: commonOpts,
});

new Chart(document.getElementById("chartPlanDist"), {
  type: "doughnut",
  data: { labels: DATA.plan_distribution.labels, datasets: [{ data: DATA.plan_distribution.values, backgroundColor: PALETTE }] },
  options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } },
});

new Chart(document.getElementById("chartModels"), {
  type: "bar",
  data: { labels: DATA.cv_model_comparison.labels, datasets: [{ data: DATA.cv_model_comparison.values, backgroundColor: PALETTE[1] }] },
  options: { ...commonOpts, scales: { y: { min: 0.5, max: 0.7 } } },
});
</script>
</body>
</html>
"""


def main():
    df = load_clean_data()
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    dashboard_data = build_dashboard_data(df, metrics)

    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(dashboard_data))
    out_path = OUT_DIR / "dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")
    print(json.dumps(dashboard_data["kpis"], indent=2))


if __name__ == "__main__":
    main()
