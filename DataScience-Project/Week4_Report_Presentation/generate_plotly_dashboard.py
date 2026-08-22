"""
Week 4 (v2) - Interactive Plotly Dashboard
==============================================
Generates a single self-contained HTML file using Plotly.js (via CDN) —
genuinely interactive charts with hover tooltips, zoom/pan, legend
toggling, and a dropdown to switch the primary breakdown dimension
(plan type / region / contract type). No server or build step required;
open directly in a browser.

This does NOT require the `plotly` Python package — chart data is
serialized to JSON and rendered client-side with Plotly.js from CDN, so
it works even in environments without network access to pip install.

Run:
    python generate_plotly_dashboard.py
Output:
    outputs/interactive_dashboard.html
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
DATA_PATH = BASE / "Week1_Project_Planning" / "data" / "churn_dataset.csv"
METRICS_V1_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "metrics.json"
METRICS_V2_PATH = BASE / "Week3_ML_Model_Development" / "outputs" / "metrics_v2.json"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_clean_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "customer_id"])
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    q1, q3 = df["monthly_charge"].quantile([0.25, 0.75])
    df["monthly_charge"] = df["monthly_charge"].clip(upper=q3 + 1.5 * (q3 - q1))
    return df


def churn_rate_by(df, dim):
    g = df.groupby(dim, observed=True)["churned"].agg(["mean", "count"])
    g["mean"] = (g["mean"] * 100).round(1)
    return g.sort_values("mean", ascending=False)


def revenue_at_risk_by(df, dim):
    return (
        df[df["churned"] == 1].groupby(dim, observed=True)["monthly_charge"]
        .sum().round(0).sort_values(ascending=False)
    )


def main():
    df = load_clean_data()
    df["engagement_bucket"] = pd.cut(
        df["days_since_last_login"], bins=[-1, 7, 30, 60, 1000],
        labels=["Active (<=7d)", "Cooling off (8-30d)", "At risk (31-60d)", "Dormant (60+d)"]
    )

    metrics_v1 = json.loads(METRICS_V1_PATH.read_text()) if METRICS_V1_PATH.exists() else None
    metrics_v2 = json.loads(METRICS_V2_PATH.read_text()) if METRICS_V2_PATH.exists() else None

    dims = ["plan_type", "region", "contract_type", "payment_method"]
    churn_by_dim = {d: churn_rate_by(df, d) for d in dims}
    revenue_by_dim = {d: revenue_at_risk_by(df, d) for d in dims}
    engagement_churn = churn_rate_by(df, "engagement_bucket")

    total_customers = int(len(df))
    churned = int(df["churned"].sum())
    churn_rate = round(df["churned"].mean() * 100, 1)
    total_revenue_at_risk = int(df[df["churned"] == 1]["monthly_charge"].sum())

    # Scatter sample (cap points for a snappy browser render)
    scatter_df = df.sample(n=min(1200, len(df)), random_state=42)
    scatter_traces = []
    for plan in sorted(scatter_df["plan_type"].unique()):
        sub = scatter_df[scatter_df["plan_type"] == plan]
        scatter_traces.append({
            "name": plan,
            "tenure": sub["tenure_months"].tolist(),
            "charge": sub["monthly_charge"].tolist(),
            "churn": sub["churned"].tolist(),
        })

    # Model comparison (v1 vs v2, if available)
    model_compare = None
    if metrics_v1 and metrics_v2:
        model_compare = {
            "labels": ["v1 (baseline)", "v2 (engineered features + ensemble)"],
            "roc_auc": [
                round(metrics_v1["test_metrics"]["roc_auc"], 3),
                round(metrics_v2["test_metrics_default_threshold"]["roc_auc"], 3),
            ],
            "f1": [
                round(metrics_v1["test_metrics"]["f1_score"], 3),
                round(metrics_v2["test_metrics_optimal_threshold"]["f1_score"], 3),
            ],
        }

    data_payload = {
        "dims": dims,
        "churn_by_dim": {d: {"labels": v.index.astype(str).tolist(), "rate": v["mean"].tolist(),
                              "count": v["count"].tolist()} for d, v in churn_by_dim.items()},
        "revenue_by_dim": {d: {"labels": v.index.astype(str).tolist(), "values": v.values.tolist()}
                           for d, v in revenue_by_dim.items()},
        "engagement": {"labels": engagement_churn.index.astype(str).tolist(),
                       "rate": engagement_churn["mean"].tolist()},
        "scatter": scatter_traces,
        "model_compare": model_compare,
        "kpis": {
            "total_customers": total_customers, "churned": churned,
            "churn_rate": churn_rate, "revenue_at_risk": total_revenue_at_risk,
        },
    }

    html = build_html(data_payload)
    out_path = OUT_DIR / "interactive_dashboard.html"
    out_path.write_text(html)
    print(f"Interactive Plotly dashboard written to {out_path}")


def build_html(data):
    data_json = json.dumps(data)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Churn Analytics — Interactive Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
          background: #f5f7fa; margin: 0; padding: 24px; color: #222; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 24px; font-size: 13px; }}
  .kpi-row {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .kpi-card {{ background: white; border-radius: 10px; padding: 16px 20px; flex: 1;
               min-width: 150px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .kpi-label {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: .04em; }}
  .kpi-value {{ font-size: 26px; font-weight: 700; margin-top: 4px; }}
  .kpi-value.red {{ color: #C44E52; }}
  .kpi-value.blue {{ color: #4C72B0; }}
  .kpi-value.green {{ color: #55A868; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
  .chart-card {{ background: white; border-radius: 10px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .chart-card h3 {{ font-size: 14px; margin: 4px 8px 0; color: #333; }}
  select {{ margin: 4px 8px; padding: 4px 8px; border-radius: 6px; border: 1px solid #ccc; }}
</style>
</head>
<body>
  <h1>Streaming Service Churn — Interactive Analytics Dashboard</h1>
  <div class="subtitle">Synthetic dataset · generated by Week 4 technical pipeline · filter charts with the dropdowns below</div>

  <div class="kpi-row">
    <div class="kpi-card"><div class="kpi-label">Total Customers</div><div class="kpi-value blue" id="kpi-total"></div></div>
    <div class="kpi-card"><div class="kpi-label">Churned</div><div class="kpi-value red" id="kpi-churned"></div></div>
    <div class="kpi-card"><div class="kpi-label">Churn Rate</div><div class="kpi-value red" id="kpi-rate"></div></div>
    <div class="kpi-card"><div class="kpi-label">Revenue at Risk / mo</div><div class="kpi-value green" id="kpi-revenue"></div></div>
  </div>

  <div class="grid">
    <div class="chart-card">
      <h3>Churn Rate by Dimension</h3>
      <select id="churn-dim-select"></select>
      <div id="churn-by-dim-chart" style="height:360px;"></div>
    </div>
    <div class="chart-card">
      <h3>Revenue at Risk by Dimension ($/mo)</h3>
      <select id="revenue-dim-select"></select>
      <div id="revenue-by-dim-chart" style="height:360px;"></div>
    </div>
    <div class="chart-card">
      <h3>Churn Rate by Engagement Level</h3>
      <div id="engagement-chart" style="height:360px;"></div>
    </div>
    <div class="chart-card">
      <h3>Tenure vs. Monthly Charge (colored by plan — click legend to isolate)</h3>
      <div id="scatter-chart" style="height:360px;"></div>
    </div>
    <div class="chart-card" id="model-card" style="display:none;">
      <h3>Model Comparison: v1 vs v2</h3>
      <div id="model-compare-chart" style="height:320px;"></div>
    </div>
  </div>

<script>
const DATA = {data_json};

document.getElementById('kpi-total').textContent = DATA.kpis.total_customers.toLocaleString();
document.getElementById('kpi-churned').textContent = DATA.kpis.churned.toLocaleString();
document.getElementById('kpi-rate').textContent = DATA.kpis.churn_rate + '%';
document.getElementById('kpi-revenue').textContent = '$' + DATA.kpis.revenue_at_risk.toLocaleString();

const layoutBase = {{
  margin: {{t: 10, r: 10, b: 60, l: 50}},
  font: {{family: 'Helvetica, Arial, sans-serif', size: 11}},
  paper_bgcolor: 'white', plot_bgcolor: 'white',
}};

// --- Churn by dimension (dropdown-driven) ---
const churnSelect = document.getElementById('churn-dim-select');
DATA.dims.forEach(d => {{
  const opt = document.createElement('option');
  opt.value = d; opt.textContent = d.replace('_', ' ');
  churnSelect.appendChild(opt);
}});

function renderChurnByDim(dim) {{
  const d = DATA.churn_by_dim[dim];
  Plotly.react('churn-by-dim-chart', [{{
    x: d.labels, y: d.rate, type: 'bar',
    marker: {{color: '#4C72B0'}},
    text: d.rate.map(v => v + '%'), textposition: 'auto',
    hovertemplate: '%{{x}}<br>Churn rate: %{{y}}%<extra></extra>',
  }}], {{...layoutBase, yaxis: {{title: 'Churn Rate (%)'}}}}, {{responsive: true, displayModeBar: false}});
}}
churnSelect.addEventListener('change', e => renderChurnByDim(e.target.value));
renderChurnByDim(DATA.dims[0]);

// --- Revenue at risk by dimension (dropdown-driven) ---
const revSelect = document.getElementById('revenue-dim-select');
DATA.dims.forEach(d => {{
  const opt = document.createElement('option');
  opt.value = d; opt.textContent = d.replace('_', ' ');
  revSelect.appendChild(opt);
}});

function renderRevenueByDim(dim) {{
  const d = DATA.revenue_by_dim[dim];
  Plotly.react('revenue-by-dim-chart', [{{
    x: d.labels, y: d.values, type: 'bar',
    marker: {{color: '#C44E52'}},
    hovertemplate: '%{{x}}<br>$%{{y:,.0f}}/mo<extra></extra>',
  }}], {{...layoutBase, yaxis: {{title: 'Revenue at Risk ($/mo)'}}}}, {{responsive: true, displayModeBar: false}});
}}
revSelect.addEventListener('change', e => renderRevenueByDim(e.target.value));
renderRevenueByDim(DATA.dims[0]);

// --- Engagement chart (static) ---
Plotly.newPlot('engagement-chart', [{{
  x: DATA.engagement.labels, y: DATA.engagement.rate, type: 'bar',
  marker: {{color: '#E0922C'}},
  text: DATA.engagement.rate.map(v => v + '%'), textposition: 'auto',
  hovertemplate: '%{{x}}<br>Churn rate: %{{y}}%<extra></extra>',
}}], {{...layoutBase, yaxis: {{title: 'Churn Rate (%)'}}}}, {{responsive: true, displayModeBar: false}});

// --- Scatter (one trace per plan, click legend to isolate) ---
const palette = ['#4C72B0', '#55A868', '#C44E52', '#8172B2'];
const scatterTraces = DATA.scatter.map((s, i) => ({{
  x: s.tenure, y: s.charge, mode: 'markers', name: s.name,
  marker: {{
    color: s.churn.map(c => c === 1 ? '#C44E52' : palette[i % palette.length]),
    size: 6, opacity: 0.6,
  }},
  text: s.churn.map(c => c === 1 ? 'Churned' : 'Retained'),
  hovertemplate: 'Tenure: %{{x}}mo<br>Charge: $%{{y}}<br>%{{text}}<extra></extra>',
}}));
Plotly.newPlot('scatter-chart', scatterTraces, {{
  ...layoutBase, xaxis: {{title: 'Tenure (months)'}}, yaxis: {{title: 'Monthly Charge ($)'}},
  legend: {{orientation: 'h', y: -0.2}},
}}, {{responsive: true, displayModeBar: false}});

// --- Model comparison (if v2 metrics exist) ---
if (DATA.model_compare) {{
  document.getElementById('model-card').style.display = 'block';
  Plotly.newPlot('model-compare-chart', [
    {{x: DATA.model_compare.labels, y: DATA.model_compare.roc_auc, name: 'ROC-AUC', type: 'bar', marker: {{color: '#4C72B0'}}}},
    {{x: DATA.model_compare.labels, y: DATA.model_compare.f1, name: 'F1 (optimal threshold)', type: 'bar', marker: {{color: '#55A868'}}}},
  ], {{...layoutBase, barmode: 'group', yaxis: {{title: 'Score', range: [0, 1]}}}}, {{responsive: true, displayModeBar: false}});
}}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
