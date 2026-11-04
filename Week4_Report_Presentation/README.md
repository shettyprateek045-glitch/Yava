# Week 4 — Comprehensive Report & Presentation Plan (Technical Component)

Pulls together the EDA (Week 2) and model results (Week 3) into a single
executive-facing report with business-framed (not purely statistical)
visualizations.

| File | Purpose |
|---|---|
| `generate_report.py` | Builds the executive markdown report and two storytelling charts. |
| `diagrams.py` | Generates the insight-to-action pipeline and stakeholder communication framework diagrams (Graphviz). |
| `outputs/executive_report.md` | Final report: executive summary, methodology recap, key insights, presentation strategy, recommendations. |
| `outputs/01_revenue_at_risk.png` | Monthly recurring revenue at risk from churn, broken down by plan type. |
| `outputs/02_engagement_vs_churn.png` | Churn rate by customer engagement bucket — the single chart recommended for stakeholder presentations. |
| `outputs/diagrams/01_insight_to_action_pipeline.png` | How a model score becomes a business action: score → rank → segment → outreach → measure → feed back into next retrain. |
| `outputs/diagrams/02_communication_framework.png` | How technical results get translated for non-technical stakeholders: metric → business meaning → narrative → single anchor visual → audience. |

## To run
```bash
pip install pandas numpy matplotlib seaborn graphviz
python generate_report.py
python diagrams.py
```
Requires `Week3_ML_Model_Development/outputs/metrics.json` to already exist
(run Week 3's script first).

## Interactive Plotly dashboard (`generate_plotly_dashboard.py`)
A second, genuinely interactive dashboard built with Plotly.js (via CDN
— no `plotly` Python package needed). Unlike the static Chart.js
dashboard above, this one supports hover tooltips, zoom/pan, legend
toggling, and dropdown filters to switch the breakdown dimension
(plan / region / contract / payment method) live in the browser. It also
plots a v1-vs-v2 model comparison if `metrics_v2.json` exists.

```bash
python generate_plotly_dashboard.py
```
Output: `outputs/interactive_dashboard.html` — open directly in a browser.
