# Week 2 — EDA & Visualization Framework (Technical Component)

Runs a complete exploratory data analysis pass on the Week 1 synthetic churn
dataset and produces the visualization set referenced in the DOC framework.

| File | Purpose |
|---|---|
| `eda_analysis.py` | Cleans the data (missing values, duplicates, outliers), then runs univariate, bivariate, and multivariate analysis with matplotlib/seaborn. |
| `diagrams.py` | Generates the EDA process flowchart (Graphviz) showing the decision points for cleaning and the three analysis tiers. |
| `outputs/*.png` | Generated charts: distribution histograms, class balance, boxplots by churn status, churn-by-plan bar chart, correlation heatmap, pairplot. |
| `outputs/diagrams/01_eda_process_flowchart.png` | EDA workflow: quality check → clean/dedupe decision → outlier decision → univariate → bivariate → multivariate → insights → report. |
| `outputs/eda_summary_report.md` | Auto-generated written summary of findings (data quality notes + key correlations). |

## To run
```bash
pip install pandas numpy matplotlib seaborn tabulate graphviz
python eda_analysis.py
python diagrams.py
```

## Key generated visuals
1. `01_univariate_histograms.png` — distribution of all numeric features
2. `02_churn_class_balance.png` — churned vs. retained customer counts
3. `03_bivariate_boxplots.png` — tenure / watch hours / login recency vs. churn
4. `04_churn_by_plan_type.png` — churn rate stacked by subscription plan
5. `05_correlation_heatmap.png` — correlation matrix across numeric features
6. `06_pairplot.png` — pairwise relationships colored by churn status
