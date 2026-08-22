# Technical Implementation Guide

The assignment asks for four DOC planning/report deliverables. This repository is the **technical companion** that makes those plans reproducible on GitHub.

| Week | Technical work | Main output |
|---|---|---|
| Week 1 | Generate a reproducible synthetic churn dataset, scaffold the proposed project structure, and generate planning diagrams | `data/churn_dataset.csv`, architecture/methodology PNGs |
| Week 2 | Clean the synthetic data and perform univariate, bivariate and multivariate EDA | Six EDA charts + EDA summary |
| Week 3 | Build preprocessing + model comparison + hyperparameter tuning + held-out evaluation pipeline | Trained `.joblib` model, metrics, confusion matrix, ROC curve, model comparison, coefficient chart |
| Week 4 | Convert analytical results into business-facing insights and recommendations | Executive report + revenue-risk and engagement visuals |

## Reproducibility

Run:

```bash
pip install -r requirements.txt
python run_all.py
```

The dataset is synthetic and generated with a fixed random seed. No real or proprietary customer data is included.

## Important interpretation

The Week 3 model is a **classification prototype**, not a production churn predictor. Its test ROC-AUC is approximately 0.63, so the results should be presented as a proof of workflow and risk-prioritization example rather than a claim of production readiness.
