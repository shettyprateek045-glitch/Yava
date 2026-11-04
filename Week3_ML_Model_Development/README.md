# Week 3 — ML Model Development & Evaluation (Technical Component)

Full model development pipeline for churn prediction: preprocessing,
feature engineering, model comparison, hyperparameter tuning, and
evaluation on a held-out test set.

| File | Purpose |
|---|---|
| `model_training.py` | End-to-end training pipeline (see workflow below). |
| `diagrams.py` | Generates the ML pipeline flowchart and an algorithm-mechanics diagram (Graphviz). |
| `outputs/churn_model.joblib` | Final serialized, tuned model pipeline (preprocessing + classifier). |
| `outputs/metrics.json` | Best model, best hyperparameters, cross-validated and test-set metrics. |
| `outputs/01_confusion_matrix.png` | Confusion matrix on the test set. |
| `outputs/02_roc_curve.png` | ROC curve / AUC on the test set. |
| `outputs/03_model_comparison.png` | Baseline comparison of Logistic Regression, Random Forest, Gradient Boosting. |
| `outputs/04_feature_coefficients.png` | Top absolute logistic-regression coefficient effects for the selected model. |
| `outputs/diagrams/01_ml_pipeline_flowchart.png` | Full ML pipeline: split → preprocess → train candidates → cross-validate → tune (with retry loop) → test evaluation → save. |
| `outputs/diagrams/02_logistic_regression_mechanics.png` | How the selected logistic-regression classifier converts engineered features into churn probabilities. |

## Workflow
1. Load and clean the Week 1 dataset; engineer two extra features
   (`charge_per_watch_hour`, `is_inactive`).
2. `ColumnTransformer`: standard-scale numeric features, one-hot encode
   categoricals.
3. Compare three candidate models with 5-fold cross-validated ROC-AUC.
4. Grid-search-tune the winning model.
5. Evaluate on a stratified 80/20 held-out test set: accuracy, precision,
   recall, F1, ROC-AUC, confusion matrix.
6. Save the fitted pipeline and metrics for reuse in Week 4's report.

## To run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib graphviz
python model_training.py
python diagrams.py
```

## v2 — improved model (`model_training_v2.py`)
Adds 6 engineered features (engagement score, tenure buckets, log
transforms, support-to-tenure ratio), a wider candidate pool
(+ HistGradientBoosting, + a Stacking ensemble), RandomizedSearchCV
tuning, and F1-optimized threshold selection via the precision-recall
curve. Outputs `churn_model_v2.joblib`, `metrics_v2.json`, an updated
model-comparison chart, and a stacking-ensemble architecture diagram.

**See `../MODEL_CARD.md` for full, honest performance comparison** —
ROC-AUC barely moves (the synthetic data has a ~0.66 AUC noise ceiling
by design); the real win is threshold tuning, which lifts F1 from 0.446
to 0.458 and recall from 0.60 to 0.81.

```bash
pip install scipy   # in addition to the above
python model_training_v2.py
python diagrams_v2.py
```
