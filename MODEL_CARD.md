# Model Card — Streaming Service Churn Predictor

## Model Details
- **Task:** Binary classification — predict whether a customer will churn
  in the observed period.
- **Two model versions ship in this repo:**
  - **v1** (`Week3_ML_Model_Development/outputs/churn_model.joblib`) —
    Logistic Regression, tuned via grid search over 3 candidates
    (Logistic Regression, Random Forest, Gradient Boosting).
  - **v2** (`Week3_ML_Model_Development/outputs/churn_model_v2.joblib`) —
    Stacking ensemble (Logistic Regression + Random Forest +
    HistGradientBoostingClassifier, meta-learner: Logistic Regression),
    trained on an expanded feature set and evaluated at both the default
    0.5 threshold and an F1-optimized threshold.
- **Training data:** Synthetic customer dataset (5,000+ rows) generated
  in Week 1 — see `Week1_Project_Planning/generate_dataset.py`. Not real
  customer data.
- **Framework:** scikit-learn 1.8, Python 3.

## Intended Use
Prioritization tool for a customer retention team — ranks active
customers by estimated churn risk so outreach effort can be focused on
the highest-risk segment. **Not** intended as an automated
accept/reject decision system, and not validated on real-world data.

## Features Used

| v1 | v2 (adds) |
|---|---|
| age, tenure, monthly charge, watch hours, devices, support tickets, days since login, region, plan type, contract type, payment method | + charge-per-watch-hour, inactivity flag, log(days since login), watch-hours-per-device, engagement score, support-to-tenure ratio, tenure bucket |

## Performance

| Metric | v1 (Logistic Regression) | v2 default threshold (Stacking) | v2 optimal threshold (0.206) |
|---|---|---|---|
| ROC-AUC | 0.630 | 0.627 | — (threshold-independent, same as default) |
| Accuracy | 0.592 | 0.727 | 0.472 |
| Precision | 0.355 | 0.519 | 0.319 |
| Recall | 0.599 | 0.051 | **0.814** |
| F1 | 0.446 | 0.093 | **0.458** |

*(5-fold CV ROC-AUC across candidates: Logistic Regression 0.660, Random
Forest 0.633, HistGradientBoosting 0.628, Stacking Ensemble 0.661.)*

## Honest Assessment — What Actually Improved and What Didn't
This is a deliberately transparent write-up rather than a "line went up"
narrative:

- **ROC-AUC barely moved** (0.660 → 0.661 in cross-validation; test-set
  ROC-AUC is actually marginally *lower* for v2, within noise). The
  synthetic dataset was generated with an injected noise term, so ~0.66
  AUC is close to the practical ceiling for this data — no amount of
  feature engineering or ensembling was going to push it to 0.85+, and a
  model card that claimed otherwise would be misleading.
- **The real improvement is in decision-threshold optimization, not the
  algorithm.** At the default 0.5 cutoff, the v2 ensemble's recall
  collapses to 0.05 (it rarely predicts "churn" at all). Optimizing the
  threshold for F1 instead (0.206) lifts recall to 0.81 and F1 to 0.458
  — a meaningful gain over v1's F1 of 0.446, and far more useful for a
  retention use case where **missing a churner is costlier than a wasted
  outreach email.**
- **Recommendation:** for production, prefer **v1 (Logistic Regression)**
  unless the threshold-tuned v2 recall/F1 profile is specifically
  desired — it's simpler, faster, fully interpretable via coefficients
  (see `outputs/diagrams/02_logistic_regression_mechanics.png`), and
  performs within noise of the more complex ensemble. The stacking
  ensemble is included to demonstrate the technique and because its
  threshold-optimized F1 is the best result obtained, not because it's
  unambiguously the better production choice.

## Limitations
- Trained and evaluated entirely on **synthetic data** with a known,
  intentionally noisy churn-generating process — real-world performance
  will differ and must be re-validated on actual customer data before
  any production use.
- Class imbalance (~27% churn rate) means accuracy alone is a misleading
  metric; this card leads with ROC-AUC/F1/recall for that reason.
- No fairness/bias audit has been performed across the `region` feature;
  this would be required before any real deployment.
- Point-in-time model — no drift monitoring is implemented. The plan
  (see Week 4 report) recommends quarterly retraining.

## How to Reproduce
```bash
python Week1_Project_Planning/generate_dataset.py
python Week3_ML_Model_Development/model_training.py       # v1
python Week3_ML_Model_Development/model_training_v2.py    # v2
```
