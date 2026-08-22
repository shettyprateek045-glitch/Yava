# Executive Report — Streaming Service Churn Prediction

## Executive Summary
Out of **5,000** customers analyzed, **1,371
(27.4%)** churned in the observed period, representing an
estimated **$18,549/month** in recurring revenue at
risk. A predictive model was developed to flag at-risk customers in
advance, achieving **63% ROC-AUC** and
**60% recall** on churners in the held-out test
set — meaning the model correctly identifies roughly 60%
of the observed churn cases, making it useful as a customer-risk
prioritization tool.

## Methodology Recap
1. Assembled and cleaned customer usage, billing, and engagement data.
2. Explored patterns across tenure, plan type, engagement, and support
   history to understand what separates retained customers from churners.
3. Engineered features capturing engagement recency and value-for-money.
4. Trained and compared multiple classification models
   (LogisticRegression, RandomForest, GradientBoosting), selected **LogisticRegression**
   as the best performer, and tuned it via cross-validated grid search.
5. Validated performance on a held-out test set never seen during training.

## Key Insights

### 1. Engagement recency is the strongest churn signal
Customers who haven't logged in recently churn at dramatically higher
rates than active users:

| Engagement Level | Churn Rate |
|---|---|
| Active (<=7 days) | 23.2% |
| Cooling off (8-30 days) | 31.0% |
| At risk (31-60 days) | 46.3% |
| Dormant (60+ days) | 72.7% |

*(See `outputs/02_engagement_vs_churn.png`)*

### 2. Revenue exposure varies significantly by plan
- **Standard**: $8,169/month at risk
- **Basic**: $6,827/month at risk
- **Premium**: $3,553/month at risk

*(See `outputs/01_revenue_at_risk.png`)*

### 3. Monthly (vs. annual) contracts and low tenure correlate with churn
Shorter-tenure, monthly-billed customers are meaningfully more likely to
churn than customers on annual plans with longer histories — consistent
with the exploratory analysis in Week 2.

### 4. Model performance
| Metric | Value |
|---|---|
| Accuracy | 0.592 |
| Precision | 0.355 |
| Recall | 0.599 |
| F1 Score | 0.446 |
| Roc Auc | 0.630 |

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
   60%.
4. Re-train the model quarterly as usage patterns evolve, and monitor for
   data/model drift.
