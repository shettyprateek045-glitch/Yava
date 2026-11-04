"""
Week 3 (Advanced) - Model Interpretability & Threshold Optimization
=======================================================================
Goes beyond raw coefficients to explain *how* the model behaves and to
pick a decision threshold that matches the business problem (churn is
imbalanced, so the default 0.5 cutoff is rarely the right one).

Adds four things a coefficient chart alone does not give you:

  1. Permutation importance — how much test-set ROC-AUC drops when each
     feature is randomly shuffled (model-agnostic, unlike raw coefficients
     which can be misleading with correlated/encoded features).
  2. Partial dependence plots — how predicted churn probability changes
     as a single feature varies, holding others constant.
  3. Precision-Recall curve with an F1-optimal threshold — more
     informative than ROC-AUC alone on an imbalanced target, and turns
     "0.63 ROC-AUC" into an actionable operating point.
  4. Calibration curve — checks whether "70% predicted probability" really
     means ~70% of those customers churn, which matters if the business
     wants to act on the probability itself (e.g. discount size scaled by
     risk), not just the rank order.

Run:
    python advanced_interpretability.py
Requires the Week 3 model to already be trained (outputs/churn_model.joblib).
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import PrecisionRecallDisplay, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE = Path(__file__).parent
DATA_PATH = BASE.parent / "Week1_Project_Planning" / "data" / "churn_dataset.csv"
MODEL_PATH = BASE / "outputs" / "churn_model.joblib"
OUT_DIR = BASE / "outputs" / "advanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42


def load_clean_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "customer_id"])
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    q1, q3 = df["monthly_charge"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    df["monthly_charge"] = df["monthly_charge"].clip(upper=upper)
    df["charge_per_watch_hour"] = df["monthly_charge"] / (df["avg_watch_hours_week"] + 0.1)
    df["is_inactive"] = (df["days_since_last_login"] > 30).astype(int)
    return df.drop(columns=["customer_id"])


def permutation_importance_plot(model, X_test, y_test):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns)
    errors = pd.Series(result.importances_std, index=X_test.columns)
    top = importances.abs().sort_values(ascending=False).head(10).index

    fig, ax = plt.subplots(figsize=(8, 6))
    order = importances[top].sort_values()
    ax.barh(order.index, order.values, xerr=errors[order.index], color="#4C72B0")
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Mean decrease in test ROC-AUC when feature is shuffled")
    ax.set_title("Permutation Importance (model-agnostic)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_permutation_importance.png", dpi=120)
    plt.close()
    print("Saved: 01_permutation_importance.png")
    return importances.sort_values(ascending=False)


def partial_dependence_plots(model, X_train, top_features):
    features_to_plot = [f for f in top_features[:4] if f in X_train.select_dtypes(include=np.number).columns]
    if not features_to_plot:
        features_to_plot = ["tenure_months", "days_since_last_login"]

    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(
        model, X_train, features_to_plot, ax=ax, n_cols=2,
        line_kw={"color": "#C44E52", "linewidth": 2},
    )
    fig.suptitle("Partial Dependence — Effect of Each Feature on Predicted Churn Probability", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_partial_dependence.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: 02_partial_dependence.png")


def precision_recall_threshold(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])  # last point has no threshold
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=ax, color="#4C72B0")
    ax.scatter(recall[best_idx], precision[best_idx], color="#C44E52", zorder=5, s=80,
               label=f"F1-optimal threshold = {best_threshold:.2f}")
    ax.set_title("Precision-Recall Curve with Optimal Operating Point")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_precision_recall_curve.png", dpi=120)
    plt.close()
    print(f"Saved: 03_precision_recall_curve.png (best threshold={best_threshold:.3f}, F1={best_f1:.3f})")

    default_f1 = f1_score(y_test, (y_proba >= 0.5).astype(int))
    tuned_f1 = f1_score(y_test, (y_proba >= best_threshold).astype(int))
    return {
        "default_threshold_0.5_f1": round(float(default_f1), 4),
        "optimal_threshold": round(float(best_threshold), 4),
        "optimal_threshold_f1": round(float(tuned_f1), 4),
    }


def calibration_plot(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 6))
    CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=10, ax=ax, color="#55A868")
    ax.set_title("Calibration Curve — Is a 70% Prediction Really ~70% Churn?")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_calibration_curve.png", dpi=120)
    plt.close()
    print("Saved: 04_calibration_curve.png")


def main():
    df = load_clean_data()
    numeric_cols = ["age", "tenure_months", "monthly_charge", "avg_watch_hours_week",
                     "num_devices", "support_tickets", "days_since_last_login",
                     "charge_per_watch_hour", "is_inactive"]
    categorical_cols = ["region", "plan_type", "contract_type", "payment_method"]
    X = df[numeric_cols + categorical_cols]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = joblib.load(MODEL_PATH)

    importances = permutation_importance_plot(model, X_test, y_test)
    partial_dependence_plots(model, X_train, list(importances.index))
    threshold_results = precision_recall_threshold(model, X_test, y_test)
    calibration_plot(model, X_test, y_test)

    summary = {
        "top_permutation_features": importances.head(5).round(4).to_dict(),
        "threshold_optimization": threshold_results,
    }
    with open(OUT_DIR / "interpretability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: interpretability_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
