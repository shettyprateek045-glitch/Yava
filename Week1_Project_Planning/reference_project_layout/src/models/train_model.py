"""
src.models.train_model
========================
A lean, importable training entrypoint built from processed data. Uses the
same feature set, preprocessor shape, and stacking-ensemble architecture as
Week3_ML_Model_Development/model_training_v2.py, with fixed hyperparameters
instead of that script's RandomizedSearchCV sweep — this module is meant to
be a fast, reusable retraining path (e.g., for a CI job or a notebook),
not a replacement for the full tuned experiment in Week 3.

Run directly to train on data/processed/ and save to models/:

    python -m src.models.train_model
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LAYOUT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = LAYOUT_ROOT / "data" / "processed" / "churn_dataset_processed.csv"
MODEL_PATH = LAYOUT_ROOT / "models" / "churn_model_reference.joblib"
METRICS_PATH = LAYOUT_ROOT / "models" / "churn_model_reference_metrics.json"
RANDOM_SEED = 42

NUMERIC_COLS = [
    "age", "tenure_months", "monthly_charge", "avg_watch_hours_week",
    "num_devices", "support_tickets", "days_since_last_login",
    "charge_per_watch_hour", "is_inactive", "log_days_since_login",
    "watch_hours_per_device", "engagement_score", "support_tenure_ratio",
]
CATEGORICAL_COLS = ["region", "plan_type", "contract_type", "payment_method", "tenure_bucket"]
TARGET = "churned"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


def build_pipeline() -> Pipeline:
    """A stacking ensemble (LR + RF + HistGB) behind the shared preprocessor —
    architecturally identical to the v2 model in Week3_ML_Model_Development,
    with fixed, reasonable hyperparameters rather than a randomized search."""
    stack = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("rf", RandomForestClassifier(
                random_state=RANDOM_SEED, class_weight="balanced", n_estimators=150)),
            ("hgb", HistGradientBoostingClassifier(random_state=RANDOM_SEED)),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5, n_jobs=-1,
    )
    return Pipeline([("preprocess", build_preprocessor()), ("model", stack)])


def optimize_threshold(y_true, y_proba) -> tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1s[:-1]))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def train(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Train on a processed dataframe and return (fitted_pipeline, metrics)."""
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    threshold, best_f1 = optimize_threshold(y_test.values, y_proba)
    y_pred_opt = (y_proba >= threshold).astype(int)

    metrics = {
        "model_type": "StackingEnsemble",
        "roc_auc": round(float(roc_auc), 4),
        "optimal_threshold": round(threshold, 4),
        "precision_at_threshold": round(precision_score(y_test, y_pred_opt), 4),
        "recall_at_threshold": round(recall_score(y_test, y_pred_opt), 4),
        "f1_at_threshold": round(best_f1, 4),
        "accuracy_at_threshold": round(accuracy_score(y_test, y_pred_opt), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return pipe, metrics


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_PATH)
    pipe, metrics = train(df)

    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model  -> {MODEL_PATH.relative_to(LAYOUT_ROOT)}")
    print(f"Saved metrics -> {METRICS_PATH.relative_to(LAYOUT_ROOT)}")


if __name__ == "__main__":
    main()
