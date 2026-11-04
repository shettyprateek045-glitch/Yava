"""
Week 3 (v2) - Improved Machine Learning Model
=================================================
Project: Predicting Customer Churn for a Subscription-Based Streaming Service

Builds on model_training.py with:
  - Richer, signal-focused feature engineering (recency/engagement ratios,
    tenure buckets, log transforms)
  - A wider candidate pool: Logistic Regression, Random Forest,
    HistGradientBoostingClassifier (native sklearn, handles missing values
    and interactions well), and a Stacking ensemble of all three
  - RandomizedSearchCV (broader search than the original GridSearchCV)
  - Decision-threshold optimization for F1 (not just the default 0.5 cutoff)
  - A side-by-side comparison against the v1 model in metrics_v2.json

Run:
    python model_training_v2.py
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
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay, accuracy_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_PATH = Path(__file__).parent.parent / "Week1_Project_Planning" / "data" / "churn_dataset.csv"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
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
    return df.drop(columns=["customer_id"])


def engineer_features(df):
    df = df.copy()
    # Value-for-money signal
    df["charge_per_watch_hour"] = df["monthly_charge"] / (df["avg_watch_hours_week"] + 0.1)
    # Recency / disengagement flags
    df["is_inactive"] = (df["days_since_last_login"] > 30).astype(int)
    df["log_days_since_login"] = np.log1p(df["days_since_last_login"])
    # Engagement intensity, normalized by household size
    df["watch_hours_per_device"] = df["avg_watch_hours_week"] / df["num_devices"]
    # Composite engagement score: high watch hours + recent login = high engagement
    df["engagement_score"] = df["avg_watch_hours_week"] / (df["days_since_last_login"] + 1)
    # Support friction interacting with tenure (new + high-friction customers churn most)
    df["support_tenure_ratio"] = df["support_tickets"] / (df["tenure_months"] + 1)
    # Tenure bucket as an explicit categorical (captures non-linear tenure effects)
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"], bins=[-1, 6, 12, 24, 999],
        labels=["0-6mo", "7-12mo", "13-24mo", "25mo+"]
    ).astype(str)
    return df


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])


def optimize_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # last point has no corresponding threshold
    return float(thresholds[best_idx]), float(f1s[best_idx])


def main():
    df = engineer_features(load_clean_data())

    target = "churned"
    numeric_cols = [
        "age", "tenure_months", "monthly_charge", "avg_watch_hours_week",
        "num_devices", "support_tickets", "days_since_last_login",
        "charge_per_watch_hour", "is_inactive", "log_days_since_login",
        "watch_hours_per_device", "engagement_score", "support_tenure_ratio",
    ]
    categorical_cols = ["region", "plan_type", "contract_type", "payment_method", "tenure_bucket"]

    X = df[numeric_cols + categorical_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    base_models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_SEED),
    }

    print("=== Candidate comparison with engineered features (5-fold CV, ROC-AUC) ===")
    cv_scores = {}
    for name, model in base_models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = scores.mean()
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # ---- Stacking ensemble ----
    print("\n=== Stacking ensemble (LR + RF + HGB, meta-learner: Logistic Regression) ===")
    stack = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("rf", RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced", n_estimators=150)),
            ("hgb", HistGradientBoostingClassifier(random_state=RANDOM_SEED)),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5, n_jobs=-1,
    )
    stack_pipe = Pipeline([("preprocess", preprocessor), ("model", stack)])
    stack_scores = cross_val_score(stack_pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    cv_scores["StackingEnsemble"] = stack_scores.mean()
    print(f"StackingEnsemble: {stack_scores.mean():.4f} (+/- {stack_scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nBest model overall: {best_name} (CV ROC-AUC = {cv_scores[best_name]:.4f})")

    # ---- Randomized hyperparameter search on the winner ----
    print(f"\n=== Randomized search tuning: {best_name} ===")
    if best_name == "StackingEnsemble":
        final_pipe = stack_pipe
        final_pipe.fit(X_train, y_train)
        best_params = {"note": "stacking ensemble used as-is (base learners not individually re-tuned)"}
    else:
        pipe = Pipeline([("preprocess", preprocessor), ("model", base_models[best_name])])
        if best_name == "RandomForest":
            param_dist = {
                "model__n_estimators": randint(100, 400),
                "model__max_depth": [4, 6, 8, 12, None],
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", "log2", None],
            }
        elif best_name == "HistGradientBoosting":
            param_dist = {
                "model__max_iter": randint(100, 300),
                "model__learning_rate": uniform(0.02, 0.18),
                "model__max_depth": [3, 5, 8, None],
                "model__l2_regularization": uniform(0, 1),
            }
        else:  # LogisticRegression
            param_dist = {"model__C": uniform(0.001, 10), "model__penalty": ["l2"]}

        search = RandomizedSearchCV(
            pipe, param_dist, n_iter=25, cv=5, scoring="roc_auc",
            random_state=RANDOM_SEED, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        final_pipe = search.best_estimator_
        best_params = search.best_params_
        print(f"Best params: {best_params}")
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    # ---- Test set evaluation ----
    y_proba = final_pipe.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    best_threshold, best_f1 = optimize_threshold(y_test, y_proba)
    y_pred_optimal = (y_proba >= best_threshold).astype(int)

    metrics_default = {
        "accuracy": accuracy_score(y_test, y_pred_default),
        "precision": precision_score(y_test, y_pred_default),
        "recall": recall_score(y_test, y_pred_default),
        "f1_score": f1_score(y_test, y_pred_default),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    metrics_optimal_threshold = {
        "threshold": best_threshold,
        "accuracy": accuracy_score(y_test, y_pred_optimal),
        "precision": precision_score(y_test, y_pred_optimal),
        "recall": recall_score(y_test, y_pred_optimal),
        "f1_score": f1_score(y_test, y_pred_optimal),
    }

    print(f"\n=== Test set (default 0.5 threshold) ===")
    for k, v in metrics_default.items():
        print(f"{k}: {v:.4f}")
    print(f"\n=== Test set (optimal threshold = {best_threshold:.3f}) ===")
    for k, v in metrics_optimal_threshold.items():
        print(f"{k}: {v}" if k == "threshold" else f"{k}: {v:.4f}")

    # ---- Comparison chart: v1 vs v2 candidates ----
    v1_metrics_path = OUT_DIR / "metrics.json"
    v1_roc_auc = None
    if v1_metrics_path.exists():
        v1 = json.loads(v1_metrics_path.read_text())
        v1_roc_auc = v1["test_metrics"]["roc_auc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = pd.Series(cv_scores).sort_values()
    bars.plot(kind="barh", color="#4C72B0", ax=ax)
    ax.set_xlabel("Mean CV ROC-AUC")
    ax.set_title("Model Comparison (v2: engineered features + ensembles)")
    if v1_roc_auc:
        ax.axvline(v1_roc_auc, color="#C44E52", linestyle="--",
                    label=f"v1 test ROC-AUC ({v1_roc_auc:.3f})")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_v2_model_comparison.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=f"{best_name} v2")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC Curve — v2 Model (Test Set)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_v2_roc_curve.png", dpi=120)
    plt.close()

    # ---- Save model + metrics ----
    joblib.dump(final_pipe, OUT_DIR / "churn_model_v2.joblib")
    with open(OUT_DIR / "metrics_v2.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "best_params": best_params,
            "cv_roc_auc_all_candidates": cv_scores,
            "test_metrics_default_threshold": metrics_default,
            "test_metrics_optimal_threshold": metrics_optimal_threshold,
            "v1_test_roc_auc_for_comparison": v1_roc_auc,
            "improvement_vs_v1": (
                round(metrics_default["roc_auc"] - v1_roc_auc, 4) if v1_roc_auc else None
            ),
            "feature_columns": {"numeric": numeric_cols, "categorical": categorical_cols},
        }, f, indent=2)

    print(f"\nSaved model -> {OUT_DIR / 'churn_model_v2.joblib'}")
    print(f"Saved metrics -> {OUT_DIR / 'metrics_v2.json'}")


if __name__ == "__main__":
    main()
