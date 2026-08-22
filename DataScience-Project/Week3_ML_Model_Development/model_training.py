"""
Week 3 - Machine Learning Model Development & Evaluation
============================================================
Project: Predicting Customer Churn for a Subscription-Based Streaming Service

Preprocesses the churn dataset, trains and compares three candidate models
(Logistic Regression, Random Forest, Gradient Boosting), tunes the best one
with cross-validated grid search, evaluates on a held-out test set, and
saves the final model + evaluation artifacts.

Run:
    python model_training.py
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
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


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])


def evaluate_model(name, pipeline, X_test, y_test, results):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    results[name] = metrics
    print(f"\n{name} — " + ", ".join(f"{k}: {v:.3f}" for k, v in metrics.items()))
    return y_pred, y_proba


def main():
    df = load_clean_data()

    # ---- Feature engineering ----
    df["charge_per_watch_hour"] = df["monthly_charge"] / (df["avg_watch_hours_week"] + 0.1)
    df["is_inactive"] = (df["days_since_last_login"] > 30).astype(int)

    target = "churned"
    numeric_cols = ["age", "tenure_months", "monthly_charge", "avg_watch_hours_week",
                     "num_devices", "support_tickets", "days_since_last_login",
                     "charge_per_watch_hour", "is_inactive"]
    categorical_cols = ["region", "plan_type", "contract_type", "payment_method"]

    X = df[numeric_cols + categorical_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train churn rate: {y_train.mean():.2%}, Test churn rate: {y_test.mean():.2%}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }

    results = {}
    cv_scores = {}
    pipelines = {}

    print("\n=== Baseline model comparison (5-fold CV, ROC-AUC) ===")
    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
        cv_scores[name] = scores.mean()
        pipelines[name] = pipe
        print(f"{name}: mean ROC-AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nBest baseline model: {best_name}")

    # ---- Hyperparameter tuning on the best candidate ----
    print(f"\n=== Hyperparameter tuning: {best_name} ===")
    if best_name == "RandomForest":
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None],
            "model__min_samples_leaf": [1, 5],
        }
    elif best_name == "GradientBoosting":
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }
    else:
        param_grid = {"model__C": [0.01, 0.1, 1, 10]}

    grid = GridSearchCV(pipelines[best_name], param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.3f}")

    final_model = grid.best_estimator_

    # ---- Final evaluation on held-out test set ----
    print("\n=== Final Test Set Evaluation ===")
    y_pred, y_proba = evaluate_model(f"{best_name} (tuned)", final_model, X_test, y_test, results)

    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # ---- Confusion matrix ----
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name} (tuned)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_confusion_matrix.png", dpi=120)
    plt.close()

    # ---- ROC curve ----
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=f"{best_name} (tuned)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC Curve — Test Set")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_roc_curve.png", dpi=120)
    plt.close()

    # ---- Model comparison bar chart ----
    fig, ax = plt.subplots(figsize=(7, 5))
    pd.Series(cv_scores).sort_values().plot(kind="barh", color="#4C72B0", ax=ax)
    ax.set_xlabel("Mean CV ROC-AUC")
    ax.set_title("Baseline Model Comparison")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_model_comparison.png", dpi=120)
    plt.close()

    # ---- Model explainability chart ----
    feature_names = (numeric_cols +
                     list(final_model.named_steps["preprocess"]
                          .named_transformers_["cat"]
                          .get_feature_names_out(categorical_cols)))
    estimator = final_model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        title = f"Top Feature Importances — {best_name}"
        filename = "04_feature_importance.png"
    else:
        # Logistic regression uses coefficients rather than tree importances.
        values = np.abs(estimator.coef_[0])
        title = "Top Feature Effects — Logistic Regression"
        filename = "04_feature_coefficients.png"
    explain_df = pd.Series(values, index=feature_names).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 6))
    explain_df.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Absolute effect / importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=120)
    plt.close()

    # ---- Save model + metrics ----
    joblib.dump(final_model, OUT_DIR / "churn_model.joblib")
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "best_params": grid.best_params_,
            "cv_roc_auc": cv_scores,
            "test_metrics": results[f"{best_name} (tuned)"],
        }, f, indent=2)

    print(f"\nSaved model -> {OUT_DIR / 'churn_model.joblib'}")
    print(f"Saved metrics -> {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
