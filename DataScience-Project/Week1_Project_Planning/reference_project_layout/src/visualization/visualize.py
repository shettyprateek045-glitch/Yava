"""
src.visualization.visualize
=============================
Reusable plotting functions, factored out of the one-off scripts in
Week2_EDA_Visualization/ and Week3_ML_Model_Development/ so a notebook or
report can call them directly instead of copy-pasting matplotlib code.

Run directly to regenerate reports/figures/ from data/processed/:

    python -m src.visualization.visualize
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

sns.set_theme(style="whitegrid")

LAYOUT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = LAYOUT_ROOT / "data" / "processed" / "churn_dataset_processed.csv"
FIGURES_DIR = LAYOUT_ROOT / "reports" / "figures"


def plot_churn_balance(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of churned vs retained customer counts."""
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["churned"].value_counts().sort_index()
    ax.bar(["Retained", "Churned"], counts.values, color=["#4FBF97", "#E5574F"])
    ax.set_title("Churn Class Balance")
    ax.set_ylabel("Customers")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Correlation heatmap over the numeric engineered features."""
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Engineered Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_engagement_vs_churn(df: pd.DataFrame, out_path: Path) -> None:
    """Boxplot of the engineered engagement_score, split by churn outcome."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(data=df, x="churned", y="engagement_score", hue="churned",
                palette={0: "#4FBF97", 1: "#E5574F"}, legend=False, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained", "Churned"])
    ax.set_title("Engagement Score by Churn Outcome")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, out_path: Path, label: str = "Model") -> None:
    """ROC curve for a fitted model's held-out predictions."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, name=label, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"ROC Curve — {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_PATH)
    plot_churn_balance(df, FIGURES_DIR / "reference_churn_balance.png")
    plot_correlation_heatmap(df, FIGURES_DIR / "reference_correlation_heatmap.png")
    plot_engagement_vs_churn(df, FIGURES_DIR / "reference_engagement_vs_churn.png")
    print(f"Wrote 3 figures to {FIGURES_DIR.relative_to(LAYOUT_ROOT)}")


if __name__ == "__main__":
    main()
