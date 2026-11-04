"""
Week 2 - Exploratory Data Analysis (EDA) & Visualization Framework
====================================================================
Project: Predicting Customer Churn for a Subscription-Based Streaming Service

Runs a full EDA pass over the Week 1 synthetic dataset: data quality checks,
cleaning, univariate/bivariate/multivariate analysis, and a set of
visualizations (histograms, boxplots, correlation heatmap, bar charts,
pairplot). Saves all figures to outputs/ and a markdown summary report.

Run:
    python eda_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")

DATA_PATH = Path(__file__).parent.parent / "Week1_Project_Planning" / "data" / "churn_dataset.csv"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    log(f"## Data Quality Summary\n")
    log(f"- Raw rows: {len(df)}")

    n_dupes = df.duplicated(subset=[c for c in df.columns if c != "customer_id"]).sum()
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "customer_id"])
    log(f"- Duplicate rows removed: {n_dupes}")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    log(f"- Columns with missing values: {dict(missing)}")

    # Impute: numeric -> median, categorical -> mode
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Outlier handling on monthly_charge via IQR capping
    q1, q3 = df["monthly_charge"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    n_outliers = (df["monthly_charge"] > upper).sum()
    df["monthly_charge"] = df["monthly_charge"].clip(upper=upper)
    log(f"- Outliers capped in monthly_charge (IQR method): {n_outliers}")
    log(f"- Final clean rows: {len(df)}\n")

    return df


def univariate_analysis(df):
    log("## Univariate Analysis\n")
    numeric_cols = ["age", "tenure_months", "monthly_charge", "avg_watch_hours_week",
                     "num_devices", "support_tickets", "days_since_last_login"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#4C72B0")
        axes[i].set_title(f"Distribution of {col}")
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_univariate_histograms.png", dpi=120)
    plt.close()
    log("- Saved: 01_univariate_histograms.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    df["churned"].value_counts().rename({0: "Retained", 1: "Churned"}).plot(
        kind="bar", color=["#55A868", "#C44E52"], ax=ax
    )
    ax.set_title("Churn Class Balance")
    ax.set_ylabel("Customer Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_churn_class_balance.png", dpi=120)
    plt.close()
    log("- Saved: 02_churn_class_balance.png")

    log(f"\nSummary statistics:\n\n{df[numeric_cols].describe().round(2).to_markdown()}\n")


def bivariate_analysis(df):
    log("## Bivariate Analysis\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.boxplot(data=df, x="churned", y="tenure_months", ax=axes[0])
    axes[0].set_title("Tenure vs Churn")
    sns.boxplot(data=df, x="churned", y="avg_watch_hours_week", ax=axes[1])
    axes[1].set_title("Watch Hours vs Churn")
    sns.boxplot(data=df, x="churned", y="days_since_last_login", ax=axes[2])
    axes[2].set_title("Days Since Login vs Churn")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_bivariate_boxplots.png", dpi=120)
    plt.close()
    log("- Saved: 03_bivariate_boxplots.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    churn_by_plan = pd.crosstab(df["plan_type"], df["churned"], normalize="index")
    churn_by_plan.rename(columns={0: "Retained", 1: "Churned"}).plot(
        kind="bar", stacked=True, color=["#55A868", "#C44E52"], ax=ax
    )
    ax.set_title("Churn Rate by Plan Type")
    ax.set_ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_churn_by_plan_type.png", dpi=120)
    plt.close()
    log("- Saved: 04_churn_by_plan_type.png")


def multivariate_analysis(df):
    log("## Multivariate Analysis\n")

    numeric_cols = ["age", "tenure_months", "monthly_charge", "avg_watch_hours_week",
                     "num_devices", "support_tickets", "days_since_last_login", "churned"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_correlation_heatmap.png", dpi=120)
    plt.close()
    log("- Saved: 05_correlation_heatmap.png")

    pair_cols = ["tenure_months", "avg_watch_hours_week", "days_since_last_login", "churned"]
    g = sns.pairplot(df[pair_cols], hue="churned", diag_kind="kde", palette=["#55A868", "#C44E52"])
    g.fig.suptitle("Pairwise Relationships by Churn Status", y=1.02)
    g.savefig(OUT_DIR / "06_pairplot.png", dpi=120)
    plt.close()
    log("- Saved: 06_pairplot.png")

    log(f"\nStrongest correlations with `churned`:\n\n"
        f"{corr['churned'].drop('churned').sort_values(key=abs, ascending=False).round(3).to_markdown()}\n")


def main():
    df = load_and_clean()
    univariate_analysis(df)
    bivariate_analysis(df)
    multivariate_analysis(df)

    report_path = OUT_DIR / "eda_summary_report.md"
    report_path.write_text(
        "# EDA Summary Report — Streaming Service Churn Dataset\n\n" + "\n".join(report_lines)
    )
    print(f"\nFull EDA report written to {report_path}")


if __name__ == "__main__":
    main()
