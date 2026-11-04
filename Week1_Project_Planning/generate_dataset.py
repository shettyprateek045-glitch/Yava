"""
Week 1 - Project Planning & Strategy Design
============================================
Project: Predicting Customer Churn for a Subscription-Based Streaming Service

This script produces the synthetic dataset that stands in for the "real"
data source described in the Week 1 project plan document. Since the
assignment is conceptual (no real dataset provided), we simulate a
realistic customer-churn dataset so that the later weeks (EDA, modeling,
reporting) have something concrete to run against.

Run:
    python generate_dataset.py
Output:
    data/churn_dataset.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_CUSTOMERS = 5000

np.random.seed(RANDOM_SEED)


def generate_churn_dataset(n=N_CUSTOMERS) -> pd.DataFrame:
    customer_id = np.arange(1, n + 1)

    tenure_months = np.random.gamma(shape=2.0, scale=12, size=n).clip(0, 72).astype(int)
    monthly_charge = np.round(np.random.normal(loc=13.5, scale=4.0, size=n).clip(4.99, 24.99), 2)

    plan_type = np.random.choice(
        ["Basic", "Standard", "Premium"], size=n, p=[0.35, 0.45, 0.20]
    )
    contract_type = np.random.choice(
        ["Monthly", "Annual"], size=n, p=[0.7, 0.3]
    )
    payment_method = np.random.choice(
        ["Credit Card", "PayPal", "Bank Transfer", "Gift Card"],
        size=n, p=[0.45, 0.30, 0.15, 0.10]
    )

    avg_watch_hours_week = np.random.gamma(shape=2.2, scale=3.0, size=n).clip(0, 40)
    num_devices = np.random.choice([1, 2, 3, 4], size=n, p=[0.4, 0.35, 0.18, 0.07])
    support_tickets = np.random.poisson(lam=0.6, size=n)
    days_since_last_login = np.random.exponential(scale=10, size=n).clip(0, 180).astype(int)

    age = np.random.normal(35, 12, size=n).clip(13, 85).astype(int)
    region = np.random.choice(
        ["North America", "Europe", "Asia-Pacific", "Latin America", "Other"],
        size=n, p=[0.35, 0.30, 0.20, 0.10, 0.05]
    )

    # Latent churn probability driven by a plausible combination of factors
    churn_logit = (
        -1.2
        + 0.035 * days_since_last_login
        - 0.05 * avg_watch_hours_week
        - 0.02 * tenure_months
        + 0.25 * support_tickets
        + np.where(contract_type == "Monthly", 0.6, -0.4)
        + np.where(plan_type == "Basic", 0.3, 0.0)
        + np.random.normal(0, 0.6, size=n)  # noise
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = (np.random.rand(n) < churn_prob).astype(int)

    # Inject realistic messiness for the EDA/cleaning stage
    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "region": region,
        "tenure_months": tenure_months,
        "plan_type": plan_type,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "monthly_charge": monthly_charge,
        "avg_watch_hours_week": np.round(avg_watch_hours_week, 2),
        "num_devices": num_devices,
        "support_tickets": support_tickets,
        "days_since_last_login": days_since_last_login,
        "churned": churned,
    })

    # Introduce missing values (~3-5%) to simulate real-world data quality issues
    for col in ["age", "avg_watch_hours_week", "payment_method", "monthly_charge"]:
        missing_idx = df.sample(frac=np.random.uniform(0.03, 0.05), random_state=RANDOM_SEED).index
        df.loc[missing_idx, col] = np.nan

    # Introduce a handful of outliers
    outlier_idx = df.sample(n=15, random_state=RANDOM_SEED).index
    df.loc[outlier_idx, "monthly_charge"] = df.loc[outlier_idx, "monthly_charge"] * 6

    # Introduce a few duplicate rows
    dup_rows = df.sample(n=10, random_state=RANDOM_SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def main():
    df = generate_churn_dataset()
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "churn_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Churn rate: {df['churned'].mean():.2%}")


if __name__ == "__main__":
    main()
