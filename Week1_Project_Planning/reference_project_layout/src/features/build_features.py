"""
src.features.build_features
=============================
Engineers the signal-focused features introduced in the v2 model
(Week3_ML_Model_Development/model_training_v2.py): value-for-money ratio,
disengagement flags, engagement score, support/tenure friction, and a
tenure bucket. This is the "interim -> processed" stage — the output is
model-ready.

Run directly to regenerate data/processed/churn_dataset_processed.csv
from data/interim/churn_dataset_interim.csv:

    python -m src.features.build_features
"""
from pathlib import Path

import numpy as np
import pandas as pd

LAYOUT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_PATH = LAYOUT_ROOT / "data" / "interim" / "churn_dataset_interim.csv"
PROCESSED_PATH = LAYOUT_ROOT / "data" / "processed" / "churn_dataset_processed.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features on top of the cleaned dataset.

    Mirrors `engineer_features()` in
    Week3_ML_Model_Development/model_training_v2.py.
    """
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
        labels=["0-6mo", "7-12mo", "13-24mo", "25mo+"],
    ).astype(str)
    return df


def main() -> None:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INTERIM_PATH)
    processed = engineer_features(df)
    processed.to_csv(PROCESSED_PATH, index=False)
    added = set(processed.columns) - set(df.columns)
    print(f"Added {len(added)} engineered columns: {sorted(added)}")
    print(f"Wrote {PROCESSED_PATH.relative_to(LAYOUT_ROOT)}")


if __name__ == "__main__":
    main()
