"""
src.data.make_dataset
======================
Loads the raw churn CSV and applies the same cleaning steps used
throughout Week 3 (Week3_ML_Model_Development/model_training_v2.py):
drop duplicate customers, impute missing values, and clip the
monthly_charge outlier tail. This is the "raw -> interim" stage.

Run directly to regenerate data/interim/churn_dataset_interim.csv from
data/raw/churn_dataset_raw.csv:

    python -m src.data.make_dataset
"""
from pathlib import Path

import numpy as np
import pandas as pd

LAYOUT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = LAYOUT_ROOT / "data" / "raw" / "churn_dataset_raw.csv"
INTERIM_PATH = LAYOUT_ROOT / "data" / "interim" / "churn_dataset_interim.csv"


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw, unmodified churn dataset."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, impute missing values, and clip outliers.

    Mirrors `load_clean_data()` in
    Week3_ML_Model_Development/model_training_v2.py so results here stay
    consistent with the tuned model in that folder.
    """
    df = df.copy()
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "customer_id"])

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    q1, q3 = df["monthly_charge"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    df["monthly_charge"] = df["monthly_charge"].clip(upper=upper)

    return df


def main() -> None:
    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_raw_data()
    cleaned = clean_data(df)
    cleaned.to_csv(INTERIM_PATH, index=False)
    print(f"Loaded {len(df)} raw rows -> {len(cleaned)} cleaned rows")
    print(f"Wrote {INTERIM_PATH.relative_to(LAYOUT_ROOT)}")


if __name__ == "__main__":
    main()
