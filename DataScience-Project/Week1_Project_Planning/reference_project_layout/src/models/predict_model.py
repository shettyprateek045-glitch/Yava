"""
src.models.predict_model
==========================
Loads a serialized pipeline (from src.models.train_model or one of the
Week3_ML_Model_Development/outputs/*.joblib files) and scores new,
already-feature-engineered rows (see src.features.build_features).

Run directly to score the processed dataset with the reference model and
print a risk-band summary:

    python -m src.models.predict_model
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .train_model import CATEGORICAL_COLS, MODEL_PATH, NUMERIC_COLS, PROCESSED_PATH

DEFAULT_THRESHOLD = 0.206  # matches the tuned v2 threshold in MODEL_CARD.md


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    """Return churn probabilities for each row of an already feature-engineered dataframe."""
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    return model.predict_proba(X)[:, 1]


def risk_band(probability: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    if probability >= threshold:
        return "High"
    if probability >= threshold * 0.6:
        return "Moderate"
    return "Low"


def main() -> None:
    model = load_model()
    df = pd.read_csv(PROCESSED_PATH)
    probabilities = predict_proba(model, df)
    bands = pd.Series(probabilities).apply(risk_band)

    print(f"Scored {len(df)} customers with {MODEL_PATH.name}")
    print(bands.value_counts().to_string())


if __name__ == "__main__":
    main()
