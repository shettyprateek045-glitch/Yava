"""
Week 3 (Advanced) - Automated Tests
=======================================
Sanity-checks the trained pipeline so a broken model (bad schema, NaN
outputs, out-of-range probabilities) fails loudly in CI instead of
silently shipping. Uses only the standard library's unittest plus
packages already required by the pipeline (pandas, numpy, joblib) — no
extra install needed to run these.

Run:
    python -m unittest discover -s tests -v
(from the Week3_ML_Model_Development/ directory)
"""

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
MODEL_PATH = BASE / "outputs" / "churn_model.joblib"
METRICS_PATH = BASE / "outputs" / "metrics.json"
DATA_PATH = BASE.parent / "Week1_Project_Planning" / "data" / "churn_dataset.csv"

REQUIRED_NUMERIC_COLS = [
    "age", "tenure_months", "monthly_charge", "avg_watch_hours_week",
    "num_devices", "support_tickets", "days_since_last_login",
]
REQUIRED_CATEGORICAL_COLS = ["region", "plan_type", "contract_type", "payment_method"]


def _sample_customer(**overrides):
    base = {
        "age": 34.0, "tenure_months": 12, "monthly_charge": 13.99,
        "avg_watch_hours_week": 5.0, "num_devices": 2, "support_tickets": 0,
        "days_since_last_login": 5, "region": "North America",
        "plan_type": "Standard", "contract_type": "Annual",
        "payment_method": "Credit Card",
    }
    base.update(overrides)
    base["charge_per_watch_hour"] = base["monthly_charge"] / (base["avg_watch_hours_week"] + 0.1)
    base["is_inactive"] = int(base["days_since_last_login"] > 30)
    return pd.DataFrame([base])


class TestArtifactsExist(unittest.TestCase):
    def test_dataset_exists(self):
        self.assertTrue(DATA_PATH.exists(), f"Missing dataset at {DATA_PATH}")

    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.exists(), f"Missing model at {MODEL_PATH}")

    def test_metrics_file_exists(self):
        self.assertTrue(METRICS_PATH.exists(), f"Missing metrics at {METRICS_PATH}")


class TestDatasetSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)

    def test_required_columns_present(self):
        for col in REQUIRED_NUMERIC_COLS + REQUIRED_CATEGORICAL_COLS + ["churned"]:
            self.assertIn(col, self.df.columns)

    def test_target_is_binary(self):
        self.assertTrue(set(self.df["churned"].dropna().unique()).issubset({0, 1}))

    def test_no_negative_tenure_or_charges(self):
        self.assertTrue((self.df["tenure_months"].dropna() >= 0).all())
        self.assertTrue((self.df["monthly_charge"].dropna() >= 0).all())


class TestModelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise unittest.SkipTest("Model not trained yet — run model_training.py first.")
        cls.model = joblib.load(MODEL_PATH)

    def test_predict_proba_shape_and_range(self):
        X = _sample_customer()
        proba = self.model.predict_proba(X)
        self.assertEqual(proba.shape, (1, 2))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        self.assertAlmostEqual(proba.sum(), 1.0, places=5)

    def test_predict_returns_valid_class(self):
        X = _sample_customer()
        pred = self.model.predict(X)
        self.assertIn(pred[0], (0, 1))

    def test_higher_risk_profile_scores_higher(self):
        """A disengaged, monthly, high-support-ticket customer should score
        at least as risky as an engaged, annual, low-ticket customer."""
        low_risk = _sample_customer(
            tenure_months=48, contract_type="Annual", avg_watch_hours_week=15,
            support_tickets=0, days_since_last_login=1,
        )
        high_risk = _sample_customer(
            tenure_months=1, contract_type="Monthly", avg_watch_hours_week=0.5,
            support_tickets=4, days_since_last_login=60,
        )
        p_low = self.model.predict_proba(low_risk)[0, 1]
        p_high = self.model.predict_proba(high_risk)[0, 1]
        self.assertGreater(p_high, p_low)

    def test_batch_prediction_matches_row_count(self):
        X = pd.concat([_sample_customer(), _sample_customer(age=50)], ignore_index=True)
        proba = self.model.predict_proba(X)
        self.assertEqual(len(proba), 2)


class TestMetricsSanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not METRICS_PATH.exists():
            raise unittest.SkipTest("Metrics not generated yet — run model_training.py first.")
        cls.metrics = json.loads(METRICS_PATH.read_text())

    def test_test_metrics_in_valid_range(self):
        for key, value in self.metrics["test_metrics"].items():
            self.assertGreaterEqual(value, 0.0, f"{key} below 0")
            self.assertLessEqual(value, 1.0, f"{key} above 1")

    def test_best_model_matches_highest_cv_score(self):
        cv_scores = self.metrics["cv_roc_auc"]
        best_by_cv = max(cv_scores, key=cv_scores.get)
        self.assertEqual(self.metrics["best_model"], best_by_cv,
                          "metrics.json best_model does not match the highest CV ROC-AUC score")


if __name__ == "__main__":
    unittest.main(verbosity=2)
