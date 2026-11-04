# Reference Project Layout

The target repository structure a production implementation of this churn-prediction project would follow, as proposed in the Week 1 planning document — populated with real, working content, not just empty example folders:

- `data/raw/` — The raw, unmodified churn dataset
- `data/interim/` — Cleaned data — see src/data/make_dataset.py
- `data/processed/` — Feature-engineered, model-ready data — see src/features/build_features.py
- `notebooks/` — An executed end-to-end walkthrough notebook
- `src/data/` — Data loading & cleaning module (make_dataset.py)
- `src/features/` — Feature engineering module (build_features.py)
- `src/models/` — Model training & prediction modules (train_model.py, predict_model.py)
- `src/visualization/` — Reusable plotting functions (visualize.py)
- `reports/figures/` — Generated charts (churn balance, correlation heatmap, engagement boxplot)
- `reports/final/` — A copy of the Week 4 executive report
- `models/` — A serialized trained model + its metrics

Start with `notebooks/01_churn_eda_and_modeling.ipynb` for an executed, end-to-end walkthrough that calls every module in `src/` for real. Each `src/` module is also runnable on its own, e.g. `python -m src.models.train_model` from this folder's parent directory (`Week1_Project_Planning/`).

This mirrors — rather than replaces — the Week 1-4 scripts in the parent project, which are written as standalone, run-top-to-bottom files for a course/assignment context. This folder shows the same pipeline organized as an importable `src/` package instead. Fixed hyperparameters are used here rather than the `RandomizedSearchCV` sweep in `Week3_ML_Model_Development/model_training_v2.py`, so results are close to but not identical to the numbers in `MODEL_CARD.md`.
