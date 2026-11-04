"""
Week 1 - Project Planning & Strategy Design
============================================
Scaffolds the standard project directory structure recommended in the
project plan (README/report deliverable). Demonstrates the intended
repository layout for the full 4-week churn prediction project.

Run:
    python setup_project_structure.py
"""

"""
Week 1 - Project Planning & Strategy Design
============================================
Scaffolds the standard project directory structure recommended in the
project plan (README/report deliverable): the intended repository layout
for a production implementation of the churn-prediction project.

This folder (reference_project_layout/) is no longer just an empty
example — src/, data/, models/, reports/, and notebooks/ all contain real,
working content (see reference_project_layout/README.md for what's in
each). This script is idempotent: it only creates a `.gitkeep` placeholder
in a folder that's genuinely still empty, and it will not overwrite a
folder's real files — it just makes sure the folder structure itself
exists and regenerates the top-level README describing it.

Run:
    python setup_project_structure.py
"""

from pathlib import Path

STRUCTURE = {
    "data/raw": "The raw, unmodified churn dataset",
    "data/interim": "Cleaned data — see src/data/make_dataset.py",
    "data/processed": "Feature-engineered, model-ready data — see src/features/build_features.py",
    "notebooks": "An executed end-to-end walkthrough notebook",
    "src/data": "Data loading & cleaning module (make_dataset.py)",
    "src/features": "Feature engineering module (build_features.py)",
    "src/models": "Model training & prediction modules (train_model.py, predict_model.py)",
    "src/visualization": "Reusable plotting functions (visualize.py)",
    "reports/figures": "Generated charts (churn balance, correlation heatmap, engagement boxplot)",
    "reports/final": "A copy of the Week 4 executive report",
    "models": "A serialized trained model + its metrics",
}


def main():
    base = Path(__file__).parent / "reference_project_layout"
    for rel_path, description in STRUCTURE.items():
        folder = base / rel_path
        folder.mkdir(parents=True, exist_ok=True)
        # Only place a .gitkeep if the folder doesn't already have real
        # content in it — don't clutter populated folders on re-run.
        existing = [p for p in folder.iterdir() if p.name != ".gitkeep"]
        if not existing:
            (folder / ".gitkeep").write_text("")
        print(f"{rel_path:30s} -> {description}")

    readme = base / "README.md"
    readme.write_text(
        "# Reference Project Layout\n\n"
        "The target repository structure a production implementation of "
        "this churn-prediction project would follow, as proposed in the "
        "Week 1 planning document — populated with real, working content, "
        "not just empty example folders:\n\n"
        + "\n".join(f"- `{p}/` — {d}" for p, d in STRUCTURE.items())
        + "\n\nStart with `notebooks/01_churn_eda_and_modeling.ipynb` for an "
        "executed, end-to-end walkthrough that calls every module in "
        "`src/` for real. Each `src/` module is also runnable on its own, "
        "e.g. `python -m src.models.train_model` from this folder's parent "
        "directory (`Week1_Project_Planning/`).\n\n"
        "This mirrors — rather than replaces — the Week 1-4 scripts in the "
        "parent project, which are written as standalone, run-top-to-"
        "bottom files for a course/assignment context. This folder shows "
        "the same pipeline organized as an importable `src/` package "
        "instead. Fixed hyperparameters are used here rather than the "
        "`RandomizedSearchCV` sweep in "
        "`Week3_ML_Model_Development/model_training_v2.py`, so results are "
        "close to but not identical to the numbers in `MODEL_CARD.md`.\n"
    )
    print(f"\nScaffold present at: {base}")


if __name__ == "__main__":
    main()
