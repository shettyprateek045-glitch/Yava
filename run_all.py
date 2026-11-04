"""Run the four-week technical implementation end-to-end.

The DOC files are the primary assignment deliverables; this script reproduces
all supporting technical artifacts in the GitHub repository.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
COMMANDS = [
    ROOT / "Week1_Project_Planning" / "generate_dataset.py",
    ROOT / "Week1_Project_Planning" / "setup_project_structure.py",
    ROOT / "Week1_Project_Planning" / "diagrams.py",
    ROOT / "Week2_EDA_Visualization" / "eda_analysis.py",
    ROOT / "Week2_EDA_Visualization" / "diagrams.py",
    ROOT / "Week3_ML_Model_Development" / "model_training.py",
    ROOT / "Week3_ML_Model_Development" / "model_training_v2.py",
    ROOT / "Week3_ML_Model_Development" / "diagrams.py",
    ROOT / "Week3_ML_Model_Development" / "diagrams_v2.py",
    ROOT / "Week4_Report_Presentation" / "generate_report.py",
    ROOT / "Week4_Report_Presentation" / "diagrams.py",
    ROOT / "Week4_Report_Presentation" / "generate_dashboard.py",
    ROOT / "Week4_Report_Presentation" / "generate_plotly_dashboard.py",
]

# The reference_project_layout/src package (Week1_Project_Planning/) is run
# as modules, not scripts, since it uses relative imports within src/.
# Re-running this doesn't touch notebooks/01_churn_eda_and_modeling.ipynb —
# that notebook is committed with its outputs already executed.
REFERENCE_LAYOUT_MODULES = [
    "reference_project_layout.src.data.make_dataset",
    "reference_project_layout.src.features.build_features",
    "reference_project_layout.src.models.train_model",
    "reference_project_layout.src.visualization.visualize",
]

for script in COMMANDS:
    print(f"\n>>> Running {script.relative_to(ROOT)}")
    subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)

for module in REFERENCE_LAYOUT_MODULES:
    print(f"\n>>> Running -m {module}")
    subprocess.run(
        [sys.executable, "-m", module], check=True, cwd=ROOT / "Week1_Project_Planning"
    )

print("\nPipeline completed successfully.")
