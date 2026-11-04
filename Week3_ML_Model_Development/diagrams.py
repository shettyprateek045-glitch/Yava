"""
Week 3 - ML Pipeline & Algorithm Diagrams
=============================================
Generates:
  1. End-to-end ML development pipeline flowchart (preprocessing through
     deployment, with a tuning feedback loop)
  2. A model-mechanics diagram explaining the selected Logistic Regression
     classifier and how it converts features into churn probabilities

Run:
    python diagrams.py
"""

from pathlib import Path
from graphviz import Digraph

OUT_DIR = Path(__file__).parent / "outputs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#4C72B0"
GREEN = "#55A868"
RED = "#C44E52"
GREY = "#8C8C8C"
LIGHT = "#EAF0F8"
ORANGE = "#E0922C"
ORANGE_FILL = "#FDF3E7"
PURPLE = "#8172B2"
PURPLE_FILL = "#F1EDF9"


def ml_pipeline_flowchart():
    g = Digraph("ml_pipeline", format="png")
    g.attr(rankdir="TB", bgcolor="white", fontname="Helvetica")
    g.attr("node", fontname="Helvetica", fontsize="11")
    g.attr("edge", fontname="Helvetica", fontsize="9", color=GREY)

    def box(name, label, color=BLUE, fill=LIGHT):
        g.node(name, label, shape="box", style="filled,rounded", fillcolor=fill, color=color, penwidth="1.8")

    g.node("start", "Cleaned Dataset\n(from Week 2)", shape="cylinder", style="filled", fillcolor=LIGHT, color=BLUE)
    box("split", "Train / Test Split\n(stratified 80/20)")
    box("preprocess", "Preprocessing Pipeline\n(scale numeric,\none-hot encode categorical)")
    box("candidates", "Train Candidate Models\n(Logistic Regression,\nRandom Forest, Gradient Boosting)")
    box("cv", "5-Fold Cross-Validation\n(ROC-AUC scoring)")
    box("select", "Select Best\nBaseline Model")
    box("tune", "Grid Search\nHyperparameter Tuning", color=PURPLE, fill=PURPLE_FILL)
    g.node("gate", "CV score\nimproved?", shape="diamond", style="filled", fillcolor=ORANGE_FILL, color=ORANGE, fontsize="10")
    box("test_eval", "Evaluate on\nHeld-Out Test Set\n(accuracy, precision, recall,\nF1, ROC-AUC)")
    box("save", "Save Model + Metrics\n(joblib, metrics.json)", color=GREEN, fill="#EAF6EE")
    g.node("end", "Hand off to\nReporting (Week 4)", shape="ellipse", style="filled", fillcolor=RED, color=RED, fontcolor="white")

    g.edge("start", "split")
    g.edge("split", "preprocess")
    g.edge("preprocess", "candidates")
    g.edge("candidates", "cv")
    g.edge("cv", "select")
    g.edge("select", "tune")
    g.edge("tune", "gate")
    g.edge("gate", "tune", label="No, retry\ngrid")
    g.edge("gate", "test_eval", label="Yes")
    g.edge("test_eval", "save")
    g.edge("save", "end")

    g.render(OUT_DIR / "01_ml_pipeline_flowchart", cleanup=True)
    print("Saved: 01_ml_pipeline_flowchart.png")


def algorithm_mechanics_diagram():
    """Explains the selected logistic-regression classifier."""
    g = Digraph("logistic_regression_mechanics", format="png")
    g.attr(rankdir="LR", bgcolor="white", fontname="Helvetica")
    g.attr("node", fontname="Helvetica", fontsize="10")
    g.node("features", "Customer features\n(tenure, engagement, billing, support)", shape="box", style="filled,rounded", fillcolor=LIGHT, color=BLUE)
    g.node("linear", "Weighted sum\nz = β₀ + β₁x₁ + … + βₙxₙ", shape="box", style="filled,rounded", fillcolor=PURPLE_FILL, color=PURPLE)
    g.node("sigmoid", "Sigmoid function\np = 1 / (1 + e⁻ᶻ)", shape="box", style="filled,rounded", fillcolor=ORANGE_FILL, color=ORANGE)
    g.node("risk", "Churn probability\n+ classification threshold", shape="box", style="filled,rounded", fillcolor="#EAF6EA", color=GREEN)
    g.edge("features", "linear")
    g.edge("linear", "sigmoid")
    g.edge("sigmoid", "risk")
    g.render(OUT_DIR / "02_logistic_regression_mechanics", cleanup=True)
    print("Saved: 02_logistic_regression_mechanics.png")


if __name__ == "__main__":
    ml_pipeline_flowchart()
    algorithm_mechanics_diagram()
