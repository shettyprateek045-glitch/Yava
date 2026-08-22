"""
Week 1 - Project Planning Diagrams
=====================================
Generates two professional flowchart diagrams using Graphviz:
  1. End-to-end project methodology flowchart (with a tuning feedback loop)
  2. High-level system/data-flow architecture diagram

Run:
    python diagrams.py
Requires the `graphviz` Python package AND the Graphviz `dot` binary
(apt install graphviz / brew install graphviz).
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


def methodology_flowchart():
    g = Digraph("methodology", format="png")
    g.attr(rankdir="TB", splines="ortho", bgcolor="white", fontname="Helvetica")
    g.attr("node", fontname="Helvetica", fontsize="11")
    g.attr("edge", fontname="Helvetica", fontsize="9", color=GREY)

    def box(name, label, color=BLUE, shape="box", fill=LIGHT):
        g.node(name, label, shape=shape, style="filled,rounded" if shape == "box" else "filled",
               fillcolor=fill, color=color, penwidth="1.8", fontcolor="#222222")

    g.node("start", "Start", shape="ellipse", style="filled", fillcolor=GREEN, color=GREEN, fontcolor="white")
    box("collect", "1. Data Collection\n(source identification,\nacquisition, licensing)")
    box("clean", "2. Data Cleaning\n(missing values, duplicates,\noutlier handling)")
    box("eda", "3. Exploratory Data\nAnalysis\n(univariate/bivariate/\nmultivariate)")
    box("feat", "4. Feature Engineering\n(transform, encode,\nselect features)")
    box("model", "5. Model Development\n(candidate algorithms,\ntraining)")
    g.node("check", "Meets performance\ncriteria?", shape="diamond", style="filled",
           fillcolor="#FDF3E7", color="#E0922C", fontsize="10")
    box("tune", "6. Hyperparameter\nTuning", color="#E0922C", fill="#FDF3E7")
    box("evaluate", "7. Evaluation\n(hold-out test,\nbusiness metrics)")
    box("report", "8. Reporting &\nCommunication")
    box("deploy", "9. Deployment &\nMonitoring", color=GREEN, fill="#EAF6EE")
    g.node("end", "End", shape="ellipse", style="filled", fillcolor=RED, color=RED, fontcolor="white")

    g.edge("start", "collect")
    g.edge("collect", "clean")
    g.edge("clean", "eda")
    g.edge("eda", "feat")
    g.edge("feat", "model")
    g.edge("model", "check")
    g.edge("check", "tune", label="No")
    g.edge("tune", "model")
    g.edge("check", "evaluate", label="Yes")
    g.edge("evaluate", "report")
    g.edge("report", "deploy")
    g.edge("deploy", "end")

    g.render(OUT_DIR / "01_methodology_flowchart", cleanup=True)
    print("Saved: 01_methodology_flowchart.png")


def architecture_diagram():
    g = Digraph("architecture", format="png")
    g.attr(rankdir="LR", bgcolor="white", fontname="Helvetica")
    g.attr("node", fontname="Helvetica", fontsize="11", shape="box",
           style="filled,rounded", fillcolor=LIGHT, color=BLUE, penwidth="1.8")
    g.attr("edge", fontname="Helvetica", fontsize="9", color=GREY)

    g.node("raw", "Raw Data\n(usage, billing,\nsupport logs)", shape="cylinder", fillcolor="#EAF0F8")
    g.node("etl", "ETL / Cleaning\nPipeline\n(src/data)")
    g.node("store", "Processed Data\nStore", shape="cylinder", fillcolor="#EAF0F8")
    g.node("eda", "EDA &\nVisualization\n(src/visualization)")
    g.node("feat", "Feature\nEngineering\n(src/features)")
    g.node("train", "Model Training\n& Tuning\n(src/models)")
    g.node("registry", "Model Registry\n(models/)", shape="cylinder", fillcolor="#EAF6EE", color=GREEN)
    g.node("report", "Reports &\nDashboards\n(reports/)", fillcolor="#FDF3E7", color="#E0922C")
    g.node("stake", "Stakeholders", shape="ellipse", style="filled", fillcolor=RED, color=RED, fontcolor="white")

    g.edge("raw", "etl")
    g.edge("etl", "store")
    g.edge("store", "eda")
    g.edge("store", "feat")
    g.edge("feat", "train")
    g.edge("train", "registry")
    g.edge("eda", "report")
    g.edge("registry", "report")
    g.edge("report", "stake")

    g.render(OUT_DIR / "02_system_architecture", cleanup=True)
    print("Saved: 02_system_architecture.png")


if __name__ == "__main__":
    methodology_flowchart()
    architecture_diagram()
