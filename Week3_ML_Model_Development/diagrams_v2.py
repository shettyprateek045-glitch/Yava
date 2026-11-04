"""
Week 3 - Stacking Ensemble Architecture Diagram (v2)
========================================================
Visualizes how the v2 model combines three base learners via a
meta-learner, for use in the model card / documentation.

Run:
    python diagrams_v2.py
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
PURPLE = "#8172B2"
PURPLE_FILL = "#F1EDF9"


def stacking_architecture():
    g = Digraph("stacking_architecture", format="png")
    g.attr(rankdir="LR", bgcolor="white", fontname="Helvetica")
    g.attr("node", fontname="Helvetica", fontsize="10")
    g.attr("edge", fontname="Helvetica", fontsize="9", color=GREY)

    g.node("features", "Engineered\nFeature Set\n(18 features)", shape="cylinder",
           style="filled", fillcolor=LIGHT, color=BLUE)

    with g.subgraph(name="cluster_base") as c:
        c.attr(label="Base Learners (5-fold out-of-fold predictions)", fontsize="11",
               style="rounded", color=GREY, fontname="Helvetica")
        c.node("lr", "Logistic\nRegression", shape="box", style="filled,rounded",
               fillcolor=PURPLE_FILL, color=PURPLE)
        c.node("rf", "Random\nForest\n(150 trees)", shape="box", style="filled,rounded",
               fillcolor=PURPLE_FILL, color=PURPLE)
        c.node("hgb", "HistGradient\nBoosting", shape="box", style="filled,rounded",
               fillcolor=PURPLE_FILL, color=PURPLE)

    g.node("meta", "Meta-Learner\n(Logistic Regression\non base predictions)",
           shape="box", style="filled,rounded", fillcolor="#FDF3E7", color="#E0922C")
    g.node("thresh", "Decision Threshold\n(0.206, F1-optimized\nvia precision-recall curve)",
           shape="box", style="filled,rounded", fillcolor="#EAF6EE", color=GREEN)
    g.node("out", "Churn\nPrediction", shape="ellipse", style="filled", fillcolor=RED, color=RED, fontcolor="white")

    g.edge("features", "lr")
    g.edge("features", "rf")
    g.edge("features", "hgb")
    g.edge("lr", "meta")
    g.edge("rf", "meta")
    g.edge("hgb", "meta")
    g.edge("meta", "thresh")
    g.edge("thresh", "out")

    g.render(OUT_DIR / "03_stacking_ensemble_architecture", cleanup=True)
    print("Saved: 03_stacking_ensemble_architecture.png")


if __name__ == "__main__":
    stacking_architecture()
