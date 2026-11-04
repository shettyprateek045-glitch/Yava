"""
Week 4 - Reporting & Communication Diagrams
================================================
Generates:
  1. Insight-to-action pipeline (how a model prediction becomes a
     business decision)
  2. Stakeholder communication framework (technical -> narrative funnel)

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


def insight_to_action_pipeline():
    g = Digraph("insight_to_action", format="png")
    g.attr(rankdir="LR", bgcolor="white", fontname="Helvetica")
    g.attr("node", fontname="Helvetica", fontsize="10")
    g.attr("edge", fontname="Helvetica", fontsize="9", color=GREY)

    def box(name, label, color=BLUE, fill=LIGHT):
        g.node(name, label, shape="box", style="filled,rounded", fillcolor=fill, color=color, penwidth="1.8")

    box("model", "Trained Model\n(Week 3)", color=GREY, fill="#F2F2F2")
    box("score", "Score Every\nActive Customer\n(churn probability)")
    box("rank", "Rank by Risk\n(prioritized list)")
    box("segment", "Segment by\nEngagement Bucket\n(active / cooling off /\nat risk / dormant)")
    box("action", "Retention Team\nOutreach Campaign", color=ORANGE, fill=ORANGE_FILL)
    box("measure", "Measure Impact\n(churn rate change,\nrevenue retained)", color=GREEN, fill="#EAF6EE")
    g.node("loop", "Feed results back\ninto next\nmodel refresh", shape="ellipse", style="filled", fillcolor=RED, color=RED, fontcolor="white")

    g.edge("model", "score")
    g.edge("score", "rank")
    g.edge("rank", "segment")
    g.edge("segment", "action")
    g.edge("action", "measure")
    g.edge("measure", "loop")
    g.edge("loop", "model", style="dashed", label="quarterly\nretrain")

    g.render(OUT_DIR / "01_insight_to_action_pipeline", cleanup=True)
    print("Saved: 01_insight_to_action_pipeline.png")


def communication_framework():
    g = Digraph("comm_framework", format="png")
    g.attr(rankdir="TB", bgcolor="white", fontname="Helvetica")
    g.attr("node", fontname="Helvetica", fontsize="10")
    g.attr("edge", fontname="Helvetica", fontsize="9", color=GREY)

    g.node("technical", "Technical Analysis\n(model metrics, statistical tests,\nfeature importances)",
           shape="box", style="filled,rounded", fillcolor=LIGHT, color=BLUE)
    g.node("translate", "Translate:\nmetric -> business meaning\n(e.g. ROC-AUC -> \"ranks risk reliably\")",
           shape="box", style="filled,rounded", fillcolor=ORANGE_FILL, color=ORANGE)
    g.node("narrative", "Build Narrative\n(problem -> evidence -> so-what -> action)",
           shape="box", style="filled,rounded", fillcolor="#F1EDF9", color="#8172B2")
    g.node("visual", "One Anchor Visual\nper Key Message\n(avoid chart overload)",
           shape="box", style="filled,rounded", fillcolor="#EAF6EE", color=GREEN)
    g.node("audience", "Executive / Non-Technical\nStakeholders", shape="ellipse",
           style="filled", fillcolor=RED, color=RED, fontcolor="white")

    g.edge("technical", "translate")
    g.edge("translate", "narrative")
    g.edge("narrative", "visual")
    g.edge("visual", "audience")

    g.render(OUT_DIR / "02_communication_framework", cleanup=True)
    print("Saved: 02_communication_framework.png")


if __name__ == "__main__":
    insight_to_action_pipeline()
    communication_framework()
