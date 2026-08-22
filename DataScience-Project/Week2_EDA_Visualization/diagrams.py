"""
Week 2 - EDA Process Flowchart
=================================
Generates a flowchart of the exploratory data analysis workflow, showing
the decision points for handling missing values / outliers and the three
tiers of analysis (univariate -> bivariate -> multivariate).

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


def eda_flowchart():
    g = Digraph("eda_process", format="png")
    g.attr(rankdir="TB", bgcolor="white", fontname="Helvetica")
    g.attr("node", fontname="Helvetica", fontsize="11")
    g.attr("edge", fontname="Helvetica", fontsize="9", color=GREY)

    def box(name, label, color=BLUE, fill=LIGHT):
        g.node(name, label, shape="box", style="filled,rounded", fillcolor=fill, color=color, penwidth="1.8")

    g.node("start", "Load Raw\nDataset", shape="ellipse", style="filled", fillcolor=GREEN, color=GREEN, fontcolor="white")
    box("quality", "Data Quality Check\n(dtypes, shape,\nnull %, duplicates)")
    g.node("missing_q", "Missing values\nor duplicates?", shape="diamond", style="filled", fillcolor=ORANGE_FILL, color=ORANGE, fontsize="10")
    box("clean", "Impute / Drop /\nDeduplicate", color=ORANGE, fill=ORANGE_FILL)
    g.node("outlier_q", "Outliers\npresent?", shape="diamond", style="filled", fillcolor=ORANGE_FILL, color=ORANGE, fontsize="10")
    box("outlier", "IQR / Z-score\nCapping", color=ORANGE, fill=ORANGE_FILL)

    box("uni", "Univariate Analysis\n(histograms, boxplots,\nvalue counts)")
    box("bi", "Bivariate Analysis\n(grouped boxplots,\ncrosstabs, scatterplots)")
    box("multi", "Multivariate Analysis\n(correlation heatmap,\npairplot)")
    box("insight", "Synthesize Insights\n(feature-target\nrelationships)")
    box("report", "EDA Summary Report\n(markdown + charts)", color=GREEN, fill="#EAF6EE")
    g.node("end", "Hand off to\nModeling (Week 3)", shape="ellipse", style="filled", fillcolor=RED, color=RED, fontcolor="white")

    g.edge("start", "quality")
    g.edge("quality", "missing_q")
    g.edge("missing_q", "clean", label="Yes")
    g.edge("clean", "outlier_q")
    g.edge("missing_q", "outlier_q", label="No")
    g.edge("outlier_q", "outlier", label="Yes")
    g.edge("outlier", "uni")
    g.edge("outlier_q", "uni", label="No")
    g.edge("uni", "bi")
    g.edge("bi", "multi")
    g.edge("multi", "insight")
    g.edge("insight", "report")
    g.edge("report", "end")

    g.render(OUT_DIR / "01_eda_process_flowchart", cleanup=True)
    print("Saved: 01_eda_process_flowchart.png")


if __name__ == "__main__":
    eda_flowchart()
