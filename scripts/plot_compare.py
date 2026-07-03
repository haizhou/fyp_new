#!/usr/bin/env python3
"""Plot all compare_<label>.summary.json series as a grouped bar chart (accuracy by category) + CSV.

Picks up whatever ran: compare_ours, compare_rag_naive, compare_rag_strong, ... So you can show
our system vs naive RAG vs strong RAG side by side.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CMP = ROOT / "data" / "qa" / "eval" / "compare"
# v1:conjunction excluded: its golden answers AND-in supplier_count>=1 & buyer_count>=1,
# conditions absent from the question text, so it penalises any question-faithful system
# (oracle-generation artifact, not a capability gap). Overalls are recomputed without it.
EXCLUDE = {"v1:conjunction"}
# preferred display order + colour; anything else appended after.
STYLE = {
    "ours": ("Our system (KG reasoning)", "#2b8cbe"),
    "rag_naive": ("Baseline RAG (naive)", "#fdae6b"),
    "rag_strong": ("Strong RAG (boosted retrieval)", "#d95f0e"),
    "rag": ("Baseline RAG", "#d95f0e"),
}


def main() -> None:
    files = sorted(CMP.glob("compare_*.summary.json"))
    series = []
    for f in files:
        label = f.name[len("compare_"):-len(".summary.json")]
        series.append((label, json.loads(f.read_text(encoding="utf-8"))))
    if not series:
        print("no compare_*.summary.json found — run run_compare.py first"); return
    order = ["ours", "rag_naive", "rag_strong", "rag"]
    series.sort(key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order))

    cats = sorted({c for _, s in series for c in s["by_category"]} - EXCLUDE)
    labels = [lbl for lbl, _ in series]

    def overall(s):  # recompute over shown categories only
        tot = sum(s["by_category"].get(c, {}).get("total", 0) for c in cats)
        cor = sum(s["by_category"].get(c, {}).get("correct", 0) for c in cats)
        return cor / tot if tot else 0.0
    ov = {lbl: overall(s) for lbl, s in series}

    # CSV
    with (CMP / "compare_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category"] + [f"{lbl}_%" for lbl in labels])
        for c in cats:
            w.writerow([c] + [f"{s['by_category'].get(c, {}).get('accuracy', 0)*100:.0f}" for _, s in series])
        w.writerow(["OVERALL"] + [f"{ov[lbl]*100:.0f}" for lbl, _ in series])

    # grouped bars
    n = len(series)
    width = 0.8 / n
    x = range(len(cats))
    fig, ax = plt.subplots(figsize=(14, 6.5))
    for j, (lbl, s) in enumerate(series):
        vals = [s["by_category"].get(c, {}).get("accuracy", 0.0) * 100 for c in cats]
        name, color = STYLE.get(lbl, (lbl, None))
        offset = (j - (n - 1) / 2) * width
        bars = ax.bar([i + offset for i in x], vals, width,
                      label=f"{name}  (overall {ov[lbl]:.0%})", color=color)
        for i, v in zip(x, vals):
            ax.text(i + offset, v + 1, f"{v:.0f}", ha="center", fontsize=6)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Structured KG reasoning vs RAG baselines — accuracy by question category (20 each)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    if EXCLUDE:
        fig.text(0.01, 0.01, "Excludes " + ", ".join(sorted(EXCLUDE)) +
                 " (golden answers encode filters absent from the question text).",
                 fontsize=7, style="italic", color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CMP / "compare_chart.png", dpi=150)
    print(f"wrote {CMP / 'compare_chart.png'} and compare_table.csv")
    for lbl, _ in series:
        print(f"  {lbl:12s} overall {ov[lbl]:.0%}")


if __name__ == "__main__":
    main()
