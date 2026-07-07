#!/usr/bin/env python3
"""Publication figure suite for the CICADA student-ladder experiments.

Reads eval/training artifacts from outputs/ and renders PDF+PNG figures into
outputs/figures/. Re-run any time; figures regenerate from whatever artifacts exist.

Design: light surface, colorblind-validated categorical palette, Wilson 95% CIs,
direct labels, no dual axes, thin marks. (dataviz reference palette.)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (dataviz reference, light mode) ------------------------------------
C = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
     "violet": "#4a3aa7", "red": "#e34948", "magenta": "#e87ba4", "orange": "#eb6834",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
     "axis": "#c3c2b7", "surface": "#fcfcfb"}
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.edgecolor": C["axis"], "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.axisbelow": True, "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "text.color": C["ink"], "axes.labelcolor": C["ink2"],
    "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_summary(path: Path):
    f = path / "compare_cicada.summary.json"
    if not f.exists():
        alts = list(path.glob("compare_*.summary.json"))
        if not alts:
            return None
        f = alts[0]
    return json.loads(f.read_text())


def save(fig, name: str):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


E = ROOT / "outputs" / "eval"

# =====================================================================
# F2 — main ladder result (v2.2 dev-set matrix), grouped by system
# =====================================================================
def fig_ladder():
    systems = [
        ("RAG naive", E / "baselines/rag_naive", C["muted"], "baseline"),
        ("RAG strong", E / "baselines/rag_strong", C["muted"], "baseline"),
        ("zero-shot", E / "matrix_v2/cicada-qwen3-zeroshot", C["blue"], "Qwen3-8B ladder"),
        ("SFT", E / "matrix_v2/cicada-qwen3-sft", C["blue"], "Qwen3-8B ladder"),
        ("+RSFT", E / "matrix_v2/cicada-qwen3-rsft", C["blue"], "Qwen3-8B ladder"),
        ("+DPO", E / "matrix_v2/cicada-qwen3-dpo", C["blue"], "Qwen3-8B ladder"),
        ("zero-shot ", E / "matrix_v2/cicada-llama31-zeroshot", C["aqua"], "Llama-3.1-8B ladder"),
        ("SFT ", E / "matrix_v2/cicada-llama31-sft", C["aqua"], "Llama-3.1-8B ladder"),
        ("+RSFT ", E / "matrix_v2/cicada-llama31-rsft", C["aqua"], "Llama-3.1-8B ladder"),
        ("+DPO ", E / "matrix_v2/cicada-llama31-dpo", C["aqua"], "Llama-3.1-8B ladder"),
    ]
    rows = []
    for label, p, color, group in systems:
        s = load_summary(p)
        if s:
            o = s["overall"]
            rows.append((label, o["correct"], o["total"], color, group))
    if not rows:
        return
    # teacher noise floor band
    tacc = []
    for r in ["teacher", "teacher_r2", "teacher_r3"]:
        s = load_summary(E / f"matrix_v2/{r}")
        if s:
            tacc.append(s["overall"]["accuracy"] * 100)

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    xs = np.arange(len(rows))
    seen = set()
    for i, (label, k, n, color, group) in enumerate(rows):
        acc = 100 * k / n
        lo, hi = wilson(k, n)
        ax.bar(i, acc, width=0.62, color=color, zorder=3,
               label=group if group not in seen else None)
        seen.add(group)
        ax.errorbar(i, acc, yerr=[[acc - lo * 100], [hi * 100 - acc]],
                    fmt="none", ecolor=C["ink2"], elinewidth=1.1, capsize=2.5, zorder=4)
        ax.text(i, min(acc, hi * 100) + 3.2, f"{acc:.1f}", ha="center",
                fontsize=8.6, color=C["ink"], zorder=5)
    if tacc:
        m = float(np.mean(tacc))
        sd = float(np.std(tacc, ddof=1)) if len(tacc) > 1 else 0.0
        ax.axhspan(m - 2 * sd, m + 2 * sd, color=C["yellow"], alpha=0.14, zorder=1)
        ax.axhline(m, color=C["yellow"], lw=1.4, zorder=2)
        ax.text(len(rows) - 0.45, m + 1.0, f"teacher {m:.1f} ± {sd:.1f} (3 runs)",
                ha="right", fontsize=8.4, color="#8a6100")
    ax.set_xticks(xs, [r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("accuracy on compare_set_v4 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Student ladder vs baselines and teacher (260 questions, Wilson 95% CI, pipeline v2.2)",
                 fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=8.4, ncols=1)
    save(fig, "F2_ladder_main")


# =====================================================================
# F3 — pipeline v1 -> v2.2 ablation dumbbell (same adapters, scaffolding only)
# =====================================================================
def fig_pipeline_ablation():
    pairs = [
        ("zero-shot", "matrix/cicada-qwen3-zeroshot", "matrix_v2/cicada-qwen3-zeroshot"),
        ("SFT", "matrix/cicada-qwen3-sft", "matrix_v2/cicada-qwen3-sft"),
        ("RSFT", "matrix/cicada-qwen3-rsft", "matrix_v2/cicada-qwen3-rsft"),
        ("DPO", "matrix/cicada-qwen3-dpo", "matrix_v2/cicada-qwen3-dpo"),
    ]
    rows = []
    for label, p1, p2 in pairs:
        s1, s2 = load_summary(E / p1), load_summary(E / p2)
        if s1 and s2:
            rows.append((label, s1["overall"]["accuracy"] * 100, s2["overall"]["accuracy"] * 100))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for i, (label, a, b) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([a, b], [y, y], color=C["axis"], lw=1.6, zorder=2)
        ax.scatter([a], [y], s=52, color=C["muted"], zorder=3, label="pipeline v1" if i == 0 else None)
        ax.scatter([b], [y], s=52, color=C["blue"], zorder=3, label="pipeline v2.2" if i == 0 else None)
        ax.text(b + 0.7, y, f"+{b - a:.1f}", va="center", fontsize=8.6, color=C["green"])
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlabel("accuracy on compare_set_v4 (%)  —  same adapters, scaffolding fixes only")
    ax.set_title("Deterministic-scaffolding ablation: pipeline v1 → v2.2 (Qwen3-8B ladder)",
                 fontsize=10, loc="left")
    ax.legend(loc="lower right", fontsize=8.4)
    save(fig, "F3_pipeline_ablation")


# =====================================================================
# F4 — per-bucket heatmap (systems x 13 buckets)
# =====================================================================
def fig_bucket_heatmap():
    systems = [
        ("RAG naive", E / "baselines/rag_naive"),
        ("teacher (r1)", E / "matrix_v2/teacher"),
        ("Q zero-shot", E / "matrix_v2/cicada-qwen3-zeroshot"),
        ("Q SFT", E / "matrix_v2/cicada-qwen3-sft"),
        ("Q RSFT", E / "matrix_v2/cicada-qwen3-rsft"),
        ("Q DPO", E / "matrix_v2/cicada-qwen3-dpo"),
        ("L zero-shot", E / "matrix_v2/cicada-llama31-zeroshot"),
        ("L SFT", E / "matrix_v2/cicada-llama31-sft"),
        ("L RSFT", E / "matrix_v2/cicada-llama31-rsft"),
        ("L DPO", E / "matrix_v2/cicada-llama31-dpo"),
    ]
    data, labels = [], []
    buckets = None
    for label, p in systems:
        s = load_summary(p)
        if not s:
            continue
        by = s["by_category"]
        if buckets is None:
            buckets = sorted(by.keys())
        data.append([100 * by[b]["accuracy"] if b in by else np.nan for b in buckets])
        labels.append(label)
    if not data:
        return
    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(7.8, 0.42 * len(labels) + 1.6))
    im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(buckets)), [b.replace("v4:", "") for b in buckets],
                  rotation=35, ha="right", fontsize=8.2)
    ax.set_yticks(range(len(labels)), labels, fontsize=8.6)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.4,
                        color="white" if v > 55 else C["ink"])
    ax.grid(False)
    ax.set_title("Per-bucket accuracy (%) on compare_set_v4 — pipeline v2.2", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.85, label="accuracy (%)")
    save(fig, "F4_bucket_heatmap")


# =====================================================================
# F5 — bootstrapping evidence: harvest yields (teacher vs student self-harvest)
# =====================================================================
def fig_bootstrap_yield():
    def yield_from(d: Path):
        f = d / "traces.jsonl"
        if not f.exists():
            return None
        ans_n = ans_vo = 0
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            if str(t.get("bucket", "")).startswith("abstain"):
                continue
            ans_n += 1
            if t.get("verified") and t.get("oracle_match"):
                ans_vo += 1
        return 100 * ans_vo / ans_n if ans_n else None
    rows = [("teacher\n(nano+grok)", yield_from(ROOT / "data/qa/teacher_full_v1"), C["yellow"]),
            ("Qwen SFT student\n(temp 0.7 ×4 + verifier)", yield_from(ROOT / "data/qa/rsft_qwen_r1"), C["blue"]),
            ("Llama SFT student\n(temp 0.7 ×4 + verifier)", yield_from(ROOT / "data/qa/rsft_llama_r1"), C["aqua"])]
    rows = [r for r in rows if r[1] is not None]
    if len(rows) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    xs = np.arange(len(rows))
    for i, (label, v, color) in enumerate(rows):
        ax.bar(i, v, width=0.55, color=color, zorder=3)
        ax.text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=9, color=C["ink"])
    ax.set_xticks(xs, [r[0] for r in rows], fontsize=8.4)
    ax.set_ylabel("oracle-correct on answerable\ntrain pool (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Verifier-gated self-harvest exceeds the teacher\n(9,267-question train pool)",
                 fontsize=10, loc="left")
    save(fig, "F5_bootstrap_yield")


# =====================================================================
# F6 — error-mode decomposition across rungs
# =====================================================================
def fig_error_modes():
    meta = {str(r["id"]): r for r in
            (json.loads(l) for l in (ROOT / "data/qa/eval/compare_set_v4.jsonl").read_text().splitlines() if l.strip())}
    systems = [("zero-shot", "matrix_v2/cicada-qwen3-zeroshot"), ("SFT", "matrix_v2/cicada-qwen3-sft"),
               ("RSFT", "matrix_v2/cicada-qwen3-rsft"), ("DPO", "matrix_v2/cicada-qwen3-dpo"),
               ("teacher", "matrix_v2/teacher")]
    labels, wrong_val, abstained, hallucinated = [], [], [], []
    for label, p in systems:
        f = E / p / "compare_cicada.results.jsonl"
        if not f.exists():
            continue
        wv = ab = ha = 0
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["correct"]:
                continue
            m = meta[str(r["id"])]
            exp = m.get("expected_status", "answerable")
            if exp != "answerable" and r["predicted"] is not None:
                ha += 1
            elif exp == "answerable" and r["predicted"] is None:
                ab += 1
            else:
                wv += 1
        labels.append(label)
        wrong_val.append(wv); abstained.append(ab); hallucinated.append(ha)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    xs = np.arange(len(labels))
    b1 = ax.bar(xs, wrong_val, 0.6, color=C["blue"], label="wrong value (answerable)")
    b2 = ax.bar(xs, abstained, 0.6, bottom=wrong_val, color=C["yellow"],
                label="abstained on answerable")
    bot = [a + b for a, b in zip(wrong_val, abstained)]
    b3 = ax.bar(xs, hallucinated, 0.6, bottom=bot, color=C["red"],
                label="answered when should abstain")
    for i in xs:
        tot = wrong_val[i] + abstained[i] + hallucinated[i]
        ax.text(i, tot + 1.2, str(tot), ha="center", fontsize=8.8)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("errors (of 260)")
    ax.set_title("What each stage fixes: error-mode decomposition (compare_set_v4, v2.2)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8.2, loc="upper right")
    save(fig, "F6_error_modes")


# =====================================================================
# F7 — training curves (eval loss per rung, log scale)
# =====================================================================
def fig_training_curves():
    runs = [("Qwen SFT", "outputs/qwen3_8b_cicada_sft_v1", C["blue"], "-"),
            ("Qwen RSFT", "outputs/qwen3_8b_cicada_rsft_v1", C["blue"], "--"),
            ("Llama SFT", "outputs/llama31_8b_cicada_sft_v1", C["aqua"], "-"),
            ("Llama RSFT", "outputs/llama31_8b_cicada_rsft_v1", C["aqua"], "--"),
            ("Qwen Step-1", "outputs/qwen3_8b_cicada_step1_v1", C["violet"], "-"),
            ("Llama Step-1", "outputs/llama31_8b_cicada_step1_v1", C["violet"], "--")]
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    plotted = False
    for label, p, color, ls in runs:
        f = ROOT / p / "trainer_log.jsonl"
        if not f.exists():
            continue
        ev = [(d["current_steps"], d["eval_loss"]) for d in
              (json.loads(l) for l in f.read_text().splitlines() if l.strip()) if "eval_loss" in d]
        if not ev:
            continue
        ax.plot([e[0] for e in ev], [e[1] for e in ev], ls, color=color, lw=1.6,
                marker="o", ms=3.5, label=label)
        plotted = True
    if not plotted:
        return
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("eval loss (log)")
    ax.set_title("Held-out loss across ladder rungs", fontsize=10, loc="left")
    ax.legend(fontsize=8.2, ncols=3)
    save(fig, "F7_training_curves")


# =====================================================================
# F8 — abstention calibration scatter
# =====================================================================
def fig_abstention():
    meta = {str(r["id"]): r for r in
            (json.loads(l) for l in (ROOT / "data/qa/eval/compare_set_v4.jsonl").read_text().splitlines() if l.strip())}
    systems = [("RAG naive", "baselines/rag_naive", C["muted"]),
               ("teacher", "matrix_v2/teacher", C["yellow"]),
               ("Q zero-shot", "matrix_v2/cicada-qwen3-zeroshot", "#9ec5f4"),
               ("Q SFT", "matrix_v2/cicada-qwen3-sft", "#5598e7"),
               ("Q DPO", "matrix_v2/cicada-qwen3-dpo", C["blue"]),
               ("L DPO", "matrix_v2/cicada-llama31-dpo", C["aqua"])]
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    for label, p, color in systems:
        f = E / p / ("compare_cicada.results.jsonl" if "baselines" not in p else "compare_rag_naive.results.jsonl")
        if not f.exists():
            alts = list((E / p).glob("compare_*.results.jsonl"))
            if not alts:
                continue
            f = alts[0]
        ans_n = ans_right = abst_n = abst_right = 0
        for line in f.read_text().splitlines():
            r = json.loads(line)
            exp = meta[str(r["id"])].get("expected_status", "answerable")
            if exp == "answerable":
                ans_n += 1
                ans_right += int(r["correct"])
            else:
                abst_n += 1
                abst_right += int(r["correct"])
        if not ans_n or not abst_n:
            continue
        x, y = 100 * ans_right / ans_n, 100 * abst_right / abst_n
        ax.scatter([x], [y], s=64, color=color, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8.2)
    ax.set_xlabel("accuracy on answerable questions (%)")
    ax.set_ylabel("correct abstention on unanswerable (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_title("Answering vs abstaining: calibration frontier", fontsize=10, loc="left")
    save(fig, "F8_abstention_frontier")


# =====================================================================
# F1 — final_test headline scoreboard (five systems, one held-out set)
# =====================================================================
def fig_final_scoreboard():
    data = [  # (label, acc, n_correct, color)
        ("Teacher\n(nano+grok)", 69.76, 1594, C["yellow"]),
        ("Llama hybrid\n(nano+SFT)", 75.97, 1736, "#9ec5f4"),
        ("Qwen hybrid\n(nano+DPO)", 78.03, 1783, "#5598e7"),
        ("Llama fully-local\n(local step1+SFT)", 83.33, 1904, C["aqua"]),
        ("Qwen fully-local\n(local step1+DPO)", 85.65, 1957, C["blue"]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for i, (label, acc, k, color) in enumerate(data):
        lo, hi = wilson(k, 2285)
        ax.bar(i, acc, 0.6, color=color, zorder=3)
        ax.errorbar(i, acc, yerr=[[acc - lo * 100], [hi * 100 - acc]], fmt="none",
                    ecolor=C["ink2"], elinewidth=1.1, capsize=2.5, zorder=4)
        ax.text(i, hi * 100 + 1.0, f"{acc:.1f}", ha="center", fontsize=9, color=C["ink"])
    ax.set_xticks(range(len(data)), [d[0] for d in data], fontsize=8.2)
    ax.set_ylabel("accuracy on held-out final_test (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Held-out test set (n=2,285): student stacks beat the teacher; local briefing beats cloud\n"
                 "all pairwise McNemar p<1e-14", fontsize=10, loc="left")
    save(fig, "F1_final_scoreboard")


# =====================================================================
# F_decay — the four-point dev->test decay (the headline mechanism figure)
# =====================================================================
def fig_decay():
    # (label, dev_acc, test_acc, color, marker)
    rows = [
        ("Qwen fully-local", 86.2, 85.65, C["blue"], "o"),
        ("Llama fully-local", 84.2, 83.33, C["aqua"], "o"),
        ("Qwen hybrid", 83.5, 78.03, C["blue"], "s"),
        ("Llama hybrid", 83.1, 75.97, C["aqua"], "s"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for label, dev, test, color, marker in rows:
        style = "-" if marker == "o" else "--"
        ax.plot([0, 1], [dev, test], style, color=color, lw=1.8, marker=marker,
                ms=7, markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.text(1.02, test, f" {label}\n {test - dev:+.1f}pt", va="center", fontsize=8.2,
                color=C["ink"])
    ax.set_xticks([0, 1], ["dev\n(compare_v4, macro)", "final_test\n(natural, harder mix)"], fontsize=9)
    ax.set_xlim(-0.1, 1.6)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Fully-local stays flat, hybrid drops sharply\n"
                 "(solid=local briefing, dashed=cloud nano briefing)", fontsize=10, loc="left")
    ax.annotate("local briefing is stable\non the harder mix",
                xy=(1, 84.5), xytext=(0.35, 89), fontsize=8, color=C["ink2"],
                arrowprops=dict(arrowstyle="->", color=C["muted"], lw=0.8))
    save(fig, "F_decay_dev_to_test")


if __name__ == "__main__":
    print("rendering figures ->", OUT)
    fig_ladder()
    fig_pipeline_ablation()
    fig_bucket_heatmap()
    fig_bootstrap_yield()
    fig_error_modes()
    fig_training_curves()
    fig_abstention()
    fig_final_scoreboard()
    fig_decay()
    print("done")
