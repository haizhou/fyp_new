#!/usr/bin/env python3
"""Figure 5.1 / fig:architecture — CICADA system diagram (vector PDF, journal quality).

Two-step planning -> deterministic core (compile/ground/execute/verify) -> answer|abstain,
gated reflector loop, four-check chain with the semantic seam marked, three layers annotated.
Palette matches the figure suite (dataviz reference, light mode)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

C = {"blue": "#2a78d6", "blue_soft": "#e3edf9", "aqua": "#1baf7a", "amber": "#eda100",
     "amber_soft": "#fdf3dd", "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e",
     "muted": "#898781", "line": "#c3c2b7", "core_bg": "#f4f4f1", "green_soft": "#e2f2e2",
     "green": "#008300", "gray_soft": "#ececea"}

fig, ax = plt.subplots(figsize=(7.4, 4.15))
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=7.6, lw=1.0, tc=None, bold=False, r=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.4,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc or C["ink"], zorder=4, fontweight="bold" if bold else "normal",
            linespacing=1.35)

def arrow(x1, y1, x2, y2, color=None, lw=1.2, style="-", rad=0.0, z=2, mut=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mut,
                 linewidth=lw, color=color or C["ink2"], linestyle=style,
                 connectionstyle=f"arc3,rad={rad}", zorder=z))

def chip(x, y, n, color):
    ax.add_patch(plt.Circle((x, y), 1.55, facecolor="white", edgecolor=color, lw=1.2, zorder=6))
    ax.text(x, y, n, ha="center", va="center", fontsize=7.2, color=color, zorder=7, fontweight="bold")

# ---------- main row (y ~ 32..46) ----------
box(1.5, 34.5, 10.5, 9, "Question", "white", C["ink2"], fs=8, bold=True)

box(17, 34.5, 15.5, 9, "Step 1 · Briefing\n8B + LoRA (local)\nintent program", C["blue_soft"], C["blue"], fs=7.2)
box(37.5, 34.5, 16, 9, "Step 2 · Graph plan\n8B + LoRA (local)\nguided JSON", C["blue_soft"], C["blue"], fs=7.2)

# deterministic core container
ax.add_patch(FancyBboxPatch((58, 26.5), 40, 21.5, boxstyle="round,pad=0.4,rounding_size=2",
             facecolor=C["core_bg"], edgecolor=C["line"], linewidth=1.1, zorder=1))
ax.text(78, 45.6, "Deterministic core  (sole answer authority)", ha="center",
        fontsize=7.6, color=C["ink2"], style="italic", zorder=4)
cw, ch_, cy = 9.0, 8.2, 33.0
labels = ["Compile", "Ground", "Execute", "Verify"]
subs   = ["T-rewrites", "schema + ER", "KG 215k", "checks"]
xs = [59.5, 69.3, 79.1, 88.9]
for x, t, sub in zip(xs, labels, subs):
    box(x, cy, cw, ch_, "", "white", C["ink2"])
    ax.text(x+cw/2, cy+ch_/2+1.5, t, ha="center", va="center", fontsize=7.2, fontweight="bold", zorder=5)
    ax.text(x+cw/2, cy+ch_/2-1.9, sub, ha="center", va="center", fontsize=5.4, color=C["ink2"], zorder=5)
for i in range(3):
    arrow(xs[i]+cw+0.45, cy+ch_/2, xs[i+1]-0.5, cy+ch_/2, lw=1.0, mut=7)

# main flow arrows
arrow(12.4, 39, 16.4, 39)
arrow(33, 39, 36.9, 39)
arrow(54, 39, 58.9, 37.4, rad=-0.06)

# outputs
box(88.5, 16.5, 10, 5.6, "Answer", C["green_soft"], C["green"], fs=8, bold=True)
box(76.5, 16.5, 10, 5.6, "Abstain", C["gray_soft"], C["muted"], fs=8, bold=True)
arrow(93.2, 26.2, 93.4, 22.6, lw=1.2)             # verify -> answer
arrow(83.5, 26.2, 81.8, 22.6, lw=1.2, color=C["muted"])  # core -> abstain

# ---------- reflector loop (below, amber) ----------
box(62, 8.0, 22, 7, "Reflector · repair proposals\nLLM — uncertain, never a verifier", C["amber_soft"], C["amber"], fs=6.8)
arrow(73, 26.0, 72.5, 15.6, color=C["amber"], lw=1.2, rad=0.15)
ax.text(69.3, 21.0, "diagnostic\nsignals", fontsize=6.2, color=C["amber"], ha="right", linespacing=1.2)
arrow(61.5, 11.5, 45, 33.9, color=C["amber"], lw=1.2, style=(0,(4,2)), rad=0.25)
ax.text(46.5, 18.5, "repair re-enters\nall four checks", fontsize=6.2, color=C["amber"], ha="center", linespacing=1.2)

# ---------- four-check chips ----------
chip(56.2, 43.5, "1", C["ink2"]); ax.text(56.2, 51.4, "plan ↔ executor\n(hard)", fontsize=5.9, ha="center", color=C["ink2"], linespacing=1.15)
ax.plot([56.2,56.2],[45,49.6], color=C["ink2"], lw=0.7)
chip(35, 43.5, "2", C["ink2"]);  ax.text(35, 51.4, "plan ↔ briefing\n(hard)", fontsize=5.9, ha="center", color=C["ink2"], linespacing=1.15)
ax.plot([35,35],[45,49.6], color=C["ink2"], lw=0.7)
chip(14.5, 43.5, "3", C["red"]); ax.text(14.5, 51.4, "briefing ↔ question\nSEMANTIC SEAM", fontsize=5.9, ha="center", color=C["red"], linespacing=1.15, fontweight="bold")
ax.plot([14.5,14.5],[45,49.6], color=C["red"], lw=0.9)
chip(97.3, 12.5, "4", C["muted"]); ax.text(97.3, 7.4, "answer ↔ oracle\n(training only)", fontsize=5.9, ha="center", color=C["muted"], linespacing=1.15)
ax.plot([97.3,97.3],[10.9,9.4], color=C["muted"], lw=0.7, linestyle=":")
arrow(96.2, 16.3, 94.8, 16.4, color=C["muted"], lw=0.8, style=":", mut=6)

# seam leakage note
ax.text(14.5, 47.0, "", fontsize=1)  # spacer
ax.annotate("hard negatives leak here", xy=(14.5, 45.2), xytext=(4.5, 28.5),
            fontsize=6.0, color=C["red"],
            arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.8,
                            connectionstyle="arc3,rad=0.25"))

# ---------- learning layer (aqua loop, bottom-left) ----------
box(4, 8.5, 30, 7, "Learning layer\nverified traces → SFT · RSFT · DPO\n→ the two LoRA adapters", "#e4f6ef", C["aqua"], fs=6.6)
arrow(88.5, 16.4, 34.6, 11.2, color=C["aqua"], lw=1.3, rad=0.34, style=(0,(1,0)))
ax.text(30, 1.2, "supervision flows only from the verified region", fontsize=6.3,
        color=C["aqua"], ha="center", style="italic")
arrow(13, 15.9, 21, 33.9, color=C["aqua"], lw=1.1, rad=-0.18)
arrow(18, 15.9, 43, 33.9, color=C["aqua"], lw=1.1, rad=-0.12)

# abstention layer caption


OUT = Path("outputs/figures")
fig.savefig(OUT/"architecture.pdf", bbox_inches="tight")
fig.savefig(OUT/"architecture.png", dpi=230, bbox_inches="tight")
print("wrote architecture.pdf/.png")
