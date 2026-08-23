#!/usr/bin/env python3
"""Build the full-runtime method figure from current code and a stored trace."""

from __future__ import annotations

import argparse
import html
import json
import sys
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "L2::bridge_join_0502#L2a"
DEFAULT_OUT = ROOT / "paper" / "tmlr" / "figures" / "src" / "fig02_full_runtime.svg"


def row_by_id(path: Path, row_id: str) -> tuple[dict[str, Any], int]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("id") == row_id:
                return row, line_no
    raise KeyError(f"{row_id!r} not found in {path}")


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: str, cls: str = "body", anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{esc(value)}</text>'


def lines(x: float, y: float, values: list[str], cls: str = "body", leading: float = 15) -> str:
    out = [f'<text x="{x}" y="{y}" class="{cls}">']
    for index, value in enumerate(values):
        out.append(f'<tspan x="{x}" dy="{0 if index == 0 else leading}">{esc(value)}</tspan>')
    out.append("</text>")
    return "".join(out)


def box(x: float, y: float, w: float, h: float, accent: str = "black") -> str:
    colors = {"learned": "#2f6fa3", "deterministic": "#333333", "release": "#4f6f5a", "black": "#333333"}
    color = colors[accent]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#444444" stroke-width="0.9"/>'
        f'<line x1="{x}" y1="{y}" x2="{x + w}" y2="{y}" stroke="{color}" stroke-width="3"/>'
    )


def arrow(x1: float, y1: float, x2: float, y2: float, cls: str = "arrow") -> str:
    return f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}" marker-end="url(#arrowhead)"/>'


def load_trace() -> dict[str, Any]:
    trace_path = ROOT / "data/qa/teacher_full_v1/traces.jsonl"
    sft_path = ROOT / "data/qa/teacher_full_v1/verified_sft.jsonl"
    trace, trace_line = row_by_id(trace_path, TRACE_ID)
    sft, sft_line = row_by_id(sft_path, TRACE_ID)
    if not (trace.get("verified") is True and trace.get("oracle_match") is True):
        raise RuntimeError("selected runtime trace is no longer verified and oracle-correct")
    if trace.get("answer") != 1940 or trace.get("n_repair_attempts") != 0:
        raise RuntimeError("selected runtime trace changed; review the method figure")
    briefing = trace.get("briefing")
    if not isinstance(briefing, dict):
        raise RuntimeError("selected trace no longer contains its Step-1 intent program")

    sys.path.insert(0, str(ROOT / "src"))
    from procurement_graph.reasoning.graph_planning import normalise_graph_plan
    from procurement_graph.reasoning.typed_planning import (
        _fastpath_veto,
        _intent_program_to_graph_payload,
        _verify_intent_program,
    )

    issues = _verify_intent_program(briefing)
    payload, error = _intent_program_to_graph_payload(sft["question"], {"intent_program": briefing})
    if issues or error or not payload:
        raise RuntimeError(f"stored Step-1 program no longer compiles: {issues!r} / {error}")
    compiled_graph = normalise_graph_plan(sft["question"], payload["graph_plan"])
    saved_graph = normalise_graph_plan(sft["question"], trace["graph_plan"])
    if compiled_graph != saved_graph:
        raise RuntimeError("current deterministic compiler no longer reproduces the stored graph plan")
    veto = _fastpath_veto(sft["question"], compiled_graph)
    if veto:
        raise RuntimeError(f"selected trace is no longer a deterministic fast-path example: {veto}")

    return {
        "trace": trace,
        "sft": sft,
        "briefing": briefing,
        "graph": compiled_graph,
        "sources": {
            "trace": f"{trace_path.relative_to(ROOT)}:{trace_line}",
            "sft": f"{sft_path.relative_to(ROOT)}:{sft_line}",
            "runtime_control_flow": [
                "src/procurement_graph/reasoning/typed_planning.py:2583-2702",
                "src/procurement_graph/reasoning/pipeline.py:318-623",
            ],
        },
    }


def build_svg(data: dict[str, Any]) -> str:
    question = data["sft"]["question"]
    out: list[str] = ['''<svg xmlns="http://www.w3.org/2000/svg" width="6.5in" height="2.34in" viewBox="0 0 1000 360" role="img" aria-labelledby="svg-title svg-desc">
<title id="svg-title">Implemented full question-answering runtime with a stored multi-hop trace</title>
<desc id="svg-desc">A question is converted to a typed intent program. Valid intent programs take a deterministic compiler path, while invalid or vetoed programs use a conditional graph planner. The resulting executable graph is grounded, run over the procurement graph and checked before an answer is released. A stored multi-hop trace illustrates the deterministic fast path and returns one thousand nine hundred forty notices.</desc>
<defs>
  <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L7,4 L0,8 z" fill="#222222"/></marker>
  <style>
    text { font-family: "DejaVu Sans", Arial, sans-serif; fill: #171717; }
    .panel { font-size: 17px; font-weight: 700; }
    .label { font-size: 14px; font-weight: 700; }
    .body { font-size: 13px; }
    .small { font-size: 12px; fill: #505050; }
    .mini { font-size: 11.2px; fill: #5d5d5d; }
    .mono { font-family: "DejaVu Sans Mono", monospace; font-size: 11.5px; }
    .arrow { fill: none; stroke: #222222; stroke-width: 1.25; }
    .feedback { fill: none; stroke: #666666; stroke-width: 1.1; stroke-dasharray: 5 4; }
    .rule { stroke: #777777; stroke-width: 0.9; }
    .learned { fill: #2f6fa3; font-weight: 700; }
    .correct { fill: #2e7d32; font-size: 13px; font-weight: 700; }
    .abstain { fill: #7a5814; font-size: 13px; font-weight: 700; }
    .tracebox { fill: #ffffff; stroke: #777777; stroke-width: 0.8; }
  </style>
</defs>
<rect width="1000" height="360" fill="#ffffff"/>
''']

    # Panel (a): code-accurate runtime control flow.
    out += [text(15, 23, "(a) Full runtime", "panel")]
    stages = [
        (15, 45, 72, 66, "black", "Question", ["+ schema"]),
        (105, 45, 122, 66, "learned", "Interpret", ["intent program", "answer signature"]),
        (326, 45, 119, 66, "deterministic", "Compile", ["entities · DAG", "executable graph"]),
        (463, 45, 119, 66, "deterministic", "Ground", ["schema · values", "guards · preflight"]),
        (600, 45, 118, 66, "deterministic", "Execute", ["bound variables", "exact reduction"]),
        (736, 45, 143, 66, "release", "Release checks", ["evidence · sanity", "postflight"]),
    ]
    for x, y, w, h, accent, title, subtitle in stages:
        out += [box(x, y, w, h, accent), text(x + 8, y + 23, title, "label"), lines(x + 8, y + 42, subtitle, "mini", 14)]
    for x1, x2 in [(87, 105), (290, 326), (445, 463), (582, 600), (718, 736)]:
        out.append(arrow(x1, 78, x2, 78))

    # Conditional gate and Step-2 planner.
    out += [
        '<polygon points="259,49 290,78 259,107 228,78" fill="#ffffff" stroke="#444444" stroke-width="0.9"/>',
        text(259, 73, "valid", "mini", "middle"), text(259, 87, "intent?", "mini", "middle"),
        arrow(227, 78, 228, 78),
        text(297, 68, "yes", "mini"),
        box(235, 132, 110, 45, "learned"), text(244, 151, "Step-2 planner", "label"),
        text(244, 167, "fallback graph plan", "mini"),
        '<path d="M259,107 L259,132" class="arrow" marker-end="url(#arrowhead)"/>',
        text(266, 124, "no / veto", "mini"),
        '<path d="M345,154 L385,154 L385,111" class="arrow" marker-end="url(#arrowhead)"/>',
    ]

    # Outputs and bounded repair.
    out += [
        arrow(879, 63, 900, 63), arrow(879, 96, 900, 96),
        text(907, 60, "Answer + evidence", "correct"),
        text(907, 79, "or", "mini"),
        text(907, 99, "No verified answer", "abstain"),
        '<path d="M807,111 L807,190 L290,190 L290,177" class="feedback" marker-end="url(#arrowhead)"/>',
        text(520, 185, "structured failure · bounded replan · rerun ground → execute → checks", "mini", "middle"),
        '<line x1="15" y1="211" x2="985" y2="211" class="rule"/>',
    ]

    # Panel (b): a real stored trace aligned with the main method.
    out += [
        text(15, 236, "(b) Stored multi-hop trace — deterministic fast path", "panel"),
        box(15, 253, 190, 87, "black"),
        lines(25, 274, textwrap.wrap(question, width=29), "small", 15),
        arrow(205, 296, 222, 296),
        box(224, 253, 250, 87, "learned"),
        text(234, 274, "Step-1 intent program", "label"),
        lines(234, 293, ["A filter supplier = Dodd Group", "B = distinct buyers(A)",
                         "C filter buyer ∈ B", "D = count(C)"], "mono", 15),
        arrow(474, 296, 492, 296),
        box(494, 253, 178, 87, "deterministic"),
        text(504, 274, "Compiled graph", "label"),
        lines(504, 295, ["A record_set", "→ B buyer_set", "→ C bound records"], "mono", 15),
        arrow(672, 296, 690, 296),
        box(692, 253, 112, 87, "deterministic"),
        text(702, 274, "Execute", "label"),
        text(748, 304, "count(C)", "mono", "middle"),
        text(748, 326, "no repair", "mini", "middle"),
        arrow(804, 296, 822, 296),
        box(824, 253, 161, 87, "release"),
        text(834, 274, "Verified result", "label"),
        text(904, 307, "1,940", "panel", "middle"),
        text(904, 328, "passed · oracle match ✓", "correct", "middle"),
        text(480, 351, "Valid Step-1 program → deterministic compiler; the conditional Step-2 planner is bypassed for this trace.", "mini", "middle"),
        "</svg>",
    ]
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = load_trace()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_svg(data), encoding="utf-8")
    trace = data["trace"]
    manifest = {
        "figure": "fig02_full_runtime",
        "system_name": None,
        "trace_id": TRACE_ID,
        "question": data["sft"]["question"],
        "current_code_branch": "deterministic intent compiler; Step-2 planner bypassed",
        "verified": trace["verified"],
        "oracle_match": trace["oracle_match"],
        "answer": trace["answer"],
        "repair_attempts": trace["n_repair_attempts"],
        "sources": data["sources"],
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.out)


if __name__ == "__main__":
    main()
