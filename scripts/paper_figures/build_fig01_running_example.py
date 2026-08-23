#!/usr/bin/env python3
"""Build the Introduction running-example figure from saved PACS artifacts.

This is intentionally an example figure, not the full runtime architecture.  The
selected PACS row was evaluated by direct typed-tree emission, type checking and
deterministic algebra execution; it did not call the two-stage runtime planner.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ITEM_ID = "PACS::F6:L3:f6_other_buyers_via_suppliers:0011#a"
DEFAULT_OUT = ROOT / "paper" / "tmlr" / "figures" / "src" / "fig01_running_example.svg"


def read_row(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("id") == ITEM_ID:
                row["_line"] = line_no
                return row
    raise KeyError(f"missing {ITEM_ID} in {path}")


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: str, cls: str = "body", anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{esc(value)}</text>'


def lines(x: float, y: float, values: list[str], cls: str = "body", leading: float = 16) -> str:
    out = [f'<text x="{x}" y="{y}" class="{cls}">']
    for index, value in enumerate(values):
        out.append(f'<tspan x="{x}" dy="{0 if index == 0 else leading}">{esc(value)}</tspan>')
    out.append("</text>")
    return "".join(out)


def rect(x: float, y: float, w: float, h: float, cls: str, rx: float = 0) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'


def arrow(x1: float, y1: float, x2: float, y2: float, cls: str = "arrow") -> str:
    marker = "red-arrow" if cls == "red-arrow-line" else "arrow"
    return f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}" marker-end="url(#{marker})"/>'


def load_data() -> dict[str, Any]:
    question_path = ROOT / "data/qa/pacs_v1/test_channel_a.jsonl"
    base_path = ROOT / "data/qa/compose_probe_v1/eval_pacstest_base_a.jsonl"
    v3_path = ROOT / "data/qa/compose_probe_v1/eval_pacstest_v3_a.jsonl"
    question, base, v3 = read_row(question_path), read_row(base_path), read_row(v3_path)

    sys.path.insert(0, str(ROOT / "src"))
    from procurement_graph.compose.eval_runtime import RuntimeAlgebraEvaluator
    from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend
    from procurement_graph.reasoning.models import QueryConstraint

    backend = ParquetKGQueryBackend.from_directory(ROOT / "data/kg", include_evidence=False)
    evaluator = RuntimeAlgebraEvaluator(backend)
    tree = v3["tree"]
    supplier_values_tree = tree["left"]["of"]["where"][0]["expr"]
    supplier_records_tree = supplier_values_tree["of"]
    outward_records_tree = tree["left"]["of"]
    all_buyers_tree = tree["left"]

    states = {
        "anchor_records": evaluator.run(supplier_records_tree),
        "suppliers": evaluator.run(supplier_values_tree),
        "outward_records": evaluator.run(outward_records_tree),
        "buyers_before_difference": evaluator.run(all_buyers_tree),
        "answer": evaluator.run(tree),
    }
    if states["answer"].get("answer") != question["oracle_answer"]:
        raise RuntimeError("saved tree no longer reproduces the frozen PACS oracle")
    if base.get("answer") != [] or base.get("correct") is not False:
        raise RuntimeError("saved baseline failure changed; review the figure")

    relation_specs = [
        ("DfI Transport and Road Asset Management TRAM", "JOHN MCQUILLAN CONTRACTS LTD", "030700-2025-1-1"),
        ("Ards and North Down Borough Council", "JOHN MCQUILLAN CONTRACTS LTD", "011224-2024-2-2"),
        ("NI Water and its subsidiaries", "JOHN MCQUILLAN CONTRACTS LTD", "004375-2022-1-1"),
        ("DfI Transport and Road Asset Management TRAM", "W.D.M.LIMITED", "013310-2025-1-1"),
        ("Leeds City Council", "W.D.M.LIMITED", "017146-2024-3-3"),
    ]
    relation_rows: list[dict[str, str]] = []
    for buyer, supplier, contract_suffix in relation_specs:
        matches = backend.query((QueryConstraint("supplier_name", "eq", supplier),))
        row = next(
            (
                record
                for record in matches
                if record.get("buyer_name") == buyer
                and str(record.get("contract_node_id") or "").endswith(contract_suffix)
            ),
            None,
        )
        if row is None:
            raise RuntimeError(f"display relation disappeared: {buyer} / {supplier} / {contract_suffix}")
        relation_rows.append({"buyer": buyer, "supplier": supplier, "contract": contract_suffix})

    return {
        "question": question,
        "base": base,
        "v3": v3,
        "states": states,
        "relation_rows": relation_rows,
        "sources": {
            "question": f"{question_path.relative_to(ROOT)}:{question['_line']}",
            "baseline": f"{base_path.relative_to(ROOT)}:{base['_line']}",
            "compose_v3": f"{v3_path.relative_to(ROOT)}:{v3['_line']}",
            "protocol": "scripts/run_compose_probe_eval.py:185-307",
        },
    }


def _build_svg_slide_draft(data: dict[str, Any]) -> str:
    q = data["question"]["question"]
    states = data["states"]
    n_anchor = len(states["anchor_records"]["answer"])
    n_suppliers = len(states["suppliers"]["answer"])
    n_outward = len(states["outward_records"]["answer"])
    n_buyers = len(states["buyers_before_difference"]["answer"])
    n_answer = len(states["answer"]["answer"])

    out: list[str] = ['''<svg xmlns="http://www.w3.org/2000/svg" width="6.5in" height="3.315in" viewBox="0 0 1000 510" role="img" aria-labelledby="svg-title svg-desc">
<title id="svg-title">A real compositional procurement question: baseline failure and typed execution</title>
<desc id="svg-desc">A procurement graph and question are shared by two paths. The base model confuses buyer and supplier roles and returns an empty set. The trained planner emits a recursive typed program that is checked and deterministically executed through five intermediate states, returning seventeen stored buyer-name strings.</desc>
<defs>
  <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#30343b"/></marker>
  <marker id="red-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#d13d4f"/></marker>
  <style>
    text { font-family: "DejaVu Sans", Arial, sans-serif; fill: #17191d; }
    .heading { font-size: 18px; font-weight: 700; }
    .subheading { font-size: 14px; font-weight: 700; }
    .body { font-size: 13px; }
    .small { font-size: 11.5px; fill: #41464f; }
    .tiny { font-size: 10px; fill: #555c67; }
    .mono { font-family: "DejaVu Sans Mono", monospace; font-size: 11.5px; }
    .mono-small { font-family: "DejaVu Sans Mono", monospace; font-size: 10.5px; }
    .input-panel { fill: #f0f0f0; stroke: #9b9b9b; stroke-width: 1.2; }
    .question-box { fill: #ffd99f; stroke: #2e2e2e; stroke-width: 1.2; }
    .code-box { fill: #fafafa; stroke: #737373; stroke-width: 1.1; }
    .ours-box { fill: #ffffff; stroke: #5a78a8; stroke-width: 1.3; stroke-dasharray: 5 4; }
    .op-blue { fill: #dcecff; stroke: #3976b8; stroke-width: 1.1; }
    .op-green { fill: #dff3df; stroke: #3f8b4f; stroke-width: 1.1; }
    .op-yellow { fill: #fff3a8; stroke: #b29a27; stroke-width: 1.1; }
    .result-box { fill: #f3fbf3; stroke: #2f8a46; stroke-width: 1.3; }
    .callout { fill: #ffffff; stroke: #e2485a; stroke-width: 1.5; }
    .buyer { fill: #e7f0ff; stroke: #4c78a8; stroke-width: 1.1; }
    .supplier { fill: #fff2bc; stroke: #a88826; stroke-width: 1.1; }
    .graph-edge { fill: none; stroke: #8a8e95; stroke-width: 1.1; }
    .arrow { fill: none; stroke: #30343b; stroke-width: 1.5; }
    .red-arrow-line { fill: none; stroke: #d13d4f; stroke-width: 1.5; }
    .divider { stroke: #707070; stroke-width: 1.2; stroke-dasharray: 6 5; }
    .error { fill: #d5223a; font-weight: 700; }
    .success { fill: #187b36; font-weight: 700; }
    .blue { fill: #276fb0; font-weight: 700; }
    .purple { fill: #7655b5; font-weight: 700; }
  </style>
</defs>
<rect width="1000" height="510" fill="#ffffff"/>
''']

    # Shared input, deliberately compact and data-centric.
    out += [
        rect(15, 16, 245, 477, "input-panel", 8),
        text(28, 39, "Input", "heading"),
        rect(28, 50, 219, 118, "question-box"),
        text(38, 69, "Question", "subheading"),
        lines(38, 88, textwrap.wrap(q, width=35), "small", 14),
        text(28, 194, "Observed procurement relations", "subheading"),
    ]

    # Mini relation graph. It is explicitly a subset; counts come from deterministic execution.
    coords = {
        "k": (58, 321), "s1": (135, 234), "s2": (135, 321), "s3": (135, 408),
        "b1": (220, 218), "b2": (220, 270), "b3": (220, 322), "b4": (220, 374), "b5": (220, 426),
    }
    for left, right in [("k", "s1"), ("k", "s2"), ("k", "s3"),
                        ("s1", "b1"), ("s1", "b2"), ("s2", "b3"), ("s3", "b4"), ("s3", "b5")]:
        x1, y1 = coords[left]
        x2, y2 = coords[right]
        out.append(f'<path d="M{x1 + 25},{y1} C{x1 + 46},{y1} {x2 - 46},{y2} {x2 - 25},{y2}" class="graph-edge"/>')
    for key, label in [("k", ["DfI", "TRAM"]), ("b1", ["Dept. for", "Infrastructure"]),
                       ("b2", ["Dept. for", "Communities"]), ("b3", ["Lisburn &", "Castlereagh"]),
                       ("b4", ["DAERA", "Forest Service"]), ("b5", ["Education", "Authority NI"])]:
        x, y = coords[key]
        out += [f'<circle cx="{x}" cy="{y}" r="24" class="buyer"/>',
                text(x, y - 2, label[0], "tiny", "middle"), text(x, y + 10, label[1], "tiny", "middle")]
    for key, label in [("s1", ["FP McCann", "Ltd"]), ("s2", ["GIBSON", "QUARRIES"]),
                       ("s3", ["GREENTOWN", "ENV."])]:
        x, y = coords[key]
        out += [f'<circle cx="{x}" cy="{y}" r="24" class="supplier"/>',
                text(x, y - 2, label[0], "tiny", "middle"), text(x, y + 10, label[1], "tiny", "middle")]
    out += [
        text(30, 470, "buyer", "tiny"), '<circle cx="70" cy="467" r="7" class="buyer"/>',
        text(92, 470, "supplier", "tiny"), '<circle cx="147" cy="467" r="7" class="supplier"/>',
        text(170, 470, "subset shown", "tiny"),
        arrow(260, 91, 291, 91), arrow(260, 321, 278, 321),
    ]

    # Baseline comparison band.
    out += [
        text(282, 36, "(a) Base planning", "heading"),
        f'<circle cx="319" cy="91" r="26" fill="#ffffff" stroke="#30343b" stroke-width="1.2"/>',
        text(319, 87, "BASE", "tiny", "middle"), text(319, 100, "8B", "subheading", "middle"),
        arrow(345, 91, 364, 91),
        rect(366, 48, 263, 112, "code-box"),
        lines(379, 68, [
            "DIFFERENCE : VALUES", "├─ VALUES contract_node_id", "│  └─ FILTER supplier_name = k",
            "└─ VALUES contract_node_id", "   └─ FILTER buyer_name = k",
        ], "mono-small", 18),
        arrow(629, 103, 665, 103),
        text(688, 92, "∅", "heading", "middle"), text(688, 115, "wrong", "error", "middle"),
        rect(735, 49, 247, 105, "callout", 6),
        text(748, 70, "Wrong role and return type", "subheading"),
        lines(748, 90, ["The anchor is a buyer, but the plan", "filters it as a supplier and returns", "contract IDs instead of buyer names."], "small", 16),
        arrow(735, 116, 711, 109, "red-arrow-line"),
        '<line x1="279" y1="181" x2="984" y2="181" class="divider"/>',
    ]

    # Proposed typed program and deterministic execution.
    out += [
        text(282, 210, "(b) Typed compositional planning", "heading"),
        rect(279, 226, 250, 249, "ours-box"),
        text(293, 248, "Planner output: recursive program", "subheading"),
        lines(294, 271, [
            "DIFFERENCE : VALUES", "├─ VALUES buyer_name", "│  └─ FILTER supplier_name IN_EXPR",
            "│     └─ VALUES supplier_name", "│        └─ FILTER buyer_name = k",
            "└─ VALUES buyer_name", "   └─ FILTER buyer_name = k",
        ], "mono-small", 22),
        rect(294, 436, 216, 25, "op-blue", 3),
        text(402, 453, "validate_tree → VALUES  ✓", "small", "middle"),
        arrow(529, 349, 553, 349),
        text(556, 248, "Deterministic execution on 215,221 records", "subheading"),
    ]

    step_x = [556, 642, 728, 814, 900]
    step_cls = ["op-blue", "op-green", "op-blue", "op-green", "op-yellow"]
    step_titles = ["FILTER", "VALUES", "FILTER", "VALUES", "DIFFERENCE"]
    step_lines = [["buyer = k"], ["supplier"], ["supplier ∈ S"], ["buyer"], ["minus {k}"]]
    step_counts = [f"{n_anchor} records", f"{n_suppliers} strings", f"{n_outward} records", f"{n_buyers} strings", f"{n_answer} strings"]
    for index, x in enumerate(step_x):
        out += [rect(x, 267, 72, 67, step_cls[index], 3), text(x + 36, 286, step_titles[index], "tiny", "middle"),
                text(x + 36, 310, step_lines[index][0], "mono-small", "middle"),
                text(x + 36, 352, step_counts[index], "tiny", "middle")]
        if index < len(step_x) - 1:
            out.append(arrow(x + 72, 300, step_x[index + 1] - 3, 300))
    out += [
        text(556, 380, "Final stored buyer-name values", "subheading"),
        rect(556, 393, 426, 68, "result-box", 4),
        lines(568, 412, ["Ards and North Down Borough Council", "Cardiff and Vale University Health Board",
                         "Leeds City Council   ·   +14 more"], "small", 17),
        text(963, 422, "17", "heading", "end"), text(963, 444, "exact set match ✓", "success", "end"),
        text(282, 496, "Saved F6–L3 unseen example; intermediate counts are recomputed from the same frozen KG. Semantic labels are author annotations.", "tiny"),
        "</svg>",
    ]
    return "".join(out)


def build_svg(data: dict[str, Any]) -> str:
    """Academic editorial layout inspired by example-first conference figures."""
    q = data["question"]["question"]
    states = data["states"]
    counts = [
        len(states["anchor_records"]["answer"]),
        len(states["suppliers"]["answer"]),
        len(states["outward_records"]["answer"]),
        len(states["buyers_before_difference"]["answer"]),
        len(states["answer"]["answer"]),
    ]
    out: list[str] = ['''<svg xmlns="http://www.w3.org/2000/svg" width="6.5in" height="3.25in" viewBox="0 0 1000 500" role="img" aria-labelledby="svg-title svg-desc">
<title id="svg-title">Running example for a typed compositional procurement query</title>
<desc id="svg-desc">The same procurement records feed a base program and a typed compositional program. The base program reverses buyer and supplier roles and returns contract identifiers, producing an empty set. The typed program materialises supplier values, follows them to buyer-name values, removes the anchor string, and returns seventeen stored names.</desc>
<defs>
  <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L7,4 L0,8 z" fill="#1d1d1d"/></marker>
  <style>
    text { font-family: "DejaVu Sans", Arial, sans-serif; fill: #171717; }
    .panel { font-size: 17.5px; font-weight: 700; }
    .label { font-size: 14.5px; font-weight: 700; }
    .body { font-size: 14px; }
    .small { font-size: 13px; fill: #4e4e4e; }
    .mini { font-size: 12.5px; fill: #555555; }
    .mono { font-family: "DejaVu Sans Mono", monospace; font-size: 13.2px; }
    .mono-small { font-family: "DejaVu Sans Mono", monospace; font-size: 12.5px; }
    .rule { stroke: #222222; stroke-width: 1.1; }
    .light-rule { stroke: #777777; stroke-width: 0.8; }
    .divider { stroke: #777777; stroke-width: 1; stroke-dasharray: 6 5; }
    .arrow { fill: none; stroke: #1d1d1d; stroke-width: 1.35; }
    .buyer-dot { fill: #dce9f4; stroke: #2f6fa3; stroke-width: 1; }
    .supplier-dot { fill: #f4e7a1; stroke: #8d7518; stroke-width: 1; }
    .data-edge { fill: none; stroke: #555555; stroke-width: 0.9; }
    .state { fill: #ffffff; stroke: #555555; stroke-width: 0.9; }
    .state-blue { fill: #ffffff; stroke: #2f6fa3; stroke-width: 1.1; }
    .result { fill: #ffffff; stroke: #2e7d32; stroke-width: 1; }
    .error { fill: #b3261e; font-weight: 700; }
    .correct { fill: #2e7d32; font-weight: 700; }
    .ours { fill: #2f6fa3; font-weight: 700; }
    .highlight { fill: #f4e7a1; }
    .answer-highlight { fill: #edf5fa; }
  </style>
</defs>
<rect width="1000" height="500" fill="#ffffff"/>
''']

    # Shared question across the complete width: no slide title, badge, or panel background.
    out += [
        text(15, 24, "Example: a three-hop procurement query", "label"),
        lines(15, 47, textwrap.wrap(q, width=125), "body", 18),
        '<line x1="15" y1="82" x2="985" y2="82" class="rule"/>',
        '<line x1="275" y1="96" x2="275" y2="486" class="light-rule"/>',
    ]

    # Left: observed graph excerpt and exact record projection.
    out += [
        text(15, 104, "Observed buyer–supplier", "label"),
        text(15, 121, "records (excerpt)", "label"),
        text(15, 139, "Frozen record universe", "mini"),
    ]
    graph = {
        "anchor": (38, 218), "john": (122, 166), "wdm": (122, 270),
        "ards": (230, 143), "ni": (230, 190), "leeds": (230, 270),
    }
    for left, right in [("anchor", "john"), ("anchor", "wdm"), ("john", "ards"),
                        ("john", "ni"), ("wdm", "leeds")]:
        x1, y1 = graph[left]
        x2, y2 = graph[right]
        out.append(f'<path d="M{x1 + 7},{y1} C{x1 + 34},{y1} {x2 - 34},{y2} {x2 - 7},{y2}" class="data-edge"/>')
    for key in ("anchor", "ards", "ni", "leeds"):
        x, y = graph[key]
        out.append(f'<circle cx="{x}" cy="{y}" r="6" class="buyer-dot"/>')
    for key in ("john", "wdm"):
        x, y = graph[key]
        out.append(f'<rect x="{x - 6}" y="{y - 6}" width="12" height="12" class="supplier-dot"/>')
    out += [
        lines(15, 240, ["DfI … TRAM", "(anchor buyer)"] , "mini", 14),
        lines(78, 150, ["JOHN MCQUILLAN", "CONTRACTS LTD"], "mini", 14),
        text(88, 291, "W.D.M. LIMITED", "mini"),
        lines(166, 127, ["Ards &", "North Down"], "mini", 14),
        text(181, 211, "NI Water", "mini"),
        text(181, 291, "Leeds CC", "mini"),
        text(246, 239, "⋮", "panel", "middle"),
        text(15, 319, "Record projection", "mini"),
    ]

    # Mini table (square rules, no UI chrome).
    tx, ty, tw = 15, 330, 245
    col = [tx, tx + 76, tx + 158, tx + tw]
    row_h = 24
    out += [
        f'<rect x="{tx}" y="{ty}" width="{tw}" height="{row_h * 6}" fill="#ffffff" stroke="#333333" stroke-width="0.8"/>',
        f'<rect x="{col[1]}" y="{ty + row_h}" width="{col[2] - col[1]}" height="{row_h * 5}" class="highlight"/>',
    ]
    for x in col[1:-1]:
        out.append(f'<line x1="{x}" y1="{ty}" x2="{x}" y2="{ty + row_h * 6}" class="light-rule"/>')
    for row in range(1, 6):
        y = ty + row_h * row
        out.append(f'<line x1="{tx}" y1="{y}" x2="{tx + tw}" y2="{y}" class="light-rule"/>')
    out += [
        text(tx + 4, ty + 17, "buyer", "mini"), text(col[1] + 4, ty + 17, "supplier", "mini"),
        text(col[2] + 4, ty + 17, "contract suffix", "mini"),
    ]
    table_rows = [
        ("DfI … TRAM", "JOHN MCQ.", "…030700"), ("Ards & N. Down", "JOHN MCQ.", "…011224"),
        ("NI Water", "JOHN MCQ.", "…004375"), ("DfI … TRAM", "W.D.M.", "…013310"),
        ("Leeds CC", "W.D.M.", "…017146"),
    ]
    for index, (buyer, supplier, cid) in enumerate(table_rows):
        y = ty + row_h * (index + 1) + 17
        out += [text(tx + 4, y, buyer, "mini"), text(col[1] + 4, y, supplier, "mini"), text(col[2] + 4, y, cid, "mini")]
    out += [
        '<circle cx="20" cy="486" r="5" class="buyer-dot"/>', text(30, 490, "buyer value", "mini"),
        '<rect x="116" y="481" width="10" height="10" class="supplier-dot"/>', text(132, 490, "supplier value", "mini"),
    ]

    # Right top: one compact baseline expression and author diagnosis.
    out += [
        text(291, 108, "(a) Direct plan generation (base model)", "panel"),
        text(295, 139, "VALUES_contract_id[supplier_name = k]  −", "mono"),
        text(600, 139, "VALUES_contract_id[buyer_name = k]  →  [ ]", "mono"),
        '<line x1="424" y1="145" x2="558" y2="145" stroke="#b3261e" stroke-width="1.2"/>',
        '<line x1="349" y1="145" x2="422" y2="145" stroke="#b3261e" stroke-width="1.2"/>',
        '<line x1="654" y1="145" x2="727" y2="145" stroke="#b3261e" stroke-width="1.2"/>',
        text(430, 165, "buyer/supplier role reversed", "error"),
        text(642, 184, "wrong return field", "error"),
        text(960, 165, "✗", "error", "middle"),
        text(293, 184, "k = DfI Transport and Road Asset Management TRAM", "mini"),
        '<line x1="291" y1="210" x2="985" y2="210" class="divider"/>',
    ]

    # Right bottom: typed expression plus a Chain-of-Table-like sequence of actual states.
    out += [
        text(291, 236, "(b) Typed compositional query", "panel"),
        text(295, 263, "S = VALUES_supplier( FILTER_buyer=k(R) )", "mono"),
        text(295, 284, "A = VALUES_buyer( FILTER_supplier∈S(R) ) \\ {k}", "mono"),
        text(804, 263, "validate_tree(A) → VALUES  ✓", "ours"),
    ]
    step_x = [295, 430, 565, 700, 835]
    step_names = ["R₀", "S", "R₁", "B", "A"]
    step_ops = [["buyer_name", "= k"], ["VALUES", "supplier_name"], ["supplier_name", "∈ S"],
                ["VALUES", "buyer_name"], ["B \\ {k}", ""]]
    step_counts = [f"{counts[0]} records", f"{counts[1]} supplier values", f"{counts[2]} records",
                   f"{counts[3]} buyer-name values", f"{counts[4]} stored values"]
    for index, x in enumerate(step_x):
        cls = "state-blue" if index in {1, 3, 4} else "state"
        out += [
            rect(x, 305, 112, 65, cls), text(x + 8, 325, step_names[index], "label"),
            text(x + 8, 346, step_ops[index][0], "mono-small"),
            text(x + 8, 361, step_ops[index][1], "mono-small"),
            text(x + 56, 389, step_counts[index], "mini", "middle"),
        ]
        if index < 4:
            out.append(arrow(x + 112, 335, step_x[index + 1] - 4, 335))
    out += [
        text(295, 418, "17 stored buyer-name values", "label"),
        f'<rect x="295" y="428" width="690" height="54" class="result"/>',
        text(307, 449, "Ards and North Down Borough Council", "small"),
        text(307, 470, "Leeds City Council  ·  NI Water and its subsidiaries  ·  Education Authority NI  ·  …", "small"),
        text(970, 449, "exact oracle-set match ✓", "correct", "end"),
        "</svg>",
    ]
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = load_data()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_svg(data), encoding="utf-8")
    states = data["states"]
    manifest = {
        "figure": "fig01_running_example",
        "example_id": ITEM_ID,
        "protocol_boundary": "Direct PACS Compose evaluation; no separate Step-1 or ReasoningPipeline.",
        "question": data["question"]["question"],
        "displayed_counts": {
            "record_universe": 215221,
            "anchor_records": len(states["anchor_records"]["answer"]),
            "supplier_strings": len(states["suppliers"]["answer"]),
            "outward_records": len(states["outward_records"]["answer"]),
            "buyer_strings_before_difference": len(states["buyers_before_difference"]["answer"]),
            "final_stored_buyer_name_strings": len(states["answer"]["answer"]),
        },
        "base_answer": data["base"]["answer"],
        "sources": data["sources"],
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.out)


if __name__ == "__main__":
    main()
