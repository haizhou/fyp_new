#!/usr/bin/env python3
"""Build Figure 1 from auditable code and PACS artifacts.

The generated figure deliberately separates two real paths:

1. The full CICADA runtime (Step-1 understanding, conditional Step-2 planning,
   deterministic graph execution, verification, and bounded repair).
2. The direct PACS Compose capability evaluation used by the running example.

The PACS example did not run through the CICADA Step-1/ReasoningPipeline.  Keeping
the two panels visually separate prevents the paper figure from claiming that it
did.  The JSON sidecar records every source and displayed value.
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
DEFAULT_OUT = ROOT / "paper" / "tmlr" / "figures" / "src" / "fig01_system_overview.svg"
ITEM_ID = "PACS::F6:L3:f6_other_buyers_via_suppliers:0011#a"


def _read_row(path: Path, item_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("id") == item_id:
                row["_source_line"] = line_no
                return row
    raise KeyError(f"{item_id!r} not found in {path}")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _wrap(value: str, width: int) -> list[str]:
    return textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False)


def _text(x: float, y: float, value: str, cls: str = "body", anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{_escape(value)}</text>'


def _multiline(x: float, y: float, lines: list[str], cls: str = "body", leading: float = 15) -> str:
    chunks = [f'<text x="{x}" y="{y}" class="{cls}">']
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else leading
        chunks.append(f'<tspan x="{x}" dy="{dy}">{_escape(line)}</tspan>')
    chunks.append("</text>")
    return "".join(chunks)


def _round_rect(x: float, y: float, w: float, h: float, cls: str, radius: float = 12) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" class="{cls}"/>'


def _pill(x: float, y: float, w: float, label: str, cls: str) -> str:
    return _round_rect(x, y, w, 23, cls, 11.5) + _text(x + w / 2, y + 15.5, label, "pill-text", "middle")


def _arrow(x1: float, y1: float, x2: float, y2: float, cls: str = "arrow") -> str:
    return f'<path d="M {x1} {y1} L {x2} {y2}" class="{cls}" marker-end="url(#arrowhead)"/>'


def _orth_arrow(points: list[tuple[float, float]], cls: str = "arrow", marker: str = "arrowhead") -> str:
    path = "M " + " L ".join(f"{x} {y}" for x, y in points)
    return f'<path d="{path}" class="{cls}" marker-end="url(#{marker})"/>'


def _load_example() -> dict[str, Any]:
    question_path = ROOT / "data" / "qa" / "pacs_v1" / "test_channel_a.jsonl"
    base_path = ROOT / "data" / "qa" / "compose_probe_v1" / "eval_pacstest_base_a.jsonl"
    v3_path = ROOT / "data" / "qa" / "compose_probe_v1" / "eval_pacstest_v3_a.jsonl"
    question = _read_row(question_path, ITEM_ID)
    base = _read_row(base_path, ITEM_ID)
    v3 = _read_row(v3_path, ITEM_ID)

    sys.path.insert(0, str(ROOT / "src"))
    from procurement_graph.compose.eval_runtime import RuntimeAlgebraEvaluator
    from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend
    from procurement_graph.reasoning.models import QueryConstraint

    backend = ParquetKGQueryBackend.from_directory(ROOT / "data" / "kg", include_evidence=False)
    evaluator = RuntimeAlgebraEvaluator(backend)
    tree = v3["tree"]
    supplier_tree = tree["left"]["of"]["where"][0]["expr"]
    suppliers = evaluator.run(supplier_tree)["answer"]
    buyers_before_difference = evaluator.run(tree["left"])["answer"]
    full = evaluator.run(tree)
    if full.get("status") != "ok" or full.get("answer") != question["oracle_answer"]:
        raise RuntimeError("the displayed Compose-v3 tree no longer reproduces the PACS oracle")

    selected_edges: list[tuple[str, str]] = []
    preferred_buyers = {
        "Department for Infrastructure",
        "Cardiff and Vale University Health Board",
        "Leeds City Council",
        "DAERA - Forest Service",
    }
    for supplier in suppliers:
        rows = backend.query((QueryConstraint("supplier_name", "eq", supplier),))
        for record in rows:
            buyer = str(record.get("buyer_name") or "")
            edge = (str(supplier), buyer)
            if buyer in preferred_buyers and edge not in selected_edges:
                selected_edges.append(edge)
    sources = {
        "question": f"{question_path.relative_to(ROOT)}:{question['_source_line']}",
        "base_result": f"{base_path.relative_to(ROOT)}:{base['_source_line']}",
        "compose_v3_result": f"{v3_path.relative_to(ROOT)}:{v3['_source_line']}",
        "compose_eval_protocol": "scripts/run_compose_probe_eval.py:185-307",
        "full_runtime": [
            "src/procurement_graph/reasoning/typed_planning.py:2583-2702",
            "src/procurement_graph/reasoning/pipeline.py:318-623",
        ],
    }
    return {
        "question": question,
        "base": base,
        "v3": v3,
        "suppliers": suppliers,
        "buyers_before_difference": buyers_before_difference,
        "answer": full["answer"],
        "edges": selected_edges,
        "sources": sources,
    }


def _build_svg(data: dict[str, Any]) -> str:
    W, H = 1200, 760
    q = data["question"]["question"]
    pieces: list[str] = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="6.5in" height="4.1167in" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
<title id="title">CICADA system overview and an auditable compositional planning example</title>
<desc id="desc">The upper overview shows the full deployed CICADA path from question understanding through conditional planning, deterministic graph execution, verification, and bounded repair. The lower comparison shows the separate direct PACS Compose evaluation in which a base model returns an empty set while Compose-v3 produces a valid three-stage algebra tree returning seventeen buyers.</desc>
<defs>
  <marker id="arrowhead" markerWidth="9" markerHeight="9" refX="7.2" refY="3.6" orient="auto"><path d="M0,0 L7.2,3.6 L0,7.2 z" fill="#52606d"/></marker>
  <marker id="arrowhead-red" markerWidth="9" markerHeight="9" refX="7.2" refY="3.6" orient="auto"><path d="M0,0 L7.2,3.6 L0,7.2 z" fill="#c53d4a"/></marker>
  <style>
    text {{ font-family: "DejaVu Sans", Arial, sans-serif; fill: #172033; }}
    .title {{ font-size: 20px; font-weight: 700; letter-spacing: -0.2px; }}
    .subtitle {{ font-size: 11px; font-weight: 500; fill: #64748b; }}
    .section {{ font-size: 12px; font-weight: 700; letter-spacing: 0.5px; fill: #334155; }}
    .node-title {{ font-size: 12px; font-weight: 700; }}
    .body {{ font-size: 10.5px; font-weight: 400; }}
    .small {{ font-size: 9px; font-weight: 400; fill: #52606d; }}
    .tiny {{ font-size: 8px; font-weight: 400; fill: #64748b; }}
    .mono {{ font-family: "DejaVu Sans Mono", monospace; font-size: 9.2px; }}
    .mono-small {{ font-family: "DejaVu Sans Mono", monospace; font-size: 8.3px; }}
    .pill-text {{ font-size: 8.5px; font-weight: 700; fill: #334155; }}
    .card {{ fill: #ffffff; stroke: #d6deea; stroke-width: 1.3; }}
    .question {{ fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1.2; }}
    .blue-card {{ fill: #eaf2ff; stroke: #7ca7e8; stroke-width: 1.3; }}
    .violet-card {{ fill: #f3edff; stroke: #a88ddd; stroke-width: 1.3; }}
    .cyan-card {{ fill: #e8f7f8; stroke: #70b8be; stroke-width: 1.3; }}
    .green-card {{ fill: #eaf7ef; stroke: #62aa79; stroke-width: 1.3; }}
    .red-card {{ fill: #fff0f1; stroke: #d66a75; stroke-width: 1.3; }}
    .amber-card {{ fill: #fff8e8; stroke: #d7a842; stroke-width: 1.3; }}
    .gray-pill {{ fill: #eef2f6; stroke: #cbd5e1; stroke-width: 1; }}
    .blue-pill {{ fill: #dbeafe; stroke: #93b7e8; stroke-width: 1; }}
    .green-pill {{ fill: #dcf5e5; stroke: #8dc69f; stroke-width: 1; }}
    .red-pill {{ fill: #ffe1e4; stroke: #e6a0a7; stroke-width: 1; }}
    .amber-pill {{ fill: #fff0c7; stroke: #e2be63; stroke-width: 1; }}
    .lane {{ fill: #f8fafc; stroke: #d9e2ec; stroke-width: 1.1; }}
    .pacs-lane {{ fill: #fbfaff; stroke: #d9d0ec; stroke-width: 1.1; }}
    .arrow {{ fill: none; stroke: #52606d; stroke-width: 1.8; stroke-linejoin: round; stroke-linecap: round; }}
    .arrow-dashed {{ fill: none; stroke: #64748b; stroke-width: 1.5; stroke-dasharray: 5 4; }}
    .arrow-red {{ fill: none; stroke: #c53d4a; stroke-width: 1.7; }}
    .divider {{ stroke: #cbd5e1; stroke-width: 1; }}
    .graph-edge {{ stroke: #9aa8b8; stroke-width: 1.5; fill: none; }}
    .graph-node-buyer {{ fill: #eaf2ff; stroke: #6c9bd2; stroke-width: 1.2; }}
    .graph-node-supplier {{ fill: #f3edff; stroke: #9b7cc7; stroke-width: 1.2; }}
    .error {{ fill: #bd3342; font-weight: 700; }}
    .success {{ fill: #287a45; font-weight: 700; }}
  </style>
</defs>
<rect width="1200" height="760" fill="#ffffff"/>
''']

    pieces += [
        _text(30, 31, "CICADA: understand first, then plan and verify", "title"),
        _text(30, 49, "Full runtime above · separate PACS capability evidence below", "subtitle"),
        _pill(1021, 20, 149, "AUDITABLE BY DESIGN", "green-pill"),
        _text(30, 73, "A  FULL DEPLOYED KGQA RUNTIME", "section"),
        _round_rect(25, 84, 1150, 250, "lane", 16),
    ]

    # Full runtime: question + schema.
    pieces += [
        _round_rect(45, 108, 175, 104, "question", 12),
        _pill(58, 119, 91, "USER INPUT", "gray-pill"),
        _text(58, 157, "Question", "node-title"),
        _multiline(58, 175, ["+ retrieved KG", "schema context"], "body", 15),
    ]

    # Step 1.
    pieces += [
        _round_rect(259, 102, 190, 116, "blue-card", 13),
        _pill(272, 113, 72, "STEP 1", "blue-pill"),
        _text(272, 154, "Intent understander", "node-title"),
        _multiline(272, 173, ["answer signature", "typed intent program", "unsupported cues"], "small", 14),
        _arrow(220, 160, 259, 160),
    ]

    # Decision / conditional step 2.
    pieces += [
        '<polygon points="490,160 548,112 606,160 548,208" fill="#fff8e8" stroke="#d7a842" stroke-width="1.3"/>',
        _text(548, 151, "valid", "small", "middle"),
        _text(548, 165, "intent", "small", "middle"),
        _text(548, 179, "program?", "small", "middle"),
        _arrow(449, 160, 490, 160),
        _text(563, 222, "yes", "tiny"),
        _text(611, 153, "no", "tiny"),
        _round_rect(625, 96, 190, 93, "violet-card", 13),
        _pill(638, 107, 72, "STEP 2", "gray-pill"),
        _text(638, 145, "Graph planner", "node-title"),
        _multiline(638, 163, ["variables · bindings · return"], "small", 14),
        _orth_arrow([(606, 160), (625, 160)], "arrow"),
        _round_rect(625, 224, 190, 82, "cyan-card", 13),
        _text(638, 250, "Deterministic compiler", "node-title"),
        _multiline(638, 269, ["resolve entities · check DAG", "ground executable variables"], "small", 14),
        _orth_arrow([(548, 208), (548, 265), (625, 265)], "arrow"),
        _arrow(720, 189, 720, 224),
    ]

    # Execution and verification.
    pieces += [
        _round_rect(854, 102, 148, 116, "cyan-card", 13),
        _text(868, 132, "Graph executor", "node-title"),
        _multiline(868, 153, ["bind → ground", "exhaustive query", "deterministic reduce"], "small", 14),
        _orth_arrow([(815, 265), (835, 265), (835, 160), (854, 160)], "arrow"),
        _round_rect(1022, 102, 133, 116, "green-card", 13),
        _text(1035, 132, "Verification", "node-title"),
        _multiline(1035, 153, ["schema + guards", "evidence + sanity", "postflight checks"], "small", 14),
        _arrow(1002, 160, 1022, 160),
        _round_rect(854, 247, 301, 59, "card", 12),
        _text(870, 271, "Verified AnswerCard", "node-title"),
        _text(870, 291, "answer + evidence + limitations   or   abstain", "small"),
        _orth_arrow([(1088, 218), (1088, 247)], "arrow"),
        _orth_arrow([(1088, 218), (1088, 235), (780, 235), (780, 189)], "arrow-dashed", marker="arrowhead"),
        _pill(842, 222, 174, "ELIGIBLE DEFECT → REPLAN", "amber-pill"),
        _text(57, 319, "Step 2 is conditional: a valid Step-1 intent program is compiled deterministically; only the fallback branch calls the graph-plan LLM.", "tiny"),
    ]

    # Separate PACS panel.
    pieces += [
        _text(30, 360, "B  DIRECT PACS COMPOSE EVALUATION · REAL HELD-OUT EXAMPLE", "section"),
        _round_rect(25, 371, 1150, 356, "pacs-lane", 16),
        _pill(895, 385, 253, "NO STEP 1 · NO REASONINGPIPELINE", "amber-pill"),
        _round_rect(45, 395, 365, 100, "question", 12),
        _pill(58, 406, 112, "F6 · L3 · UNSEEN", "violet-card"),
        _multiline(58, 443, _wrap(q, 57)[:4], "body", 15),
    ]

    # Small real graph fragment.
    gx, gy = 45, 510
    pieces += [_round_rect(gx, gy, 365, 195, "card", 12), _text(gx + 13, gy + 23, "Procurement relation fragment", "node-title")]
    # Nodes and edges are an explanatory subgraph; the counts come from execution.
    coords = {
        "anchor": (91, 610),
        "fp": (211, 558),
        "gibson": (211, 610),
        "green": (211, 662),
        "infra": (345, 551),
        "comm": (345, 588),
        "lisburn": (345, 625),
        "daera": (345, 662),
    }
    edge_pairs = [("anchor", "fp"), ("anchor", "gibson"), ("anchor", "green"),
                  ("fp", "infra"), ("fp", "comm"), ("gibson", "lisburn"), ("green", "daera")]
    for a, b in edge_pairs:
        x1, y1 = coords[a]
        x2, y2 = coords[b]
        pieces.append(f'<path d="M{x1 + 33},{y1} C{x1 + 62},{y1} {x2 - 62},{y2} {x2 - 33},{y2}" class="graph-edge"/>')
    pieces += [
        '<circle cx="91" cy="610" r="25" class="graph-node-buyer"/>',
        _text(91, 606, "DfI", "small", "middle"), _text(91, 620, "TRAM", "small", "middle"),
        '<circle cx="211" cy="558" r="25" class="graph-node-supplier"/>',
        _text(211, 555, "FP McCann", "tiny", "middle"), _text(211, 567, "Ltd", "tiny", "middle"),
        '<circle cx="211" cy="610" r="25" class="graph-node-supplier"/>',
        _text(211, 606, "GIBSON", "tiny", "middle"), _text(211, 618, "QUARRIES", "tiny", "middle"),
        '<circle cx="211" cy="662" r="25" class="graph-node-supplier"/>',
        _text(211, 659, "GREENTOWN", "tiny", "middle"), _text(211, 671, "ENV.", "tiny", "middle"),
    ]
    for key, lines in [("infra", ["Dept. for", "Infrastructure"]), ("comm", ["Dept. for", "Communities"]),
                       ("lisburn", ["Lisburn &", "Castlereagh"]), ("daera", ["DAERA", "Forest Service"])]:
        x, y = coords[key]
        pieces.append(f'<circle cx="{x}" cy="{y}" r="24" class="graph-node-buyer"/>')
        pieces.append(_text(x, y - 3, lines[0], "tiny", "middle"))
        pieces.append(_text(x, y + 9, lines[1], "tiny", "middle"))
    pieces += [
        _text(gx + 13, gy + 188,
              f"{len(data['suppliers'])} supplier strings → {len(data['buyers_before_difference'])} buyer strings · subset shown",
              "tiny"),
    ]

    # Baseline and ours cards.
    pieces += [
        _round_rect(442, 395, 316, 126, "red-card", 12),
        _pill(456, 406, 112, "BASE QWEN3-8B", "red-pill"),
        _text(456, 445, "Wrong role + wrong projection", "node-title"),
        _multiline(456, 466, ["filter supplier = DfI TRAM", "project contract_node_id", "difference(contract IDs)"], "mono-small", 14),
        _text(731, 499, "∅  ×", "error", "end"),
        _round_rect(442, 537, 700, 168, "green-card", 12),
        _pill(456, 548, 138, "COMPOSE-v3 PLANNER", "green-pill"),
        _text(456, 587, "Typed compositional plan", "node-title"),
        _round_rect(456, 603, 189, 70, "violet-card", 9),
        _text(468, 624, "1  discover suppliers", "small"),
        _multiline(468, 643, ["S = suppliers(" , "  buyer = DfI TRAM)"], "mono-small", 13),
        _round_rect(672, 603, 189, 70, "blue-card", 9),
        _text(684, 624, "2  traverse outward", "small"),
        _multiline(684, 643, ["B = buyers(", "  supplier IN S)"], "mono-small", 13),
        _round_rect(888, 603, 137, 70, "cyan-card", 9),
        _text(900, 624, "3  exclude anchor", "small"),
        _multiline(900, 643, ["answer =", "B − {DfI TRAM}"], "mono-small", 13),
        _arrow(645, 638, 672, 638),
        _arrow(861, 638, 888, 638),
        _round_rect(1030, 590, 94, 96, "green-card", 12),
        _text(1077, 617, "VALID", "small", "middle"),
        _text(1077, 648, "17", "title", "middle"),
        _text(1077, 665, "stored names ✓", "success", "middle"),
        _arrow(1025, 638, 1030, 638),
        _text(456, 692, "validate_tree → deterministic full-universe execution → exact oracle match", "tiny"),
        _orth_arrow([(410, 436), (426, 436), (426, 458), (442, 458)], "arrow"),
        _orth_arrow([(410, 452), (426, 452), (426, 621), (442, 621)], "arrow"),
    ]

    # Footer and legend.
    pieces += [
        _text(30, 748, "Solid arrows: executed path   ·   Dashed arrow: bounded repair   ·   Lower panel is capability evidence, not an end-to-end runtime trace", "tiny"),
        "</svg>",
    ]
    return "".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = _load_example()
    svg = _build_svg(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg, encoding="utf-8")
    manifest = {
        "figure": "fig01_system_overview",
        "claim_boundary": {
            "full_runtime": "Code-audited deployed CICADA control flow; not the PACS example trace.",
            "pacs_example": "Direct Compose evaluation; no Step-1 understanding or ReasoningPipeline.",
        },
        "example_id": ITEM_ID,
        "question": data["question"]["question"],
        "base_answer": data["base"]["answer"],
        "compose_v3_answer_count": len(data["answer"]),
        "oracle_answer_count": len(data["question"]["oracle_answer"]),
        "intermediate_supplier_count": len(data["suppliers"]),
        "intermediate_buyer_count_before_difference": len(data["buyers_before_difference"]),
        "displayed_graph_is_subset": True,
        "sources": data["sources"],
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    print(manifest_path)


if __name__ == "__main__":
    main()
