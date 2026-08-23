#!/usr/bin/env python3
"""Build the Introduction running example from a saved Step-1 artifact.

The layout deliberately follows the visual grammar of Chain-of-Table Figure 1:
a shared input remains on the left, a compact interpretation/planning path sits
above, and the method's real intermediate states occupy the larger lower band.
It does not copy artwork or prose from that paper.

No model is called. The script reads the saved Step-1 artifact, deterministically
recompiles it with the current code, and replays it over the frozen local KG. The
rendered example is therefore a reconstruction rather than a saved final trace.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ITEM_ID = "L2::bridge_join_0502#L2a"
DEFAULT_DIR = ROOT / "paper" / "tmlr" / "figures" / "src"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: str, cls: str = "body", anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{esc(value)}</text>'


def lines(
    x: float,
    y: float,
    values: list[str] | tuple[str, ...],
    cls: str = "body",
    leading: float = 21,
    anchor: str = "start",
) -> str:
    spans = []
    for index, value in enumerate(values):
        spans.append(
            f'<tspan x="{x}" dy="{0 if index == 0 else leading}">{esc(value)}</tspan>'
        )
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{"".join(spans)}</text>'


def rect(x: float, y: float, w: float, h: float, cls: str, rx: float = 0) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'


def arrow(x1: float, y1: float, x2: float, y2: float, cls: str = "arrow") -> str:
    marker = "gray-arrow" if cls == "branch" else "arrow"
    return f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}" marker-end="url(#{marker})"/>'


def read_row(path: Path, item_id: str) -> tuple[dict[str, Any], int]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("id") == item_id:
                return row, line_no
    raise KeyError(f"missing {item_id} in {path}")


@dataclass(frozen=True)
class StaticPlanner:
    candidate: Any

    def plan(self, question: str) -> tuple[Any, ...]:
        return (self.candidate,)


def load_and_replay() -> dict[str, Any]:
    """Read the saved artifact and replay only deterministic local stages."""
    verified_path = ROOT / "data/qa/teacher_full_v1/verified_sft.jsonl"
    trace_path = ROOT / "data/qa/teacher_full_v1/traces.jsonl"
    saved, saved_line = read_row(verified_path, ITEM_ID)
    compact, compact_line = read_row(trace_path, ITEM_ID)

    sys.path.insert(0, str(ROOT / "src"))
    from procurement_graph.reasoning import ReasoningPipeline
    from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
    from procurement_graph.reasoning.models import QueryConstraint
    from procurement_graph.reasoning.typed_planning import _fastpath_veto, compile_typed_plan

    backend = RuntimeKGBackend.from_directory(ROOT / "data/kg", include_evidence=True)
    candidate = compile_typed_plan(
        saved["question"],
        {"intent_program": saved["step1_briefing"]},
        org_resolver=backend.org_resolver(),
    )
    if candidate.status != "planned" or candidate.graph_plan is None:
        raise RuntimeError(f"saved intent no longer compiles: {candidate.status}")
    raw_graph = candidate.graph_plan.raw_graph_plan
    vetoes = _fastpath_veto(saved["question"], raw_graph)
    if vetoes:
        raise RuntimeError(f"example no longer takes deterministic fast path: {vetoes}")

    replay = ReasoningPipeline(
        backend=backend,
        planner=StaticPlanner(candidate),
        org_resolver=backend.org_resolver(),
        max_feedback_replans=0,
    ).run(saved["question"])
    attempt = (replay.metadata.get("attempts") or [{}])[0]
    graph_execution = attempt.get("graph_execution") or {}
    variable_traces = graph_execution.get("variables") or []
    by_id = {str(row.get("var_id")): row for row in variable_traces}
    answer = getattr(replay.answer_card, "answer", None)
    if answer != compact.get("answer") or answer != compact.get("oracle"):
        raise RuntimeError("current deterministic replay differs from the frozen accepted answer")
    expected_sizes = {"A": 21, "B": 16, "C": 1940}
    actual_sizes = {key: int((by_id.get(key) or {}).get("output_size", -1)) for key in expected_sizes}
    if actual_sizes != expected_sizes:
        raise RuntimeError(f"intermediate cardinalities changed: {actual_sizes}")

    supplier = "Dodd Group (Midlands) Limited"
    source_constraint = (QueryConstraint("supplier_name", "eq", supplier),)
    source_rows = backend.query(source_constraint)
    source_examples = []
    preferred = (
        "Dudley MBC",
        "Solihull Metropolitan Borough Council",
        "Fusion21 Members Consortium",
    )
    for buyer in preferred:
        row = next(record for record in source_rows if record.get("buyer_name") == buyer)
        source_examples.append(
            {
                "buyer": buyer,
                "contract_id": str(row.get("contract_node_id") or ""),
            }
        )

    bridge_constraint = next(
        constraint
        for constraint in (by_id.get("C") or {}).get("constraints", [])
        if constraint.get("field") == "buyer_name" and constraint.get("op") == "in"
    )
    buyers = tuple(str(value) for value in bridge_constraint.get("value") or ())
    target_counts = Counter()
    for buyer in buyers:
        target_counts[buyer] = backend.count((QueryConstraint("buyer_name", "eq", buyer),))
    if sum(target_counts.values()) != answer:
        raise RuntimeError("per-buyer counts no longer sum to the terminal count")

    evidence_verdict = attempt.get("evidence_verdict") or {}
    postflight = attempt.get("postflight") or {}
    coverage = next(
        (check for check in postflight.get("checks", []) if check.get("check") == "population_coverage"),
        {},
    )
    answer_sanity = attempt.get("answer_sanity") or {}
    if evidence_verdict.get("status") != "kg_supported" or not answer_sanity.get("ok"):
        raise RuntimeError("release-check status changed; review the figure")

    return {
        "question": saved["question"],
        "briefing": saved["step1_briefing"],
        "graph": raw_graph,
        "answer": answer,
        "source_examples": source_examples,
        "bridge_buyers": buyers,
        "target_counts": target_counts,
        "variable_traces": variable_traces,
        "evidence_verdict": evidence_verdict,
        "answer_sanity": answer_sanity,
        "coverage": coverage,
        "confidence": getattr(replay.answer_card, "confidence_label", ""),
        "limitations": list(getattr(replay.answer_card, "limitations", ()) or ()),
        "compact_trace": compact,
        "sources": {
            "saved_step1_and_graph": f"{verified_path.relative_to(ROOT)}:{saved_line}",
            "accepted_compact_trace": f"{trace_path.relative_to(ROOT)}:{compact_line}",
            "current_runtime": "src/procurement_graph/reasoning/pipeline.py",
            "current_typed_planning": "src/procurement_graph/reasoning/typed_planning.py",
            "current_kg": "data/kg",
        },
    }


SVG_HEAD = '''<svg xmlns="http://www.w3.org/2000/svg" width="6.5in" height="4.0625in" viewBox="0 0 1040 650" role="img" aria-labelledby="svg-title svg-desc">
<title id="svg-title">A procurement bridge query reconstructed through deterministic replay</title>
<desc id="svg-desc">A saved Step 1 artifact is recompiled and replayed without a model call. The replay finds twenty one source records, derives sixteen buyer values, binds them into a target query, and counts one thousand nine hundred forty records. This reconstruction is not a saved final trace.</desc>
<defs>
  <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#222222"/></marker>
  <marker id="gray-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#888888"/></marker>
  <style>
    text { font-family: "DejaVu Sans", Arial, sans-serif; fill: #181818; }
    .panel { font-size: 20px; font-weight: 700; }
    .heading { font-size: 18px; font-weight: 700; }
    .body { font-size: 16px; }
    .small { font-size: 15.5px; fill: #444444; }
    .tiny { font-size: 14.5px; fill: #666666; }
    .mono { font-family: "DejaVu Sans Mono", monospace; font-size: 14.5px; }
    .input { fill: #f2f2f2; stroke: #777777; stroke-width: 1.2; }
    .artifact { fill: #ffffff; stroke: #333333; stroke-width: 1.25; }
    .learned { fill: #edf4fb; stroke: #2f6fa3; stroke-width: 1.4; }
    .deterministic { fill: #eef8f6; stroke: #177e75; stroke-width: 1.4; }
    .bypassed { fill: #fafafa; stroke: #888888; stroke-width: 1.1; stroke-dasharray: 6 5; }
    .check { fill: #ffffff; stroke: #2e7d32; stroke-width: 1.4; }
    .warning { fill: #fff7df; stroke: #b07818; stroke-width: 1.2; }
    .rule { stroke: #666666; stroke-width: 1.3; stroke-dasharray: 7 6; }
    .grid { stroke: #777777; stroke-width: 0.8; }
    .arrow { fill: none; stroke: #222222; stroke-width: 1.8; }
    .branch { fill: none; stroke: #888888; stroke-width: 1.25; stroke-dasharray: 6 5; }
    .blue { fill: #2f6fa3; font-weight: 700; }
    .teal { fill: #177e75; font-weight: 700; }
    .green { fill: #2e7d32; font-weight: 700; }
    .amber { fill: #9a6410; font-weight: 700; }
    .gray { fill: #777777; }
  </style>
</defs>
<rect width="1040" height="650" fill="#ffffff"/>
'''


def build_wireframe() -> str:
    """First design pass: geometry and reading order only."""
    out = [SVG_HEAD]
    out += [
        rect(14, 18, 232, 516, "input", 7),
        text(29, 48, "Input and bridge", "heading"),
        rect(29, 67, 202, 145, "artifact"),
        lines(43, 96, ("[exact bridge question]", "[source · bridge · target]"), "body", 27),
        rect(29, 242, 202, 176, "artifact"),
        text(43, 270, "[relation sketch]", "small"),
        text(43, 394, "[shared record schema]", "small"),
        text(266, 38, "(a) Question → executable graph", "panel"),
        rect(273, 61, 205, 128, "learned"),
        text(288, 91, "[structured understanding]", "heading"),
        lines(288, 119, ("[A source]", "[B bridge]", "[C target]", "[D return]"), "small", 20),
        arrow(478, 125, 522, 125),
        rect(526, 91, 82, 66, "artifact"),
        text(567, 116, "[gate]", "small", "middle"),
        text(567, 141, "[pass]", "small", "middle"),
        arrow(608, 109, 650, 109),
        rect(654, 72, 177, 73, "deterministic"),
        text(742, 102, "[taken compiler]", "small", "middle"),
        rect(654, 160, 177, 55, "bypassed"),
        text(742, 193, "[conditional planner]", "small", "middle"),
        arrow(831, 109, 864, 109),
        rect(868, 62, 158, 144, "artifact"),
        text(947, 92, "[typed DAG]", "heading", "middle"),
        lines(884, 125, ("A → B → C", "      ↓", "   return"), "mono", 24),
        '<line x1="263" y1="235" x2="1027" y2="235" class="rule"/>',
        text(266, 270, "(b) Grounded bridge execution → release", "panel"),
        rect(273, 294, 172, 194, "artifact"),
        rect(477, 294, 142, 194, "artifact"),
        rect(651, 294, 172, 194, "artifact"),
        rect(855, 294, 171, 194, "check"),
        text(359, 326, "[source records]", "heading", "middle"),
        text(548, 326, "[bridge set]", "heading", "middle"),
        text(737, 326, "[bound records]", "heading", "middle"),
        text(941, 326, "[verified result]", "heading", "middle"),
        lines(290, 359, ("[constraint]", "[mini rows]", "[N₀ records]"), "small", 32),
        lines(493, 359, ("[emit field]", "[values]", "[N₁ values]"), "small", 32),
        lines(667, 359, ("[IN binding]", "[mini rows]", "[N₂ records]"), "small", 32),
        lines(871, 359, ("[return]", "[checks]", "[caveat]"), "small", 32),
        arrow(445, 390, 477, 390),
        arrow(619, 390, 651, 390),
        arrow(823, 390, 855, 390),
        "</svg>",
    ]
    return "".join(out)


def network_glyph(cx: float, cy: float) -> list[str]:
    nodes = ((cx - 20, cy), (cx, cy - 18), (cx, cy + 18), (cx + 20, cy))
    out = []
    for left, right in ((0, 1), (0, 2), (1, 3), (2, 3), (1, 2)):
        x1, y1 = nodes[left]
        x2, y2 = nodes[right]
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#2f6fa3" stroke-width="1.2"/>')
    for x, y in nodes:
        out.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="#ffffff" stroke="#2f6fa3" stroke-width="1.3"/>')
    return out


def build_filled(data: dict[str, Any]) -> str:
    coverage = data["coverage"]
    sample_size = int(coverage.get("coverage_sample_size") or 0)
    without_supplier = int(coverage.get("without_supplier") or 0)
    evidence_count = int(data["evidence_verdict"]["kg_support"][0]["evidence_count"])
    variable_sizes = {
        str(row.get("var_id")): int(row.get("output_size") or 0)
        for row in data["variable_traces"]
    }
    source_size = variable_sizes["A"]
    bridge_size = variable_sizes["B"]
    target_size = variable_sizes["C"]

    out = [SVG_HEAD]
    # Eligibility statement separates the saved artifact from replayed states.
    out += [
        rect(18, 16, 1004, 62, "deterministic", 5),
        text(34, 43, "Figure eligibility", "heading"),
        text(205, 40, "Saved Step 1 artifact reconstructed by deterministic replay", "small"),
        text(205, 63, "No model call. This is not a saved final trace.", "tiny"),
        text(18, 108, "(a) Question and executable graph", "panel"),
        rect(18, 126, 200, 170, "input"),
        text(34, 153, "Input question", "heading"),
        lines(
            34,
            179,
            (
                "Record check",
                "How many contract",
                "notices were published",
                "by buyers that",
                "awarded a contract",
                "to Dodd Group",
                "(Midlands) Limited?",
            ),
            "tiny",
            18,
        ),
        arrow(218, 211, 246, 211),
        rect(250, 126, 250, 170, "learned"),
        text(266, 153, "Saved Step 1 artifact", "heading"),
        text(266, 177, "Typed intent", "small"),
        lines(
            266,
            203,
            (
                "A  supplier is Dodd Group",
                "B  distinct awarding buyers",
                "C  notices from buyers in B",
                "D  count C",
            ),
            "mono",
            22,
        ),
        arrow(500, 169, 544, 169),
        rect(548, 126, 180, 86, "deterministic"),
        text(638, 151, "Deterministic", "heading", "middle"),
        text(638, 175, "intent compiler", "small", "middle"),
        text(638, 198, "valid and no veto", "tiny", "middle"),
        '<path d="M500,252 L522,252 L522,264 L544,264" class="branch" marker-end="url(#gray-arrow)"/>',
        rect(548, 232, 180, 64, "bypassed"),
        text(638, 257, "Step 2 planner", "heading", "middle"),
        text(638, 280, "not used here", "tiny", "middle"),
        arrow(728, 169, 772, 169),
        rect(776, 126, 246, 170, "artifact"),
        text(899, 153, "Typed DAG", "heading", "middle"),
        lines(
            899,
            178,
            ("A  source records", "↓", "B  distinct buyers", "↓", "C  bound records", "↓", "D  count C"),
            "mono",
            17,
            "middle",
        ),
        '<line x1="18" y1="320" x2="1022" y2="320" class="rule"/>',
    ]

    # Lower path contains states produced by the deterministic replay.
    out += [
        text(18, 352, "(b) Deterministic bridge execution", "panel"),
        rect(18, 375, 220, 240, "artifact"),
        text(34, 403, "A  source records", "heading"),
        lines(34, 431, ("supplier is Dodd Group", "(Midlands) Limited"), "mono", 20),
        '<line x1="34" y1="468" x2="222" y2="468" class="grid"/>',
        text(34, 493, "Dudley MBC", "small"),
        text(34, 517, "Solihull MBC", "small"),
        text(34, 541, "Fusion21", "small"),
        text(222, 597, f"{source_size:,} records", "teal", "end"),
        arrow(238, 493, 266, 493),
        rect(270, 375, 200, 240, "artifact"),
        text(286, 403, "B  buyer values", "heading"),
        text(286, 431, "emit distinct buyers", "tiny"),
        '<line x1="286" y1="450" x2="454" y2="450" class="grid"/>',
        text(286, 477, "Dudley MBC", "small"),
        text(286, 501, "Solihull MBC", "small"),
        text(286, 525, "Education", "small"),
        text(286, 549, "and 13 more", "small"),
        text(454, 597, f"{bridge_size:,} values", "teal", "end"),
        arrow(470, 493, 498, 493),
        rect(502, 375, 220, 240, "artifact"),
        text(518, 403, "C  bound records", "heading"),
        text(518, 431, "buyer name is in B", "tiny"),
        '<line x1="518" y1="450" x2="706" y2="450" class="grid"/>',
        text(518, 477, "DfE", "small"),
        text(518, 501, "Sandwell", "small"),
        text(518, 525, "Fusion21", "small"),
        text(518, 549, "13 other buyers", "small"),
        text(706, 597, f"{target_size:,} records", "teal", "end"),
        arrow(722, 493, 750, 493),
        rect(754, 375, 268, 240, "check"),
        text(888, 403, "Released result", "heading", "middle"),
        text(888, 435, f"COUNT(C) = {target_size:,}", "teal", "middle"),
        text(770, 470, "graph execution passed", "green"),
        text(770, 494, f"matched records {evidence_count:,}", "green"),
        text(770, 518, "answer sanity passed", "green"),
        rect(770, 536, 236, 64, "warning"),
        lines(
            888,
            554,
            ("Coverage note", f"{without_supplier} / {sample_size} sampled rows", "lack supplier data"),
            "amber",
            19,
            "middle",
        ),
        "</svg>",
    ]
    return "".join(out)


def manifest(data: dict[str, Any]) -> dict[str, Any]:
    coverage = data["coverage"]
    return {
        "figure": "fig01_runtime_bridge",
        "example_id": ITEM_ID,
        "question": data["question"],
        "branch_taken": "saved Step-1 understanding -> deterministic intent compiler",
        "branch_not_taken": "Step-2 typed graph planner",
        "replay": "local deterministic replay; no model call",
        "intermediate_cardinalities": {
            row["var_id"]: row["output_size"] for row in data["variable_traces"]
        },
        "answer": data["answer"],
        "runtime_checks": {
            "graph_execution": "passed",
            "evidence_verdict": data["evidence_verdict"].get("status"),
            "evidence_count": data["evidence_verdict"]["kg_support"][0].get("evidence_count"),
            "document_status": data["evidence_verdict"].get("document_status"),
            "answer_sanity_ok": data["answer_sanity"].get("ok"),
            "population_coverage_check_passed": coverage.get("passed"),
            "coverage_sample_size": coverage.get("coverage_sample_size"),
            "coverage_sample_without_supplier": coverage.get("without_supplier"),
        },
        "offline_audit": {
            "oracle_match": data["compact_trace"].get("oracle_match"),
            "repair_attempts": data["compact_trace"].get("n_repair_attempts"),
        },
        "limitations": data["limitations"],
        "sources": data["sources"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("wireframe", "filled", "both"), default="both")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"wireframe", "both"}:
        wireframe = args.out_dir / "fig01_runtime_bridge_wireframe.svg"
        wireframe.write_text(build_wireframe(), encoding="utf-8")
        print(wireframe)

    if args.stage in {"filled", "both"}:
        data = load_and_replay()
        filled = args.out_dir / "fig01_runtime_bridge.svg"
        filled.write_text(build_filled(data), encoding="utf-8")
        filled.with_suffix(".manifest.json").write_text(
            json.dumps(manifest(data), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(filled)


if __name__ == "__main__":
    main()
