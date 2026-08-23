#!/usr/bin/env python3
"""Generate the TMLR v3 method and data figures as self-contained SVG.

The figures are intentionally editorial rather than slide-like: white background,
square cards, few semantic colours, and real implementation objects.  No model is
called.  Figure 3 reads Parquet metadata so displayed graph sizes cannot silently
drift from the local snapshot.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "paper" / "tmlr_v3" / "figs" / "src"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: str, cls: str = "body", anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{esc(value)}</text>'


def lines(
    x: float,
    y: float,
    values: tuple[str, ...] | list[str],
    cls: str = "body",
    leading: float = 21,
    anchor: str = "start",
) -> str:
    spans = []
    for index, value in enumerate(values):
        spans.append(f'<tspan x="{x}" dy="{0 if index == 0 else leading}">{esc(value)}</tspan>')
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{"".join(spans)}</text>'


def rect(x: float, y: float, w: float, h: float, cls: str = "box", rx: float = 0) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'


def arrow(x1: float, y1: float, x2: float, y2: float, cls: str = "arrow") -> str:
    marker = "gray-arrow" if cls == "feedback" else "arrow"
    return f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}" marker-end="url(#{marker})"/>'


def path(points: list[tuple[float, float]], cls: str = "arrow") -> str:
    marker = "gray-arrow" if cls == "feedback" else "arrow"
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    return f'<path d="{d}" class="{cls}" marker-end="url(#{marker})"/>'


def svg_head(width: int, height: int, inches_high: float, title_value: str, desc: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="6.5in" height="{inches_high}in" viewBox="0 0 {width} {height}" role="img" aria-labelledby="svg-title svg-desc">
<title id="svg-title">{esc(title_value)}</title>
<desc id="svg-desc">{esc(desc)}</desc>
<defs>
  <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#222222"/></marker>
  <marker id="gray-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#777777"/></marker>
  <style>
    text {{ font-family: "DejaVu Sans", Arial, sans-serif; fill: #171717; }}
    .panel {{ font-size: 20px; font-weight: 700; }}
    .heading {{ font-size: 18px; font-weight: 700; }}
    .body {{ font-size: 17px; }}
    .small {{ font-size: 16px; fill: #444444; }}
    .tiny {{ font-size: 15px; fill: #666666; }}
    .mono {{ font-family: "DejaVu Sans Mono", monospace; font-size: 16px; }}
    .box {{ fill: #ffffff; stroke: #333333; stroke-width: 1.25; }}
    .learned {{ fill: #edf4fb; stroke: #2f6fa3; stroke-width: 1.4; }}
    .det {{ fill: #eef8f6; stroke: #177e75; stroke-width: 1.4; }}
    .gate {{ fill: #fff8df; stroke: #9a6410; stroke-width: 1.25; }}
    .offline {{ fill: #fff9e8; stroke: #b07818; stroke-width: 1.3; }}
    .negative {{ fill: #fff4f3; stroke: #b64a42; stroke-width: 1.2; }}
    .muted {{ fill: #fafafa; stroke: #888888; stroke-width: 1.1; stroke-dasharray: 6 5; }}
    .arrow {{ fill: none; stroke: #222222; stroke-width: 1.7; }}
    .feedback {{ fill: none; stroke: #777777; stroke-width: 1.25; stroke-dasharray: 6 5; }}
    .rule {{ stroke: #777777; stroke-width: 1.0; }}
    .grid {{ stroke: #aaaaaa; stroke-width: 0.8; }}
    .blue {{ fill: #2f6fa3; font-weight: 700; }}
    .teal {{ fill: #177e75; font-weight: 700; }}
    .green {{ fill: #2e7d32; font-weight: 700; }}
    .amber {{ fill: #8a5b10; font-weight: 700; }}
    .red {{ fill: #a63d36; font-weight: 700; }}
  </style>
</defs>
<rect width="{width}" height="{height}" fill="#ffffff"/>
'''


def build_online_offline() -> str:
    out = [svg_head(
        1040,
        650,
        4.0625,
        "Online typed procurement QA and offline supervision construction",
        "The upper path shows online typed planning and deterministic execution without an oracle. The lower path shows a hidden oracle applying routing gates before candidates contribute to supervision views that may overlap.",
    )]

    out += [text(18, 28, "(a) Online inference without an oracle", "panel")]
    # Main online path.
    stages = [
        (18, 54, 82, 76, "box", "Input", ("question", "schema")),
        (122, 48, 120, 88, "learned", "Step 1", ("typed intent", "answer type")),
        (476, 54, 105, 76, "box", "DAG", ("variables", "dependencies")),
        (613, 54, 112, 76, "det", "Ground", ("roles, guards", "preflight")),
        (757, 54, 112, 76, "det", "Execute", ("complete sets", "exact reduce")),
        (901, 48, 120, 88, "box", "Release", ("evidence", "final checks")),
    ]
    for x, y, w, h, cls, heading, body in stages:
        out += [rect(x, y, w, h, cls), text(x + 9, y + 25, heading, "heading"), lines(x + 9, y + 48, body, "tiny", 19)]
    out += [
        arrow(100, 92, 122, 92),
        arrow(242, 92, 250, 92),
        '<polygon points="280,62 310,92 280,122 250,92" class="gate"/>',
        lines(280, 86, ("valid?", "no veto?"), "tiny", 18, "middle"),
        arrow(310, 76, 338, 76),
        rect(340, 48, 104, 56, "det"),
        lines(392, 70, ("intent", "compiler"), "tiny", 18, "middle"),
        arrow(444, 76, 476, 76),
        path([(280, 122), (280, 181), (338, 181)], "feedback"),
        rect(340, 150, 104, 62, "learned"),
        lines(392, 175, ("Step 2", "graph plan"), "tiny", 18, "middle"),
        path([(444, 181), (460, 181), (460, 116), (476, 116)]),
        text(319, 68, "yes", "tiny"),
        text(288, 143, "no or veto", "tiny"),
        arrow(581, 92, 613, 92),
        arrow(725, 92, 757, 92),
        arrow(869, 92, 901, 92),
        arrow(961, 136, 961, 153),
        text(1018, 174, "Answer with provenance", "green", "end"),
        text(1018, 195, "or no answer", "amber", "end"),
        path([(961, 205), (961, 238), (392, 238), (392, 212)], "feedback"),
        text(690, 232, "One structured failure may prompt one revised plan", "tiny", "middle"),
        '<line x1="18" y1="266" x2="1022" y2="266" class="rule"/>',
    ]

    out += [text(18, 298, "(b) Offline curation with a hidden oracle", "panel")]
    out += [
        rect(18, 326, 138, 168, "box"),
        lines(87, 351, ("Candidate", "trace"), "heading", 20, "middle"),
        lines(31, 399, ("typed plan", "grounded data", "runtime status", "failure feedback"), "tiny", 24),
        arrow(156, 410, 178, 410),
        rect(180, 326, 228, 168, "offline"),
        text(294, 354, "Routing gates", "heading", "middle"),
        lines(196, 377, ("runtime status recorded", "runtime checks applied", "expected status check", "oracle on answerable items", "shape on answerable items", "exportable graph JSON"), "tiny", 19),
        arrow(408, 410, 438, 410),
        rect(442, 316, 580, 200, "muted"),
        text(732, 340, "Overlapping supervision views after the gates", "heading", "middle"),
        text(732, 360, "Views can overlap.", "tiny", "middle"),
        rect(458, 369, 170, 59, "learned"),
        text(543, 392, "Direct plan", "heading", "middle"),
        text(543, 414, "5,598 before caps", "tiny", "middle"),
        rect(646, 369, 170, 59, "learned"),
        text(731, 392, "Repair target", "heading", "middle"),
        text(731, 414, "1,725 repair targets", "tiny", "middle"),
        rect(834, 369, 172, 59, "learned"),
        text(920, 392, "Preference pair", "heading", "middle"),
        text(920, 414, "390 pairs within repair", "tiny", "middle"),
        rect(458, 442, 170, 65, "negative"),
        text(543, 463, "Hard negative", "heading", "middle"),
        lines(543, 483, ("1,262 candidates", "oracle or shape differs"), "tiny", 17, "middle"),
        rect(646, 442, 170, 65, "box"),
        text(731, 465, "Abstention", "heading", "middle"),
        lines(731, 483, ("590 target pool", "declared no answer"), "tiny", 17, "middle"),
        rect(834, 442, 172, 65, "offline"),
        text(920, 465, "Export rule", "heading", "middle"),
        text(920, 487, "oracle value omitted", "tiny", "middle"),
        text(18, 536, "Teacher harvest", "small"),
        text(190, 535, "9,267", "blue", "middle"),
        text(190, 557, "questions", "tiny", "middle"),
        arrow(258, 546, 304, 546),
        text(398, 535, "6,860", "teal", "middle"),
        text(398, 557, "answer and runtime pass", "tiny", "middle"),
        text(535, 550, "=", "heading", "middle"),
        text(650, 535, "5,598", "green", "middle"),
        text(650, 557, "positive pool", "tiny", "middle"),
        text(765, 550, "+", "heading", "middle"),
        text(900, 535, "1,262", "red", "middle"),
        text(900, 557, "hard negatives", "tiny", "middle"),
        text(520, 585, "5,605 match the oracle. Seven fail the expected shape check and join the hard negatives.", "tiny", "middle"),
        text(18, 628, "The hidden oracle filters candidates but contributes no content to student targets.", "small"),
        "</svg>",
    ]
    return "".join(out)


def parquet_rows(path_value: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path_value).metadata.num_rows)


def graph_stats() -> dict[str, int]:
    kg = ROOT / "data" / "kg"
    return {
        "contracts": parquet_rows(kg / "nodes" / "contract_nodes.parquet"),
        "organisations": parquet_rows(kg / "nodes" / "org_nodes.parquet"),
        "cpv": parquet_rows(kg / "nodes" / "cpv_nodes.parquet"),
        "evidence": parquet_rows(kg / "nodes" / "evidence_nodes.parquet"),
        "buyer_edges": parquet_rows(kg / "edges" / "buyer_of.parquet"),
        "supplier_edges": parquet_rows(kg / "edges" / "supplier_of.parquet"),
        "category_edges": parquet_rows(kg / "edges" / "categorized_by.parquet"),
        "evidence_edges": parquet_rows(kg / "edges" / "evidence_for.parquet"),
    }


def build_data_graph(stats: dict[str, int]) -> str:
    out = [svg_head(
        1040,
        680,
        4.25,
        "Procurement snapshot, organisation resolution and property graph",
        "Three panels show latest release selection, ordered organisation resolution rules, and the procurement property graph with current node and edge counts.",
    )]
    out += [
        text(18, 29, "(a) Snapshot selection", "panel"),
        text(356, 29, "(b) Organisation resolution", "panel"),
        text(710, 29, "(c) Property graph", "panel"),
        '<line x1="338" y1="15" x2="338" y2="660" class="rule"/>',
        '<line x1="690" y1="15" x2="690" y2="660" class="rule"/>',
    ]

    # Snapshot panel.
    out += [
        rect(18, 55, 304, 138, "box"),
        text(34, 82, "Yearly OCDS releases", "heading"),
        text(34, 107, "grouped by the same OCID", "small"),
        text(42, 137, "release r₁", "mono"),
        text(190, 137, "date d₁", "mono"),
        text(42, 163, "release r₂", "mono"),
        text(190, 163, "date d₂", "mono"),
        text(42, 189, "release r₃", "mono"),
        text(306, 189, "maximum date", "green", "end"),
        arrow(170, 193, 170, 220),
        rect(42, 224, 256, 78, "det"),
        text(170, 252, "Latest release per OCID", "heading", "middle"),
        text(170, 280, "exact OCID and release ID", "small", "middle"),
        arrow(170, 302, 170, 329),
        rect(18, 333, 304, 120, "box"),
        text(34, 361, "Retained records", "heading"),
        lines(34, 390, ("tender, lots, awards and bids", "text evidence", "document metadata"), "small", 23),
        arrow(170, 453, 170, 483),
        lines(170, 509, ("Older releases remain", "outside the snapshot"), "tiny", 20, "middle"),
    ]

    # ER panel.
    out += [
        rect(356, 55, 320, 58, "box"),
        text(516, 90, "Organisation observations", "heading", "middle"),
        arrow(516, 113, 516, 136),
        rect(356, 140, 320, 330, "box"),
        text(516, 168, "Ordered resolution rules", "heading", "middle"),
        text(372, 193, "Rule", "tiny"),
        text(660, 193, "Disposition", "tiny", "end"),
        '<line x1="372" y1="202" x2="660" y2="202" class="grid"/>',
        text(372, 226, "1  Official scheme and ID", "tiny"),
        text(660, 226, "canonical", "green", "end"),
        lines(372, 253, ("2  FTS with the same OCID", "   name and role"), "tiny", 18),
        text(660, 262, "unique only", "amber", "end"),
        text(372, 298, "3  Government lookup", "tiny"),
        text(372, 328, "4  Exact name and region", "tiny"),
        lines(372, 355, ("5  Name only", "   when region is absent"), "tiny", 18),
        '<line x1="372" y1="389" x2="660" y2="389" class="grid"/>',
        lines(372, 411, ("Jaro Winkler", "similarity at least 0.92"), "tiny", 18),
        text(660, 420, "review only", "amber", "end"),
        text(372, 454, "Empty or unresolved name", "tiny"),
        text(660, 454, "singleton", "red", "end"),
        arrow(516, 470, 516, 494),
        rect(405, 498, 222, 68, "det"),
        text(516, 524, "Alias map", "heading", "middle"),
        text(516, 549, "with cluster audit", "small", "middle"),
    ]

    # Graph panel.
    out += [
        rect(714, 58, 102, 56, "learned"),
        text(765, 80, "Buyer", "heading", "middle"),
        text(765, 101, "organisation", "tiny", "middle"),
        rect(918, 58, 102, 56, "learned"),
        text(969, 80, "Supplier", "heading", "middle"),
        text(969, 101, "organisation", "tiny", "middle"),
        rect(819, 164, 96, 66, "det"),
        lines(867, 190, ("contract", "award"), "heading", 22, "middle"),
        arrow(765, 114, 838, 164),
        arrow(969, 114, 896, 164),
        text(794, 137, "buyer_of", "tiny", "middle"),
        text(940, 137, "supplier_of", "tiny", "middle"),
        '<polygon points="766,258 798,288 766,318 734,288" fill="#fff8df" stroke="#9a6410" stroke-width="1.3"/>',
        text(766, 294, "CPV", "small", "middle"),
        rect(924, 259, 96, 58, "box"),
        lines(972, 282, ("evidence", "node"), "tiny", 18, "middle"),
        arrow(840, 230, 785, 267),
        arrow(894, 230, 950, 259),
        text(812, 250, "categorized_by", "tiny", "end"),
        text(922, 250, "evidence_for", "tiny"),
        text(714, 349, "Node counts", "heading"),
        text(714, 376, "contracts", "tiny"),
        text(1020, 376, f"{stats['contracts']:,}", "tiny", "end"),
        text(714, 400, "organisations", "tiny"),
        text(1020, 400, f"{stats['organisations']:,}", "tiny", "end"),
        text(714, 424, "CPV", "tiny"),
        text(1020, 424, f"{stats['cpv']:,}", "tiny", "end"),
        text(714, 448, "evidence", "tiny"),
        text(1020, 448, f"{stats['evidence']:,}", "tiny", "end"),
        '<line x1="714" y1="464" x2="1020" y2="464" class="grid"/>',
        text(714, 491, "Edge counts", "heading"),
        text(714, 518, "buyer_of", "tiny"),
        text(1020, 518, f"{stats['buyer_edges']:,}", "tiny", "end"),
        text(714, 542, "supplier_of", "tiny"),
        text(1020, 542, f"{stats['supplier_edges']:,}", "tiny", "end"),
        text(714, 566, "categorized_by", "tiny"),
        text(1020, 566, f"{stats['category_edges']:,}", "tiny", "end"),
        text(714, 590, "evidence_for", "tiny"),
        text(1020, 590, f"{stats['evidence_edges']:,}", "tiny", "end"),
        '<line x1="714" y1="606" x2="1020" y2="606" class="grid"/>',
        text(714, 634, "Parquet backed graph", "teal"),
        text(714, 657, "Source metadata retained", "small"),
        "</svg>",
    ]
    return "".join(out)


def write(path_value: Path, value: str) -> None:
    path_value.parent.mkdir(parents=True, exist_ok=True)
    path_value.write_text(value, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig2 = args.out_dir / "fig02_online_offline.svg"
    write(fig2, build_online_offline())
    fig2.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "figure": "fig02_online_offline",
                "model_calls": 0,
                "teacher_counts": {
                    "input_questions": 9267,
                    "answer_and_runtime_pass": 6860,
                    "oracle_match": 5605,
                    "positive_direct_before_caps": 5598,
                    "repair_target_view": 1725,
                    "preference_pair_view": 390,
                    "hard_negative_pool": 1262,
                    "abstention_target_pool": 590,
                },
                "sources": [
                    "src/procurement_graph/reasoning/typed_planning.py",
                    "src/procurement_graph/reasoning/pipeline.py",
                    "scripts/run_teacher.py",
                    "scripts/export_llamafactory.py",
                    "data/qa/teacher_full_v1",
                ],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    stats = graph_stats()
    fig3 = args.out_dir / "fig03_data_graph.svg"
    write(fig3, build_data_graph(stats))
    fig3.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "figure": "fig03_data_graph",
                "model_calls": 0,
                "graph_stats": stats,
                "sources": [
                    "src/procurement_graph/ingest/loader.py",
                    "src/procurement_graph/er/phase1.py",
                    "src/procurement_graph/er/phase2.py",
                    "scripts/apply_safe_er_merges.py",
                    "src/procurement_graph/kg/build.py",
                    "data/kg/nodes",
                    "data/kg/edges",
                ],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(fig2)
    print(fig3)


if __name__ == "__main__":
    main()
