"""
Pipeline step 50: local deterministic QA benchmark Stage 1.

No LLM/API calls are made here.

Writes:
    data/qa/generated/answer_specs.jsonl
    data/qa/generated/gate_a_report.jsonl
    reports/qa/stage1_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_graph.qa.benchmark.samplers import SamplerConfig
from procurement_graph.qa.benchmark.stage1 import build_stage1


ROOT = Path(__file__).resolve().parents[1]


def main(args: argparse.Namespace) -> int:
    summary = build_stage1(
        kg_dir=ROOT / "data" / "kg",
        output_dir=ROOT / "data" / "qa" / "generated",
        report_dir=ROOT / "reports" / "qa",
        config=SamplerConfig(
            seed=args.seed,
            target_specs=args.target_specs,
            min_evidence_rows=args.min_evidence_rows,
            min_aggregation_evidence_rows=args.min_aggregation_evidence_rows,
            min_sum_evidence_rows=args.min_sum_evidence_rows,
            min_conjunction_evidence_rows=args.min_conjunction_evidence_rows,
            min_temporal_evidence_rows=args.min_temporal_evidence_rows,
            min_cpv_evidence_rows=args.min_cpv_evidence_rows,
            max_evidence_rows=args.max_evidence_rows,
        ),
        exclude_sum_anomalies=args.exclude_sum_anomalies,
        progress=args.progress,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build local QA Stage 1 answer specs")
    parser.add_argument("--target-specs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-evidence-rows", type=int, default=1)
    parser.add_argument("--min-aggregation-evidence-rows", type=int, default=3)
    parser.add_argument("--min-sum-evidence-rows", type=int, default=3)
    parser.add_argument("--min-conjunction-evidence-rows", type=int, default=3)
    parser.add_argument("--min-temporal-evidence-rows", type=int, default=3)
    parser.add_argument("--min-cpv-evidence-rows", type=int, default=3)
    parser.add_argument("--max-evidence-rows", type=int, default=5000)
    parser.add_argument("--progress", action="store_true", help="Print Stage 1 progress while building specs.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Progress print interval in specs.")
    parser.add_argument(
        "--exclude-sum-anomalies",
        action="store_true",
        help="Reject flagged aggregation_sum specs (FAIL) instead of keeping them with a WARN.",
    )
    args = parser.parse_args()
    raise SystemExit(main(args))
