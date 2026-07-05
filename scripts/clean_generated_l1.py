"""Clean the original generated L1 QA set for plan-supervision use.

This is intentionally narrower than a full oracle revalidation pass. Stage 1 already executed the
gold specs; this cleaner removes L1 rows whose question cannot express the hidden constraints used
to compute the answer, then joins each surviving question back to the full gold spec.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data" / "qa" / "generated" / "benchmark.jsonl"
DEFAULT_SPECS = ROOT / "data" / "qa" / "generated" / "answer_specs.jsonl"
DEFAULT_OUT = ROOT / "data" / "qa" / "generated_clean_l1"

DROP_HIDDEN_FIELDS = {"supplier_count", "buyer_count"}
DROP_OPERATION_FAMILIES = {"conjunction"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def constraint_fields(row_or_spec: dict[str, Any]) -> set[str]:
    return {str(item.get("field", "")) for item in row_or_spec.get("constraints", []) if item.get("field")}


def load_specs(path: Path) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        spec = row.get("spec") or {}
        spec_id = str(spec.get("spec_id", ""))
        if spec_id:
            specs[spec_id] = row
    return specs


def drop_reasons(row: dict[str, Any], spec: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    row_fields = constraint_fields(row)
    spec_fields = constraint_fields(spec or {})
    hidden = sorted((row_fields | spec_fields) & DROP_HIDDEN_FIELDS)
    if hidden:
        reasons.append("hidden_count_presence_filter:" + ",".join(hidden))
    family = str(row.get("operation_family", ""))
    if family in DROP_OPERATION_FAMILIES:
        reasons.append(f"operation_family:{family}")
    if spec is None:
        reasons.append("missing_stage1_gold_spec")
    return reasons


def clean_row(row: dict[str, Any], spec_row: dict[str, Any]) -> dict[str, Any]:
    spec = spec_row["spec"]
    merged = dict(row)
    merged["gold_plan"] = spec
    merged["source_stage1"] = {
        "stage": spec_row.get("stage"),
        "value_sanity_status": spec_row.get("value_sanity_status"),
    }
    merged["cleaning"] = {
        "version": "generated_l1_clean_v1",
        "status": "clean",
        "rule": "drop supplier_count/buyer_count hidden presence filters",
    }
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--answer-specs", type=Path, default=DEFAULT_SPECS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    benchmark_rows = read_jsonl(args.benchmark)
    specs_by_id = load_specs(args.answer_specs)

    clean: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in benchmark_rows:
        spec_id = str(row.get("spec_id", ""))
        spec_row = specs_by_id.get(spec_id)
        spec = spec_row.get("spec") if spec_row else None
        reasons = drop_reasons(row, spec)
        if reasons:
            dropped.append({
                "spec_id": spec_id,
                "question": row.get("question"),
                "answer_operation": row.get("answer_operation"),
                "operation_family": row.get("operation_family"),
                "drop_reasons": reasons,
                "constraints": row.get("constraints", []),
                "gold_plan": spec,
            })
            continue
        clean.append(clean_row(row, spec_row))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "clean.jsonl", clean)
    write_jsonl(args.out_dir / "dropped.jsonl", dropped)

    summary = {
        "source_benchmark": str(args.benchmark),
        "source_answer_specs": str(args.answer_specs),
        "total_benchmark_rows": len(benchmark_rows),
        "clean_rows": len(clean),
        "dropped_rows": len(dropped),
        "drop_reasons": dict(Counter(reason for row in dropped for reason in row["drop_reasons"])),
        "clean_by_operation_family": dict(Counter(str(row.get("operation_family", "")) for row in clean)),
        "dropped_by_operation_family": dict(Counter(str(row.get("operation_family", "")) for row in dropped)),
        "clean_by_answer_operation": dict(Counter(str(row.get("answer_operation", "")) for row in clean)),
        "outputs": {
            "clean": str(args.out_dir / "clean.jsonl"),
            "dropped": str(args.out_dir / "dropped.jsonl"),
            "summary": str(args.out_dir / "summary.json"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
