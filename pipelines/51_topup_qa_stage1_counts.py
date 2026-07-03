"""
Pipeline step 51: append relaxed aggregation-count Stage 1 specs.

Use this after pipeline 50 has completed but undershot the requested total because
the stricter v0.2 count templates ran out of capacity. This script does not rebuild
or overwrite existing specs; it only appends verified top-up specs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_graph.qa.benchmark.constraints import detect_constraint_conflicts, normalize_constraints
from procurement_graph.qa.benchmark.executor import AnswerExecutionError, execute_answer_spec_rows
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend
from procurement_graph.qa.benchmark.models import AnswerSpec, Constraint, GateReport
from procurement_graph.qa.benchmark.reference_index import ReferenceKGIndex
from procurement_graph.qa.benchmark.samplers import (
    SamplerConfig,
    _combined_groups,
    _constraints_from_group,
    _domain_slice,
    _logic_from_group,
    _make_spec,
    _metadata,
    _rng,
    _sample_frame,
)
from procurement_graph.qa.benchmark.serialization import gate_report_to_dict, json_default, spec_to_dict
from procurement_graph.qa.benchmark.stage1 import _SAMPLE_COLUMNS, _completeness_gate, _csv_value, _sample_row


ROOT = Path(__file__).resolve().parents[1]

# value_source removed from question-facing filters. With the v0.3 sampler the base
# aggregation_count family already fills from release_year+tender_cpv_id, so this top-up is
# usually a no-op; the templates below stay value_source-free and rely on the constraint
# signature dedupe to append only genuinely new (natural) count combinations.
RELAXED_COUNT_TEMPLATES = [
    ["release_year", "tender_cpv_id"],
    ["tender_category", "tender_cpv_id"],
    ["release_year", "tender_category", "tender_cpv_id"],
]


def main(args: argparse.Namespace) -> int:
    generated_dir = ROOT / "data" / "qa" / "generated"
    report_dir = ROOT / "reports" / "qa"
    answer_specs_path = generated_dir / "answer_specs.jsonl"
    gate_path = generated_dir / "gate_a_report.jsonl"
    sample_csv_path = report_dir / "stage1_answer_spec_sample.csv"
    summary_path = report_dir / "stage1_summary.json"

    existing = _load_existing(answer_specs_path)
    existing_count = len(existing["spec_ids"])
    needed = max(0, args.target_total - existing_count)
    if args.max_new is not None:
        needed = min(needed, args.max_new)

    _progress(args.progress, f"existing accepted specs={existing_count:,}; target={args.target_total:,}; needed={needed:,}")
    if needed <= 0:
        print(json.dumps({"existing": existing_count, "needed": 0, "appended": 0}, indent=2))
        return 0

    backend = ParquetKGQueryBackend.from_directory(ROOT / "data" / "kg", include_evidence=False)
    reference = ReferenceKGIndex.from_directory(ROOT / "data" / "kg")
    config = SamplerConfig(
        seed=args.seed,
        target_specs=args.target_total,
        min_evidence_rows=args.min_evidence_rows,
        min_aggregation_evidence_rows=args.min_evidence_rows,
        max_evidence_rows=args.max_evidence_rows,
    )
    rng = _rng(args.seed)

    _progress(args.progress, "building relaxed aggregation_count candidate groups")
    groups = _combined_groups(
        backend.records_df,
        RELAXED_COUNT_TEMPLATES,
        config,
        min_rows=args.min_evidence_rows,
    )
    groups = _sample_frame(groups, len(groups), rng)
    _progress(args.progress, f"candidate groups={len(groups):,}")

    appended = 0
    rejected = 0
    duplicate_skipped = 0
    attempted = 0
    gate_fail_counter: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []

    with answer_specs_path.open("a", encoding="utf-8") as accepted_file, gate_path.open(
        "a", encoding="utf-8"
    ) as gate_file:
        for idx, row in enumerate(groups.itertuples(index=False), start=1):
            if appended >= needed:
                break
            constraints = normalize_constraints(_constraints_from_group(row))
            signature = _constraint_signature(constraints)
            if signature in existing["constraint_signatures"]:
                duplicate_skipped += 1
                continue

            spec = _make_spec(
                backend,
                spec_id=f"count_relaxed_topup_{idx:05d}",
                constraints=constraints,
                answer_operation="count",
                answer_field="contract_node_id",
                answer_value_type="integer",
                dedupe_key="contract_node_id",
                logic_chain=(*_logic_from_group(row), "count contracts"),
                metadata=_metadata(
                    "aggregation_count",
                    row,
                    operation_family="filtered_count_relaxed",
                    domain_slice=_domain_slice(row),
                    evidence_floor=args.min_evidence_rows,
                )
                | {
                    "sampler_version": "stage1_v0.2_topup_counts",
                    "topup_reason": "strict aggregation_count capacity below target",
                },
            )
            attempted += 1
            gate_record, accepted_record, sample_row, failed_gate = _validate_topup_spec(backend, reference, spec)
            gate_file.write(json.dumps(gate_record, default=json_default) + "\n")

            if accepted_record is None:
                rejected += 1
                if failed_gate:
                    gate_fail_counter[failed_gate] += 1
                continue

            accepted_file.write(json.dumps(accepted_record, default=json_default) + "\n")
            sample_rows.append(sample_row)
            existing["spec_ids"].add(spec.spec_id)
            existing["constraint_signatures"].add(signature)
            appended += 1

            if args.progress and (appended == 1 or appended % args.progress_every == 0 or appended == needed):
                _progress(
                    True,
                    f"appended {appended:,}/{needed:,} "
                    f"(attempted={attempted:,}, rejected={rejected:,}, duplicate_skipped={duplicate_skipped:,})",
                )

    _append_sample_csv(sample_csv_path, sample_rows)
    _update_summary(summary_path, appended, rejected, attempted, duplicate_skipped, gate_fail_counter, args)

    result = {
        "existing_before": existing_count,
        "target_total": args.target_total,
        "needed": needed,
        "attempted": attempted,
        "appended": appended,
        "rejected": rejected,
        "duplicate_skipped": duplicate_skipped,
        "answer_specs": str(answer_specs_path),
        "gate_a_report": str(gate_path),
        "sample_csv": str(sample_csv_path),
        "summary": str(summary_path),
    }
    print(json.dumps(result, indent=2, default=json_default))
    return 0


def _load_existing(path: Path) -> dict[str, set[Any]]:
    spec_ids: set[str] = set()
    signatures: set[tuple[tuple[str, str, str], ...]] = set()
    if not path.exists():
        return {"spec_ids": spec_ids, "constraint_signatures": signatures}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            spec = record.get("spec", {})
            spec_id = str(spec.get("spec_id", ""))
            if spec_id:
                spec_ids.add(spec_id)
            constraints = tuple(
                Constraint(item.get("field", ""), item.get("op", "eq"), item.get("value"))
                for item in spec.get("constraints", [])
            )
            if constraints:
                signatures.add(_constraint_signature(constraints))
    return {"spec_ids": spec_ids, "constraint_signatures": signatures}


def _validate_topup_spec(
    backend: ParquetKGQueryBackend,
    reference: ReferenceKGIndex,
    spec: AnswerSpec,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, str]:
    conflicts = detect_constraint_conflicts(spec.constraints)
    constraint_gate = GateReport(
        gate="gate_a_constraints",
        passed=not conflicts,
        status="PASS" if not conflicts else "FAIL",
        reason="no contradictory constraints" if not conflicts else "; ".join(conflicts),
        metrics={"constraint_count": len(spec.constraints), "conflicts": conflicts},
    )

    full_rows = backend.query(spec.constraints)
    full_id_list = [backend.record_id(row) for row in full_rows if backend.record_id(row)]
    full_ids = set(full_id_list)
    duplicate_rows = len(full_id_list) - len(full_ids)
    sampled_ids = set(spec.sampled_evidence_ids)
    completeness = _completeness_gate(reference, spec.constraints, sampled_ids, full_ids, duplicate_rows)

    golden_answer: Any = None
    execution_error = ""
    try:
        golden_answer = execute_answer_spec_rows(full_rows, spec)
    except AnswerExecutionError as exc:
        uniqueness = GateReport(
            gate="gate_a_uniqueness",
            passed=False,
            status="FAIL",
            reason=str(exc),
            metrics={"answer_operation": spec.answer_operation, "answer_field": spec.answer_field},
        )
        execution_error = str(exc)
    else:
        uniqueness = GateReport(
            gate="gate_a_uniqueness",
            passed=True,
            status="PASS",
            reason="answer_spec produced one deterministic golden answer",
            metrics={"answer": str(golden_answer), "answer_field": spec.answer_field},
        )

    gate_reports = [constraint_gate, completeness, uniqueness]
    statuses = [report.effective_status for report in gate_reports]
    worst = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
    gate_passed = "FAIL" not in statuses
    failed_gate = next((report.gate for report in gate_reports if report.effective_status == "FAIL"), "")
    gate_record = {
        "spec_id": spec.spec_id,
        "accepted": gate_passed,
        "status": worst,
        "question_type": spec.metadata.get("question_type", "aggregation_count"),
        "gate_reports": [gate_report_to_dict(report) for report in gate_reports],
        "execution_error": execution_error,
        "evidence_count": len(spec.sampled_evidence_ids),
    }
    if not gate_passed:
        return gate_record, None, None, failed_gate
    accepted_record = {
        "spec": spec_to_dict(spec),
        "golden_answer": golden_answer,
        "value_sanity_status": "PASS",
        "stage": "stage1_gate_a_passed_topup",
    }
    return gate_record, accepted_record, _sample_row(spec, golden_answer, "PASS"), ""


def _append_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SAMPLE_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in _SAMPLE_COLUMNS})


def _update_summary(
    path: Path,
    appended: int,
    rejected: int,
    attempted: int,
    duplicate_skipped: int,
    gate_fail_counter: Counter[str],
    args: argparse.Namespace,
) -> None:
    summary: dict[str, Any] = {}
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
    summary["attempted"] = int(summary.get("attempted", 0)) + attempted
    summary["accepted"] = int(summary.get("accepted", 0)) + appended
    summary["rejected"] = int(summary.get("rejected", 0)) + rejected
    summary["target_specs"] = args.target_total
    for key in ("type_counts_attempted", "type_counts_accepted"):
        counts = dict(summary.get(key, {}))
        increment = attempted if key == "type_counts_attempted" else appended
        counts["aggregation_count"] = int(counts.get("aggregation_count", 0)) + increment
        summary[key] = dict(sorted(counts.items()))
    status_rollup = dict(summary.get("status_rollup", {}))
    status_rollup["PASS"] = int(status_rollup.get("PASS", 0)) + appended
    status_rollup["FAIL"] = int(status_rollup.get("FAIL", 0)) + rejected
    summary["status_rollup"] = status_rollup
    gate_a = dict(summary.get("gate_a", {}))
    gate_a["topup_count_attempted"] = attempted
    gate_a["topup_count_appended"] = appended
    gate_a["topup_count_rejected"] = rejected
    gate_a["topup_duplicate_skipped"] = duplicate_skipped
    gate_a["topup_constraint_conflict_fail"] = gate_fail_counter.get("gate_a_constraints", 0)
    gate_a["topup_completeness_fail"] = gate_fail_counter.get("gate_a_completeness", 0)
    gate_a["topup_uniqueness_fail"] = gate_fail_counter.get("gate_a_uniqueness", 0)
    summary["gate_a"] = gate_a
    summary["stage1_topup"] = {
        "mode": "relaxed_aggregation_count",
        "templates": ["|".join(template) for template in RELAXED_COUNT_TEMPLATES],
        "min_evidence_rows": args.min_evidence_rows,
        "max_evidence_rows": args.max_evidence_rows,
    }
    path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")


def _constraint_signature(constraints: tuple[Constraint, ...]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((c.field, c.op, json.dumps(c.value, sort_keys=True, default=json_default)) for c in constraints))


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[stage1-topup] {message}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top up Stage 1 with relaxed aggregation_count specs")
    parser.add_argument("--target-total", type=int, default=10000)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--min-evidence-rows", type=int, default=3)
    parser.add_argument("--max-evidence-rows", type=int, default=5000)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250)
    raise SystemExit(main(parser.parse_args()))
