"""Local deterministic Stage 1 QA benchmark build."""

from __future__ import annotations

import csv
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from .anomaly import SumAnomalyConfig, absolute_sum_flags, distribution_outliers
from .constraints import detect_constraint_conflicts
from .executor import AnswerExecutionError, execute_answer_spec_rows, numeric_row_values
from .kg_interface import ParquetKGQueryBackend
from .models import GateReport
from .reference_index import ReferenceKGIndex, UnsupportedReferenceOp
from .samplers import SamplerConfig, sample_answer_specs
from .serialization import gate_report_to_dict, json_default, spec_to_dict


def build_stage1(
    *,
    kg_dir: Path,
    output_dir: Path,
    report_dir: Path,
    config: SamplerConfig = SamplerConfig(),
    anomaly_config: SumAnomalyConfig | None = None,
    exclude_sum_anomalies: bool = False,
    progress: bool = False,
    progress_every: int = 1000,
) -> dict[str, Any]:
    anomaly_config = anomaly_config or SumAnomalyConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    _progress(progress, f"loading KG from {kg_dir}")
    backend = ParquetKGQueryBackend.from_directory(kg_dir, include_evidence=False)
    # Independent re-derivation oracle for Gate A completeness, built straight from the
    # source KG tables (separate code path from the backend's flattened records_df).
    _progress(progress, "building independent Gate A reference index")
    reference = ReferenceKGIndex.from_directory(kg_dir)
    _progress(progress, f"sampling answer specs target={config.target_specs:,}")
    specs = sample_answer_specs(
        backend,
        config,
        progress_callback=(lambda message: _progress(progress, message)) if progress else None,
        progress_every=progress_every,
    )
    _progress(progress, f"sampled {len(specs):,} answer specs")

    # Pass 1: context-free gates + golden answers. We keep only small per-spec records
    # (never the full matched rows) so the run scales to 10k specs.
    interim: list[dict[str, Any]] = []
    sum_stats: list[tuple[str, float, int]] = []
    type_counter: Counter[str] = Counter()

    for idx, spec in enumerate(specs, start=1):
        qtype = str(spec.metadata.get("question_type", "unknown"))
        type_counter[qtype] += 1

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

        core_passed = constraint_gate.passed and completeness.passed and uniqueness.passed
        row_values: list[float] = []
        if qtype == "aggregation_sum" and core_passed and golden_answer is not None:
            row_values = [float(value) for value in numeric_row_values(full_rows, spec)]
            total = float(golden_answer)
            sum_stats.append((spec.spec_id, total, len(spec.sampled_evidence_ids)))

        interim.append(
            {
                "spec": spec,
                "qtype": qtype,
                "gates": [constraint_gate, completeness, uniqueness],
                "golden_answer": golden_answer,
                "execution_error": execution_error,
                "row_values": row_values,
            }
        )
        if progress and progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(specs)):
            _progress(progress, f"Gate A pass 1 checked {idx:,}/{len(specs):,}")

    # Distribution-level outlier fence over the accepted sum cohort.
    _progress(progress, "checking aggregation-sum distribution anomalies")
    outlier_flags, anomaly_report = distribution_outliers(sum_stats, anomaly_config)

    # Pass 2: add the sum value-sanity gate, finalise accept/reject, write outputs.
    accepted_path = output_dir / "answer_specs.jsonl"
    gate_path = output_dir / "gate_a_report.jsonl"
    sample_csv_path = report_dir / "stage1_answer_spec_sample.csv"
    summary_path = report_dir / "stage1_summary.json"

    accepted = 0
    rejected = 0
    accepted_type_counter: Counter[str] = Counter()
    status_rollup: Counter[str] = Counter()
    gate_fail_counter: Counter[str] = Counter()
    sum_value_warn: list[str] = []
    sum_value_excluded: list[str] = []
    placeholder_flagged: list[str] = []
    sample_rows: list[dict[str, Any]] = []

    with accepted_path.open("w", encoding="utf-8") as accepted_file, gate_path.open(
        "w", encoding="utf-8"
    ) as gate_file:
        for idx, item in enumerate(interim, start=1):
            spec = item["spec"]
            qtype = item["qtype"]
            gate_reports: list[GateReport] = list(item["gates"])
            golden_answer = item["golden_answer"]
            core_passed = all(report.passed for report in gate_reports)

            value_status = "PASS"
            if qtype == "aggregation_sum" and core_passed and golden_answer is not None:
                value_gate = _sum_value_gate(
                    spec_id=spec.spec_id,
                    row_values=item["row_values"],
                    total=float(golden_answer),
                    n_evidence=len(spec.sampled_evidence_ids),
                    outlier_flags=outlier_flags,
                    anomaly_config=anomaly_config,
                    exclude=exclude_sum_anomalies,
                )
                gate_reports.append(value_gate)
                value_status = value_gate.effective_status
                if value_status == "WARN":
                    sum_value_warn.append(spec.spec_id)
                elif value_status == "FAIL":
                    sum_value_excluded.append(spec.spec_id)
                if value_gate.metrics.get("absolute_flags"):
                    placeholder_flagged.append(spec.spec_id)

            statuses = [report.effective_status for report in gate_reports]
            worst = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")
            status_rollup[worst] += 1
            for report in gate_reports:
                if report.effective_status == "FAIL":
                    gate_fail_counter[report.gate] += 1

            gate_passed = "FAIL" not in statuses

            gate_record = {
                "spec_id": spec.spec_id,
                "accepted": gate_passed,
                "status": worst,
                "question_type": qtype,
                "gate_reports": [gate_report_to_dict(report) for report in gate_reports],
                "execution_error": item["execution_error"],
                "evidence_count": len(spec.sampled_evidence_ids),
            }
            gate_file.write(json.dumps(gate_record, default=json_default) + "\n")

            if not gate_passed:
                rejected += 1
                continue

            accepted += 1
            accepted_type_counter[qtype] += 1
            accepted_record = {
                "spec": spec_to_dict(spec),
                "golden_answer": golden_answer,
                "value_sanity_status": value_status,
                "stage": "stage1_gate_a_passed",
            }
            accepted_file.write(json.dumps(accepted_record, default=json_default) + "\n")
            sample_rows.append(_sample_row(spec, golden_answer, value_status))
            if progress and progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == len(interim)):
                _progress(
                    progress,
                    f"Gate A pass 2 wrote {idx:,}/{len(interim):,} "
                    f"(accepted={accepted:,}, rejected={rejected:,})",
                )

    _progress(progress, f"writing sample CSV and summary to {report_dir}")
    _write_sample_csv(sample_csv_path, sample_rows)

    summary = {
        "attempted": len(interim),
        "accepted": accepted,
        "rejected": rejected,
        "target_specs": config.target_specs,
        "type_counts_attempted": dict(sorted(type_counter.items())),
        "type_counts_accepted": dict(sorted(accepted_type_counter.items())),
        "status_rollup": {status: status_rollup.get(status, 0) for status in ("PASS", "WARN", "FAIL")},
        "gate_a": {
            "constraint_conflict_fail": gate_fail_counter.get("gate_a_constraints", 0),
            "completeness_fail": gate_fail_counter.get("gate_a_completeness", 0),
            "uniqueness_fail": gate_fail_counter.get("gate_a_uniqueness", 0),
            "sum_value_warn": len(sum_value_warn),
            "sum_value_excluded": len(sum_value_excluded),
        },
        "sum_anomaly_report": {
            **anomaly_report,
            "placeholder_or_tiny_warn": sorted(placeholder_flagged),
            "distribution_outlier_warn": sorted(outlier_flags),
            "exclude_sum_anomalies": exclude_sum_anomalies,
        },
        "outputs": {
            "answer_specs": str(accepted_path),
            "gate_a_report": str(gate_path),
            "sample_csv": str(sample_csv_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    _progress(progress, f"done accepted={accepted:,} rejected={rejected:,}")
    return summary


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[stage1] {message}", flush=True)


def _completeness_gate(
    reference: ReferenceKGIndex,
    constraints: Any,
    sampled_ids: set[str],
    records_df_ids: set[str],
    duplicate_rows: int,
) -> GateReport:
    """Verify recorded evidence against an independent re-derivation from source tables.

    This is the non-circular completeness check: ``sampled_ids`` came from the backend's
    flattened ``records_df``; ``reference`` recomputes the matching set from the raw
    ``contract_nodes`` + edge tables. Agreement is genuine evidence of completeness;
    disagreement means the backend dropped, duplicated, or miscounted contracts.
    """
    metrics: dict[str, Any] = {
        "sampled_count": len(sampled_ids),
        "records_df_count": len(records_df_ids),
        "duplicate_rows": duplicate_rows,
    }
    consistent = sampled_ids == records_df_ids
    nonempty = bool(sampled_ids)

    try:
        reference_ids = reference.matching_ids(tuple(constraints))
    except UnsupportedReferenceOp as exc:
        # Cannot cross-check this op independently: keep but flag (WARN), unless a basic
        # integrity check already fails.
        ok = nonempty and duplicate_rows == 0 and consistent
        metrics["reference_available"] = False
        metrics["reference_note"] = str(exc)
        return GateReport(
            gate="gate_a_completeness",
            passed=ok,
            status="WARN" if ok else "FAIL",
            reason=f"independent reference unavailable ({exc}); basic integrity checks only"
            if ok
            else "basic integrity checks failed without an independent reference",
            metrics=metrics,
        )

    missing = sorted(reference_ids - sampled_ids)
    extra = sorted(sampled_ids - reference_ids)
    metrics.update(
        {
            "reference_available": True,
            "reference_count": len(reference_ids),
            "missing_vs_reference": missing[:20],
            "extra_vs_reference": extra[:20],
        }
    )
    passed = nonempty and duplicate_rows == 0 and consistent and not missing and not extra
    if passed:
        reason = "recorded evidence exactly matches the independent re-derivation"
    elif not nonempty:
        reason = "empty sampled evidence"
    elif duplicate_rows:
        reason = f"{duplicate_rows} duplicate contract rows in backend result"
    elif not consistent:
        reason = "sampler and Stage 1 backend queries disagree (non-deterministic)"
    else:
        reason = "recorded evidence disagrees with independent re-derivation"
    return GateReport(
        gate="gate_a_completeness",
        passed=passed,
        status="PASS" if passed else "FAIL",
        reason=reason,
        metrics=metrics,
    )


def _sum_value_gate(
    *,
    spec_id: str,
    row_values: list[float],
    total: float,
    n_evidence: int,
    outlier_flags: dict[str, str],
    anomaly_config: SumAnomalyConfig,
    exclude: bool,
) -> GateReport:
    absolute_flags = absolute_sum_flags(row_values, total, anomaly_config)
    flags = list(absolute_flags)
    if spec_id in outlier_flags:
        flags.append(outlier_flags[spec_id])
    if not flags:
        return GateReport(
            gate="gate_a_value_sanity",
            passed=True,
            status="PASS",
            reason="sum value within plausible range",
            metrics={"total": f"{total:g}", "n_evidence": n_evidence},
        )
    return GateReport(
        gate="gate_a_value_sanity",
        passed=not exclude,
        status="FAIL" if exclude else "WARN",
        reason="; ".join(flags),
        metrics={
            "total": f"{total:g}",
            "n_evidence": n_evidence,
            "absolute_flags": absolute_flags,
            "distribution_outlier": spec_id in outlier_flags,
        },
    )


def _sample_row(spec: Any, golden_answer: Any, value_status: str) -> dict[str, Any]:
    metadata = spec.metadata or {}
    constraints = [
        {"field": c.field, "op": c.op, "value": c.value} for c in spec.constraints
    ]
    return {
        "spec_id": spec.spec_id,
        "question_type": metadata.get("question_type", ""),
        "operation_family": metadata.get("operation_family", ""),
        "domain_slice": metadata.get("domain_slice", ""),
        "difficulty": metadata.get("difficulty", ""),
        "hop_class": metadata.get("hop_class", ""),
        "generalization_class": metadata.get("generalization_class", ""),
        "evidence_floor": metadata.get("evidence_floor", ""),
        "template_fields": metadata.get("template_fields", ""),
        "operation": spec.answer_operation,
        "answer_field": spec.answer_field,
        "golden_answer": _scalar(golden_answer),
        "evidence_count": len(spec.sampled_evidence_ids),
        "value_sanity_status": value_status,
        "constraints": json.dumps(constraints, default=json_default),
        "logic_chain": " | ".join(spec.logic_chain),
    }


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


_SAMPLE_COLUMNS = [
    "spec_id",
    "question_type",
    "operation_family",
    "domain_slice",
    "difficulty",
    "hop_class",
    "generalization_class",
    "evidence_floor",
    "template_fields",
    "operation",
    "answer_field",
    "golden_answer",
    "evidence_count",
    "value_sanity_status",
    "constraints",
    "logic_chain",
]


def _write_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SAMPLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in _SAMPLE_COLUMNS})


def _csv_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    return value


__all__ = ["build_stage1"]
