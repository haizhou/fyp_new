"""Summarise small LLM adjudication batches before scaling up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from procurement_graph.experiments.reference_llm_adjudication import (
    QUEUE_PATH as REFERENCE_QUEUE_PATH,
    read_jsonl,
    validate_decisions,
)

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports" / "ablation" / "llm"
DEFAULT_DECISIONS_PATH = ROOT / "data" / "ablation" / "reference" / "llm_decisions.jsonl"
DEFAULT_RAW_PATH = ROOT / "data" / "ablation" / "reference" / "llm_raw_responses.jsonl"
DEFAULT_ERRORS_PATH = ROOT / "data" / "ablation" / "reference" / "llm_errors.jsonl"
SUMMARY_PATH = REPORT_DIR / "batch_review_summary.md"
DECISION_REVIEW_PATH = REPORT_DIR / "batch_decision_review.csv"
ERROR_REVIEW_PATH = REPORT_DIR / "batch_error_review.csv"


def _load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _raw_usage(raw_records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in raw_records:
        response = record.get("response") or {}
        usage = response.get("usage") or {}
        rows.append({
            "task_id": record.get("task_id", ""),
            "prompt_version": record.get("prompt_version", ""),
            "schema_version": record.get("schema_version", ""),
            "schema_hash": record.get("schema_hash", ""),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        })
    return pd.DataFrame(rows)


def _md_counts(title: str, series: pd.Series) -> list[str]:
    lines = [f"## {title}", ""]
    if series.empty:
        lines.append("_No rows._")
    else:
        for key, value in series.items():
            lines.append(f"- `{key}`: `{int(value)}`")
    lines.append("")
    return lines


def review_batch(
    queue_path: Path = REFERENCE_QUEUE_PATH,
    decisions_path: Path = DEFAULT_DECISIONS_PATH,
    raw_path: Path = DEFAULT_RAW_PATH,
    errors_path: Path = DEFAULT_ERRORS_PATH,
    sample_rows: int = 50,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    queue = read_jsonl(queue_path)
    decisions = _load_jsonl_if_exists(decisions_path)
    raw_records = _load_jsonl_if_exists(raw_path)
    error_records = _load_jsonl_if_exists(errors_path)

    validated = validate_decisions(queue, decisions) if decisions else pd.DataFrame()
    usage = _raw_usage(raw_records)
    errors = pd.DataFrame(error_records)

    if not validated.empty:
        review_cols = [
            "task_id",
            "task_type",
            "prompt_version",
            "schema_version",
            "schema_hash",
            "reference_source",
            "reference_canonical_id",
            "reference_matched_name",
            "decision",
            "llm_confidence",
            "approved_entity_ids",
            "excluded_entity_ids",
            "candidate_entity_ids",
            "risk_flags",
            "validation_errors",
            "is_valid_json_schema",
            "reason",
        ]
        review_cols = [col for col in review_cols if col in validated.columns]
        validated.sort_values(["is_valid_json_schema", "decision", "llm_confidence"], ascending=[True, True, False])[
            review_cols
        ].head(sample_rows).to_csv(DECISION_REVIEW_PATH, index=False)
    else:
        pd.DataFrame().to_csv(DECISION_REVIEW_PATH, index=False)

    if not errors.empty:
        errors.head(sample_rows).to_csv(ERROR_REVIEW_PATH, index=False)
    else:
        pd.DataFrame().to_csv(ERROR_REVIEW_PATH, index=False)

    lines = [
        "# LLM Batch Review Summary",
        "",
        f"- Queue tasks: `{len(queue)}`",
        f"- Decision rows: `{len(decisions)}`",
        f"- Raw response rows: `{len(raw_records)}`",
        f"- Error rows: `{len(error_records)}`",
        "",
    ]
    if not validated.empty:
        lines.extend(_md_counts("Decision Counts", validated["decision"].value_counts()))
        lines.extend(_md_counts("Validation Error Counts", validated["validation_errors"].value_counts()))
        lines.extend(_md_counts("Risk Flag Counts", validated["risk_flags"].value_counts().head(20)))
        lines.append(f"- Valid schema rows: `{int(validated['is_valid_json_schema'].sum())}`")
        lines.append(f"- Invalid schema rows: `{int((~validated['is_valid_json_schema']).sum())}`")
        lines.append("")
    if not usage.empty:
        lines.extend([
            "## Token Usage",
            "",
            f"- Total input tokens: `{int(usage['input_tokens'].sum())}`",
            f"- Total output tokens: `{int(usage['output_tokens'].sum())}`",
            f"- Total tokens: `{int(usage['total_tokens'].sum())}`",
            f"- Median tokens/task: `{float(usage['total_tokens'].median()):.1f}`",
            "",
        ])
    if not errors.empty and "message" in errors.columns:
        lines.extend(_md_counts("Error Types", errors.get("error", pd.Series(dtype=str)).value_counts()))
        enum_like = errors[errors["message"].astype(str).str.contains("enum|schema|json", case=False, na=False)]
        lines.append(f"- Enum/schema-like errors: `{len(enum_like)}`")
        lines.append("")

    lines.extend([
        "## Review Files",
        "",
        f"- `{DECISION_REVIEW_PATH}`",
        f"- `{ERROR_REVIEW_PATH}`",
    ])
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written: {SUMMARY_PATH}")
    print(f"Written: {DECISION_REVIEW_PATH}")
    print(f"Written: {ERROR_REVIEW_PATH}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review a small LLM decision batch")
    parser.add_argument("--queue", type=Path, default=REFERENCE_QUEUE_PATH)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS_PATH)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS_PATH)
    parser.add_argument("--sample-rows", type=int, default=50)
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    review_batch(
        queue_path=args.queue,
        decisions_path=args.decisions,
        raw_path=args.raw,
        errors_path=args.errors,
        sample_rows=args.sample_rows,
    )


__all__ = ["cli_main", "review_batch"]
