"""JSONL serialization helpers for QA benchmark Stage 1 outputs."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from typing import Any

from .models import AnswerSpec, GateReport


def spec_to_dict(spec: AnswerSpec) -> dict[str, Any]:
    return asdict(spec)


def gate_report_to_dict(report: GateReport) -> dict[str, Any]:
    return asdict(report)


def json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, set):
        return sorted(value)
    return str(value)
