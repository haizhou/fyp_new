"""Answer-sanity gate for aggregations (unifies the benchmark sum-anomaly WARN with the runtime).

Generalised from the prior pipeline's `step5.aggregation_anomaly_features` and the QA benchmark's
sum-anomaly gate so that "is this total trustworthy?" has ONE definition offline and online. For a
computed sum it inspects the contributing row values and flags:

- placeholder/nominal totals (every contributor <= a small threshold, e.g. GBP 1);
- a single dominant contributor (>= a share of the total) -> a framework CEILING value or a
  data-entry error rather than real spend.

The runtime uses this to downgrade confidence and disclose the caveat instead of presenting a
suspicious number as fact; the benchmark uses the same signal to WARN-flag a spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ExecutionResult


@dataclass(frozen=True)
class SanityVerdict:
    ok: bool
    flags: tuple[str, ...] = ()
    dominant_share: float = 0.0
    contributor_count: int = 0

    @property
    def caveat(self) -> str:
        return "; ".join(self.flags)


def check_sum_sanity(
    values: list[float],
    total: float,
    *,
    placeholder_max: float = 1.0,
    dominant_share_threshold: float = 0.9,
) -> SanityVerdict:
    floats = [float(value) for value in values]
    count = len(floats)
    if not floats:
        return SanityVerdict(ok=True)
    flags: list[str] = []
    if all(value <= placeholder_max for value in floats):
        flags.append(f"all {count} contributing values are <= {placeholder_max:g} (nominal/placeholder)")
    largest = max(floats, key=abs)
    denom = abs(total) if total else (sum(abs(value) for value in floats) or 1.0)
    share = abs(largest) / denom if denom else 0.0
    if count > 1 and share >= dominant_share_threshold:
        flags.append(f"one contributor is {share:.0%} of the total (framework-ceiling or data-entry risk)")
    return SanityVerdict(ok=not flags, flags=tuple(flags), dominant_share=round(share, 4), contributor_count=count)


def check_sum_sanity_summary(
    max_value: float,
    count: int,
    total: float,
    *,
    placeholder_max: float = 1.0,
    dominant_share_threshold: float = 0.9,
) -> SanityVerdict:
    """Same verdict as check_sum_sanity from a (max, count, total) summary -- no per-row scan.
    Valid because both signals reduce to the max contributor: placeholder = max <= placeholder_max;
    dominant = |max| / |total| >= threshold. (Contract values are non-negative.)"""
    if count <= 0:
        return SanityVerdict(ok=True)
    flags: list[str] = []
    if max_value <= placeholder_max:
        flags.append(f"all {count} contributing values are <= {placeholder_max:g} (nominal/placeholder)")
    denom = abs(total) if total else 1.0
    share = abs(max_value) / denom if denom else 0.0
    if count > 1 and share >= dominant_share_threshold:
        flags.append(f"one contributor is {share:.0%} of the total (framework-ceiling or data-entry risk)")
    return SanityVerdict(ok=not flags, flags=tuple(flags), dominant_share=round(share, 4), contributor_count=count)


def sum_contributor_values(result: ExecutionResult, field: str = "value_amount") -> list[float]:
    """Per-row numeric values behind a sum ExecutionResult, deduped by contract_node_id."""
    values: list[float] = []
    seen: set[str] = set()
    for row in result.evidence.rows:
        key = str(row.get("contract_node_id") or row.get("record_id") or id(row))
        if key in seen:
            continue
        seen.add(key)
        number = _to_number(row.get(field))
        if number is not None:
            values.append(number)
    return values


def sanity_for_execution(result: ExecutionResult) -> SanityVerdict:
    """Return a sanity verdict for a passed sum; trivially OK for everything else.

    Prefers the contributor summary (max, count) the executor carries in metrics, so a large sum need
    not re-scan every row; falls back to scanning evidence rows when the summary is absent."""
    if not result.passed or result.query_spec.answer_operation != "sum":
        return SanityVerdict(ok=True)
    total = _to_number(result.answer) or 0.0
    metrics = result.metrics or {}
    if "contributor_max" in metrics and "contributor_count" in metrics:
        return check_sum_sanity_summary(_to_number(metrics["contributor_max"]) or 0.0,
                                        int(metrics["contributor_count"]), total)
    values = sum_contributor_values(result, result.query_spec.answer_field or "value_amount")
    return check_sum_sanity(values, total)


def _to_number(value: object) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


__all__ = ["SanityVerdict", "check_sum_sanity", "check_sum_sanity_summary",
           "sum_contributor_values", "sanity_for_execution"]
