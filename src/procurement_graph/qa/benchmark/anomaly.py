"""Deterministic anomaly checks for aggregation-sum answer specs.

Pilot spot-checks surfaced two failure modes in ``aggregation_sum`` golden answers:

- nominal placeholders: a single contract with ``value_amount == 1`` produces a sum of
  ``1.0``, which is a data artefact rather than a real procurement total;
- magnitude outliers: a sum of ~GBP 1.17bn over only two contracts (~GBP 586m per row)
  is plausibly a data-entry error (extra zeros) rather than a genuine value.

The checks are deterministic and split into two layers:

- :func:`absolute_sum_flags` is per-spec and needs no global context (placeholders and
  implausibly small totals);
- :func:`distribution_outliers` fences the per-evidence mean on a log10 scale using an IQR
  rule, so it catches both very small and very large magnitudes relative to the cohort.

Neither layer mutates data. Stage 1 turns the flags into a visible WARN gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SumAnomalyConfig:
    placeholder_value_max: float = 1.0
    min_plausible_total: float = 1.0
    # Standard Tukey outlier fence (1.5*IQR). This gate only WARNs (flags for review),
    # so a tighter fence than "far-out" (3.0) is preferred: it surfaces the few-records /
    # huge-per-record cases that are most likely data-entry errors.
    iqr_multiplier: float = 1.5
    min_specs_for_distribution: int = 8


def absolute_sum_flags(values: Iterable[float], total: float, config: SumAnomalyConfig) -> list[str]:
    """Per-spec flags that do not depend on the rest of the cohort."""
    rows = [float(value) for value in values]
    flags: list[str] = []
    if rows and all(value <= config.placeholder_value_max for value in rows):
        flags.append(
            f"all {len(rows)} summed values <= {config.placeholder_value_max:g} (nominal/placeholder)"
        )
    if total <= config.min_plausible_total:
        flags.append(f"total {total:g} <= min plausible {config.min_plausible_total:g}")
    return flags


def distribution_outliers(
    stats: list[tuple[str, float, int]],
    config: SumAnomalyConfig,
) -> tuple[dict[str, str], dict[str, object]]:
    """Flag specs whose log10 per-evidence mean is an IQR outlier within the cohort.

    ``stats`` is a list of ``(spec_id, total, n_evidence)``. Returns a mapping of flagged
    ``spec_id`` to reason, plus a small report describing the fence for the summary.
    """
    usable = [(spec_id, total, n) for spec_id, total, n in stats if n > 0 and total > 0]
    report: dict[str, object] = {
        "sum_specs_total": len(stats),
        "sum_specs_usable": len(usable),
        "fence_applied": False,
        "log10_mean_lower_fence": None,
        "log10_mean_upper_fence": None,
    }
    if len(usable) < config.min_specs_for_distribution:
        report["reason"] = (
            f"only {len(usable)} usable sum specs (< {config.min_specs_for_distribution}); "
            "distribution fence skipped"
        )
        return {}, report

    log_means = [(spec_id, math.log10(total / n)) for spec_id, total, n in usable]
    sorted_logs = sorted(value for _, value in log_means)
    q1 = _quantile(sorted_logs, 0.25)
    q3 = _quantile(sorted_logs, 0.75)
    iqr = q3 - q1
    lower = q1 - config.iqr_multiplier * iqr
    upper = q3 + config.iqr_multiplier * iqr
    report.update(
        fence_applied=True,
        log10_mean_lower_fence=round(lower, 4),
        log10_mean_upper_fence=round(upper, 4),
    )

    flagged: dict[str, str] = {}
    for spec_id, log_mean in log_means:
        mean = 10 ** log_mean
        if log_mean < lower:
            flagged[spec_id] = f"per-evidence mean {mean:,.2f} below cohort fence (low outlier)"
        elif log_mean > upper:
            flagged[spec_id] = f"per-evidence mean {mean:,.2f} above cohort fence (high outlier)"
    return flagged, report


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[int(position)]
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


__all__ = ["SumAnomalyConfig", "absolute_sum_flags", "distribution_outliers"]
