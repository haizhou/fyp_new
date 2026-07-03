"""Procurement-specific deterministic AnswerSpec samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

from .constraints import normalize_constraints
from .kg_interface import QueryBackend
from .models import AnswerSpec, AnswerValueType, Constraint

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class SamplerConfig:
    seed: int = 42
    target_specs: int = 10000
    min_evidence_rows: int = 1
    min_aggregation_evidence_rows: int = 3
    min_sum_evidence_rows: int = 3
    min_conjunction_evidence_rows: int = 3
    min_temporal_evidence_rows: int = 3
    min_cpv_evidence_rows: int = 3
    max_evidence_rows: int = 20000


def sample_answer_specs(
    backend: QueryBackend,
    config: SamplerConfig = SamplerConfig(),
    *,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 1000,
) -> list[AnswerSpec]:
    if not hasattr(backend, "records_df"):
        raise TypeError("Procurement samplers require ParquetKGQueryBackend.records_df")
    records = getattr(backend, "records_df")
    rng = _rng(config.seed)
    specs: list[AnswerSpec] = []
    budgets, capacities = _rebalance_budgets(records, _budgets(config.target_specs), config)

    _emit(progress_callback, f"sampler capacities: {capacities}")
    _emit(progress_callback, f"sampler budgets: {budgets}")
    specs.extend(_sample_factoid_specs(backend, records, rng, budgets["factoid"], progress_callback, progress_every))
    specs.extend(
        _sample_count_specs(backend, records, rng, budgets["aggregation_count"], config, progress_callback, progress_every)
    )
    specs.extend(_sample_sum_specs(backend, records, rng, budgets["aggregation_sum"], config, progress_callback, progress_every))
    specs.extend(
        _sample_conjunction_specs(
            backend, records, rng, budgets["conjunction_constraint"], config, progress_callback, progress_every
        )
    )
    specs.extend(_sample_temporal_specs(backend, records, rng, budgets["temporal"], config, progress_callback, progress_every))
    specs.extend(_sample_cpv_specs(backend, records, rng, budgets["categorical_cpv"], config, progress_callback, progress_every))

    deduped: dict[str, AnswerSpec] = {}
    for spec in specs:
        deduped.setdefault(spec.spec_id, spec)
    _emit(progress_callback, f"sampler deduped {len(specs):,} raw specs to {len(deduped):,}")
    return list(deduped.values())[: config.target_specs]


def _sample_factoid_specs(
    backend: QueryBackend,
    records: pd.DataFrame,
    rng: pd.Series,
    target: int,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 1000,
) -> list[AnswerSpec]:
    _emit(progress_callback, f"sampler factoid: selecting up to {target:,}")
    candidates = records[
        records["buyer_count"].gt(0)
        & records["supplier_count"].gt(0)
        & records["value_source"].astype(str).ne("")
        & records["has_award_signed_date"].fillna(False)
        & records["tender_category"].astype(str).str.strip().ne("")
    ]
    candidates = _sample_frame(candidates, min(target, len(candidates)), rng)
    specs: list[AnswerSpec] = []
    fields: list[tuple[str, AnswerValueType]] = [
        ("buyer_name", "string"),
        ("supplier_name", "string"),
        ("tender_category", "string"),
        ("value_source", "string"),
        ("award_date_signed", "date"),
    ]
    total = len(candidates)
    _emit(progress_callback, f"sampler factoid: {total:,} candidates selected")
    for idx, row in enumerate(candidates.itertuples(index=False)):
        field, value_type = fields[idx % len(fields)]
        constraints = (Constraint("contract_node_id", "eq", row.contract_node_id),)
        specs.append(
            _make_spec(
                backend,
                spec_id=f"factoid_{idx:04d}_{field}",
                constraints=constraints,
                answer_operation="select_unique",
                answer_field=field,
                answer_value_type=value_type,
                logic_chain=("contract_node_id exact match", f"select {field}"),
                metadata={
                    "question_type": "factoid",
                    "difficulty": "easy",
                    "hop_class": "1-hop",
                    "operation_family": "contract_factoid",
                    "domain_slice": "contract_identity",
                    "generalization_class": "iid",
                    "evidence_floor": "1",
                    "template_fields": "contract_node_id",
                },
            )
        )
        _emit_loop(progress_callback, "sampler factoid", idx + 1, total, progress_every)
    return specs


def _sample_count_specs(
    backend: QueryBackend,
    records: pd.DataFrame,
    rng: pd.Series,
    target: int,
    config: SamplerConfig,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 1000,
) -> list[AnswerSpec]:
    specs = []
    _emit(progress_callback, f"sampler aggregation_count: grouping for target {target:,}")
    grouped = _combined_groups(
        records,
        [
            ["release_year", "tender_category"],
            ["release_year", "tender_cpv_id"],
        ],
        config,
        min_rows=config.min_aggregation_evidence_rows,
    )
    grouped = _sample_frame(grouped, min(target, len(grouped)), rng)
    total = len(grouped)
    _emit(progress_callback, f"sampler aggregation_count: {total:,} groups selected")
    for idx, row in enumerate(grouped.itertuples(index=False)):
        constraints = _constraints_from_group(row)
        specs.append(
            _make_spec(
                backend,
                spec_id=f"count_year_category_{idx:04d}",
                constraints=constraints,
                answer_operation="count",
                answer_field="contract_node_id",
                answer_value_type="integer",
                dedupe_key="contract_node_id",
                logic_chain=(*_logic_from_group(row), "count contracts"),
                metadata=_metadata(
                    "aggregation_count",
                    row,
                    operation_family="filtered_count",
                    domain_slice=_domain_slice(row),
                    evidence_floor=config.min_aggregation_evidence_rows,
                ),
            )
        )
        _emit_loop(progress_callback, "sampler aggregation_count", idx + 1, total, progress_every)
    return specs


def _sample_sum_specs(
    backend: QueryBackend,
    records: pd.DataFrame,
    rng: pd.Series,
    target: int,
    config: SamplerConfig,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 1000,
) -> list[AnswerSpec]:
    additive = records[records["value_is_additive"].fillna(False)]
    _emit(progress_callback, f"sampler aggregation_sum: grouping additive rows for target {target:,}")
    grouped = _combined_groups(
        additive,
        [
            ["release_year", "tender_category"],
            ["release_year", "tender_cpv_id"],
            ["tender_category", "tender_cpv_id"],
        ],
        config,
        min_rows=config.min_sum_evidence_rows,
    )
    grouped = _sample_frame(grouped, min(target, len(grouped)), rng)
    specs = []
    total = len(grouped)
    _emit(progress_callback, f"sampler aggregation_sum: {total:,} groups selected")
    for idx, row in enumerate(grouped.itertuples(index=False)):
        constraints = _constraints_from_group(row) + (Constraint("value_is_additive", "eq", True),)
        specs.append(
            _make_spec(
                backend,
                spec_id=f"sum_additive_year_category_{idx:04d}",
                constraints=constraints,
                answer_operation="sum",
                answer_field="value_amount",
                answer_value_type="currency",
                dedupe_key="contract_node_id",
                logic_chain=(*_logic_from_group(row), "value_is_additive=True", "sum value_amount"),
                metadata=_metadata(
                    "aggregation_sum",
                    row,
                    operation_family="additive_sum",
                    domain_slice="value",
                    evidence_floor=config.min_sum_evidence_rows,
                ),
            )
        )
        _emit_loop(progress_callback, "sampler aggregation_sum", idx + 1, total, progress_every)
    return specs


def _sample_conjunction_specs(
    backend: QueryBackend,
    records: pd.DataFrame,
    rng: pd.Series,
    target: int,
    config: SamplerConfig,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 1000,
) -> list[AnswerSpec]:
    _emit(progress_callback, f"sampler conjunction_constraint: filtering eligible rows for target {target:,}")
    eligible = records[
        records["supplier_count"].gt(0)
        & records["buyer_count"].gt(0)
        & records["tender_category"].astype(str).ne("")
    ]
    grouped = _combined_groups(
        eligible,
        [
            ["release_year", "tender_category", "tender_cpv_id"],
        ],
        config,
        min_rows=config.min_conjunction_evidence_rows,
    )
    grouped = _sample_frame(grouped, min(target, len(grouped)), rng)
    specs = []
    total = len(grouped)
    _emit(progress_callback, f"sampler conjunction_constraint: {total:,} groups selected")
    for idx, row in enumerate(grouped.itertuples(index=False)):
        constraints = _constraints_from_group(row) + (
            Constraint("supplier_count", "gte", 1),
            Constraint("buyer_count", "gte", 1),
        )
        specs.append(
            _make_spec(
                backend,
                spec_id=f"conjunction_count_{idx:04d}",
                constraints=constraints,
                answer_operation="count",
                answer_field="contract_node_id",
                answer_value_type="integer",
                dedupe_key="contract_node_id",
                logic_suffix=("count contracts",),
                metadata=_metadata(
                    "conjunction_constraint",
                    row,
                    operation_family="conjunction",
                    domain_slice=_domain_slice(row),
                    evidence_floor=config.min_conjunction_evidence_rows,
                ),
            )
        )
        _emit_loop(progress_callback, "sampler conjunction_constraint", idx + 1, total, progress_every)
    return specs


def _sample_temporal_specs(
    backend: QueryBackend,
    records: pd.DataFrame,
    rng: pd.Series,
    target: int,
    config: SamplerConfig,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 1000,
) -> list[AnswerSpec]:
    temporal = records[records["has_award_signed_date"].fillna(False)]
    _emit(progress_callback, f"sampler temporal: grouping signed-date rows for target {target:,}")
    grouped = _combined_groups(
        temporal,
        [
            ["release_year", "tender_category"],
            ["release_year", "tender_cpv_id"],
            ["tender_category", "tender_cpv_id"],
        ],
        config,
        min_rows=config.min_temporal_evidence_rows,
    )
    grouped = _sample_frame(grouped, min(target, len(grouped)), rng)
    specs = []
    total = len(grouped)
    _emit(progress_callback, f"sampler temporal: {total:,} groups selected")
    for idx, row in enumerate(grouped.itertuples(index=False)):
        constraints = _constraints_from_group(row) + (Constraint("has_award_signed_date", "eq", True),)
        specs.append(
            _make_spec(
                backend,
                spec_id=f"temporal_signed_count_{idx:04d}",
                constraints=constraints,
                answer_operation="count",
                answer_field="contract_node_id",
                answer_value_type="integer",
                dedupe_key="contract_node_id",
                logic_suffix=("count contracts",),
                metadata=_metadata(
                    "temporal",
                    row,
                    operation_family="temporal_count",
                    domain_slice="temporal",
                    evidence_floor=config.min_temporal_evidence_rows,
                ),
            )
        )
        _emit_loop(progress_callback, "sampler temporal", idx + 1, total, progress_every)
    return specs


def _sample_cpv_specs(
    backend: QueryBackend,
    records: pd.DataFrame,
    rng: pd.Series,
    target: int,
    config: SamplerConfig,
    progress_callback: ProgressCallback | None = None,
    progress_every: int = 1000,
) -> list[AnswerSpec]:
    cpv = records[records["tender_cpv_id"].astype(str).str.strip().ne("")]
    _emit(progress_callback, f"sampler categorical_cpv: grouping CPV rows for target {target:,}")
    grouped = _combined_groups(
        cpv,
        [
            ["tender_category", "tender_cpv_id"],
        ],
        config,
        min_rows=config.min_cpv_evidence_rows,
    )
    grouped = _sample_frame(grouped, min(target, len(grouped)), rng)
    specs = []
    total = len(grouped)
    _emit(progress_callback, f"sampler categorical_cpv: {total:,} groups selected")
    for idx, row in enumerate(grouped.itertuples(index=False)):
        constraints = _constraints_from_group(row)
        specs.append(
            _make_spec(
                backend,
                spec_id=f"cpv_count_{idx:04d}",
                constraints=constraints,
                answer_operation="count",
                answer_field="contract_node_id",
                answer_value_type="integer",
                dedupe_key="contract_node_id",
                logic_chain=(*_logic_from_group(row), "count contracts"),
                metadata=_metadata(
                    "categorical_cpv",
                    row,
                    operation_family="cpv_slice",
                    domain_slice="cpv",
                    evidence_floor=config.min_cpv_evidence_rows,
                ),
            )
        )
        _emit_loop(progress_callback, "sampler categorical_cpv", idx + 1, total, progress_every)
    return specs


def _make_spec(
    backend: QueryBackend,
    *,
    spec_id: str,
    constraints: tuple[Constraint, ...],
    answer_operation: str,
    answer_field: str,
    answer_value_type: AnswerValueType,
    dedupe_key: str = "",
    logic_chain: tuple[str, ...] = (),
    logic_suffix: tuple[str, ...] = (),
    metadata: dict[str, str],
) -> AnswerSpec:
    constraints = normalize_constraints(constraints)
    if not logic_chain:
        logic_chain = (*_logic_from_constraints(constraints), *logic_suffix)
    rows = backend.query(constraints)
    sampled_ids = tuple(sorted({backend.record_id(row) for row in rows if backend.record_id(row)}))
    return AnswerSpec(
        spec_id=spec_id,
        constraints=constraints,
        answer_operation=answer_operation,  # type: ignore[arg-type]
        answer_field=answer_field,
        answer_value_type=answer_value_type,
        dedupe_key=dedupe_key,
        logic_chain=logic_chain,
        sampled_evidence_ids=sampled_ids,
        metadata=metadata,
    )


_OP_RENDER = {"eq": "=", "gte": ">=", "lte": "<="}


def _render_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _logic_from_constraints(constraints: tuple[Constraint, ...]) -> tuple[str, ...]:
    """Render each constraint with its concrete field/value, for example
    ``release_year=2024 | tender_cpv_id=85141100 | supplier_count>=1``."""
    parts: list[str] = []
    for constraint in constraints:
        if constraint.op == "exists":
            parts.append(f"{constraint.field} exists")
        elif constraint.op in _OP_RENDER:
            parts.append(f"{constraint.field}{_OP_RENDER[constraint.op]}{_render_value(constraint.value)}")
        else:
            parts.append(f"{constraint.field} {constraint.op} {_render_value(constraint.value)}")
    return tuple(parts)


def _moderate_groups(
    records: pd.DataFrame,
    columns: list[str],
    config: SamplerConfig,
    *,
    min_rows: int | None = None,
) -> pd.DataFrame:
    grouped = records.groupby(columns, dropna=False).size().rename("n").reset_index()
    lower = config.min_evidence_rows if min_rows is None else min_rows
    grouped = grouped[
        grouped["n"].between(lower, config.max_evidence_rows)
    ]
    for column in columns:
        text = grouped[column].astype(str).str.strip()
        grouped = grouped[grouped[column].notna() & text.ne("") & text.ne("nan") & text.ne("None")]
    return grouped.sort_values(["n", *columns]).reset_index(drop=True)


def _combined_groups(
    records: pd.DataFrame,
    groupings: list[list[str]],
    config: SamplerConfig,
    *,
    min_rows: int | None = None,
) -> pd.DataFrame:
    frames = []
    for columns in groupings:
        grouped = _moderate_groups(records, columns, config, min_rows=min_rows)
        if grouped.empty:
            continue
        grouped["group_columns"] = "|".join(columns)
        frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.sort_values(["n", "group_columns"]).reset_index(drop=True)


def _constraints_from_group(row: object) -> tuple[Constraint, ...]:
    columns = str(getattr(row, "group_columns")).split("|")
    constraints = []
    for column in columns:
        value = getattr(row, column)
        if column == "release_year":
            value = int(value)
        if column == "supplier_count":
            value = int(value)
        constraints.append(Constraint(column, "eq", value))
    return tuple(constraints)


def _logic_from_group(row: object) -> tuple[str, ...]:
    columns = str(getattr(row, "group_columns")).split("|")
    parts = []
    for column in columns:
        value = getattr(row, column)
        if column == "release_year":
            value = int(value)
        parts.append(f"{column}={value}")
    return tuple(parts)


def _metadata(
    question_type: str,
    row: object,
    *,
    operation_family: str,
    domain_slice: str,
    evidence_floor: int,
) -> dict[str, str]:
    n = int(getattr(row, "n", 0))
    columns = str(getattr(row, "group_columns", "")).split("|")
    return {
        "question_type": question_type,
        "difficulty": _difficulty(columns, n),
        "hop_class": _hop_class(columns),
        "operation_family": operation_family,
        "domain_slice": domain_slice,
        "generalization_class": _generalization_class(columns, n),
        "evidence_floor": str(evidence_floor),
        "template_fields": "|".join(columns),
    }


def _difficulty(columns: list[str], n: int) -> str:
    semantic_constraints = len([column for column in columns if column not in {"supplier_count", "buyer_count"}])
    if semantic_constraints <= 1 and n < 25:
        return "easy"
    if semantic_constraints >= 3 or n >= 50:
        return "hard"
    return "medium"


def _hop_class(columns: list[str]) -> str:
    semantic_constraints = len([column for column in columns if column not in {"supplier_count", "buyer_count"}])
    if semantic_constraints <= 1:
        return "1-hop"
    if semantic_constraints == 2:
        return "2-hop"
    return "3-hop"


def _generalization_class(columns: list[str], n: int) -> str:
    semantic_constraints = len([column for column in columns if column not in {"supplier_count", "buyer_count"}])
    if semantic_constraints >= 3 or n <= 5:
        return "compositional"
    if n >= 100:
        return "iid"
    return "iid"


def _domain_slice(row: object) -> str:
    columns = set(str(getattr(row, "group_columns", "")).split("|"))
    if "tender_cpv_id" in columns:
        return "cpv"
    if "value_source" in columns:
        return "value"
    if "tender_category" in columns:
        return "category"
    if "release_year" in columns:
        return "temporal"
    return "contract_identity"


def _sample_frame(df: pd.DataFrame, n: int, rng: pd.Series) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.head(0)
    if len(df) <= n:
        return df.copy()
    seed = int(rng.iloc[0])
    rng.iloc[0] = (seed * 1103515245 + 12345) % (2**31)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _rng(seed: int) -> pd.Series:
    return pd.Series([seed])


def _rebalance_budgets(
    records: pd.DataFrame,
    budgets: dict[str, int],
    config: SamplerConfig,
) -> tuple[dict[str, int], dict[str, int]]:
    capacities = _capacity_by_type(records, config)
    balanced = {name: min(target, capacities.get(name, 0)) for name, target in budgets.items()}
    shortfall = config.target_specs - sum(balanced.values())

    while shortfall > 0:
        candidates = [name for name in budgets if balanced[name] < capacities.get(name, 0)]
        if not candidates:
            break
        total_weight = sum(max(budgets[name], 1) for name in candidates)
        grants: dict[str, int] = {}
        for name in candidates:
            remaining_capacity = capacities[name] - balanced[name]
            weighted_share = int(shortfall * max(budgets[name], 1) / total_weight)
            grants[name] = min(remaining_capacity, weighted_share)

        granted = sum(grants.values())
        if granted == 0:
            candidates = sorted(candidates, key=lambda name: budgets[name], reverse=True)
            for name in candidates:
                if shortfall <= 0:
                    break
                if balanced[name] < capacities[name]:
                    balanced[name] += 1
                    shortfall -= 1
            continue

        for name, grant in grants.items():
            balanced[name] += grant
        shortfall -= granted

    return balanced, capacities


def _capacity_by_type(records: pd.DataFrame, config: SamplerConfig) -> dict[str, int]:
    factoid = records[
        records["buyer_count"].gt(0)
        & records["supplier_count"].gt(0)
        & records["value_source"].astype(str).ne("")
        & records["has_award_signed_date"].fillna(False)
        & records["tender_category"].astype(str).str.strip().ne("")
    ]
    count = _combined_groups(
        records,
        [
            ["release_year", "tender_category"],
            ["release_year", "tender_cpv_id"],
        ],
        config,
        min_rows=config.min_aggregation_evidence_rows,
    )
    additive = records[records["value_is_additive"].fillna(False)]
    sums = _combined_groups(
        additive,
        [
            ["release_year", "tender_category"],
            ["release_year", "tender_cpv_id"],
            ["tender_category", "tender_cpv_id"],
        ],
        config,
        min_rows=config.min_sum_evidence_rows,
    )
    eligible = records[
        records["supplier_count"].gt(0)
        & records["buyer_count"].gt(0)
        & records["tender_category"].astype(str).ne("")
    ]
    conjunction = _combined_groups(
        eligible,
        [
            ["release_year", "tender_category", "tender_cpv_id"],
        ],
        config,
        min_rows=config.min_conjunction_evidence_rows,
    )
    temporal_rows = records[records["has_award_signed_date"].fillna(False)]
    temporal = _combined_groups(
        temporal_rows,
        [
            ["release_year", "tender_category"],
            ["release_year", "tender_cpv_id"],
            ["tender_category", "tender_cpv_id"],
        ],
        config,
        min_rows=config.min_temporal_evidence_rows,
    )
    cpv_rows = records[records["tender_cpv_id"].astype(str).str.strip().ne("")]
    cpv = _combined_groups(
        cpv_rows,
        [
            ["tender_category", "tender_cpv_id"],
        ],
        config,
        min_rows=config.min_cpv_evidence_rows,
    )
    return {
        "factoid": len(factoid),
        "aggregation_count": len(count),
        "aggregation_sum": len(sums),
        "conjunction_constraint": len(conjunction),
        "temporal": len(temporal),
        "categorical_cpv": len(cpv),
    }


def _emit(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _emit_loop(
    callback: ProgressCallback | None,
    label: str,
    current: int,
    total: int,
    interval: int,
) -> None:
    if callback is None or total <= 0 or interval <= 0:
        return
    if current == 1 or current % interval == 0 or current == total:
        callback(f"{label}: built {current:,}/{total:,}")


def _budgets(target_specs: int) -> dict[str, int]:
    # Weights deliberately sum to 0.88: the 0.12 comparison family is not yet executable.
    # Renormalise over the implemented types so the missing share is distributed
    # proportionally instead of being dumped onto factoid (which previously inflated
    # factoid from its 25% target to ~37% of the pool).
    weights = {
        "factoid": 0.25,
        "aggregation_count": 0.15,
        "aggregation_sum": 0.15,
        "conjunction_constraint": 0.20,
        "temporal": 0.08,
        "categorical_cpv": 0.05,
    }
    total_weight = sum(weights.values())
    exact = {name: target_specs * weight / total_weight for name, weight in weights.items()}
    budgets = {name: max(1, int(value)) for name, value in exact.items()}

    # Largest-remainder apportionment for the leftover after flooring, so the realised
    # distribution tracks the (renormalised) targets rather than favouring one bucket.
    remainder = target_specs - sum(budgets.values())
    if remainder > 0:
        order = sorted(weights, key=lambda name: exact[name] - int(exact[name]), reverse=True)
        for name in order[:remainder]:
            budgets[name] += 1
    return budgets


__all__ = ["SamplerConfig", "sample_answer_specs"]
