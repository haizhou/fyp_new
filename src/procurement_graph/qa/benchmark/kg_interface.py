"""Query backend interfaces for benchmark construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import json

from .models import Constraint


class QueryBackend(Protocol):
    """Minimal full-graph query interface used by Gate A and execution."""

    def query(self, constraints: tuple[Constraint, ...]) -> list[dict[str, Any]]:
        ...

    def record_id(self, record: dict[str, Any]) -> str:
        ...


@dataclass
class TabularQueryBackend:
    """Small in-memory backend for tests and early prototyping."""

    records: list[dict[str, Any]]
    id_field: str = "record_id"

    def record_id(self, record: dict[str, Any]) -> str:
        return str(record.get(self.id_field, ""))

    def query(self, constraints: tuple[Constraint, ...]) -> list[dict[str, Any]]:
        return [record for record in self.records if all(_matches(record, constraint) for constraint in constraints)]


@dataclass
class ParquetKGQueryBackend:
    """Real KG-backed query interface with one record per contract node."""

    records_df: pd.DataFrame
    id_field: str = "contract_node_id"

    @classmethod
    def from_directory(cls, kg_dir: Path | str, *, include_evidence: bool = True) -> "ParquetKGQueryBackend":
        kg_dir = Path(kg_dir)
        nodes = kg_dir / "nodes"
        edges = kg_dir / "edges"

        contracts = pd.read_parquet(nodes / "contract_nodes.parquet")
        orgs = pd.read_parquet(
            nodes / "org_nodes.parquet",
            columns=["canonical_id", "canonical_name", "address_region", "address_regions"],
        )
        buyer_edges = pd.read_parquet(edges / "buyer_of.parquet")
        supplier_edges = pd.read_parquet(edges / "supplier_of.parquet")

        records = contracts.copy()
        buyer_summary = _party_summary(buyer_edges, orgs, "buyer")
        supplier_summary = _party_summary(supplier_edges, orgs, "supplier")
        records = records.merge(buyer_summary, on="contract_node_id", how="left").merge(
            supplier_summary,
            on="contract_node_id",
            how="left",
        )
        if include_evidence:
            evidence_edges = pd.read_parquet(edges / "evidence_for.parquet", columns=["contract_node_id", "evidence_id"])
            evidence_summary = (
                evidence_edges.groupby("contract_node_id", as_index=False)
                .agg(evidence_count=("evidence_id", "nunique"))
            )
            records = records.merge(evidence_summary, on="contract_node_id", how="left")
        else:
            records["evidence_count"] = 0
        records["evidence_count"] = pd.to_numeric(records["evidence_count"], errors="coerce").fillna(0).astype("Int64")
        for column in [
            "buyer_canonical_ids",
            "buyer_names",
            "buyer_regions",
            "supplier_canonical_ids",
            "supplier_names",
            "supplier_regions",
        ]:
            if column not in records.columns:
                records[column] = [tuple()] * len(records)
            else:
                records[column] = records[column].map(_tuple_or_empty)
        records["buyer_name"] = records["buyer_names"].map(_first_or_empty)
        records["supplier_name"] = records["supplier_names"].map(_first_or_empty)
        records["supplier_count"] = records["supplier_canonical_ids"].map(len).astype("Int64")
        records["buyer_count"] = records["buyer_canonical_ids"].map(len).astype("Int64")
        return cls(records_df=records)

    def record_id(self, record: dict[str, Any]) -> str:
        return str(record.get(self.id_field, ""))

    def _mask(self, constraints: tuple[Constraint, ...]):
        if self.records_df.empty:
            return None
        mask = pd.Series(True, index=self.records_df.index)
        for constraint in constraints:
            if constraint.field not in self.records_df.columns:
                return None
            mask &= _series_matches(self.records_df[constraint.field], constraint)
        return mask

    def query(self, constraints: tuple[Constraint, ...]) -> list[dict[str, Any]]:
        mask = self._mask(constraints)
        if mask is None or not bool(mask.any()):
            return []
        return self.records_df.loc[mask].to_dict(orient="records")

    def count(self, constraints: tuple[Constraint, ...], *, dedupe_field: str = "contract_node_id") -> int:
        """Exact match count WITHOUT materialising row dicts (vectorised)."""
        mask = self._mask(constraints)
        if mask is None or not bool(mask.any()):
            return 0
        column = dedupe_field if dedupe_field and dedupe_field in self.records_df.columns else self.id_field
        return int(self.records_df.loc[mask, column].nunique())

    def sample(self, constraints: tuple[Constraint, ...], n: int) -> list[dict[str, Any]]:
        """First `n` matching rows as dicts (for capped evidence); avoids materialising the full set."""
        mask = self._mask(constraints)
        if mask is None or not bool(mask.any()):
            return []
        return self.records_df.loc[mask].head(max(0, n)).to_dict(orient="records")

    def project(self, constraints: tuple[Constraint, ...], fields: list[str]) -> list[dict[str, Any]]:
        """Matching rows with only `fields` -- an aggregation (sum) needs a few columns, not all 42,
        so this cuts dict-materialisation cost for a large match set by an order of magnitude."""
        mask = self._mask(constraints)
        if mask is None or not bool(mask.any()):
            return []
        cols = [f for f in fields if f in self.records_df.columns]
        return self.records_df.loc[mask, cols].to_dict(orient="records")


def _matches(record: dict[str, Any], constraint: Constraint) -> bool:
    value = record.get(constraint.field)
    target = constraint.value
    if constraint.op == "eq":
        values = _as_values(value)
        return target in values if _is_multi_value(value) else value == target
    if constraint.op == "in":
        targets = set(target or [])
        values = _as_values(value)
        return any(item in targets for item in values)
    if constraint.op == "contains":
        if value is None:
            return False
        return any(str(target).casefold() in str(item).casefold() for item in _as_values(value))
    if constraint.op == "exists":
        return bool([item for item in _as_values(value) if item not in (None, "")])
    if constraint.op == "gte":
        return value is not None and value >= target
    if constraint.op == "lte":
        return value is not None and value <= target
    raise ValueError(f"Unsupported constraint op: {constraint.op}")


def _party_summary(edges: pd.DataFrame, orgs: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(
            columns=[
                "contract_node_id",
                f"{prefix}_canonical_ids",
                f"{prefix}_names",
                f"{prefix}_regions",
            ]
        )
    joined = edges[["contract_node_id", "canonical_id"]].merge(orgs, on="canonical_id", how="left")
    return (
        joined.groupby("contract_node_id", as_index=False)
        .agg(
            **{
                f"{prefix}_canonical_ids": ("canonical_id", _unique_tuple),
                f"{prefix}_names": ("canonical_name", _unique_tuple),
                f"{prefix}_regions": ("address_regions", _unique_regions_tuple),
            }
        )
    )


def _unique_tuple(values: pd.Series) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value) and str(value) != "nan"}))


def _unique_regions_tuple(values: pd.Series) -> tuple[str, ...]:
    regions: set[str] = set()
    for value in values:
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except Exception:
                parsed = [value]
            if isinstance(parsed, list):
                regions.update(str(region).strip() for region in parsed if str(region).strip())
    return tuple(sorted(regions))


def _tuple_or_empty(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return tuple()
    return (value,)


def _first_or_empty(values: Any) -> str:
    items = _tuple_or_empty(values)
    return str(items[0]) if items else ""


def _is_multi_value(value: Any) -> bool:
    return isinstance(value, (list, tuple, set))


def _as_values(value: Any) -> tuple[Any, ...]:
    if _is_multi_value(value):
        return tuple(value)
    if value is None:
        return tuple()
    return (value,)


def _series_matches(series: pd.Series, constraint: Constraint) -> pd.Series:
    target = constraint.value
    if constraint.op == "eq":
        if series.map(_is_multi_value).any():
            return series.map(lambda value: target in _as_values(value))
        return series == target
    if constraint.op == "in":
        targets = set(target or [])
        if series.map(_is_multi_value).any():
            return series.map(lambda value: any(item in targets for item in _as_values(value)))
        return series.isin(targets)
    if constraint.op == "contains":
        needle = str(target).casefold()
        return series.map(lambda value: any(needle in str(item).casefold() for item in _as_values(value)))
    if constraint.op == "exists":
        return series.map(lambda value: bool([item for item in _as_values(value) if item not in (None, "")]))
    if constraint.op == "gte":
        return pd.to_numeric(series, errors="coerce") >= target
    if constraint.op == "lte":
        return pd.to_numeric(series, errors="coerce") <= target
    raise ValueError(f"Unsupported constraint op: {constraint.op}")


__all__ = ["ParquetKGQueryBackend", "QueryBackend", "TabularQueryBackend"]
