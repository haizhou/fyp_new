"""Adapter from the frozen KG query interface to the runtime `RuntimeQueryBackend`.

Reuses `ParquetKGQueryBackend` (the KG *query* interface over `data/kg/`), not the benchmark
*generator*, so the runtime does not depend on the sampler/Stage-1/Stage-2 code. It translates
runtime `QueryConstraint`s into the KG backend's `Constraint`s (splitting `between` into
`gte`/`lte`) and exposes `fields()` for schema preflight. Import this module only when a real KG
is needed; the rest of the reasoning package stays free of pandas/KG dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..qa.benchmark.kg_interface import ParquetKGQueryBackend
from ..qa.benchmark.models import Constraint as BenchConstraint
from .linking import OrgResolver  # noqa: F401  (re-exported concept)
from .models import EntityLinkCandidate, QueryConstraint
from .retrieval import LexicalTopKCandidateRetriever

# KG columns stored as full ISO timestamps ("2021-09-28T00:00:00+01:00"). A planner naturally emits
# a date-only value ("2021-09-28"), which can never `eq`-match the timestamp string. For these
# fields a date-only eq is matched by day via `contains` (the column is a pure timestamp, so the
# date substring is unambiguous).
_TIMESTAMP_FIELDS: frozenset[str] = frozenset(
    {
        "award_date_signed",
        "release_date",
        "award_period_start",
        "award_period_end",
        "contract_period_start",
        "contract_period_end",
        "tender_period_end",
    }
)
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class RuntimeKGBackend:
    """RuntimeQueryBackend backed by the deterministic KG v0.1 parquet outputs."""

    def __init__(self, parquet_backend: ParquetKGQueryBackend) -> None:
        self._backend = parquet_backend
        self._columns = {str(column) for column in parquet_backend.records_df.columns}
        self._org_resolver: RecordsOrgResolver | None = None

    @classmethod
    def from_directory(cls, kg_dir: Path | str, *, include_evidence: bool = False) -> "RuntimeKGBackend":
        return cls(ParquetKGQueryBackend.from_directory(kg_dir, include_evidence=include_evidence))

    def fields(self) -> set[str]:
        return set(self._columns)

    def record_id(self, record: dict[str, Any]) -> str:
        return self._backend.record_id(record)

    def query(self, constraints: tuple[QueryConstraint, ...]) -> list[dict[str, Any]]:
        return self._backend.query(self._translate_all(constraints))

    def count(self, constraints: tuple[QueryConstraint, ...], *, dedupe_field: str = "contract_node_id") -> int:
        """Exact count without materialising rows (delegates to the vectorised backend)."""
        return self._backend.count(self._translate_all(constraints), dedupe_field=dedupe_field)

    def sample(self, constraints: tuple[QueryConstraint, ...], n: int) -> list[dict[str, Any]]:
        """First n matching rows (for capped evidence)."""
        return self._backend.sample(self._translate_all(constraints), n)

    def project(self, constraints: tuple[QueryConstraint, ...], fields: list[str]) -> list[dict[str, Any]]:
        """Matching rows with only `fields` (cheap aggregation over a large match set)."""
        return self._backend.project(self._translate_all(constraints), fields)

    def distinct(self, constraints: tuple[QueryConstraint, ...], field: str) -> tuple[Any, ...]:
        """Distinct non-empty values for a single field."""
        distinct = getattr(self._backend, "distinct", None)
        if distinct is not None:
            return distinct(self._translate_all(constraints), field)
        rows = self.project(constraints, [field])
        return tuple(sorted({row.get(field) for row in rows if row.get(field) not in (None, "")}, key=str))

    def top_k(self, constraints: tuple[QueryConstraint, ...], *, group_by: str, k: int,
              metric: str = "count", metric_field: str = "value_amount",
              dedupe_field: str = "contract_node_id") -> dict[str, Any]:
        top_k = getattr(self._backend, "top_k", None)
        if top_k is None:
            return {"passed": False, "reason": "backend_top_k_unavailable", "answer": [], "groups": 0}
        return top_k(self._translate_all(constraints), group_by=group_by, k=k, metric=metric,
                     metric_field=metric_field, dedupe_field=dedupe_field)

    def _translate_all(self, constraints: tuple[QueryConstraint, ...]) -> tuple[BenchConstraint, ...]:
        translated: list[BenchConstraint] = []
        for constraint in constraints:
            translated.extend(_translate(constraint))
        return tuple(translated)

    def org_resolver(self) -> "RecordsOrgResolver":
        if self._org_resolver is None:
            self._org_resolver = RecordsOrgResolver(self._backend)
        return self._org_resolver

    def candidate_retriever(self, fields: tuple[str, ...] | None = None) -> LexicalTopKCandidateRetriever:
        """Build a local top-k retriever over text fields.

        This is a no-dependency baseline for the embedding-top-k interface. A
        vector retriever can replace it while keeping the same pipeline hook.
        """
        fields = fields or ("buyer_name", "supplier_name", "tender_title", "award_title", "tender_cpv_description")
        values: dict[str, tuple[str, ...]] = {}
        df = self._backend.records_df
        for field in fields:
            if field in df.columns:
                values[field] = tuple(sorted({str(value) for value in df[field].dropna().unique() if str(value).strip()}))
        return LexicalTopKCandidateRetriever(values)


class RecordsOrgResolver:
    """Case-insensitive organisation name resolver over the flattened contract records.

    The flattened backend filters buyer/supplier by name, so the canonical value IS the name;
    this resolver validates a mention against known names and reports ambiguity. A richer
    resolver over `org_nodes` (canonical_id + aliases) can replace this later.
    """

    def __init__(self, parquet_backend: ParquetKGQueryBackend) -> None:
        df = parquet_backend.records_df
        # The KG stores org names unnormalised, so one real-world org can appear under several case
        # variants ("The Newcastle Upon Tyne ..." vs "... upon ..."), each covering different rows.
        # Keep every variant with its row count: resolve() prefers the mention's exact surface form
        # (benchmark oracles anchor on it), else the variant covering the most rows.
        self._variants: dict[str, dict[str, int]] = {}
        for column in ("buyer_name", "supplier_name"):
            if column in df.columns:
                for name, count in df[column].dropna().astype(str).value_counts().items():
                    key = name.strip().casefold()
                    if key:
                        bucket = self._variants.setdefault(key, {})
                        bucket[name] = bucket.get(name, 0) + int(count)
        self._names: dict[str, str] = {key: max(bucket, key=bucket.get)
                                       for key, bucket in self._variants.items()}

    def resolve(self, mention: str) -> list[EntityLinkCandidate]:
        surface = str(mention).strip()
        key = surface.casefold()
        if not key:
            return []
        bucket = self._variants.get(key)
        if bucket is not None:
            exact = surface if surface in bucket else self._names[key]
            return [EntityLinkCandidate(mention=mention, linked_id=exact, linked_label=exact,
                                        entity_type="organization", score=1.0, source="records_exact")]
        matches = [name for lowered, name in self._names.items() if key in lowered]
        matches.sort(key=len)
        if len(matches) == 1 and len(key) >= 8:
            # the mention is a distinctive fragment of exactly one known org (e.g. a name cut at
            # punctuation) — uniqueness is strong evidence even when the length ratio is small.
            return [EntityLinkCandidate(mention=mention, linked_id=matches[0], linked_label=matches[0],
                                        entity_type="organization", score=0.85,
                                        source="records_substring_unique")]
        return [
            EntityLinkCandidate(mention=mention, linked_id=name, linked_label=name,
                                entity_type="organization", score=round(len(key) / max(len(name), 1), 3),
                                source="records_substring")
            for name in matches[:5]
        ]


def _translate(constraint: QueryConstraint) -> list[BenchConstraint]:
    if constraint.op == "between":
        values = list(constraint.value or [])
        if len(values) == 2:
            return [BenchConstraint(constraint.field, "gte", values[0]),
                    BenchConstraint(constraint.field, "lte", values[1])]
        return []
    if (
        constraint.op == "eq"
        and constraint.field in _TIMESTAMP_FIELDS
        and isinstance(constraint.value, str)
        and _DATE_ONLY.match(constraint.value)
    ):
        # date-only value against an ISO-timestamp column -> match by day
        return [BenchConstraint(constraint.field, "contains", constraint.value)]
    return [BenchConstraint(constraint.field, constraint.op, constraint.value)]


__all__ = ["RuntimeKGBackend", "RecordsOrgResolver"]
