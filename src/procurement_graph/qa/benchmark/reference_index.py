"""Independent re-derivation of contract evidence sets from source KG tables.

Gate A completeness is only meaningful if the evidence set is verified through a code
path that is *independent* of the one that produced it. The sampler builds
``sampled_evidence_ids`` from :class:`ParquetKGQueryBackend`'s flattened ``records_df``;
this module rebuilds the matching contract set straight from the source
``contract_nodes`` and edge parquet tables, with an independent matcher and an independent
``supplier_count`` / ``buyer_count`` recomputation.

If the two paths disagree, the flatten/merge in the backend dropped, duplicated, or
miscounted contracts -- exactly the residual risk after the design dropped local
traversal in favour of full-graph queries. On correct data the two paths agree exactly,
because every scalar constraint field is read from the same ``contract_nodes`` column and
the count recomputation mirrors the backend's distinct-canonical-id semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .models import Constraint


class UnsupportedReferenceOp(ValueError):
    """Raised when a constraint op cannot be independently verified by this index."""


@dataclass
class ReferenceKGIndex:
    contracts: pd.DataFrame
    id_field: str = "contract_node_id"

    @classmethod
    def from_directory(cls, kg_dir: Path | str) -> "ReferenceKGIndex":
        kg_dir = Path(kg_dir)
        nodes = kg_dir / "nodes"
        edges = kg_dir / "edges"
        contracts = pd.read_parquet(nodes / "contract_nodes.parquet").copy()
        ids = contracts["contract_node_id"]
        contracts["supplier_count"] = _distinct_edge_counts(edges / "supplier_of.parquet", ids)
        contracts["buyer_count"] = _distinct_edge_counts(edges / "buyer_of.parquet", ids)
        return cls(contracts=contracts)

    def matching_ids(self, constraints: tuple[Constraint, ...]) -> set[str]:
        """Independent full-graph re-derivation of the matching contract id set.

        Raises :class:`UnsupportedReferenceOp` if a constraint cannot be verified here,
        so the caller can flag the spec as unverifiable rather than silently trust it.
        """
        df = self.contracts
        mask = pd.Series(True, index=df.index)
        for constraint in constraints:
            if constraint.field not in df.columns:
                return set()
            mask &= _match(df[constraint.field], constraint)
            if not bool(mask.any()):
                return set()
        return {str(value) for value in df.loc[mask, self.id_field]}


def _distinct_edge_counts(path: Path, contract_ids: pd.Series) -> pd.Series:
    edges = pd.read_parquet(path, columns=["contract_node_id", "canonical_id"])
    canonical = edges["canonical_id"].astype(str).str.strip()
    edges = edges[canonical.ne("") & canonical.ne("nan")]
    counts = edges.groupby("contract_node_id")["canonical_id"].nunique()
    return contract_ids.map(counts).fillna(0).astype("int64").to_numpy()


def _match(series: pd.Series, constraint: Constraint) -> pd.Series:
    op = constraint.op
    target = constraint.value
    if op == "eq":
        result = series == target
    elif op == "gte":
        result = pd.to_numeric(series, errors="coerce") >= target
    elif op == "lte":
        result = pd.to_numeric(series, errors="coerce") <= target
    elif op == "in":
        result = series.isin(set(target or []))
    elif op == "exists":
        result = series.notna() & series.astype(str).str.strip().ne("")
    else:
        raise UnsupportedReferenceOp(f"op {op!r} on field {constraint.field!r}")
    return result.fillna(False).astype(bool)


__all__ = ["ReferenceKGIndex", "UnsupportedReferenceOp"]
