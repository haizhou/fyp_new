#!/usr/bin/env python3
"""Reconcile bridge_join oracles with the shared executor -- VECTORIZED.

The v2 bridge oracles used raw pandas (sums over non-additive rows; hop-1 via `df[anchor==X]` which
differs from `backend.query(anchor eq X)`). This recomputes each oracle to equal what the runtime
decomposition produces, but fast: hop-1 uses the backend (cheap, executor-consistent); hop-2 is a
vectorized pandas isin+aggregate that replicates `execute_query_spec` (dedupe by contract_node_id;
sum only over value_is_additive rows). `--verify N` cross-checks the first N against
`execute_decomposition` to prove the replication before writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
from procurement_graph.reasoning.models import QueryConstraint as QC, RuntimeQuerySpec
from procurement_graph.reasoning.decomposition import Binding, SubQuery, DecompositionPlan, execute_decomposition
from procurement_graph.reasoning.verifier import backend_fields

TARGET = ROOT / "data" / "qa" / "targeted_v2" / "full2k" / "bridge_join.full2k.accepted.jsonl"
KEYFIELD = {"buyer": "buyer_name", "supplier": "supplier_name", "cpv": "tender_cpv_id",
            "category": "tender_category", "year": "release_year"}


def _truthy(series):
    return series.map(lambda v: str(v).strip().casefold() in {"true", "1", "yes", "y"} if not isinstance(v, bool) else v)


def vectorized_oracle(row, backend, df, additive_mask):
    isq = next((c for c in row["constraints"] if c["op"] == "in_subquery"), None)
    if isq is None or row["answer_operation"] not in {"count", "sum"}:
        return None, None
    bind_field = isq["field"]
    anchors = tuple(QC(KEYFIELD[k], "eq", v) for k, v in isq["value"].items() if k in KEYFIELD)
    members = {r.get(bind_field) for r in backend.query(anchors) if r.get(bind_field) not in (None, "")}
    if not members:
        return None, None
    mask = df[bind_field].isin(members)
    for c in row["constraints"]:
        if c["op"] == "eq" and c["field"] in df.columns:
            mask &= df[c["field"]].astype(type(c["value"])) == c["value"] if df[c["field"]].dtype == object else df[c["field"]] == c["value"]
    if row["answer_operation"] == "sum":
        sub = df[mask & additive_mask & df["_val"].notna()].drop_duplicates("contract_node_id")
        return f"{sub['_val'].sum():.2f}", list(sub["contract_node_id"].astype(str))[:50]
    sub = df[mask].drop_duplicates("contract_node_id")
    return int(len(sub)), list(sub["contract_node_id"].astype(str))[:50]


def _decomp_plan(row):
    isq = next(c for c in row["constraints"] if c["op"] == "in_subquery")
    bind = isq["field"]
    anchors = tuple((KEYFIELD[k], "eq", v) for k, v in isq["value"].items() if k in KEYFIELD)
    base = tuple((c["field"], c["op"], c["value"]) for c in row["constraints"] if c["op"] != "in_subquery")

    def spec(op, cons, af="", vt="string"):
        return RuntimeQuerySpec(spec_id="s", question="q", intent="x", constraints=tuple(QC(*c) for c in cons),
                                answer_operation=op, answer_field=af, answer_value_type=vt, requires_exhaustive_retrieval=True)
    h1 = SubQuery("h1", spec("count", anchors), kind="entity_set", emit_field=bind)
    ans = spec("sum", base, "value_amount", "currency") if row["answer_operation"] == "sum" else spec("count", base)
    return DecompositionPlan(row["id"], "q", (h1, SubQuery("ans", ans, binds=(Binding("h1", bind),))), final_steps=("ans",))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", type=int, default=30)
    args = ap.parse_args()
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    allowed = frozenset(backend_fields(backend))
    df = backend._backend.records_df.copy()
    df["_val"] = pd.to_numeric(df.get("value_amount"), errors="coerce")
    additive_mask = _truthy(df["value_is_additive"]) if "value_is_additive" in df.columns else pd.Series(True, index=df.index)
    rows = [json.loads(l) for l in TARGET.read_text(encoding="utf-8").splitlines() if l.strip()]

    # verify vectorized == execute_decomposition on the SMALLEST bridges (fast, still proves the
    # replication across count + sum families).
    checked = mismatched = 0
    sample = sorted((r for r in rows if r["answer_operation"] in {"count", "sum"}),
                    key=lambda r: int(r.get("evidence_count") or 0))[: args.verify]
    for row in sample:
        vec, _ = vectorized_oracle(row, backend, df, additive_mask)
        dec = execute_decomposition(backend, _decomp_plan(row), allowed_fields=allowed)
        if vec is None or not dec.passed:
            continue
        checked += 1
        if row["answer_operation"] == "sum":
            agree = abs(float(vec) - float(dec.answer)) <= max(0.01, abs(float(dec.answer)) * 1e-9)
        else:
            agree = int(vec) == int(dec.answer)
        if not agree:
            mismatched += 1
            print(f"  MISMATCH {row['id']}: vectorized={vec} decomposition={dec.answer}")
    print(f"[verify] vectorized vs execute_decomposition: {checked - mismatched}/{checked} agree")
    if mismatched:
        print("ABORT: vectorized replication disagrees with the executor; not writing."); return

    changed = 0
    for row in rows:
        oracle, ev = vectorized_oracle(row, backend, df, additive_mask)
        if oracle is None:
            continue
        if str(oracle) != str(row["oracle_answer"]):
            changed += 1
            row["oracle_answer"] = oracle
            row["evidence_ids"] = ev
            row["evidence_count"] = len(ev)
            row["generation_notes"] = (row.get("generation_notes", "") + " | oracle reconciled to shared executor (vectorized)").strip(" |")
    TARGET.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"bridge rows: {len(rows)} | reconciled (changed): {changed} | oracle now == shared executor")


if __name__ == "__main__":
    main()
