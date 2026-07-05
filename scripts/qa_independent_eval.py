#!/usr/bin/env python3
"""Independent (second-implementation) oracle evaluator for the CICADA QA set.

Pure pandas over the RAW node/edge parquet tables — deliberately shares no code with
`kg_interface`/`executor`. Its purpose is dual verification: the benchmark oracles were produced
by the generator/executor family, so "correct" is circular until an independent implementation
recomputes the same answers from the raw tables (worklog 2026-07-04: the 0017 notice-vs-contract
count-unit drift slipped through exactly this gap).

Usage:
    python scripts/qa_independent_eval.py --qa data/qa/cicada_merged_l1_l2_trainbalanced_v1/all.jsonl \
        --kg data/kg --out data/qa/audit_dual_eval_v1
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

# families whose gold_plan does NOT serialize every parameter (thresholds/sides live only in the
# question surface) — evaluated with surface-parse assistance and reported separately.
UNDERSPECIFIED_FAMILIES = {"date_relation", "boolean_field_equality", "comparison",
                           "numeric_threshold", "supplier_set_compare"}


def load_frames_flat(kg_dir: Path) -> dict[str, pd.DataFrame]:
    """Documented dataset convention: FLAT records (one row per contract, first-party names).
    Org matching via the flat columns; op implementations remain this module's own."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend
    flat = ParquetKGQueryBackend.from_directory(kg_dir, include_evidence=False).records_df
    buyer = flat[["contract_node_id", "buyer_name"]].dropna().drop_duplicates()
    buyer = buyer[buyer["buyer_name"].astype(str).str.strip() != ""]
    supplier = flat[["contract_node_id", "supplier_name"]].dropna().drop_duplicates()
    supplier = supplier[supplier["supplier_name"].astype(str).str.strip() != ""]
    return {"contracts": flat, "buyer": buyer, "supplier": supplier}


def load_frames(kg_dir: Path) -> dict[str, pd.DataFrame]:
    contracts = pd.read_parquet(kg_dir / "nodes" / "contract_nodes.parquet")
    orgs = pd.read_parquet(kg_dir / "nodes" / "org_nodes.parquet",
                           columns=["canonical_id", "canonical_name"])
    buyer = pd.read_parquet(kg_dir / "edges" / "buyer_of.parquet",
                            columns=["canonical_id", "contract_node_id"])
    supplier = pd.read_parquet(kg_dir / "edges" / "supplier_of.parquet",
                               columns=["canonical_id", "contract_node_id"])
    buyer = buyer.merge(orgs, on="canonical_id", how="left").rename(columns={"canonical_name": "buyer_name"})
    supplier = supplier.merge(orgs, on="canonical_id", how="left").rename(columns={"canonical_name": "supplier_name"})
    buyer = buyer[["contract_node_id", "buyer_name"]].drop_duplicates()
    buyer = buyer[buyer["buyer_name"].astype(str).str.strip() != ""]
    supplier = supplier[["contract_node_id", "supplier_name"]].drop_duplicates()
    supplier = supplier[supplier["supplier_name"].astype(str).str.strip() != ""]
    return {"contracts": contracts, "buyer": buyer, "supplier": supplier}


class IndependentEvaluator:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.contracts = frames["contracts"]
        self.buyer = frames["buyer"]
        self.supplier = frames["supplier"]

    # ---- constraint application -------------------------------------------------------
    def _ids_for_org(self, kind: str, name: Any) -> set[str]:
        frame = self.buyer if kind == "buyer" else self.supplier
        col = f"{kind}_name"
        names = [str(v) for v in (name if isinstance(name, (list, tuple)) else [name])]
        return set(frame[frame[col].isin(names)]["contract_node_id"])

    def _resolve_subquery(self, value: dict[str, Any]) -> set[str] | None:
        kind = str(value.get("resolve") or "")
        if kind == "suppliers_of_buyer":
            ids = self._ids_for_org("buyer", value.get("buyer"))
            names = set(self.supplier[self.supplier["contract_node_id"].isin(ids)]["supplier_name"].dropna())
            return set(self.supplier[self.supplier["supplier_name"].isin(names)]["contract_node_id"])
        if kind == "buyers_of_supplier":
            ids = self._ids_for_org("supplier", value.get("supplier"))
            names = set(self.buyer[self.buyer["contract_node_id"].isin(ids)]["buyer_name"].dropna())
            return set(self.buyer[self.buyer["buyer_name"].isin(names)]["contract_node_id"])
        if kind == "cpvs_of_buyer":
            ids = self._ids_for_org("buyer", value.get("buyer"))
            cpvs = set(self.contracts[self.contracts["contract_node_id"].isin(ids)]["tender_cpv_id"].dropna())
            cpvs.discard("")
            return set(self.contracts[self.contracts["tender_cpv_id"].isin(cpvs)]["contract_node_id"])
        if kind == "buyers_in_category":
            cat_ids = set(self.contracts[self.contracts["tender_category"] ==
                                         str(value.get("category"))]["contract_node_id"])
            names = set(self.buyer[self.buyer["contract_node_id"].isin(cat_ids)]["buyer_name"].dropna())
            return set(self.buyer[self.buyer["buyer_name"].isin(names)]["contract_node_id"])
        if kind == "suppliers_in_year_category":
            mask = pd.Series(True, index=self.contracts.index)
            if value.get("year") is not None:
                mask &= self.contracts["release_year"] == int(value["year"])
            if value.get("category"):
                mask &= self.contracts["tender_category"] == str(value["category"])
            ids = set(self.contracts[mask]["contract_node_id"])
            names = set(self.supplier[self.supplier["contract_node_id"].isin(ids)]["supplier_name"].dropna())
            return set(self.supplier[self.supplier["supplier_name"].isin(names)]["contract_node_id"])
        if kind == "suppliers_for_cpv":
            ids = set(self.contracts[self.contracts["tender_cpv_id"] ==
                                     str(value.get("cpv"))]["contract_node_id"])
            names = set(self.supplier[self.supplier["contract_node_id"].isin(ids)]["supplier_name"].dropna())
            return set(self.supplier[self.supplier["supplier_name"].isin(names)]["contract_node_id"])
        return None

    def match(self, constraints: list[dict[str, Any]]) -> pd.DataFrame | None:
        df = self.contracts
        mask = pd.Series(True, index=df.index)
        for c in constraints or []:
            field, op, value = c.get("field"), str(c.get("op") or "eq"), c.get("value")
            if op == "in_subquery":
                ids = self._resolve_subquery(value if isinstance(value, dict) else {})
                if ids is None:
                    return None
                mask &= df["contract_node_id"].isin(ids)
            elif field in ("buyer_name", "supplier_name"):
                kind = "buyer" if field == "buyer_name" else "supplier"
                mask &= df["contract_node_id"].isin(self._ids_for_org(kind, value))
            elif field == "release_year":
                mask &= df["release_year"] == int(value)
            elif field == "value_is_additive":
                mask &= df["value_is_additive"] == bool(value)
            elif field == "has_award_signed_date":
                mask &= df["has_award_signed_date"] == bool(value)
            elif field in df.columns:
                mask &= df[field].astype(str) == str(value)
            else:
                return None
        return df[mask]

    # ---- answer operations ------------------------------------------------------------
    def evaluate(self, row: dict[str, Any]) -> tuple[Any, str]:
        """Returns (answer, note). note != '' marks partial/heuristic evaluation."""
        gp = row.get("gold_plan") or {}
        op = str(gp.get("answer_operation") or "")
        fam = str(row.get("template_family") or "")
        matched = self.match(gp.get("constraints") or [])
        if matched is None:
            return None, "unsupported_constraint"
        if op == "count":
            return int(matched["contract_node_id"].nunique()), ""
        if op == "sum":
            vals = pd.to_numeric(matched["value_amount"], errors="coerce").dropna()
            return float(round(vals.sum(), 2)), ""
        if op == "exists":
            return bool(len(matched) > 0), ""
        if op == "select_unique":
            field = str(gp.get("answer_field") or "")
            if field in ("buyer_name", "supplier_name"):
                kind = "buyer" if field == "buyer_name" else "supplier"
                frame = self.buyer if kind == "buyer" else self.supplier
                names = frame[frame["contract_node_id"].isin(matched["contract_node_id"])][field].dropna().unique()
                return (str(names[0]) if len(names) == 1 else None), ("" if len(names) == 1 else "not_unique")
            vals = matched[field].dropna().astype(str).unique() if field in matched.columns else []
            return (str(vals[0]) if len(vals) == 1 else None), ("" if len(vals) == 1 else "not_unique")
        if op == "distinct_set":
            oracle = row.get("oracle_answer")
            for field, frame in (("supplier_name", self.supplier), ("buyer_name", self.buyer)):
                pool = frame[frame["contract_node_id"].isin(matched["contract_node_id"])][field].dropna()
                names = sorted(v for v in pool.astype(str).unique() if v.strip())
                if isinstance(oracle, list) and sorted(map(str, oracle)) == list(map(str, names)):
                    return names, f"field_inferred:{field}"
            cpvs = sorted(matched["tender_cpv_id"].dropna().astype(str).unique())
            return cpvs, "field_inferred:tender_cpv_id"
        if op in ("argmin", "argmax"):
            vals = matched.assign(_v=pd.to_numeric(matched["value_amount"], errors="coerce")).dropna(subset=["_v"])
            nonzero = vals[vals["_v"] > 0]
            pick = nonzero if len(nonzero) else vals
            if not len(pick):
                return None, "empty"
            idx = pick["_v"].idxmin() if op == "argmin" else pick["_v"].idxmax()
            return str(pick.loc[idx, "contract_node_id"]), "nonzero_preferred"
        if op == "rank_top_k":
            joined = self.buyer[self.buyer["contract_node_id"].isin(matched["contract_node_id"])]
            top = joined.groupby("buyer_name")["contract_node_id"].nunique().sort_values(ascending=False)
            return [str(n) for n in top.head(3).index], "top3_buyers_by_count"
        if op == "compare":
            return self._evaluate_compare(row, gp, fam, matched)
        return None, f"unsupported_op:{op}"

    def _evaluate_compare(self, row, gp, fam, matched) -> tuple[Any, str]:
        question = str(row.get("question") or "")
        if fam == "numeric_threshold":
            vals = pd.to_numeric(matched["value_amount"], errors="coerce").dropna()
            m = re.search(r"(?:gbp|£)\s*([\d,.]+)\s*(million|billion|m|bn)?", question, re.I)
            if not m:
                return None, "threshold_unparsed"
            thr = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "").casefold()
            thr *= 1e6 if unit in ("million", "m") else 1e9 if unit in ("billion", "bn") else 1.0
            return bool(float(vals.sum()) > thr), "surface_threshold"
        if fam == "date_relation":
            dates = matched["award_date_signed"].dropna().astype(str)
            if len(dates) != 1:
                return None, "date_not_unique"
            m = re.search(r"(?:after|before)\s+(\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2})", question, re.I)
            if not m:
                return None, "date_unparsed"
            from datetime import datetime
            raw = m.group(1)
            try:
                pivot = (datetime.strptime(raw, "%d %B %Y") if "-" not in raw
                         else datetime.strptime(raw, "%Y-%m-%d")).strftime("%Y-%m-%d")
            except ValueError:
                return None, "date_unparsed"
            after = "after" in question.casefold()
            actual = str(dates.iloc[0])[:10]
            return bool(actual > pivot) if after else bool(actual < pivot), "surface_date"
        if fam == "boolean_field_equality":
            m = re.search(r"\b(goods|services|works)\b", question.casefold())
            cats = matched["tender_category"].dropna().astype(str).unique()
            if not m or len(cats) != 1:
                return None, "category_unparsed"
            return bool(cats[0] == m.group(1)), "surface_category"
        if fam in ("comparison", "supplier_set_compare"):
            oracle = row.get("oracle_answer")
            if not isinstance(oracle, dict):
                return None, "sides_unavailable"
            sides = [k for k in oracle.keys() if k != "answer"]
            if len(sides) != 2:
                return None, "sides_unavailable"
            years = re.findall(r"\b(20[2-3]\d)\b", question)
            recomputed = {}
            for side in sides:
                ids = self._ids_for_org("buyer", side)
                sub = self.contracts[self.contracts["contract_node_id"].isin(ids)]
                if years:
                    sub = sub[sub["release_year"] == int(years[0])]
                recomputed[side] = int(sub["contract_node_id"].nunique())
            if fam == "supplier_set_compare":
                return None, "family_needs_params"  # category slice per side not serialized
            answer = recomputed[sides[0]] > recomputed[sides[1]]
            return {"answer": bool(answer), **recomputed}, "sides_from_oracle_keys"
        return None, f"compare_family_unhandled:{fam}"


def answers_agree(ours: Any, oracle: Any) -> bool:
    if isinstance(oracle, dict) and isinstance(ours, dict):
        return bool(ours.get("answer")) == bool(oracle.get("answer"))
    if isinstance(oracle, bool) or isinstance(ours, bool):
        return bool(ours) == bool(oracle)
    if isinstance(oracle, list) and isinstance(ours, list):
        return sorted(map(str, ours)) == sorted(map(str, oracle))
    try:
        return abs(float(ours) - float(oracle)) < 0.01
    except (TypeError, ValueError):
        return str(ours).strip() == str(oracle).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa", type=Path, required=True)
    ap.add_argument("--kg", type=Path, default=Path("data/kg"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--convention", choices=["edge", "flat"], default="edge",
                    help="org-matching universe: edge-level any-party vs the documented flat first-party convention")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.qa.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("expected_status") == "answerable"]
    if args.limit:
        rows = rows[: args.limit]
    frames = load_frames_flat(args.kg) if args.convention == "flat" else load_frames(args.kg)
    ev = IndependentEvaluator(frames)

    args.out.mkdir(parents=True, exist_ok=True)
    mismatches, partial, unevaluated = [], Counter(), Counter()
    agree = 0
    evaluated = 0
    for r in rows:
        ours, note = ev.evaluate(r)
        fam = str(r.get("template_family"))
        if ours is None:
            unevaluated[f"{fam}:{note}"] += 1
            continue
        evaluated += 1
        if note:
            partial[note] += 1
        if answers_agree(ours, r.get("oracle_answer")):
            agree += 1
        else:
            mismatches.append({"id": r["id"], "family": fam, "bucket": r.get("train_bucket"),
                               "ours": ours if not isinstance(ours, list) else ours[:5],
                               "oracle": r.get("oracle_answer"), "note": note,
                               "question": str(r.get("question"))[:140]})
    (args.out / "mismatches.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False, default=str) for m in mismatches) + "\n", encoding="utf-8")
    summary = {
        "total_answerable": len(rows), "evaluated": evaluated, "agree": agree,
        "agree_rate": round(agree / evaluated, 4) if evaluated else None,
        "mismatches": len(mismatches),
        "mismatch_by_family": dict(Counter(m["family"] for m in mismatches).most_common()),
        "partial_eval_notes": dict(partial.most_common()),
        "unevaluated": dict(unevaluated.most_common()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
