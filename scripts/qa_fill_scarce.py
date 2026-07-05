#!/usr/bin/env python3
"""Fill scarce buckets (top_k / min_max / set) with template-generated, fully-parameterized rows.

Lessons from the 2026-07-04 dual-verification audit are baked in:
- gold plans serialize EVERY parameter (k/group_by/metric for top_k, answer_field for set,
  nonzero policy for min_max) — nothing lives only in the question surface;
- oracles come from the INDEPENDENT evaluator conventions (flat universe, additive-only money,
  empty-name filtering), so v2's 99.88% dual-agreement is preserved by construction;
- splits are assigned per plan (hash), so the train/eval integrity gate holds trivially.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from qa_independent_eval import load_frames_flat  # noqa: E402

TARGETS = {"top_k": 300, "min_max": 150, "set": 150}
SPLIT_WHEEL = ("train", "train", "train", "train", "train", "final_test", "final_test",
               "dev_tune", "dev_tune", "train")

TOPK_SURFACES = [
    "Which three buyers published the most {cat} contract notices in {year}?",
    "For {year}, list the top 3 contracting authorities by number of {cat} notices.",
    "Rank the top three buyers by {cat} notice count for {year}.",
    "In {year}, which three buyers issued the largest number of {cat} notices?",
]
MINMAX_SURFACES = {
    "argmin": [
        "Under CPV {cpv}, which contract shows the lowest non-zero value in {year}?",
        "For {year} notices under CPV {cpv}, identify the contract with the smallest non-zero value.",
    ],
    "argmax": [
        "Under CPV {cpv}, which contract shows the highest value in {year}?",
        "For {year} notices under CPV {cpv}, identify the contract with the largest recorded value.",
    ],
}
SET_SURFACES = [
    "Which distinct suppliers won contracts under CPV {cpv} in {year}?",
    "List the suppliers awarded contracts under CPV {cpv} during {year}.",
    "For {year}, which suppliers appear on notices under CPV {cpv}?",
]


def stable_pick(seq, key: str):
    return seq[int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % len(seq)]


def make_row(plan_id: str, question: str, bucket: str, qtype: str, gold: dict[str, Any],
             oracle: Any, answer_type: str) -> dict[str, Any]:
    split = stable_pick(SPLIT_WHEEL, plan_id)
    return {
        "id": f"S3::{plan_id}",
        "source": "S3", "level": 1, "source_dataset": "scarce_fill_v4",
        "plan_id": plan_id, "question": question,
        "question_type": qtype, "template_family": gold["metadata"]["template_family"],
        "answer_operation": gold["answer_operation"], "answer_field": gold.get("answer_field", ""),
        "answer_type": answer_type, "expected_status": "answerable",
        "oracle_answer": oracle, "constraints": gold["constraints"],
        "gold_plan": gold, "difficulty": "", "domain_slice": bucket,
        "hop_class": "single-hop", "generalization_class": "iid",
        "train_bucket": bucket, "split": split, "split_family": split.split("_")[0],
        "dedup_group": f"dg_{hashlib.sha1(plan_id.encode()).hexdigest()[:16]}",
        "provenance": {"generated_by": "qa_fill_scarce_v4", "oracle": "independent_evaluator_flat"},
        "metadata": {"source_subset": bucket},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kg", type=Path, default=Path("data/kg"))
    ap.add_argument("--core", type=Path, default=Path("data/qa/cicada_core_v3/all.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/qa/scarce_fill_v4b.jsonl"))
    args = ap.parse_args()

    frames = load_frames_flat(args.kg)
    flat, sup = frames["contracts"], frames["supplier"]
    flat = flat.assign(_val=pd.to_numeric(flat["value_amount"], errors="coerce"))
    existing = {json.loads(l)["plan_id"] for l in args.core.read_text(encoding="utf-8").splitlines() if l.strip()}
    rows: list[dict[str, Any]] = []

    # ---- top_k: (year x category), top-3 buyers by notice count -----------------------
    made = 0
    for year in (2022, 2023, 2024, 2025):
        for cat in ("goods", "services", "works"):
            if made >= TARGETS["top_k"]:
                break
            for shift in range(12):  # multiple rows per combo via different surface indices
                if made >= TARGETS["top_k"]:
                    break
                pid = f"topk_{year}_{cat}_{shift}"
                if pid in existing:
                    continue
                sub = flat[(flat["release_year"] == year) & (flat["tender_category"] == cat)]
                top = (sub.groupby("buyer_name")["contract_node_id"].nunique()
                       .sort_values(ascending=False))
                top = top[[bool(str(n).strip()) for n in top.index]]
                if len(top) < 5 or top.iloc[2] == top.iloc[3]:  # require a UNIQUE top-3 cut
                    break
                oracle = [str(n) for n in top.head(3).index]
                surface = TOPK_SURFACES[shift % len(TOPK_SURFACES)].format(year=year, cat=cat)
                gold = {"spec_id": pid,
                        "constraints": [{"field": "release_year", "op": "eq", "value": year},
                                        {"field": "tender_category", "op": "eq", "value": cat}],
                        "answer_operation": "rank_top_k", "answer_field": "buyer_name",
                        "answer_value_type": "top_k", "dedupe_key": "contract_node_id",
                        "metadata": {"question_type": "top_k", "template_family": "top_k_buyers",
                                     "k": 3, "group_by": "buyer_name", "metric": "count",
                                     "expected_status": "answerable"}}
                rows.append(make_row(pid, surface, "top_k", "top_k", gold, oracle, "top_k"))
                made += 1
                if shift >= len(TOPK_SURFACES) - 1:
                    break  # one row per surface variant per combo

    # ---- top_k over CPV slices (year x cpv), to reach the target ----------------------
    cpv_by_vol = flat.groupby("tender_cpv_id")["contract_node_id"].nunique().sort_values(ascending=False)
    big_cpvs = [c for c in cpv_by_vol.index if str(c).strip()][:220]
    for cpv in big_cpvs:
        if made >= TARGETS["top_k"]:
            break
        for year in (2022, 2023, 2024, 2025):
            if made >= TARGETS["top_k"]:
                break
            pid = f"topk_cpv_{cpv}_{year}"
            if pid in existing:
                continue
            sub = flat[(flat["release_year"] == year) & (flat["tender_cpv_id"] == cpv)]
            top = (sub.groupby("buyer_name")["contract_node_id"].nunique()
                   .sort_values(ascending=False))
            top = top[[bool(str(n).strip()) for n in top.index]]
            if len(top) < 5 or top.iloc[2] == top.iloc[3]:
                continue
            oracle = [str(n) for n in top.head(3).index]
            surface = stable_pick(TOPK_SURFACES, pid).replace("{cat}", f"CPV {cpv}").format(year=year)
            gold = {"spec_id": pid,
                    "constraints": [{"field": "release_year", "op": "eq", "value": year},
                                    {"field": "tender_cpv_id", "op": "eq", "value": str(cpv)}],
                    "answer_operation": "rank_top_k", "answer_field": "buyer_name",
                    "answer_value_type": "top_k", "dedupe_key": "contract_node_id",
                    "metadata": {"question_type": "top_k", "template_family": "top_k_buyers_cpv",
                                 "k": 3, "group_by": "buyer_name", "metric": "count",
                                 "expected_status": "answerable"}}
            rows.append(make_row(pid, surface, "top_k", "top_k", gold, oracle, "top_k"))
            made += 1

    # ---- min_max & set: iterate CPV x year combos --------------------------------------
    cpv_counts = flat.groupby("tender_cpv_id")["contract_node_id"].nunique().sort_values(ascending=False)
    cpvs = [c for c in cpv_counts.index if str(c).strip()][:400]
    made_mm, made_set = 0, 0
    for cpv in cpvs:
        if made_mm >= TARGETS["min_max"] and made_set >= TARGETS["set"]:
            break
        for year in (2022, 2023, 2024, 2025):
            sub = flat[(flat["tender_cpv_id"] == cpv) & (flat["release_year"] == year)]
            vals = sub[(sub["value_is_additive"] == True) & (sub["_val"] > 0)]
            # min_max: need >=3 rows and a unique extreme
            if made_mm < TARGETS["min_max"] and len(vals) >= 3:
                for op in ("argmin", "argmax"):
                    if made_mm >= TARGETS["min_max"]:
                        break
                    ordered = vals.sort_values("_val", ascending=(op == "argmin"))
                    if len(ordered) >= 2 and float(ordered["_val"].iloc[0]) == float(ordered["_val"].iloc[1]):
                        continue  # tie -> skip
                    pid = f"minmax_{op}_{cpv}_{year}"
                    if pid in existing:
                        continue
                    oracle = str(ordered["contract_node_id"].iloc[0])
                    surface = stable_pick(MINMAX_SURFACES[op], pid).format(cpv=cpv, year=year)
                    gold = {"spec_id": pid,
                            "constraints": [{"field": "tender_cpv_id", "op": "eq", "value": str(cpv)},
                                            {"field": "release_year", "op": "eq", "value": year},
                                            {"field": "value_is_additive", "op": "eq", "value": True}],
                            "answer_operation": op, "answer_field": "contract_node_id",
                            "answer_value_type": "string", "dedupe_key": "contract_node_id",
                            "metadata": {"question_type": "min_max", "template_family": f"{op}_value",
                                         "nonzero_only": True, "sort_field": "value_amount",
                                         "expected_status": "answerable"}}
                    rows.append(make_row(pid, surface, "min_max", "min_max", gold, oracle, "min_max"))
                    made_mm += 1
            # set: distinct suppliers, small clean sets
            if made_set < TARGETS["set"]:
                ids = set(sub["contract_node_id"])
                names = sorted(v for v in sup[sup["contract_node_id"].isin(ids)]["supplier_name"]
                               .astype(str).unique() if v.strip())
                if 2 <= len(names) <= 12:
                    pid = f"setlist_{cpv}_{year}"
                    if pid in existing:
                        continue
                    surface = stable_pick(SET_SURFACES, pid).format(cpv=cpv, year=year)
                    gold = {"spec_id": pid,
                            "constraints": [{"field": "tender_cpv_id", "op": "eq", "value": str(cpv)},
                                            {"field": "release_year", "op": "eq", "value": year}],
                            "answer_operation": "distinct_set", "answer_field": "supplier_name",
                            "answer_value_type": "set_list", "dedupe_key": "contract_node_id",
                            "metadata": {"question_type": "set", "template_family": "cpv_year_suppliers",
                                         "expected_status": "answerable"}}
                    rows.append(make_row(pid, surface, "set", "set", gold, oracle=names,
                                         answer_type="set_list"))
                    made_set += 1

    summary = {"generated": len(rows),
               "by_bucket": dict(Counter(r["train_bucket"] for r in rows)),
               "by_split": dict(Counter(r["split"] for r in rows))}
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
