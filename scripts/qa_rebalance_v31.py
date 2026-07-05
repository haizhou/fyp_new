#!/usr/bin/env python3
"""v3.1 rebalance: fix the bucket allocation defects found in the 2026-07-04 matrix review.

- train count bucket 35.9% -> capped at TRAIN_COUNT_CAP (family caps alone failed: count has 8
  families, 8 x 350 = 2,561);
- final_test comparison (105) / boolean (53) / top_k supply starvation -> whole-plan donations
  from train;
- bridge_join / factoid train material restored from surplus (the hard high-entropy buckets SFT
  needs most were sitting in the overflow pool);
- dev_select scarce floors (top_k was 0).

All moves are whole-plan; the uniqueness / conservation / no-cross-split gates from curation v3
are enforced at the end. Scarce-fill rows are merged in, so downstream consumers use ONE file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TRAIN_COUNT_CAP = 1200
TEST_TARGETS = {"comparison": 255, "boolean": 120, "top_k": 50}
SURPLUS_RESTORE = {"bridge_join": 10_000, "factoid": 400}  # row budgets (bridge: all available)
DEV_SELECT_FLOOR = {"top_k": 10, "min_max": 10, "set": 10}
EVAL_SPLITS = {"final_test", "dev_select", "dev_tune", "dev_smoke"}


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n",
                    encoding="utf-8")


def hkey(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    m: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        m[str(r.get("train_bucket"))][str(r.get("split"))] += 1
    return {b: dict(c) for b, c in m.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", type=Path, default=Path("data/qa/cicada_core_v3/all.jsonl"))
    ap.add_argument("--scarce", type=Path, default=Path("data/qa/scarce_fill_v4.jsonl"))
    ap.add_argument("--surplus", type=Path, default=Path("data/qa/cicada_core_v3/surplus.jsonl"))
    args = ap.parse_args()

    base = load(args.core) + load(args.scarce)
    surplus = load(args.surplus)
    n_total = len(base) + len(surplus)
    report: dict[str, Any] = {"before": matrix(base)}

    plans: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in base:
        plans[str(r.get("plan_id"))].append(r)

    def plan_all_train(pid: str) -> bool:
        return all(str(g.get("split")) == "train" for g in plans[pid])

    def move_plan(pid: str, new_split: str, tag: str) -> int:
        n = 0
        for g in plans[pid]:
            g.setdefault("metadata", {})["reassigned"] = tag
            g["split"] = new_split
            n += 1
        return n

    # ---- 1. train -> final_test donations ----------------------------------------------
    donations = Counter()
    for bucket, target in TEST_TARGETS.items():
        have = sum(1 for r in base if str(r.get("train_bucket")) == bucket
                   and r.get("split") == "final_test")
        donors = sorted((pid for pid, group in plans.items()
                         if plan_all_train(pid)
                         and all(str(g.get("train_bucket")) == bucket for g in group)),
                        key=hkey)
        for pid in donors:
            if have >= target:
                break
            have += move_plan(pid, "final_test", "train->final_test(rebalance_v31)")
            donations[bucket] = have
    report["test_donations_reached"] = dict(donations)

    # ---- 2. surplus -> train restores (plans with NO sibling anywhere in base) ----------
    base_plan_ids = set(plans)
    surplus_by_plan: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in surplus:
        surplus_by_plan[str(r.get("plan_id"))].append(r)
    restored = Counter()
    kept_surplus: list[dict[str, Any]] = []
    for pid, group in sorted(surplus_by_plan.items(), key=lambda kv: hkey(kv[0])):
        bucket = str(group[0].get("train_bucket"))
        budget = SURPLUS_RESTORE.get(bucket)
        # never restore a plan whose siblings live in an eval split of base
        eval_conflict = pid in base_plan_ids and any(
            str(g.get("split")) in EVAL_SPLITS for g in plans[pid])
        if budget is None or restored[bucket] + len(group) > budget or eval_conflict:
            kept_surplus.extend(group)
            continue
        for g in group:
            g.setdefault("metadata", {})["reassigned"] = "surplus->train(rebalance_v31)"
            g["split"] = "train"
            base.append(g)
            plans[pid].append(g)
            restored[bucket] += 1
    report["surplus_restored"] = dict(restored)

    # ---- 3. train count-bucket cap (round-robin trim across families) -------------------
    count_train = [r for r in base if str(r.get("train_bucket")) == "count"
                   and r.get("split") == "train"]
    excess = len(count_train) - TRAIN_COUNT_CAP
    trimmed = 0
    if excess > 0:
        by_fam: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in count_train:
            by_fam[str(r.get("template_family"))].append(r)
        for fam_rows in by_fam.values():
            fam_rows.sort(key=lambda r: hkey(str(r.get("id"))), reverse=True)
        # trim plan-wise, largest families first, cycling
        fams = sorted(by_fam, key=lambda f: -len(by_fam[f]))
        while trimmed < excess:
            progressed = False
            for f in fams:
                if trimmed >= excess or not by_fam[f]:
                    continue
                victim = by_fam[f][0]
                pid = str(victim.get("plan_id"))
                if not plan_all_train(pid):
                    by_fam[f].pop(0)
                    continue
                group = plans[pid]
                for g in group:
                    g.setdefault("metadata", {})["curation_surplus_from"] = "train(rebalance_v31)"
                    base.remove(g)
                    kept_surplus.append(g)
                    trimmed += 1
                for fam_rows in by_fam.values():
                    fam_rows[:] = [x for x in fam_rows if str(x.get("plan_id")) != pid]
                plans[pid] = []
                progressed = True
            if not progressed:
                break
    report["train_count_trimmed"] = trimmed

    # ---- 4. dev_select scarce floors ----------------------------------------------------
    ds_filled = Counter()
    for bucket, floor in DEV_SELECT_FLOOR.items():
        have = sum(1 for r in base if str(r.get("train_bucket")) == bucket
                   and r.get("split") == "dev_select")
        donors = sorted((pid for pid, group in plans.items()
                         if group and plan_all_train(pid)
                         and all(str(g.get("train_bucket")) == bucket for g in group)),
                        key=hkey)
        for pid in donors:
            if have >= floor:
                break
            have += move_plan(pid, "dev_select", "train->dev_select(rebalance_v31)")
            ds_filled[bucket] = have
    report["dev_select_floors"] = dict(ds_filled)

    # ---- gates ---------------------------------------------------------------------------
    ids = [str(r.get("id")) for r in base] + [str(r.get("id")) for r in kept_surplus]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert len(ids) == n_total, f"conservation violated: {len(ids)} != {n_total}"
    plan_splits: dict[str, set] = defaultdict(set)
    for r in base:
        plan_splits[str(r.get("plan_id"))].add(str(r.get("split")))
    bad = [p for p, s in plan_splits.items()
           if "train" in s and (s & (EVAL_SPLITS))]
    assert not bad, f"plan/split integrity violated: {bad[:5]}"

    dump(args.core, base)
    dump(args.surplus, kept_surplus)
    for split in sorted({str(r.get("split")) for r in base}):
        dump(args.core.parent / f"{split}.jsonl",
             [r for r in base if str(r.get("split")) == split])
    report["after"] = matrix(base)
    report["surplus_rows"] = len(kept_surplus)
    (args.core.parent / "rebalance_v31_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    after_train = sum(v.get("train", 0) for v in report["after"].values())
    after_test = sum(v.get("final_test", 0) for v in report["after"].values())
    print(json.dumps({"train": after_train, "final_test": after_test,
                      "test_donations": report["test_donations_reached"],
                      "surplus_restored": report["surplus_restored"],
                      "count_trimmed": trimmed,
                      "dev_select_floors": report["dev_select_floors"],
                      "surplus_rows": len(kept_surplus)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
