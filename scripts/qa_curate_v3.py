#!/usr/bin/env python3
"""Curate QA v3: rebalance the oversized/skewed v2 set into a stratified core.

Problems fixed (worklog 2026-07-04 QA audit):
- final_test (10,000) was larger than train (5,534): evaluation burns 5x the needed budget while
  SFT/DPO raw material starves. New final_test is a ~2k stratified core; surplus easy iid rows
  flow BACK to train (no model has trained on any row yet, and whole dedup groups move together,
  so the one-way test->train reassignment is leakage-free).
- saturated easy buckets (count 35% of all rows) drown the hard-bucket signal: per-bucket caps
  with round-robin sampling ACROSS template families (family balance without a second knob).
- scarce buckets (top_k=15, min_max, set) can now never be dropped, and dev_tune floors are
  filled by pulling whole plans from train.
- nothing is deleted: overflow goes to surplus.jsonl (original split retained in metadata) — the
  raw-material pool for the upcoming surface-diversity pass.

dev_smoke is copied through UNTOUCHED (regression continuity).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# per-bucket caps for the final_test core; buckets absent here keep every row
FINAL_TEST_CAPS: dict[str, int] = {
    "count": 250, "sum": 200, "factoid": 250, "categorical": 120,
    "bridge_join": 350, "comparison": 300,
    "abstain_unsupported": 120, "abstain_ambiguous": 120, "abstain_no_results": 120,
}
# scarce buckets: never capped anywhere, and floor-filled in dev_tune
SCARCE = {"top_k", "min_max", "set", "boolean"}
DEV_TUNE_FLOOR = 20
TRAIN_FAMILY_CAP = 350
GENERALIZATION_PRIORITY = {"ood_candidate": 0, "compositional": 1, "iid": 2}


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n",
                    encoding="utf-8")


def stable_key(row: dict[str, Any]) -> str:
    return hashlib.sha1(str(row.get("id")).encode("utf-8")).hexdigest()


def round_robin_select(rows: list[dict[str, Any]], cap: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pick up to `cap` rows, cycling across template families; within a family, prefer harder
    generalization classes, then a stable hash (no model-performance signal — Goodhart-safe)."""
    if len(rows) <= cap:
        return rows, []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_family[str(r.get("template_family"))].append(r)
    for fam_rows in by_family.values():
        fam_rows.sort(key=lambda r: (GENERALIZATION_PRIORITY.get(str(r.get("generalization_class")), 3),
                                     stable_key(r)))
    picked: list[dict[str, Any]] = []
    families = sorted(by_family)
    idx = {f: 0 for f in families}
    while len(picked) < cap:
        progressed = False
        for f in families:
            if len(picked) >= cap:
                break
            if idx[f] < len(by_family[f]):
                picked.append(by_family[f][idx[f]])
                idx[f] += 1
                progressed = True
        if not progressed:
            break
    picked_ids = {r["id"] for r in picked}
    return picked, [r for r in rows if r["id"] not in picked_ids]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa", type=Path, default=Path("data/qa/cicada_merged_l1_l2_trainbalanced_v2/all.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/qa/cicada_core_v3"))
    args = ap.parse_args()

    rows = load(args.qa)
    report: dict[str, Any] = {"input_rows": len(rows)}
    before = Counter((str(r.get("split")), str(r.get("train_bucket"))) for r in rows)

    smoke = [r for r in rows if r.get("split") == "dev_smoke"]
    rest = [r for r in rows if r.get("split") != "dev_smoke"]
    original_split = {str(r.get("id")): str(r.get("split")) for r in rest}

    surplus: list[dict[str, Any]] = []
    test_to_train: list[dict[str, Any]] = []

    # ---- final_test: stratified core --------------------------------------------------
    final_rows = [r for r in rest if r.get("split") == "final_test"]
    kept_test: list[dict[str, Any]] = []
    for bucket in sorted({str(r.get("train_bucket")) for r in final_rows}):
        brows = [r for r in final_rows if str(r.get("train_bucket")) == bucket]
        cap = FINAL_TEST_CAPS.get(bucket)
        if bucket in SCARCE or cap is None:
            kept_test.extend(brows)
            continue
        picked, overflow = round_robin_select(brows, cap)
        kept_test.extend(picked)
        # easy iid overflow becomes training material; the rest is surface-diversity stock
        for r in overflow:
            if str(r.get("generalization_class")) == "iid":
                r.setdefault("metadata", {})["reassigned"] = "final_test->train(curation_v3)"
                r["split"] = "train"
                test_to_train.append(r)
            else:
                r.setdefault("metadata", {})["curation_surplus_from"] = "final_test"
                surplus.append(r)

    # ---- train: family caps ------------------------------------------------------------
    train_rows = [r for r in rest
                  if original_split[str(r.get("id"))] == "train"] + test_to_train
    kept_train: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in train_rows:
        if str(r.get("train_bucket")) in SCARCE:
            kept_train.append(r)
        else:
            by_family[str(r.get("template_family"))].append(r)
    for fam, fam_rows in sorted(by_family.items()):
        fam_rows.sort(key=stable_key)
        kept_train.extend(fam_rows[:TRAIN_FAMILY_CAP])
        for r in fam_rows[TRAIN_FAMILY_CAP:]:
            r.setdefault("metadata", {})["curation_surplus_from"] = "train"
            surplus.append(r)

    # ---- dev splits: keep, then fill dev_tune floors from train ------------------------
    kept_dev = [r for r in rest if r.get("split") in ("dev_tune", "dev_select")]
    tune_counts = Counter(str(r.get("train_bucket")) for r in kept_dev if r.get("split") == "dev_tune")
    filled = Counter()
    from collections import OrderedDict
    train_by_plan: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in kept_train:
        train_by_plan[str(r.get("plan_id"))].append(r)
    for bucket, n in sorted(tune_counts.items()):
        need = DEV_TUNE_FLOOR - n
        if need <= 0:
            continue
        # donate WHOLE plan groups (an L1 and its L2 siblings never straddle train/eval)
        candidate_plans = sorted(
            (pid for pid, group in train_by_plan.items()
             if group and all(str(g.get("train_bucket")) == bucket for g in group)),
            key=lambda pid: hashlib.sha1(pid.encode("utf-8")).hexdigest())
        for pid in candidate_plans:
            if filled[bucket] >= need:
                break
            group = train_by_plan[pid]
            for r in group:
                kept_train.remove(r)
                r.setdefault("metadata", {})["reassigned"] = "train->dev_tune(floor_fill_v3)"
                r["split"] = "dev_tune"
                kept_dev.append(r)
                filled[bucket] += 1
            train_by_plan[pid] = []
    report["dev_tune_floor_filled"] = dict(filled)

    out_rows = smoke + kept_dev + kept_train + kept_test
    # HARD integrity gate: no plan may straddle train and any eval split
    plan_splits: dict[str, set] = defaultdict(set)
    for r in out_rows:
        plan_splits[str(r.get("plan_id"))].add(str(r.get("split")))
    crossings = {p: sorted(s) for p, s in plan_splits.items()
                 if "train" in s and ({"final_test", "dev_select", "dev_tune", "dev_smoke"} & s)}
    if crossings:
        raise SystemExit(f"plan/split integrity violated: {list(crossings.items())[:5]}")
    all_ids = [str(r.get("id")) for r in out_rows] + [str(r.get("id")) for r in surplus]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("duplicate ids across output+surplus (double-reference bug)")
    if len(set(all_ids)) != len(rows):
        raise SystemExit(f"row conservation violated: {len(set(all_ids))} != {len(rows)}")
    after = Counter((str(r.get("split")), str(r.get("train_bucket"))) for r in out_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    dump(args.out / "all.jsonl", out_rows)
    dump(args.out / "surplus.jsonl", surplus)
    for split in sorted({str(r.get("split")) for r in out_rows}):
        dump(args.out / f"{split}.jsonl", [r for r in out_rows if str(r.get("split")) == split])

    report.update({
        "output_rows": len(out_rows),
        "surplus_rows": len(surplus),
        "test_to_train_reassigned": len(test_to_train),
        "split_sizes": dict(Counter(str(r.get("split")) for r in out_rows)),
        "final_test_buckets": {b: n for (s, b), n in sorted(after.items()) if s == "final_test"},
        "train_buckets": {b: n for (s, b), n in sorted(after.items()) if s == "train"},
        "before_split_sizes": dict(Counter(s for (s, _b) in before.elements())),
    })
    (args.out / "curation_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                                   encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
