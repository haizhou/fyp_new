#!/usr/bin/env python3
"""PACS intent generator (spec v2.2, frozen). Produces the INTENT POOL:
gold trees + dual-verified oracles + status variants + all seven identifiers.
Surface rendering (three channels) is a later, separate stage.

Quotas (spec v2.1): per family x depth cell 40-50 answerable intents, incl.
15-20 unseen-shape where constructible. Unseen steering: decoration predicates
are injected into filter scopes until the shape signature leaves the training
signature set. Isolation gates enforced at generation: gold-tree-hash overlap
with training = 0; unseen rows verified against training shapes.

Usage:
  .venv/bin/python scripts/pacs/generate.py [--per-cell 45] [--unseen 15] [--seed 20260718]
  [--smoke]  # 3 per cell, quick validation
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/pacs"))

import importlib.util

from identifiers import make_identifiers, gold_tree_hash, shape_signature  # noqa: E402
from templates import TEMPLATES, Anchors, eqp  # noqa: E402

from procurement_graph.compose.algebra import AlgebraError, validate_tree  # noqa: E402
from procurement_graph.compose.eval_runtime import RuntimeAlgebraEvaluator  # noqa: E402
from procurement_graph.qa.benchmark.kg_interface import ParquetKGQueryBackend  # noqa: E402

_spec = importlib.util.spec_from_file_location("indep", ROOT / "scripts/compose_independent_eval.py")
indep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(indep)


def training_signatures() -> tuple[set, set]:
    """Shape signatures and tree hashes of the FROZEN compose-v3 training set."""
    sigs, hashes = set(), set()
    for name in ("compose_sft_train.json", "compose_sft_val.json"):
        for r in json.load(open(ROOT / "data/training/llamafactory_compose_v3" / name)):
            try:
                t = json.loads(r["output"]).get("tree")
            except Exception:
                continue
            if t:
                sigs.add(shape_signature(t))
                hashes.add(gold_tree_hash(t))
    return sigs, hashes


def sensible(answer, family: str) -> bool:
    if isinstance(answer, bool):
        return True
    if isinstance(answer, (int, float)):
        return answer != 0 and abs(float(answer)) < 1e13
    if isinstance(answer, list):
        if answer and isinstance(answer[0], list):  # ranking
            return len(answer) >= 2
        return 1 <= len(answer) <= 30
    return bool(str(answer))


def agree(a, b) -> bool:
    if type(a) is bool or type(b) is bool:
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 0.01
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def decorate(tree: dict, anchors: Anchors) -> dict:
    """Inject a decoration predicate into the first filter scope (deep copy) to
    shift the shape signature for unseen steering."""
    t = copy.deepcopy(tree)

    def first_filter(node):
        if isinstance(node, dict):
            if node.get("node") == "filter":
                return node
            for v in node.values():
                found = first_filter(v)
                if found is not None:
                    return found
        if isinstance(node, list):
            for v in node:
                found = first_filter(v)
                if found is not None:
                    return found
        return None

    f = first_filter(t)
    if f is not None:
        used = {p.get("field") for p in f["where"] if isinstance(p, dict)}
        f["where"] = list(f["where"]) + [anchors.decoration(used)]
    return t


# ----------------------------------------------------------- status variants
_MISSING_OP_Q = {
    "F1": "What is the median contract value of notices {scope}?",
    "F2": "What is the average year-over-year growth rate of spending {scope}?",
    "F3": "What is the median award value per supplier {scope}?",
    "F4": "What is the average difference in spending between the two buyers {scope}?",
    "F5": "What is the median number of shared suppliers {scope}?",
    "F6": "What is the average number of buyers per supplier {scope}?",
    "F7": "What is the median delay between publication and signing {scope}?",
}


def status_variants(family: str, inst: dict, anchors: Anchors, ev1, df2, rng) -> list[dict]:
    out = []
    # requires_missing_operator: same intent context, out-of-grammar request
    scope = "in this scope"
    q = _MISSING_OP_Q[family].format(scope=scope)
    out.append({"expected_status": "requires_missing_operator", "question_seed": q,
                "compose_tree": None, "oracle_answer": None})
    # empty_result: re-anchor the same template until the tree runs empty/zero
    fn = TEMPLATES[(family, inst["depth"])]
    for _ in range(40):
        try:
            tree, params, tid = fn(anchors)
        except Exception:
            continue
        try:
            validate_tree(tree)
        except AlgebraError:
            continue
        r1 = ev1.run(tree)
        empty = (r1.get("status") == "ok" and (r1["answer"] in (0, 0.0, False, []))) or \
                (r1.get("status") == "failed" and r1.get("reason") in ("no_results", "no_groups"))
        if empty:
            r2 = indep.run_tree(df2, tree)
            empty2 = (r2.get("status") == "ok" and (r2.get("answer") in (0, 0.0, False, []))) or \
                     (r2.get("status") == "failed" and r2.get("reason") in ("no_results", "no_groups"))
            if empty2:
                out.append({"expected_status": "empty_result", "question_seed": None,
                            "compose_tree": tree, "params": params,
                            "template_id": tid + "#empty", "oracle_answer": None})
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=45)
    ap.add_argument("--unseen", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="data/qa/pacs_v1/intent_pool.jsonl")
    args = ap.parse_args()
    if args.smoke:
        args.per_cell, args.unseen = 3, 1

    rng = random.Random(args.seed)
    backend = ParquetKGQueryBackend.from_directory(ROOT / "data/kg", include_evidence=False)
    ev1 = RuntimeAlgebraEvaluator(backend)
    df2 = indep.load_universe()
    anchors = Anchors(backend.records_df, rng)
    train_sigs, train_hashes = training_signatures()
    print(f"training: {len(train_sigs)} shapes, {len(train_hashes)} tree hashes")

    rows, anchor_use = [], Counter()
    rejects = Counter()
    for (family, depth), fn in sorted(TEMPLATES.items()):
        got, got_unseen, attempts = 0, 0, 0
        max_attempts = args.per_cell * 60
        while got < args.per_cell and attempts < max_attempts:
            attempts += 1
            try:
                tree, params, tid = fn(anchors)
            except Exception:
                rejects["template_error"] += 1
                continue
            sig = shape_signature(tree)
            seen_quota_left = (args.per_cell - args.unseen) - (got - got_unseen)
            must_be_unseen = (got_unseen < args.unseen) and (seen_quota_left <= 0)
            want_unseen = got_unseen < args.unseen
            if sig in train_sigs and want_unseen:
                # steering attempt: up to two decorations
                cand = decorate(tree, anchors)
                if shape_signature(cand) in train_sigs:
                    cand = decorate(cand, anchors)
                if shape_signature(cand) not in train_sigs:
                    tree, sig = cand, shape_signature(cand)
                elif must_be_unseen:
                    rejects["unseen_steering_failed"] += 1
                    continue
                # else: keep original as a seen row
            try:
                validate_tree(tree)
            except AlgebraError:
                rejects["invalid_tree"] += 1
                continue
            th = gold_tree_hash(tree)
            if th in train_hashes:
                rejects["tree_hash_in_training"] += 1
                continue
            anchor_key = json.dumps(sorted((k, str(v)) for k, v in params.items()))
            if anchor_use[anchor_key] >= 3:
                rejects["anchor_reuse_cap"] += 1
                continue
            r1 = ev1.run(tree)
            if r1.get("status") != "ok" or not sensible(r1["answer"], family):
                rejects["not_sensible"] += 1
                continue
            r2 = indep.run_tree(df2, tree)
            if r2.get("status") != "ok" or not agree(r1["answer"], r2["answer"]):
                rejects["dual_disagree"] += 1
                continue
            anchor_use[anchor_key] += 1
            exposure = "unseen" if sig not in train_sigs else "seen"
            got_unseen += (exposure == "unseen")
            inst = {"family": family, "depth": depth, "exposure": exposure,
                    "template_id": tid, "params": params, "compose_tree": tree,
                    "oracle_answer": r1["answer"], "expected_status": "answerable"}
            inst.update(make_identifiers(family, depth, tid, tree, params, got))
            rows.append(inst)
            got += 1
        print(f"{family}-{depth}: {got} answerable ({got_unseen} unseen), attempts {attempts}")

    # status variants: ~20% per family, >=2 types
    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
    variants = []
    for family, members in by_family.items():
        n_var = max(2, len(members) // 5)
        picks = rng.sample(members, min(n_var, len(members)))
        for base_inst in picks:
            for v in status_variants(family, base_inst, anchors, ev1, df2, rng):
                v.update({"family": family, "depth": base_inst["depth"],
                          "exposure": base_inst["exposure"],
                          "cluster_of": base_inst["intent_id"]})
                if v.get("compose_tree"):
                    v.update(make_identifiers(family, base_inst["depth"],
                                              v.get("template_id", "statusvar"),
                                              v["compose_tree"], v.get("params", {}),
                                              len(variants)))
                else:
                    v["intent_id"] = base_inst["intent_id"] + f"#st{len(variants)}"
                variants.append(v)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows + variants:
            fh.write(json.dumps(r, default=str) + "\n")

    n_unseen = sum(1 for r in rows if r["exposure"] == "unseen")
    print(f"\nwrote {len(rows)} answerable + {len(variants)} status rows -> {out}")
    print(f"unseen total: {n_unseen}; distinct shapes: {len({r['shape_signature'] for r in rows})}")
    print("rejects:", dict(rejects.most_common(8)))
    st = Counter(v["expected_status"] for v in variants)
    print("status variants:", dict(st))

    sample = random.Random(args.seed).sample(rows, min(10, len(rows)))
    print("\n=== RAW SAMPLE (10, human-scan) ===")
    for r in sample:
        print(json.dumps({"id": r["intent_id"], "exp": r["exposure"],
                          "ans": str(r["oracle_answer"])[:60]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
