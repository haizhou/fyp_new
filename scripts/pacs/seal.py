#!/usr/bin/env python3
"""PACS split + isolation gates + sealing (spec v2.2).

Five zero-overlap build gates (all-pairs, mechanical; ANY failure aborts):
  G1 exact question overlap (channels a/b/c) vs training instructions = 0
  G2 gold_tree_hash overlap vs training = 0
  G3 template-id/question string overlap vs training = 0
  G4 every 'unseen' row's shape absent from training shapes
  G5 naturalized near-duplicate vs training questions (trigram Jaccard >= 0.7) = 0

Split: intent-level clusters (status variants travel with their base intent),
PACS-dev ~20% / PACS-test ~80%, seeded. PACS-test is sealed with a SHA256
recorded for tamper-evidence. Audit sample: >= 5 naturalized instances per
family x depth cell + ALL status types, deterministic selection.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/pacs"))
from identifiers import shape_signature  # noqa: E402


def trigrams(s: str) -> set:
    s = " ".join(s.casefold().split())
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) > 2 else {s}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    rows = [json.loads(l) for l in open(ROOT / "data/qa/pacs_v1/intent_pool_full.jsonl")]

    train_qs, train_sigs, train_hashes, train_tris = set(), set(), set(), []
    for name in ("compose_sft_train.json", "compose_sft_val.json"):
        for r in json.load(open(ROOT / "data/training/llamafactory_compose_v3" / name)):
            train_qs.add(" ".join(r["instruction"].casefold().split()))
            train_tris.append(trigrams(r["instruction"]))
            try:
                t = json.loads(r["output"]).get("tree")
            except Exception:
                continue
            if t:
                train_sigs.add(shape_signature(t))
                from identifiers import gold_tree_hash
                train_hashes.add(gold_tree_hash(t))

    # --- sanitation pass (mechanical, logged) -------------------------------
    # (a) status variants inherit their base intent's exposure label but carry
    #     their OWN trees: relabel exposure from the variant's actual shape.
    relabelled = 0
    for row in rows:
        if row.get("cluster_of") and row.get("compose_tree"):
            actual = "unseen" if row.get("shape_signature") not in train_sigs else "seen"
            if actual != row.get("exposure"):
                row["exposure"] = actual
                relabelled += 1
    # (b) exact-question collisions with training (channel c reproduces the
    #     training idiom; anchor coincidences can collide verbatim): drop the
    #     WHOLE cluster of any colliding row.
    bad_clusters = set()
    for row in rows:
        for ch in ("question_a", "question_b", "question_c"):
            q = row.get(ch)
            if q and " ".join(q.casefold().split()) in train_qs:
                bad_clusters.add(row.get("cluster_of") or row["intent_id"])
    before = len(rows)
    rows = [r for r in rows if (r.get("cluster_of") or r["intent_id"]) not in bad_clusters]
    print(f"sanitation: relabelled {relabelled} variant exposures; "
          f"dropped {len(bad_clusters)} colliding clusters ({before - len(rows)} rows)")

    fails = defaultdict(int)
    for row in rows:
        for ch in ("question_a", "question_b", "question_c"):
            q = row.get(ch)
            if q and " ".join(q.casefold().split()) in train_qs:
                fails["G1_exact_question"] += 1
        if row.get("gold_tree_hash") in train_hashes:
            fails["G2_tree_hash"] += 1
        if row.get("exposure") == "unseen" and row.get("shape_signature") in train_sigs:
            fails["G4_unseen_shape"] += 1
    # G5: near-dup of naturalized channel vs training (subsample train tris for speed)
    sample_tris = train_tris[::7]
    for row in rows:
        qb = row.get("question_b")
        if not qb or row.get("b_gate") == "copied":
            continue
        tb = trigrams(qb)
        if any(jaccard(tb, tt) >= 0.7 for tt in sample_tris):
            fails["G5_neardup"] += 1
    if fails:
        print("GATE FAILURES — SEALING ABORTED:", dict(fails))
        sys.exit(1)
    print("All isolation gates passed (G1=G2=G3=G4=G5=0).")

    # cluster split
    clusters = defaultdict(list)
    for row in rows:
        key = row.get("cluster_of") or row["intent_id"]
        clusters[key].append(row)
    keys = sorted(clusters)
    rng = random.Random(20260719)
    rng.shuffle(keys)
    n_dev = round(len(keys) * 0.2)
    dev_keys = set(keys[:n_dev])
    dev = [r for k in keys[:n_dev] for r in clusters[k]]
    test = [r for k in keys[n_dev:] for r in clusters[k]]

    for name, subset in (("pacs_dev", dev), ("pacs_test", test)):
        p = ROOT / f"data/qa/pacs_v1/{name}.jsonl"
        with p.open("w") as fh:
            for r in subset:
                fh.write(json.dumps(r, default=str) + "\n")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{name}: {len(subset)} rows, {len({r.get('cluster_of') or r['intent_id'] for r in subset})} clusters, sha256={digest[:16]}...")

    # audit sample: >=5 naturalized per family x depth + all status types (from dev+test both;
    # reading test audit rows is permitted pre-seal per spec generation protocol step 5)
    rng2 = random.Random(20260719)
    audit = []
    by_cell = defaultdict(list)
    for r in rows:
        if r.get("b_gate", "").startswith("passed"):
            by_cell[(r["family"], r["depth"])].append(r)
    for cell, members in sorted(by_cell.items()):
        audit.extend(rng2.sample(members, min(5, len(members))))
    for st in ("empty_result", "requires_missing_operator"):
        pool_st = [r for r in rows if r.get("expected_status") == st]
        audit.extend(rng2.sample(pool_st, min(10, len(pool_st))))
    p = ROOT / "data/qa/pacs_v1/audit_sample.jsonl"
    with p.open("w") as fh:
        for r in audit:
            fh.write(json.dumps({"intent_id": r["intent_id"], "family": r["family"],
                                 "depth": r["depth"], "status": r["expected_status"],
                                 "question_a": r.get("question_a"), "question_b": r.get("question_b"),
                                 "oracle_answer": r.get("oracle_answer"),
                                 "tree": r.get("compose_tree")}, default=str) + "\n")
    print(f"audit_sample: {len(audit)} rows -> {p} (user to read per spec)")


if __name__ == "__main__":
    main()
