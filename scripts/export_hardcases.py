#!/usr/bin/env python3
"""Export structural hard cases from a Step-2 probe run.

Answerable questions the Step-2 planner got wrong are the interesting supervision signal: they are
NOT fixed with compiler hacks (that would mask what SFT/RSFT must learn). This joins a probe's
`step2.traces.jsonl` back to the Step-1 briefing (for the reviewable understanding) and the gold
split (for the oracle), and writes a hard-case pool grouped by structural bucket + failure kind.

    python -B scripts/export_hardcases.py \
        --traces data/qa/plan_probe/step2_grok_baseline_v3_50/step2.traces.jsonl \
        --step1  data/qa/understanding_probe/step1_nano_dense_v3_50/understanding.outputs.jsonl \
        --gold   data/qa/cicada_merged_l1_l2_trainbalanced_v1/dev_smoke.jsonl \
        --out-dir data/qa/plan_probe/step2_grok_baseline_v3_50/hardcases
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

# Buckets whose failures are structural reasoning errors worth learning (vs. simple filter slips).
STRUCTURAL_BUCKETS = {"bridge_join", "comparison", "compare_two", "min_max", "top_k", "set", "factoid"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _classify(rec: dict[str, Any]) -> str:
    """Coarse failure kind for triage."""
    status = str(rec.get("plan_status"))
    exec_status = str(rec.get("exec_status"))
    if status != "planned":
        reason = str(rec.get("plan_rationale", ""))
        if "bridge_cue_requires_bridge_join" in reason:
            return "not_planned:bridge_flattened"
        if "role_flipped" in reason:
            return "not_planned:role_flip"
        if "invalid_graph_plan" in reason:
            return "not_planned:invalid_graph"
        if "compiled_plan_dropped_atoms" in reason:
            return "not_planned:dropped_atoms"
        return "not_planned:other"
    if exec_status == "no_results":
        return "planned:no_results"
    if exec_status in {"error", "unsupported", "unsupported_operation", "schema_error"}:
        return f"planned:{exec_status}"
    if rec.get("answered") and not rec.get("match"):
        return "planned:answered_wrong"
    if not rec.get("answered"):
        return "planned:no_answer"
    return "planned:other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--step1", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--structural-only", action="store_true",
                    help="keep only STRUCTURAL_BUCKETS (bridge/comparison/...); else all answerable misses")
    args = ap.parse_args()

    traces = read_jsonl(args.traces)
    step1_by_id = {str(r.get("id")): r for r in read_jsonl(args.step1)}
    gold_by_id = {str(r.get("id")): r for r in read_jsonl(args.gold)}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hardcases: list[dict[str, Any]] = []
    for rec in traces:
        if not rec.get("scored") or rec.get("abstain_case"):
            continue  # abstain handled correctly elsewhere; hard cases are answerable misses
        if rec.get("match"):
            continue
        bucket = str(rec.get("bucket"))
        if args.structural_only and bucket not in STRUCTURAL_BUCKETS:
            continue
        rid = str(rec.get("id"))
        s1 = step1_by_id.get(rid, {})
        gold = gold_by_id.get(rid, {})
        hardcases.append({
            "id": rid,
            "bucket": bucket,
            "structural": bucket in STRUCTURAL_BUCKETS,
            "failure_kind": _classify(rec),
            "question": rec.get("question"),
            "oracle_answer": gold.get("oracle_answer"),
            "answer_type": gold.get("answer_type"),
            "expected_status": gold.get("expected_status"),
            "plan_status": rec.get("plan_status"),
            "plan_rationale": rec.get("plan_rationale"),
            "exec_status": rec.get("exec_status"),
            "submitted_answer": rec.get("answer"),
            # reviewable artifacts for SFT/RSFT: the Step-1 briefing + the Step-2 graph plan
            "step1_understanding": s1.get("ascii_understanding") or s1.get("raw_understanding"),
            "graph_plan": rec.get("graph_plan"),
            "gold_constraints": gold.get("constraints"),
            "gold_plan": gold.get("gold_plan"),
        })

    out_path = args.out_dir / "hardcases.jsonl"
    out_path.write_text("".join(json.dumps(h, ensure_ascii=False, default=str) + "\n" for h in hardcases),
                        encoding="utf-8")

    by_bucket = Counter(h["bucket"] for h in hardcases)
    by_kind = Counter(h["failure_kind"] for h in hardcases)
    by_bucket_kind = Counter((h["bucket"], h["failure_kind"]) for h in hardcases)
    summary = {
        "source_traces": str(args.traces),
        "total_hardcases": len(hardcases),
        "structural_hardcases": sum(1 for h in hardcases if h["structural"]),
        "by_bucket": dict(by_bucket.most_common()),
        "by_failure_kind": dict(by_kind.most_common()),
        "by_bucket_x_kind": {f"{b}|{k}": n for (b, k), n in by_bucket_kind.most_common()},
    }
    (args.out_dir / "hardcases.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
