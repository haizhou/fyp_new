#!/usr/bin/env python3
"""Decompose the generalization gap between two planners' multilevel results.

Turns "hybrid is +3 points" into the research-grade taxonomy:

  both_correct            neither planner needed help
  llm_recovered           rule failed, hybrid fixed it        <- the LLM's measured contribution
  hybrid_regressed        rule was right, hybrid broke it     <- the cascade's measured cost
  rule_misfire_unescalated  rule ANSWERED wrongly, so the cascade never consulted the LLM
                            (the escalation-trigger blind spot; motivates verify-then-escalate)
  escalated_llm_failed    rule abstained, the LLM was consulted and still failed

Usage:
  python -B scripts/analyze_generalization_gap.py --levels L1,L2,L3 \\
      --a rule_decomp --b hybrid --in-dir data/qa/multilevel/eval
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default="rule_decomp", help="baseline planner label")
    ap.add_argument("--b", default="hybrid", help="treatment planner label")
    ap.add_argument("--levels", default="L1,L2,L3")
    ap.add_argument("--in-dir", default=str(ROOT / "data" / "qa" / "multilevel" / "eval"))
    args = ap.parse_args()
    in_dir = Path(args.in_dir)

    report: dict[str, dict] = {}
    for level in [x.strip() for x in args.levels.split(",") if x.strip()]:
        pa = in_dir / f"{level}.pipeline.{args.a}.results.jsonl"
        pb = in_dir / f"{level}.pipeline.{args.b}.results.jsonl"
        if not (pa.exists() and pb.exists()):
            print(f"  (skip {level}: need both {pa.name} and {pb.name})")
            continue
        a, b = load(pa), load(pb)
        taxonomy: Counter[str] = Counter()
        wrong_subsets: Counter[str] = Counter()
        for item_id, ra in a.items():
            rb = b.get(item_id)
            if rb is None or not ra.get("scored"):
                continue
            am, bm = bool(ra.get("match")), bool(rb.get("match"))
            if am and bm:
                taxonomy["both_correct"] += 1
            elif not am and bm:
                taxonomy["llm_recovered"] += 1
            elif am and not bm:
                taxonomy["hybrid_regressed"] += 1
            elif ra.get("answered"):
                taxonomy["rule_misfire_unescalated"] += 1
                wrong_subsets[ra.get("subset", "")] += 1
            else:
                taxonomy["escalated_llm_failed"] += 1
                wrong_subsets[ra.get("subset", "")] += 1
        total = sum(taxonomy.values())
        report[level] = {
            "items": total,
            "taxonomy": dict(taxonomy),
            "llm_net_contribution": taxonomy["llm_recovered"] - taxonomy["hybrid_regressed"],
            "escalation_blind_spot": taxonomy["rule_misfire_unescalated"],
            "both_wrong_by_subset": dict(wrong_subsets),
        }
        print(f"{level}: {dict(taxonomy)}")
        print(f"  LLM net contribution: +{report[level]['llm_net_contribution']} items; "
              f"blind spot (rule answered wrong, LLM never consulted): "
              f"{taxonomy['rule_misfire_unescalated']}")

    out = in_dir / f"gap_decomposition.{args.a}_vs_{args.b}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
