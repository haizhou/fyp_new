#!/usr/bin/env python3
"""Post-mortem for the verified_hybrid cascade: joins rule/hybrid/verified results per item and
REPLAYS the deterministic rule-side probe offline to attribute each escalation to its trigger.

Reports, per level:
  llm_called            escalations (each = one LLM planner call in the eval run)
  recovered_vs_hybrid   verified correct where hybrid was wrong
  regressed_vs_hybrid   verified wrong where hybrid was correct   <- the damage to explain
  failed_repair         escalated, still wrong
  trigger distribution  probe_degenerate_zero / type_mismatch / probe_no_results / ...

Offline (no API): the probe replay uses the rule planner + executor read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ML_EVAL = ROOT / "data" / "qa" / "multilevel" / "eval"
ML = ROOT / "data" / "qa" / "multilevel"


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", default="L1,L2")
    args = ap.parse_args()

    from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
    from procurement_graph.reasoning.planner_decomposition import (
        DecompositionAwarePlanner, VerifyingHybridPlanner)

    print("[escalation] loading KG ...", flush=True)
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    resolver = backend.org_resolver()
    rule = DecompositionAwarePlanner(org_resolver=resolver)
    prober = VerifyingHybridPlanner(rule=rule, backend=backend, llm=None)

    report: dict[str, dict] = {}
    for level in [x.strip() for x in args.levels.split(",") if x.strip()]:
        paths = {name: ML_EVAL / f"{level}.pipeline.{name}.results.jsonl"
                 for name in ("rule_decomp", "hybrid", "verified_hybrid")}
        missing = [p.name for p in paths.values() if not p.exists()]
        if missing:
            print(f"  (skip {level}: missing {missing})")
            continue
        results = {name: load(path) for name, path in paths.items()}
        surfaces = load(ML / f"surfaces.{level}.jsonl")

        triggers: Counter[str] = Counter()
        outcome: Counter[str] = Counter()
        regressions: list[dict] = []
        failed_repairs: list[dict] = []
        for item_id, r_rule in results["rule_decomp"].items():
            r_hyb = results["hybrid"].get(item_id)
            r_ver = results["verified_hybrid"].get(item_id)
            src = surfaces.get(item_id)
            if not (r_hyb and r_ver and src and r_rule.get("scored")):
                continue
            cands = tuple(rule.plan(src["question"]))
            top = max(cands, key=lambda c: c.confidence) if cands else None
            if top is None or top.status == "unsupported":
                trigger = "not_escalated:unsupported_or_none"
            elif top.status != "planned":
                trigger = "rule_ambiguous"
            else:
                escalate, why = prober._probe(src["question"], top)
                trigger = why if escalate else "not_escalated:probe_passed"
            escalated = not trigger.startswith("not_escalated")
            if escalated:
                triggers[trigger] += 1
                outcome["llm_called"] += 1
                if r_ver["match"] and not r_hyb["match"]:
                    outcome["recovered_vs_hybrid"] += 1
                elif not r_ver["match"] and r_hyb["match"]:
                    outcome["regressed_vs_hybrid"] += 1
                    regressions.append({"id": item_id, "trigger": trigger, "subset": src.get("subset"),
                                        "question": src["question"][:110],
                                        "oracle": src.get("oracle_answer")})
                if not r_ver["match"]:
                    outcome["failed_repair"] += 1
            elif r_ver["match"] != r_hyb["match"]:
                outcome["diverged_without_escalation"] += 1

        report[level] = {"outcome": dict(outcome), "triggers": dict(triggers.most_common()),
                         "regressions": regressions[:20], "failed_repair_sample": failed_repairs[:10]}
        print(f"\n{level}: {dict(outcome)}")
        print(f"  triggers: {dict(triggers.most_common())}")
        for reg in regressions[:8]:
            print(f"  REGRESSED [{reg['trigger']}] {reg['subset']} oracle={reg['oracle']!r}: {reg['question']}")

    out = ML_EVAL / "escalation_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
