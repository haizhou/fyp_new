#!/usr/bin/env python3
"""Build the multi-level QA benchmark from the executor-validated v2 plan rows.

Plan-first: each accepted targeted-v2 row is one PLAN (constraints + operation + oracle, already
validated against the shared executor). This script never touches v1/v2 files; it writes a new
artifact under data/qa/multilevel/:

  plan_bank.jsonl      Level 0 — language-free plans (executor-validation set)
  surfaces.L1.jsonl    Level 1 — the source template question (rule-planner control)
  surfaces.L2.jsonl    Level 2 — LLM paraphrases        (needs --llm on)
  surfaces.L3.jsonl    Level 3 — LLM adversarial forms  (needs --llm on)
  build_summary.json   acceptance/rejection statistics per level

Level 2/3 surfaces pass the deterministic gate in `procurement_graph.qa.multilevel.check_surface`
(atom preservation, no new numbers, no new KG orgs, unanswerable trigger kept, actually rewritten)
with up to --retries rejection-sampling rounds per surface. All surfaces of a plan share ONE
oracle, so per-level accuracy differences isolate language-to-plan generalization.

Offline (no key):   python -B scripts/build_multilevel_qa.py                  # plan bank + L1
With the API:       python -B scripts/build_multilevel_qa.py --llm on         # + L2 + L3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.qa.multilevel import (  # noqa: E402
    check_surface, checker_accepts, checker_messages, l2_plan_ready, l2_rewrite_messages,
    persona_for, plan_bank_row, required_atoms, rewrite_messages, surface_row,
)

V2_DIR = ROOT / "data" / "qa" / "targeted_v2" / "full2k"
OUT_DIR = ROOT / "data" / "qa" / "multilevel"
SUBSETS = ("naturalized", "coverage_fixed", "unanswerable", "extended_ops", "bridge_join")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows),
                    encoding="utf-8")


def stride(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def sample_plans(per_subset: int, tag: str) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for subset in SUBSETS:
        path = V2_DIR / f"{subset}.{tag}.accepted.jsonl"
        if not path.exists():
            print(f"  (skip {subset}: {path} missing)")
            continue
        plans.extend(stride(sorted(read_jsonl(path), key=lambda r: r["id"]), per_subset))
    return plans


def rewrite_surfaces(rows, *, level, chat, model, retries, variants, org_resolver, known_orgs,
                     checker_model: str = "", progress_every: int = 25):
    """L2 = plan-generated (generator sees the PLAN, never the template; one persona per plan;
    deterministic atom gate + independent checker LLM). L3 = legacy adversarial paraphrase."""
    accepted, rejected = [], []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        if level == 2:
            ready, reason = l2_plan_ready(row)
            if not ready:
                rejected.append({"plan_id": row["id"], "level": level, "reason": f"skipped:{reason}"})
                if progress_every and (index % progress_every == 0 or index == total):
                    print(f"    L{level}: processed {index}/{total}, accepted={len(accepted)}, "
                          f"rejected={len(rejected)}", flush=True)
                continue
        atoms = required_atoms(row)
        persona = persona_for(row["id"])
        got = 0
        for attempt in range(retries + 1):
            if level == 2:
                system, user = l2_rewrite_messages(row, atoms, n_variants=variants - got, persona=persona)
            else:
                system, user = rewrite_messages(row, atoms, level=level, n_variants=variants - got)
            try:
                parsed = getattr(chat.complete_json(model=model, system=system, user=user), "parsed", {}) or {}
            except Exception as exc:  # pragma: no cover - live boundary
                rejected.append({"plan_id": row["id"], "level": level, "reason": f"llm_error:{exc!r}"})
                break
            for candidate in list(parsed.get("variants") or [])[: variants - got]:
                text = " ".join(str(candidate).split())
                verdict = check_surface(text, atoms, row["question"], level=level,
                                        org_resolver=org_resolver, known_org_names=known_orgs,
                                        relax_org_texts=bool(level == 2 and checker_model))
                if not verdict.ok:
                    rejected.append({"plan_id": row["id"], "level": level,
                                     "candidate": text[:200], "reasons": list(verdict.reasons)})
                    continue
                checker_verdict = None
                if level == 2 and checker_model:
                    c_system, c_user = checker_messages(text, row)
                    checker_verdict = getattr(
                        chat.complete_json(model=checker_model, system=c_system, user=c_user),
                        "parsed", None)
                    ok, why = checker_accepts(
                        checker_verdict,
                        expected_status=row.get("expected_status", "answerable"),
                    )
                    if not ok:
                        rejected.append({"plan_id": row["id"], "level": level,
                                         "candidate": text[:200], "reasons": [why]})
                        continue
                srow = surface_row(row, level=level, question=text, origin=model,
                                   variant=got, verdict=verdict)
                srow["persona"] = persona if level == 2 else ""
                if level == 2:
                    srow["l2_generation_mode"] = "l1_persona_rewrite"
                if checker_verdict is not None:
                    srow["checker"] = checker_verdict
                accepted.append(srow)
                got += 1
            if got >= variants:
                break
        if got == 0 and not any(r.get("plan_id") == row["id"] and str(r.get("reason", "")).startswith("llm_error")
                                for r in rejected):
            rejected.append({"plan_id": row["id"], "level": level, "reason": "exhausted_retries"})
        if progress_every and (index % progress_every == 0 or index == total):
            print(f"    L{level}: processed {index}/{total}, accepted={len(accepted)}, "
                  f"rejected={len(rejected)}", flush=True)
    return accepted, rejected


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-subset", type=int, default=100, help="plans sampled per v2 subset (stride)")
    ap.add_argument("--tag", default="full2k")
    ap.add_argument("--llm", choices=["off", "on"], default="off")
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--variants", type=int, default=1, help="accepted surfaces per plan per level")
    ap.add_argument("--retries", type=int, default=2, help="rejection-sampling rounds per surface")
    ap.add_argument("--levels", default="2",
                    help="LLM levels: 2 = plan-generated w/ persona + checker (the benchmark level); "
                         "3 = legacy adversarial paraphrase")
    ap.add_argument("--checker-model", default="gpt-5.4-nano",
                    help="independent checker LLM for L2 semantic-equivalence ('' disables)")
    ap.add_argument("--org-gate", choices=["off", "kg"], default="off",
                    help="foreign-organisation gate: off = plan atoms + checker only; "
                         "kg = load KG org resolver and reject newly introduced KG org names")
    ap.add_argument("--progress-every", type=int, default=25,
                    help="print live progress every N source plans during LLM generation (0 disables)")
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.Random(args.seed)  # reserved for future samplers; stride keeps selection deterministic

    plans = sample_plans(args.per_subset, args.tag)
    print(f"[multilevel] {len(plans)} plans sampled ({args.per_subset}/subset, stride)")
    write_jsonl(out_dir / "plan_bank.jsonl", [plan_bank_row(r) for r in plans])

    l1 = [surface_row(r, level=1, question=r["question"], origin=r.get("rewrite_model", "template"))
          for r in plans]
    write_jsonl(out_dir / "surfaces.L1.jsonl", l1)
    summary: dict[str, Any] = {"plans": len(plans), "L1": {"accepted": len(l1)},
                               "per_subset": dict(Counter(r["subset"] for r in l1))}

    if args.llm == "on":
        from procurement_graph.qa.benchmark.chat import ChatClient

        chat = ChatClient.from_env()
        resolver = None
        known = None
        if args.org_gate == "kg":
            from procurement_graph.reasoning.kg_backend import RuntimeKGBackend

            print("[multilevel] loading KG for the foreign-org gate ...", flush=True)
            backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
            resolver = backend.org_resolver()
            known = frozenset(k for k in getattr(resolver, "_names", {}).keys())
        else:
            print("[multilevel] org gate off: using plan atoms + checker only (no KG load)", flush=True)
        for level in [int(x) for x in args.levels.split(",") if x.strip()]:
            accepted, rejected = rewrite_surfaces(
                plans, level=level, chat=chat, model=args.model, retries=args.retries,
                variants=args.variants, org_resolver=resolver, known_orgs=known,
                checker_model=args.checker_model, progress_every=args.progress_every)
            write_jsonl(out_dir / f"surfaces.L{level}.jsonl", accepted)
            write_jsonl(out_dir / f"surfaces.L{level}.rejected.jsonl", rejected)
            reasons = Counter(r for row in rejected for r in row.get("reasons", [row.get("reason", "?")]))
            summary[f"L{level}"] = {"accepted": len(accepted), "rejected": len(rejected),
                                    "top_reject_reasons": dict(reasons.most_common(8))}
            print(f"  L{level}: accepted={len(accepted)} rejected={len(rejected)}")
    else:
        print("  (LLM off: only plan_bank + L1 written; run with --llm on for L2/L3)")

    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
