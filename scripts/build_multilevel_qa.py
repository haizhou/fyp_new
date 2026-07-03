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
from procurement_graph.qa.benchmark.concurrency import run_concurrent  # noqa: E402

V2_DIR = ROOT / "data" / "qa" / "targeted_v2" / "full2k"
OUT_DIR = ROOT / "data" / "qa" / "multilevel"
SUBSETS = ("naturalized", "coverage_fixed", "unanswerable", "extended_ops", "bridge_join")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows),
                    encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


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


def completed_plan_ids(accepted_path: Path, rejected_path: Path, *, variants: int) -> set[str]:
    """Plans that do not need another live call in resume mode."""
    accepted_counts: Counter[str] = Counter()
    terminal_rejected: set[str] = set()
    if accepted_path.exists():
        for row in read_jsonl(accepted_path):
            accepted_counts[str(row.get("plan_id", ""))] += 1
    if rejected_path.exists():
        for row in read_jsonl(rejected_path):
            reason = str(row.get("reason", ""))
            if reason.startswith(("skipped:", "llm_error:")) or reason == "exhausted_retries":
                terminal_rejected.add(str(row.get("plan_id", "")))
    return {pid for pid, count in accepted_counts.items() if count >= variants} | terminal_rejected


def _rewrite_one(row: dict[str, Any], *, level: int, chat: Any, model: str, retries: int,
                 variants: int, org_resolver: Any, known_orgs: frozenset[str] | None,
                 checker_model: str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if level == 2:
        ready, reason = l2_plan_ready(row)
        if not ready:
            return [], [{"plan_id": row["id"], "level": level, "reason": f"skipped:{reason}"}]
    atoms = required_atoms(row)
    persona = persona_for(row["id"])
    got = 0
    for _attempt in range(retries + 1):
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
    if got == 0 and not any(str(r.get("reason", "")).startswith("llm_error") for r in rejected):
        rejected.append({"plan_id": row["id"], "level": level, "reason": "exhausted_retries"})
    return accepted, rejected


def rewrite_surfaces(rows, *, level, chat, model, retries, variants, org_resolver, known_orgs,
                     checker_model: str = "", progress_every: int = 25,
                     accepted_path: Path | None = None, rejected_path: Path | None = None,
                     resume: bool = False, workers: int = 1, rpm: float = 0.0):
    """Generate L2/L3 surfaces with optional checkpoint append + resume."""
    accepted: list[dict[str, Any]] = read_jsonl(accepted_path) if resume and accepted_path and accepted_path.exists() else []
    rejected: list[dict[str, Any]] = read_jsonl(rejected_path) if resume and rejected_path and rejected_path.exists() else []
    done = completed_plan_ids(accepted_path, rejected_path, variants=variants) if resume and accepted_path and rejected_path else set()
    todo = [row for row in rows if row["id"] not in done]
    skipped = len(rows) - len(todo)
    if skipped:
        print(f"    L{level}: resume skipped {skipped}/{len(rows)} completed plans", flush=True)

    processed = 0

    def handle(_row: dict[str, Any], result: Any) -> None:
        nonlocal processed, accepted, rejected
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "error":
            a_rows: list[dict[str, Any]] = []
            r_rows = [{"plan_id": _row["id"], "level": level, "reason": f"worker_error:{result[1]!r}"}]
        else:
            a_rows, r_rows = result
        append_jsonl(accepted_path, a_rows) if accepted_path else None
        append_jsonl(rejected_path, r_rows) if rejected_path else None
        accepted.extend(a_rows)
        rejected.extend(r_rows)
        processed += 1
        if progress_every and (processed % progress_every == 0 or processed == len(todo)):
            print(f"    L{level}: processed {processed}/{len(todo)} this run "
                  f"({processed + skipped}/{len(rows)} total), accepted={len(accepted)}, "
                  f"rejected={len(rejected)}", flush=True)

    def call(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return _rewrite_one(row, level=level, chat=chat, model=model, retries=retries,
                            variants=variants, org_resolver=org_resolver, known_orgs=known_orgs,
                            checker_model=checker_model)

    run_concurrent(todo, call, workers=workers, rpm=rpm, on_result=handle)
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
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel plan workers for LLM surface generation/checking")
    ap.add_argument("--rpm", type=float, default=45.0,
                    help="max plan starts per minute during LLM generation; keep <= checker RPM")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="resume from existing surfaces.L*.jsonl / rejected files (default: true)")
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
            accepted_path = out_dir / f"surfaces.L{level}.jsonl"
            rejected_path = out_dir / f"surfaces.L{level}.rejected.jsonl"
            if not args.resume:
                accepted_path.unlink(missing_ok=True)
                rejected_path.unlink(missing_ok=True)
            accepted, rejected = rewrite_surfaces(
                plans, level=level, chat=chat, model=args.model, retries=args.retries,
                variants=args.variants, org_resolver=resolver, known_orgs=known,
                checker_model=args.checker_model, progress_every=args.progress_every,
                accepted_path=accepted_path, rejected_path=rejected_path, resume=args.resume,
                workers=args.workers, rpm=args.rpm)
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
