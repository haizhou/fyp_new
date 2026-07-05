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
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.qa.multilevel import (  # noqa: E402
    PERSONAS, bridge_drift_reasons, check_surface, checker_accepts, checker_messages,
    l2_plan_ready, l2_rewrite_messages, persona_for, plan_bank_row, required_atoms,
    rewrite_messages, surface_row,
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


def assign_balanced_personas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cycle personas within each subset so small pilots do not cluster in one role."""
    names = sorted(PERSONAS)
    counters: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for row in rows:
        subset = str(row.get("subset", ""))
        persona_index = counters[subset] % len(names)
        persona = names[persona_index]
        counters[subset] += 1
        out.append({**row, "_persona": persona, "_persona_index": persona_index})
    return out


def persona_for_attempt(row: dict[str, Any], attempt_index: int, *, seed: int = 20260702) -> str:
    """Pick one of the persona prompts in a deterministic random order per plan.

    The first four attempts for one plan use all four prompt styles once, but the order is random
    under ``seed``. This keeps retries diverse without making runs irreproducible.
    """
    names = sorted(PERSONAS)
    digest = hashlib.sha256(f"{seed}:{row.get('id', '')}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    shuffled = names[:]
    rng.shuffle(shuffled)
    return shuffled[attempt_index % len(shuffled)]


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
    persona = str(row.get("_persona") or persona_for(row["id"]))
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
            bridge_reasons = bridge_drift_reasons(text, row) if level == 2 else ()
            if bridge_reasons:
                rejected.append({"plan_id": row["id"], "level": level,
                                 "candidate": text[:200], "reasons": list(bridge_reasons)})
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


def candidate_path(out_dir: Path, level: int) -> Path:
    return out_dir / f"surfaces.L{level}.candidates.jsonl"


def candidate_done_path(out_dir: Path, level: int) -> Path:
    path = candidate_path(out_dir, level)
    return path.with_suffix(path.suffix + ".done")


def _existing_candidate_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if path.exists():
        for row in read_jsonl(path):
            counts[str(row.get("plan_id", ""))] += 1
    return counts


def _accepted_plan_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if path.exists():
        for row in read_jsonl(path):
            counts[str(row.get("plan_id", ""))] += 1
    return counts


def _seen_candidate_ids(*paths: Path) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            cid = str(row.get("candidate_id", ""))
            if cid:
                seen.add(cid)
    return seen


def _pending_candidate_counts(candidates_path: Path, accepted_path: Path,
                              rejected_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not candidates_path.exists():
        return counts
    seen = _seen_candidate_ids(accepted_path, rejected_path)
    for row in read_jsonl(candidates_path):
        cid = str(row.get("candidate_id", ""))
        if cid and cid not in seen:
            counts[str(row.get("plan_id", ""))] += 1
    return counts


def generate_l2_candidates(rows: list[dict[str, Any]], *, chat: Any, model: str,
                           candidates_per_plan: int, out_path: Path,
                           accepted_path: Path | None = None,
                           rejected_path: Path | None = None,
                           new_candidates_per_plan: int = 1,
                           seed: int = 20260702,
                           progress_every: int = 25, resume: bool = True,
                           workers: int = 1, rpm: float = 0.0) -> list[dict[str, Any]]:
    """Nano-only pass: write candidate rewrites, with no Grok checker call."""
    existing = _existing_candidate_counts(out_path) if resume else Counter()
    accepted_counts = _accepted_plan_counts(accepted_path) if accepted_path and resume else Counter()
    pending_counts = (
        _pending_candidate_counts(out_path, accepted_path, rejected_path)
        if accepted_path and rejected_path and resume
        else Counter()
    )
    if not resume:
        out_path.unlink(missing_ok=True)
    todo = [
        row for row in rows
        if accepted_counts[str(row["id"])] <= 0
        and pending_counts[str(row["id"])] <= 0
        and existing[str(row["id"])] < candidates_per_plan
    ]
    written: list[dict[str, Any]] = read_jsonl(out_path) if resume and out_path.exists() else []
    skipped = len(rows) - len(todo)
    if todo:
        out_path.with_suffix(out_path.suffix + ".done").unlink(missing_ok=True)
    if skipped:
        print(f"    L2-generate: skipped {skipped}/{len(rows)} plans "
              "(accepted, pending review, or max attempts reached)",
              flush=True)
    processed = 0

    def call(row: dict[str, Any]) -> list[dict[str, Any]]:
        have = existing[str(row["id"])]
        need = min(new_candidates_per_plan, max(0, candidates_per_plan - have))
        if need <= 0:
            return []
        atoms = required_atoms(row)
        out = []
        for offset in range(have, have + need):
            persona = persona_for_attempt(row, offset, seed=seed)
            system, user = l2_rewrite_messages(row, atoms, n_variants=1, persona=persona)
            parsed = getattr(chat.complete_json(model=model, system=system, user=user), "parsed", {}) or {}
            variants = list(parsed.get("variants") or [])
            if variants:
                text = " ".join(str(variants[0]).split())
                out.append({
                    "candidate_id": f"{row['id']}#L2c{offset}",
                    "plan_id": row["id"],
                    "level": 2,
                    "candidate_index": offset,
                    "question": text,
                    "persona": persona,
                    "surface_origin": model,
                })
            else:
                out.append({
                    "candidate_id": f"{row['id']}#L2c{offset}",
                    "plan_id": row["id"],
                    "level": 2,
                    "candidate_index": offset,
                    "question": "",
                    "persona": persona,
                    "surface_origin": model,
                    "generation_error": "no_variants_returned",
                })
        if not out:
            out.append({
                "candidate_id": f"{row['id']}#L2c{have}",
                "plan_id": row["id"],
                "level": 2,
                "candidate_index": have,
                "question": "",
                "persona": persona_for_attempt(row, have, seed=seed),
                "surface_origin": model,
                "generation_error": "no_variants_returned",
            })
        return out

    def handle(_row: dict[str, Any], result: Any) -> None:
        nonlocal processed, written
        rows_out = ([{"candidate_id": f"{_row['id']}#L2error", "plan_id": _row["id"], "level": 2,
                      "question": "", "persona": str(_row.get("_persona", "")),
                      "surface_origin": model, "generation_error": repr(result[1])}]
                    if isinstance(result, tuple) and len(result) == 2 and result[0] == "error"
                    else result)
        append_jsonl(out_path, rows_out)
        written.extend(rows_out)
        processed += 1
        if progress_every and (processed % progress_every == 0 or processed == len(todo)):
            print(f"    L2-generate: processed {processed}/{len(todo)} this run "
                  f"({processed + skipped}/{len(rows)} total), candidates={len(written)}",
                  flush=True)

    run_concurrent(todo, call, workers=workers, rpm=rpm, on_result=handle)
    out_path.with_suffix(out_path.suffix + ".done").write_text(
        json.dumps({"plans": len(rows), "candidates": len(written)}, ensure_ascii=False),
        encoding="utf-8",
    )
    return written


def pending_l2_candidate_plan_count(rows: list[dict[str, Any]], *, candidates_path: Path,
                                    accepted_path: Path, rejected_path: Path,
                                    variants: int) -> int:
    if not candidates_path.exists():
        return 0
    accepted = read_jsonl(accepted_path) if accepted_path.exists() else []
    rejected = read_jsonl(rejected_path) if rejected_path.exists() else []
    accepted_counts: Counter[str] = Counter(str(r.get("plan_id", "")) for r in accepted)
    seen_candidates = {str(r.get("candidate_id", "")) for r in rejected}
    seen_candidates |= {str(r.get("candidate_id", "")) for r in accepted if r.get("candidate_id")}
    valid_plans = {str(r["id"]) for r in rows}
    pending = {
        str(c.get("plan_id", ""))
        for c in read_jsonl(candidates_path)
        if str(c.get("plan_id", "")) in valid_plans
        and str(c.get("candidate_id", "")) not in seen_candidates
        and accepted_counts[str(c.get("plan_id", ""))] < variants
    }
    return len(pending)


def check_l2_candidates(rows: list[dict[str, Any]], *, chat: Any, checker_model: str,
                        candidates_path: Path, accepted_path: Path, rejected_path: Path,
                        variants: int, org_resolver: Any, known_orgs: frozenset[str] | None,
                        progress_every: int = 25, resume: bool = True,
                        workers: int = 1, rpm: float = 0.0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Grok-only pass: consume generated candidates and write accepted/rejected surfaces."""
    if not candidates_path.exists():
        raise FileNotFoundError(f"candidate file missing: {candidates_path}")
    if not resume:
        accepted_path.unlink(missing_ok=True)
        rejected_path.unlink(missing_ok=True)
    accepted = read_jsonl(accepted_path) if resume and accepted_path.exists() else []
    rejected = read_jsonl(rejected_path) if resume and rejected_path.exists() else []
    accepted_counts: Counter[str] = Counter(str(r.get("plan_id", "")) for r in accepted)
    seen_candidates = {str(r.get("candidate_id", "")) for r in rejected}
    seen_candidates |= {str(r.get("candidate_id", "")) for r in accepted if r.get("candidate_id")}
    by_plan: dict[str, list[dict[str, Any]]] = {}
    for cand in read_jsonl(candidates_path):
        pid = str(cand.get("plan_id", ""))
        if not pid or str(cand.get("candidate_id", "")) in seen_candidates:
            continue
        if accepted_counts[pid] >= variants:
            continue
        by_plan.setdefault(pid, []).append(cand)
    plan_by_id = {str(row["id"]): row for row in rows}
    items = [(plan_by_id[pid], sorted(cands, key=lambda c: int(c.get("candidate_index", 0))))
             for pid, cands in by_plan.items() if pid in plan_by_id]
    processed = 0

    def call(item: tuple[dict[str, Any], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        row, cands = item
        a_rows: list[dict[str, Any]] = []
        r_rows: list[dict[str, Any]] = []
        atoms = required_atoms(row)
        for cand in cands:
            text = " ".join(str(cand.get("question", "")).split())
            cid = str(cand.get("candidate_id", ""))
            if cand.get("generation_error"):
                r_rows.append({"candidate_id": cid, "plan_id": row["id"], "level": 2,
                               "reason": f"generation_error:{cand.get('generation_error')}"})
                continue
            verdict = check_surface(text, atoms, row["question"], level=2,
                                    org_resolver=org_resolver, known_org_names=known_orgs,
                                    relax_org_texts=bool(checker_model))
            bridge_reasons = bridge_drift_reasons(text, row)
            if (not verdict.ok) or bridge_reasons:
                r_rows.append({"candidate_id": cid, "plan_id": row["id"], "level": 2,
                               "candidate": text[:200],
                               "reasons": list(verdict.reasons) + list(bridge_reasons)})
                continue
            checker_verdict = None
            if checker_model:
                c_system, c_user = checker_messages(text, row)
                checker_verdict = getattr(
                    chat.complete_json(model=checker_model, system=c_system, user=c_user),
                    "parsed", None)
                ok, why = checker_accepts(checker_verdict,
                                          expected_status=row.get("expected_status", "answerable"))
                if not ok:
                    r_rows.append({"candidate_id": cid, "plan_id": row["id"], "level": 2,
                                   "candidate": text[:200], "reasons": [why]})
                    continue
            srow = surface_row(row, level=2, question=text, origin=str(cand.get("surface_origin", "")),
                               variant=accepted_counts[str(row["id"])] + len(a_rows),
                               verdict=verdict)
            srow["candidate_id"] = cid
            srow["persona"] = cand.get("persona", "")
            srow["l2_generation_mode"] = "l1_persona_rewrite_decoupled"
            if checker_verdict is not None:
                srow["checker"] = checker_verdict
            a_rows.append(srow)
            if accepted_counts[str(row["id"])] + len(a_rows) >= variants:
                break
        return a_rows, r_rows

    def handle(_item: tuple[dict[str, Any], list[dict[str, Any]]], result: Any) -> None:
        nonlocal processed, accepted, rejected
        row, _cands = _item
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "error":
            a_rows: list[dict[str, Any]] = []
            r_rows = [{"plan_id": row["id"], "level": 2, "reason": f"worker_error:{result[1]!r}"}]
        else:
            a_rows, r_rows = result
        append_jsonl(accepted_path, a_rows)
        append_jsonl(rejected_path, r_rows)
        accepted.extend(a_rows)
        rejected.extend(r_rows)
        processed += 1
        if progress_every and (processed % progress_every == 0 or processed == len(items)):
            print(f"    L2-check: processed {processed}/{len(items)} plans, "
                  f"accepted={len(accepted)}, rejected={len(rejected)}", flush=True)

    run_concurrent(items, call, workers=workers, rpm=rpm, on_result=handle)
    return accepted, rejected


def run_l2_auto(rows: list[dict[str, Any]], *, chat: Any, model: str, checker_model: str,
                out_dir: Path, variants: int, base_candidates_per_plan: int,
                candidate_top_up: int, max_candidates_per_plan: int, max_rounds: int,
                org_resolver: Any, known_orgs: frozenset[str] | None, seed: int,
                progress_every: int, resume: bool, workers: int, rpm: float,
                checker_workers: int | None = None, checker_rpm: float | None = None) -> dict[str, Any]:
    """Unattended L2 driver: generate -> check, raising the per-plan candidate ceiling for any
    plan still short of `variants` accepted surfaces, until every plan is resolved or the ceiling
    / round budget is exhausted. One call = the whole L2 pass; no manual re-invocation needed.

    Each round is a complete generate-then-check pass (generate only tops plans up to the
    CURRENT ceiling; check consumes every candidate on disk), so after a round every plan is
    either accepted or provably stuck at that ceiling. Bounded by `max_candidates_per_plan` and
    `max_rounds` so a handful of unanswerable-looking or hard-to-phrase plans cannot run the
    loop (and the API bill) forever; anything left over is written to `l2_auto_report.json`
    with its reject-reason history so it can be inspected/re-run by hand later.

    `checker_workers`/`checker_rpm` default to `workers`/`rpm` (backward compatible) but should
    be set independently whenever the checker is a different deployment with a different rate
    limit than the generator (e.g. nano generating at 300 rpm, grok checking at 40 rpm) -- a
    single shared cap would either throttle the generator or hammer the checker past its limit.
    """
    check_workers = workers if checker_workers is None else checker_workers
    check_rpm = rpm if checker_rpm is None else checker_rpm
    start = time.time()
    cpath = candidate_path(out_dir, 2)
    accepted_path = out_dir / "surfaces.L2.jsonl"
    rejected_path = out_dir / "surfaces.L2.rejected.jsonl"
    all_ids = [str(r["id"]) for r in rows]

    # Resuming a previous auto run: start the ceiling at least as high as whatever is already
    # on disk for still-unresolved plans, so we do not replay already-exhausted rounds.
    ceiling = base_candidates_per_plan
    if resume and cpath.exists():
        existing_counts = _existing_candidate_counts(cpath)
        already_accepted = _accepted_plan_counts(accepted_path) if accepted_path.exists() else Counter()
        unresolved_existing = [existing_counts[pid] for pid in all_ids if already_accepted[pid] < variants]
        if unresolved_existing:
            ceiling = max(ceiling, min(max(unresolved_existing), max_candidates_per_plan))

    round_index = 0
    rounds_log: list[dict[str, Any]] = []
    while True:
        round_index += 1
        print(f"  L2-auto round {round_index}: ceiling={ceiling} candidates/plan", flush=True)
        generate_l2_candidates(
            rows, chat=chat, model=model, candidates_per_plan=ceiling, out_path=cpath,
            accepted_path=accepted_path, rejected_path=rejected_path,
            new_candidates_per_plan=candidate_top_up if round_index > 1 else base_candidates_per_plan,
            seed=seed, progress_every=progress_every, resume=True,
            workers=workers, rpm=rpm)
        accepted, rejected = check_l2_candidates(
            rows, chat=chat, checker_model=checker_model, candidates_path=cpath,
            accepted_path=accepted_path, rejected_path=rejected_path, variants=variants,
            org_resolver=org_resolver, known_orgs=known_orgs, progress_every=progress_every,
            resume=True, workers=check_workers, rpm=check_rpm)

        accepted_counts = Counter(str(r.get("plan_id", "")) for r in accepted)
        existing_counts = _existing_candidate_counts(cpath)
        unresolved = [pid for pid in all_ids if accepted_counts[pid] < variants]
        stuck = [pid for pid in unresolved if existing_counts[pid] >= ceiling]
        rounds_log.append({"round": round_index, "ceiling": ceiling, "accepted_total": len(accepted_counts),
                           "unresolved": len(unresolved), "stuck_at_ceiling": len(stuck)})
        print(f"    -> accepted_plans={len(accepted_counts)}/{len(all_ids)} "
              f"unresolved={len(unresolved)} stuck_at_ceiling={len(stuck)}", flush=True)

        if not unresolved:
            break
        if ceiling >= max_candidates_per_plan or round_index >= max_rounds:
            print(f"  L2-auto: stopping (ceiling_cap={ceiling >= max_candidates_per_plan}, "
                  f"round_cap={round_index >= max_rounds}); {len(unresolved)} plans unresolved", flush=True)
            break
        if not stuck:
            # every unresolved plan still has headroom under the current ceiling (e.g. a
            # generate call errored and needs one more pass) -> retry once more before raising it
            continue
        ceiling = min(max_candidates_per_plan, ceiling + candidate_top_up)

    accepted_final = read_jsonl(accepted_path) if accepted_path.exists() else []
    rejected_final = read_jsonl(rejected_path) if rejected_path.exists() else []
    accepted_counts = Counter(str(r.get("plan_id", "")) for r in accepted_final)
    unresolved_ids = [pid for pid in all_ids if accepted_counts[pid] < variants]
    reject_reasons_by_plan: dict[str, Counter[str]] = {}
    for row in rejected_final:
        pid = str(row.get("plan_id", ""))
        if pid in unresolved_ids:
            reject_reasons_by_plan.setdefault(pid, Counter()).update(
                row.get("reasons", [row.get("reason", "?")]))
    report = {
        "total_plans": len(all_ids),
        "accepted_plans": len(all_ids) - len(unresolved_ids),
        "unresolved_plans": len(unresolved_ids),
        "rounds_run": round_index,
        "final_ceiling": ceiling,
        "wall_clock_seconds": round(time.time() - start, 1),
        "rounds_log": rounds_log,
        "unresolved_plan_ids": unresolved_ids,
        "unresolved_reject_reasons": {pid: dict(c.most_common(5)) for pid, c in reject_reasons_by_plan.items()},
    }
    (out_dir / "l2_auto_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    return report


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
    ap.add_argument("--stage", choices=["all", "generate", "check", "inline", "auto"], default="all",
                    help="L2 workflow: generate writes nano candidates; check audits existing candidates; "
                         "all runs generate then check once; inline is the older per-plan path; "
                         "auto is the unattended driver -- loops generate+check, raising the "
                         "per-plan candidate ceiling, until every plan is resolved or capped "
                         "(--max-rounds / --max-candidates-per-plan). Use this for an overnight, "
                         "single-command full-scale run.")
    ap.add_argument("--candidates-per-plan", type=int, default=3,
                    help="maximum nano attempts per plan in the decoupled L2 candidate pool "
                         "(--stage auto: the STARTING ceiling, raised automatically thereafter)")
    ap.add_argument("--new-candidates-per-plan", type=int, default=1,
                    help="new nano attempts to add per eligible plan in this generate run")
    ap.add_argument("--max-candidates-per-plan", type=int, default=9,
                    help="--stage auto: hard cap on the per-plan candidate ceiling")
    ap.add_argument("--candidate-top-up", type=int, default=3,
                    help="--stage auto: how much to raise the ceiling each round for plans still stuck")
    ap.add_argument("--max-rounds", type=int, default=6,
                    help="--stage auto: hard cap on generate+check rounds")
    ap.add_argument("--watch", action="store_true",
                    help="with --stage check, poll the candidate file until generation writes .done")
    ap.add_argument("--poll-seconds", type=float, default=5.0,
                    help="poll interval for --stage check --watch")
    ap.add_argument("--checker-model", default="gpt-5.4-nano",
                    help="independent checker LLM for L2 semantic-equivalence ('' disables)")
    ap.add_argument("--org-gate", choices=["off", "kg"], default="off",
                    help="foreign-organisation gate: off = plan atoms + checker only; "
                         "kg = load KG org resolver and reject newly introduced KG org names")
    ap.add_argument("--progress-every", type=int, default=25,
                    help="print live progress every N source plans during LLM generation (0 disables)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel plan workers for LLM generation (and checking, unless "
                         "--checker-workers is set)")
    ap.add_argument("--rpm", type=float, default=45.0,
                    help="max plan starts per minute for LLM generation (and checking, unless "
                         "--checker-rpm is set)")
    ap.add_argument("--checker-workers", type=int, default=0,
                    help="parallel workers for the CHECKER model, if different from --workers "
                         "(0 = same as --workers); set this when checker and generator are "
                         "different deployments with different concurrency limits")
    ap.add_argument("--checker-rpm", type=float, default=0.0,
                    help="rpm cap for the CHECKER model, if different from --rpm (0 = same as "
                         "--rpm); e.g. nano generating at 300 rpm but a 50-rpm grok checker "
                         "needs --checker-rpm 40")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="resume from existing surfaces.L*.jsonl / rejected files (default: true)")
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.Random(args.seed)  # reserved for future samplers; stride keeps selection deterministic

    plans = assign_balanced_personas(sample_plans(args.per_subset, args.tag))
    print(f"[multilevel] {len(plans)} plans sampled ({args.per_subset}/subset, stride)")
    if not plans:
        print(f"[multilevel] no plans found for tag={args.tag!r}; check data\\qa\\targeted_v2\\full2k")
        return 2
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
            if level == 2 and args.stage == "auto":
                report = run_l2_auto(
                    plans, chat=chat, model=args.model, checker_model=args.checker_model,
                    out_dir=out_dir, variants=args.variants,
                    base_candidates_per_plan=args.candidates_per_plan,
                    candidate_top_up=args.candidate_top_up,
                    max_candidates_per_plan=args.max_candidates_per_plan, max_rounds=args.max_rounds,
                    org_resolver=resolver, known_orgs=known, seed=args.seed,
                    progress_every=args.progress_every, resume=args.resume,
                    workers=args.workers, rpm=args.rpm,
                    checker_workers=args.checker_workers or None, checker_rpm=args.checker_rpm or None)
                summary[f"L{level}"] = report
                print(f"  L{level}: accepted={report['accepted_plans']}/{report['total_plans']} "
                      f"in {report['rounds_run']} rounds ({report['wall_clock_seconds']}s)")
                continue
            if level == 2 and args.stage in {"all", "generate", "check"}:
                cpath = candidate_path(out_dir, level)
                dpath = candidate_done_path(out_dir, level)
                if args.stage in {"all", "generate"}:
                    if not args.resume:
                        cpath.unlink(missing_ok=True)
                        dpath.unlink(missing_ok=True)
                    generate_l2_candidates(
                        plans, chat=chat, model=args.model,
                        candidates_per_plan=args.candidates_per_plan, out_path=cpath,
                        accepted_path=accepted_path, rejected_path=rejected_path,
                        new_candidates_per_plan=args.new_candidates_per_plan,
                        seed=args.seed,
                        progress_every=args.progress_every, resume=args.resume,
                        workers=args.workers, rpm=args.rpm)
                if args.stage == "generate":
                    accepted = read_jsonl(accepted_path) if accepted_path.exists() else []
                    rejected = read_jsonl(rejected_path) if rejected_path.exists() else []
                    summary[f"L{level}"] = {
                        "accepted": len(accepted),
                        "rejected": len(rejected),
                        "candidates": len(read_jsonl(cpath)) if cpath.exists() else 0,
                    }
                    print(f"  L{level}: generated candidates={summary[f'L{level}']['candidates']}")
                    continue
                while True:
                    accepted, rejected = check_l2_candidates(
                        plans, chat=chat, checker_model=args.checker_model,
                        candidates_path=cpath, accepted_path=accepted_path, rejected_path=rejected_path,
                        variants=args.variants, org_resolver=resolver, known_orgs=known,
                        progress_every=args.progress_every, resume=args.resume,
                        workers=args.checker_workers or args.workers,
                        rpm=args.checker_rpm or args.rpm)
                    pending = pending_l2_candidate_plan_count(
                        plans, candidates_path=cpath, accepted_path=accepted_path,
                        rejected_path=rejected_path, variants=args.variants)
                    if not args.watch or (dpath.exists() and pending == 0):
                        break
                    print(f"    L{level}-check: waiting for candidates "
                          f"(pending_plans={pending}, done={dpath.exists()})", flush=True)
                    time.sleep(args.poll_seconds)
            else:
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
