#!/usr/bin/env python3
"""Surface-diversity pass over the curated core (GrailQA/LC-QuAD recipe, LLM edition).

Reference recipe (LC-QuAD / GrailQA): template -> pseudo-NL -> crowd paraphrase -> crowd
cross-validation. Our adaptation:
  crowd paraphrase   -> nano rewrites CONDITIONED on an explicit style axis (six modes, cycled
                        deterministically) — naive LLM paraphrasing collapses to low diversity,
                        the mode conditioning is what buys spread;
  cross-validation   -> the project's MECHANICAL fidelity checks (all gold literals must appear
                        verbatim, no invented years/codes, category-position drift rule) — our
                        LLM checker was measured non-discriminative (9762/9762 True), the
                        mechanical line is the one that caught 0017;
  annotation inherit -> surface-only rewrite; gold plans, oracles and ids stay fixed.

Application policy (user decision 2026-07-04):
  - eval splits (final_test / dev_tune / dev_select): rewrite IN PLACE a hash-selected 50% of
    rows (original kept in metadata.original_question) — benchmark measures surface robustness
    without growing eval cost;
  - train: ADD one styled variant (id#dv) for a hash-selected 50% of plans;
  - dev_smoke and surplus.jsonl: untouched.

Every accepted rewrite records trigram-overlap-vs-source and length delta; the acceptance
report is the diversity evidence for the dataset card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODES: list[tuple[str, str]] = [
    ("terse_query", "Rewrite as a terse search-box style query, at most 12 words, keyword-like "
                    "but still unambiguous. No greetings, no filler."),
    ("verbose_context", "Rewrite as TWO sentences: first a short scene-setting sentence with "
                        "harmless context that adds NO new constraints, then the full question."),
    ("embedded_request", "Rewrite as ONE sentence that embeds the question indirectly in a "
                         "polite request (e.g. 'Could you confirm ... for me').") ,
    ("syntactic_flip", "Rewrite with a STRUCTURALLY different syntax: use passive voice, a cleft "
                       "('It was in 2024 that ...'), or front a constraint ('Under CPV X, how many "
                       "...'). At least half the word order must change; returning the original "
                       "sentence with minor edits is a failure."),
    ("multi_sentence", "Rewrite as a background statement sentence followed by a short direct "
                       "question sentence."),
    ("typo_noise", "Rewrite casually with 1-2 realistic typos or lowercase style, as a hurried "
                   "user would type. Meaning and every number/name must stay exact."),
]
EVAL_MODES = MODES[:5]  # no typo noise in evaluation splits
CAT_INJECT_RE = re.compile(r"\b(goods|services|works)\s+(?:notices?|contracts?|tenders?)\b", re.I)


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def h(key: str) -> int:
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)


def required_literals(row: dict[str, Any]) -> list[str]:
    """Every literal the rewrite must preserve verbatim (case-insensitive for org names)."""
    out: list[str] = []
    gp = row.get("gold_plan") or {}
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ("buyer", "supplier") and isinstance(v, str):
                    out.append(v)
                else:
                    walk(v)
    for c in gp.get("constraints", []) or []:
        f, v = c.get("field"), c.get("value")
        if f in ("release_year",):
            out.append(str(v))
        elif f in ("tender_cpv_id",):
            out.append(str(v))
        elif f in ("buyer_name", "supplier_name"):
            (out.extend(str(x) for x in v) if isinstance(v, list) else out.append(str(v)))
        elif f in ("tender_title",):
            out.append(str(v))
        elif c.get("op") == "in_subquery":
            walk(v)
    params = (gp.get("metadata") or {}).get("compare_params") or {}
    out.extend(str(s) for s in params.get("sides", []))
    if params.get("year"):
        out.append(str(params["year"]))
    return [x for x in dict.fromkeys(out) if x and x.casefold() != "none"]


def _abstain_cues(row: dict[str, Any]) -> list[str]:
    """Poison-pill / ambiguity cues an abstain question must keep through a rewrite."""
    try:
        from procurement_graph.reasoning.linking import UNSUPPORTED_TERMS
        terms = list(UNSUPPORTED_TERMS)
    except Exception:
        terms = []
    terms += ["bidder", "bidders", "reasonable", "judged", "fair"]
    q = str(row.get("question") or "").casefold()
    return [term for term in terms if term in q]


def fidelity_ok(row: dict[str, Any], new_q: str) -> tuple[bool, str]:
    ql = " ".join(new_q.casefold().split())
    if str(row.get("expected_status", "answerable")) != "answerable":
        for cue in _abstain_cues(row):
            if cue not in ql:
                return False, f"abstain_cue_lost:{cue}"
    for lit in required_literals(row):
        if " ".join(str(lit).casefold().split()) not in ql:
            return False, f"missing_literal:{str(lit)[:30]}"
    old_years = set(re.findall(r"\b20[2-3]\d\b", str(row.get("question"))))
    new_years = set(re.findall(r"\b20[2-3]\d\b", new_q))
    if new_years - old_years:
        return False, f"invented_year:{sorted(new_years - old_years)}"
    old_cpvs = set(re.findall(r"\b\d{8}\b", str(row.get("question"))))
    new_cpvs = set(re.findall(r"\b\d{8}\b", new_q))
    if new_cpvs - old_cpvs:
        return False, "invented_cpv"
    gp = row.get("gold_plan") or {}
    if str(gp.get("answer_operation")) in ("count", "sum"):
        m = CAT_INJECT_RE.search(new_q)
        if m:
            cat = m.group(1).casefold()
            licensed = any(c.get("field") == "tender_category" and str(c.get("value")).casefold() == cat
                           for c in gp.get("constraints", []) or [])
            in_old = bool(CAT_INJECT_RE.search(str(row.get("question"))))
            if not licensed and not in_old:
                return False, f"category_injection:{cat}"
    return True, ""


def trigram_overlap(a: str, b: str) -> float:
    def grams(s: str) -> set:
        toks = s.casefold().split()
        return {tuple(toks[i:i + 3]) for i in range(max(0, len(toks) - 2))}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 1.0
    return len(ga & gb) / len(ga | gb)  # Jaccard: wrap-style modes no longer score 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", type=Path, default=Path("data/qa/cicada_core_v3/all.jsonl"))
    ap.add_argument("--scarce", type=Path, default=Path("data/qa/scarce_fill_v4.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/qa/cicada_core_v4"))
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap rewrite jobs (smoke test)")
    args = ap.parse_args()

    from procurement_graph.qa.benchmark.chat import ChatClient
    chat = ChatClient.from_env(temperature=0.7)  # diversity needs sampling temperature

    rows = load(args.core) + load(args.scarce)
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "rewrites.cache.jsonl"
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for r in load(cache_path):
            cache[r["job_id"]] = r

    # ---- build jobs --------------------------------------------------------------------
    jobs: list[dict[str, Any]] = []
    train_plans_seen: set[str] = set()
    for r in rows:
        split = str(r.get("split"))
        if split == "dev_smoke":
            continue
        if split in ("final_test", "dev_tune", "dev_select"):
            if h("evalpick::" + str(r["id"])) % 2 == 0:
                pool = EVAL_MODES
                if str(r.get("expected_status", "answerable")) != "answerable":
                    # terse compression destroys the ambiguity/unsupported cues abstain
                    # questions rely on (measured: 1416#dv answered True on an ambiguous row)
                    pool = [m for m in EVAL_MODES if m[0] != "terse_query"]
                mode = pool[h("mode::" + str(r["id"])) % len(pool)]
                jobs.append({"job_id": f"inplace::{r['id']}", "row_id": r["id"],
                             "kind": "inplace", "mode": mode[0], "instruction": mode[1]})
        elif split == "train":
            pid = str(r.get("plan_id"))
            if pid in train_plans_seen:
                continue
            train_plans_seen.add(pid)
            if h("trainpick::" + pid) % 2 == 0:
                pool = MODES
                if str(r.get("expected_status", "answerable")) != "answerable":
                    pool = [m for m in MODES if m[0] not in ("terse_query", "typo_noise")]
                mode = pool[h("mode::" + pid) % len(pool)]
                jobs.append({"job_id": f"variant::{r['id']}", "row_id": r["id"],
                             "kind": "variant", "mode": mode[0], "instruction": mode[1]})
    if args.limit:
        jobs = jobs[: args.limit]
    by_id = {str(r["id"]): r for r in rows}
    pending = [j for j in jobs if j["job_id"] not in cache]
    print(f"[diversify] rows={len(rows)} jobs={len(jobs)} pending={len(pending)}", flush=True)

    lock = threading.Lock()
    system = ("You rewrite ONE procurement question into a target style. The rewrite must ask "
              "for EXACTLY the same thing: never add, drop, or alter any constraint, number, "
              "year, CPV code, organisation name, or quoted title (quoted titles stay verbatim "
              "in double quotes). Output ONLY the rewritten question text on a single line.")

    def run_job(job: dict[str, Any]) -> dict[str, Any]:
        row = by_id[job["row_id"]]
        user = json.dumps({
            "question": row["question"],
            "style": job["mode"],
            "style_instruction": job["instruction"],
            "must_keep_verbatim": required_literals(row),
        }, ensure_ascii=False)
        try:
            result = chat.complete_text(model=args.model, system=system, user=user)
            text = str(getattr(result, "raw_text", result) or "").strip().strip('"').strip()
            text = " ".join(text.split())
        except Exception as exc:  # noqa: BLE001 - live boundary
            return {**job, "ok": False, "reject": f"llm_error:{exc!r}"[:120]}
        if not text or len(text) < 8:
            return {**job, "ok": False, "reject": "empty_output"}
        ok, why = fidelity_ok(row, text)
        if not ok:
            return {**job, "ok": False, "reject": why}
        if job["mode"] == "syntactic_flip" and trigram_overlap(row["question"], text) > 0.8:
            return {**job, "ok": False, "reject": "flip_too_similar"}
        return {**job, "ok": True, "text": text,
                "trigram_overlap": round(trigram_overlap(row["question"], text), 3),
                "len_delta": len(text.split()) - len(str(row["question"]).split())}

    start = time.perf_counter()
    with cache_path.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_job, j): j for j in pending}
            done = 0
            for fut in as_completed(futures):
                rec = fut.result()
                with lock:
                    cache[rec["job_id"]] = rec
                    sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sink.flush()
                done += 1
                if done % 200 == 0:
                    print(f"[diversify] {done}/{len(pending)} elapsed={time.perf_counter()-start:.0f}s",
                          flush=True)

    # ---- assemble v4 --------------------------------------------------------------------
    out_rows: list[dict[str, Any]] = []
    stats = Counter()
    overlaps: list[float] = []
    for r in rows:
        rec = cache.get(f"inplace::{r['id']}")
        if rec and rec.get("ok"):
            r = dict(r)
            r.setdefault("metadata", {})["original_question"] = r["question"]
            r["metadata"]["surface_mode"] = rec["mode"]
            r["question"] = rec["text"]
            stats[f"inplace_ok:{rec['mode']}"] += 1
            overlaps.append(rec["trigram_overlap"])
        elif rec:
            stats[f"inplace_reject:{rec.get('reject','?').split(':')[0]}"] += 1
        out_rows.append(r)
        vrec = cache.get(f"variant::{r['id']}")
        if vrec and vrec.get("ok"):
            twin = json.loads(json.dumps(r, ensure_ascii=False))
            twin["id"] = r["id"] + "#dv"
            twin["question"] = vrec["text"]
            twin["dedup_group"] = str(r.get("dedup_group")) + "#dv"
            twin.setdefault("metadata", {})["surface_mode"] = vrec["mode"]
            twin["metadata"]["variant_of"] = r["id"]
            twin.setdefault("provenance", {})["generated_by"] = "qa_surface_diversify_v4"
            out_rows.append(twin)
            stats[f"variant_ok:{vrec['mode']}"] += 1
            overlaps.append(vrec["trigram_overlap"])
        elif vrec:
            stats[f"variant_reject:{vrec.get('reject','?').split(':')[0]}"] += 1

    # integrity: ids unique, no plan straddles train/eval
    ids = [str(r["id"]) for r in out_rows]
    assert len(ids) == len(set(ids)), "duplicate ids in v4"
    plan_splits: dict[str, set] = {}
    for r in out_rows:
        plan_splits.setdefault(str(r.get("plan_id")), set()).add(str(r.get("split")))
    bad = [p for p, s in plan_splits.items()
           if "train" in s and ({"final_test", "dev_select", "dev_tune", "dev_smoke"} & s)]
    assert not bad, f"plan/split integrity violated: {bad[:5]}"

    (args.out / "all.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in out_rows) + "\n",
        encoding="utf-8")
    for split in sorted({str(r.get("split")) for r in out_rows}):
        (args.out / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, default=str)
                      for r in out_rows if str(r.get("split")) == split) + "\n", encoding="utf-8")
    overlaps.sort()
    report = {
        "rows_in": len(rows), "rows_out": len(out_rows),
        "jobs": len(jobs), "accepted": sum(v for k, v in stats.items() if "_ok:" in k),
        "stats": dict(stats.most_common()),
        "trigram_overlap_median": overlaps[len(overlaps)//2] if overlaps else None,
        "trigram_overlap_p90": overlaps[int(len(overlaps)*0.9)] if overlaps else None,
        "split_sizes": dict(Counter(str(r.get("split")) for r in out_rows)),
    }
    (args.out / "diversity_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                                    encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
