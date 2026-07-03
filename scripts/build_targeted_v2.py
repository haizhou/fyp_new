#!/usr/bin/env python3
"""Targeted-v2 benchmark generators with resumable control-plane validation.

Read-only over v1 (`data/qa/generated/`) + the real KG; writes ONLY `<subset>_<rows>.jsonl` +
`<subset>_<rows>.validation.json` under --out-dir (never overwrites the pilot `_50`/`_pilot`
files, and never touches v1). Reuses the pilot's proven leaf helpers (`build_targeted_v2_pilot`)
and the benchmark LLM/Gate-B infra.

Subsets:
  naturalized     nano rewrite -> Gate-B faithfulness -> KG oracle revalidation -> retry/fallback
  coverage_fixed  drop answer-changing supplier/buyer guards, recompute oracle on KG
  unanswerable    balanced unsupported/ambiguous/no_results, each KG/schema-verified
  extended_ops    argmax/top_k/distinct_set/compare with executor_support + recomputable evidence
  bridge_join     ONLY in_subquery semijoins (requires_decomposition); quality-first, not padded

Every run emits a validation_summary (passed/failed, failure_reason, answer_type / expected_status /
executor_support distributions, evidence coverage, duplicate id/question check).

Offline test (no API, no full run):
  python -B scripts/build_targeted_v2.py --subset all --llm off --limit 8 --out-dir <scratch>
Full run (the user runs this):
  see the printed commands / README at the bottom of this file's --help.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from procurement_graph.qa.targeted_v2.builder import main as _targeted_v2_main


if __name__ == "__main__":
    raise SystemExit(_targeted_v2_main())

import build_targeted_v2_pilot as P  # proven leaf helpers (read_jsonl, cons_of, _row, renders, ...)
from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
from procurement_graph.reasoning.models import QueryConstraint as QC
from procurement_graph.qa.benchmark.models import AnswerSpec, Constraint

GEN = ROOT / "data" / "qa" / "generated"
DEFAULT_OUT = ROOT / "data" / "qa" / "targeted_v2"
DEFAULT_N = {"naturalized": 2000, "coverage_fixed": 2000, "unanswerable": 2000,
             "extended_ops": 2000, "bridge_join": 1000}


# --------------------------------------------------------------------------- naturalized (LLM)

def _answer_spec(source: dict[str, Any], answer_field: str) -> AnswerSpec:
    cons = tuple(Constraint(c["field"], c["op"], c.get("value")) for c in P.visible(source["constraints"]))
    return AnswerSpec(spec_id=source["spec_id"], constraints=cons,
                      answer_operation=source["answer_operation"], answer_field=answer_field,
                      answer_value_type=source.get("answer_value_type", "string"))


def _rewrite_messages(canonical: str, answer_type: str) -> tuple[str, str]:
    system = (
        "You rephrase a UK public-procurement QA question into a NATURAL, differently-worded "
        "question that asks for EXACTLY the same thing. Rules: keep every filter (year, category, "
        "CPV, buyer, supplier) unchanged; add or remove NOTHING; never reveal or compute the answer; "
        "vary syntax and wording (not a template). Return strict JSON: {\"question\": \"...\"}."
    )
    user = json.dumps({"canonical_question": canonical, "answer_type": answer_type}, ensure_ascii=False)
    return system, user


def build_naturalized(bench, backend, afield, *, n, limit, chat, verifier, model, retries) -> tuple[list, list]:
    rows, failures = [], []
    counts = P._pick(bench, "filtered_count", n)
    sums = P._pick(bench, "additive_sum", n)
    facts = [b for b in P._pick(bench, "contract_factoid", n * 2) if afield.get(b["spec_id"]) != "value_source"]
    plan = ([("count", b) for b in counts] + [("sum", b) for b in sums] + [("factoid", b) for b in facts])
    target = limit or n
    seq = 0
    for kind, b in plan:
        if len(rows) >= target:
            break
        vc = P.visible(b["constraints"])
        # --- oracle / evidence via KG revalidation ---
        if kind == "factoid":
            cid = P._cval(b["constraints"], "contract_node_id")
            rec = P._record(backend, cid)
            field = afield.get(b["spec_id"], "")
            if rec is None or not field:
                failures.append({"id": b["spec_id"], "reason": "factoid_source_unresolved"}); continue
            buyer, supplier, year, cpv = rec.get("buyer_name"), rec.get("supplier_name"), rec.get("release_year"), rec.get("tender_cpv_id")
            anchor = [{"field": "buyer_name", "op": "eq", "value": buyer}, {"field": "supplier_name", "op": "eq", "value": supplier},
                      {"field": "release_year", "op": "eq", "value": year}, {"field": "tender_cpv_id", "op": "eq", "value": cpv}]
            matches = backend.query(P.cons_of(anchor))
            if len({str(r.get(field)) for r in matches}) != 1:
                failures.append({"id": b["spec_id"], "reason": "anchor_not_unique"}); continue
            cons, oracle, ev, evc = anchor, rec.get(field), P.rec_ids(backend, matches), len(matches)
            canonical = P._render_factoid(buyer, supplier, year, cpv, field, 0)
            spec_field = field
        else:
            guard = P.cons_of(vc) + ((QC("value_is_additive", "eq", True),) if kind == "sum" else ())
            rs = backend.query(guard)
            if kind == "count" and len(rs) != b["golden_answer"]:
                failures.append({"id": b["spec_id"], "reason": "kg_revalidation_mismatch"}); continue
            cons, oracle, ev, evc = vc, b["golden_answer"], P.rec_ids(backend, rs), len(rs)
            canonical = P._render_count(vc, 0) if kind == "count" else P._render_sum(vc, 0)
            spec_field = "value_amount" if kind == "sum" else "contract_node_id"

        # --- rewrite: nano + Gate-B faithfulness, retry, fallback to template ---
        atype = kind
        question, source_of_q, faith_reason = canonical, "template", "llm_off"
        if chat is not None:
            spec = _answer_spec(b, spec_field)
            for attempt in range(retries):
                try:
                    res = chat.complete_json(model=model, system=_rewrite_messages(canonical, atype)[0],
                                             user=_rewrite_messages(canonical, atype)[1])
                    cand = str((res.parsed or {}).get("question", "")).strip()
                except Exception as exc:
                    faith_reason = f"rewrite_call_failed: {exc}"; continue
                if not cand:
                    faith_reason = "rewrite_empty"; continue
                outcome = verifier.verify_faithfulness(spec, cand)
                if outcome.verified:
                    question, source_of_q, faith_reason = cand, "nano", "faithful"; break
                faith_reason = f"faithfulness_failed: {outcome.reason}"
            if source_of_q != "nano":
                source_of_q = "template_fallback"  # kept, but flagged; row still valid via template

        rows.append(P._row("naturalized", seq, question, atype,
                            {"count": "count", "sum": "sum", "factoid": "select_unique"}[kind],
                            "answerable", cons, oracle, b, evidence_ids=ev, ev_count=evc,
                            notes=f"paraphrase of {b['spec_id']}; rewrite={source_of_q}; {faith_reason}",
                            extra={"canonical_question": canonical, "rewrite_source": source_of_q}))
        seq += 1
    return rows, failures


# --------------------------------------------------------------------------- coverage_fixed

def build_coverage_fixed(bench, backend, *, n, limit) -> tuple[list, list]:
    pool = sorted((b for b in bench if b["operation_family"] == "conjunction"),
                  key=lambda b: (-int(b.get("evidence_count") or 0), b["spec_id"]))
    rows, failures, target = [], [], limit or n
    for seq, b in enumerate(pool):
        if len(rows) >= target:
            break
        vc = P.visible(b["constraints"])
        removed = [c["field"] for c in b["constraints"] if c not in vc]
        rs = backend.query(P.cons_of(vc))
        rows.append(P._row("coverage_fixed", seq, b["question"], "count", "count", "answerable",
                           vc, len(rs), b, evidence_ids=P.rec_ids(backend, rs), ev_count=len(rs),
                           notes=(f"from {b['spec_id']}; removed guards {removed}; "
                                  f"v1_oracle={b['golden_answer']} -> recomputed={len(rs)}"
                                  f"{' (changed)' if len(rs) != b['golden_answer'] else ''}")))
    return rows, failures


# --------------------------------------------------------------------------- unanswerable

def build_unanswerable(bench, backend, df, *, n, limit) -> tuple[list, list]:
    rows, failures = [], []
    target = limit or n
    per = max(1, target // 3)
    facts = P._pick(bench, "contract_factoid", target)
    seq = 0

    def add(q, atype, aop, status, cons, support, notes):
        nonlocal seq
        rows.append(P._row("unanswerable", seq, q, atype, aop, status, cons, None, None,
                           support=support, notes=notes)); seq += 1

    # unsupported: real anchor + non-KG field
    for i, b in enumerate(facts):
        if sum(1 for r in rows if r["expected_status"] == "unsupported") >= per:
            break
        rec = P._record(backend, P._cval(b["constraints"], "contract_node_id"))
        if rec is None:
            failures.append({"id": b["spec_id"], "reason": "anchor_unresolved"}); continue
        concept = P.NON_KG_CONCEPTS[i % len(P.NON_KG_CONCEPTS)]
        add(f"For the contract awarded by {rec.get('buyer_name')} to {rec.get('supplier_name')} in "
            f"{rec.get('release_year')}, {concept}?", "unsupported", "none", "unsupported", [], "n/a",
            f"asks '{concept}': not a KG v0.1 field")

    # ambiguous: buyer with a field that varies across its contracts (vary the asked field for volume)
    big_buyers = [str(x) for x in df["buyer_name"].value_counts().head(1200).index if str(x).strip()]
    ask = [("tender_cpv_id", "CPV code"), ("supplier_name", "awarded supplier"),
           ("tender_category", "procurement category")]
    for buyer in big_buyers:
        if sum(1 for r in rows if r["expected_status"] == "ambiguous") >= per:
            break
        rs = backend.query((QC("buyer_name", "eq", buyer),))
        for fld, phrase in ask:
            if len({str(r.get(fld)) for r in rs}) >= 3:
                add(f"What is the {phrase} of the contract awarded by {buyer}?", "ambiguous", "none",
                    "ambiguous", [{"field": "buyer_name", "op": "eq", "value": buyer}], "n/a",
                    f"anchor matches {len(rs)} contracts / {len({str(r.get(fld)) for r in rs})} distinct {fld}")
                break

    # no_results: real buyer + a CPV it never used
    all_cpvs = [str(x) for x in df["tender_cpv_id"].dropna().unique()]
    for buyer in big_buyers:
        if sum(1 for r in rows if r["expected_status"] == "no_results") >= per or len(rows) >= target:
            break
        used = {str(r.get("tender_cpv_id")) for r in backend.query((QC("buyer_name", "eq", buyer),))}
        unused = next((c for c in all_cpvs if c not in used), None)
        if unused is None or backend.query((QC("buyer_name", "eq", buyer), QC("tender_cpv_id", "eq", unused))):
            continue
        add(f"Who was the awarded supplier for the contract by {buyer} under CPV {unused}?", "factoid",
            "select_unique", "no_results", [{"field": "buyer_name", "op": "eq", "value": buyer},
            {"field": "tender_cpv_id", "op": "eq", "value": unused}], "supported",
            f"buyer never used CPV {unused}: 0 matches (verified)")
    return rows, failures


# --------------------------------------------------------------------------- extended_ops

def build_extended_ops(df, backend, *, n, limit) -> tuple[list, list]:
    import pandas as pd
    rows, failures, target = [], [], limit or n
    d = df.copy(); d["_val"] = pd.to_numeric(d.get("value_amount"), errors="coerce")
    seq = 0

    def add(q, atype, aop, cons, oracle, ev, support, notes, evc=None, extra=None):
        nonlocal seq
        rows.append(P._row("extended_ops", seq, q, atype, aop, "answerable", cons, oracle, None,
                           support=support, decomp=False, notes=notes, evidence_ids=ev, ev_count=evc, extra=extra))
        seq += 1

    valid = d[d["_val"].notna() & d["tender_category"].notna() & (d["tender_category"].astype(str).str.strip() != "")
              & d["tender_cpv_id"].notna() & (d["tender_cpv_id"].astype(str).str.strip() != "")]
    slices = valid.groupby(["tender_category", "tender_cpv_id"]).size().sort_values(ascending=False)
    quota = target // 4
    # argmax
    for cat, cpv in slices.index:
        if sum(1 for r in rows if r["answer_type"] == "min_max") >= quota or len(rows) >= target:
            break
        sub = valid[(valid["tender_category"] == cat) & (valid["tender_cpv_id"] == cpv)]
        cid = str(backend.record_id(sub.loc[sub["_val"].idxmax()].to_dict()))
        add(f"Which contract has the highest value among {cat} notices under CPV {cpv}?", "min_max", "argmax",
            [{"field": "tender_category", "op": "eq", "value": cat}, {"field": "tender_cpv_id", "op": "eq", "value": cpv}],
            cid, [cid], "needs_op:argmax", "argmax over value_amount on a filtered slice", evc=len(sub))
    # top_k (inherently few: category x year x k)
    for cat in ["services", "goods", "works"]:
        for yr in (2022, 2023, 2024, 2025, 2026):
            sub = d[(d["tender_category"] == cat) & (d["release_year"] == yr)]
            vc = sub["buyer_name"].value_counts().head(3)
            if len(vc) < 3:
                continue
            add(f"What are the top 3 buyers by number of {cat} contract notices published in {yr}?",
                "top_k", "rank_top_k", [{"field": "tender_category", "op": "eq", "value": cat}, {"field": "release_year", "op": "eq", "value": yr}],
                [[str(b), int(x)] for b, x in vc.items()], [], "needs_op:top_k",
                "group by buyer_name, count, rank, head-3", evc=int(len(sub)),
                extra={"group_by_field": "buyer_name", "metric": "count", "k": 3, "evidence_kind": "aggregate_recomputable"})
    # distinct_set
    for buyer in [str(x) for x in df["buyer_name"].value_counts().index]:
        if sum(1 for r in rows if r["answer_type"] == "set_list") >= quota or len(rows) >= target:
            break
        rs = backend.query((QC("buyer_name", "eq", buyer), QC("tender_category", "eq", "works")))
        sup = sorted({str(r.get("supplier_name")) for r in rs if r.get("supplier_name")})
        if not sup:
            continue
        add(f"List the distinct suppliers awarded works contracts by {buyer}.", "set_list", "distinct_set",
            [{"field": "buyer_name", "op": "eq", "value": buyer}, {"field": "tender_category", "op": "eq", "value": "works"}],
            sup[:50], P.rec_ids(backend, rs), "needs_op:distinct_set", "distinct supplier_name over the match set", evc=len(rs))
    # compare (two independent counts)
    tb = [str(x) for x in df["buyer_name"].value_counts().head(400).index]
    for i in range(0, len(tb) - 1, 2):
        if len(rows) >= target:
            break
        bx, by = tb[i], tb[i + 1]
        nx = len(backend.query((QC("buyer_name", "eq", bx), QC("release_year", "eq", 2024))))
        ny = len(backend.query((QC("buyer_name", "eq", by), QC("release_year", "eq", 2024))))
        add(f"Did {bx} publish more contract notices than {by} in 2024?", "comparison", "compare",
            [{"field": "release_year", "op": "eq", "value": 2024}], {"answer": bool(nx > ny), bx: nx, by: ny}, [],
            "needs_op:compare", "two INDEPENDENT sub-counts + comparator (parallel, not a bridge)", evc=nx + ny,
            extra={"comparison_breakdown": {bx: {"count": nx}, by: {"count": ny}}, "evidence_kind": "two_subcounts_recomputable"})
    return rows, failures


# --------------------------------------------------------------------------- bridge_join (pure)

def build_bridge_join(df, backend, *, n, limit) -> tuple[list, list]:
    import pandas as pd
    rows, failures, target = [], [], limit or n
    d = df.copy(); d["_val"] = pd.to_numeric(d.get("value_amount"), errors="coerce")
    seq = 0

    def add(q, atype, aop, cons, oracle, ev, notes, evc):
        nonlocal seq
        rows.append(P._row("bridge_join", seq, q, atype, aop, "answerable", cons, oracle, None,
                           support="needs_op:in_subquery", decomp=True, notes=notes, evidence_ids=ev, ev_count=evc))
        seq += 1

    for buyer in [str(x) for x in df["buyer_name"].value_counts().head(target).index]:
        if len(rows) >= target:
            break
        suppliers = {str(r.get("supplier_name")) for r in backend.query((QC("buyer_name", "eq", buyer),)) if r.get("supplier_name")}
        sub = d[d["supplier_name"].astype(str).isin(suppliers) & d["_val"].notna()]
        if len(suppliers) < 2 or sub.empty:
            continue
        add(f"What is the total value of contracts awarded to suppliers who also won a contract from {buyer}?",
            "sum", "sum", [{"field": "supplier_name", "op": "in_subquery", "value": {"resolve": "suppliers_of_buyer", "buyer": buyer}}],
            round(float(sub["_val"].sum()), 2), P.rec_ids(backend, sub.head(50).to_dict("records")),
            "hop1: suppliers of buyer -> hop2: sum value over their contracts", int(len(sub)))
    for supplier in [str(x) for x in df["supplier_name"].value_counts().head(target).index if str(x).strip()]:
        if len(rows) >= target:
            break
        buyers = {str(r.get("buyer_name")) for r in backend.query((QC("supplier_name", "eq", supplier),)) if r.get("buyer_name")}
        if len(buyers) < 2:
            continue
        sub = d[d["buyer_name"].astype(str).isin(buyers)]
        add(f"How many contract notices were published by buyers who have awarded a contract to {supplier}?",
            "count", "count", [{"field": "buyer_name", "op": "in_subquery", "value": {"resolve": "buyers_of_supplier", "supplier": supplier}}],
            int(len(sub)), P.rec_ids(backend, sub.head(50).to_dict("records")),
            "hop1: buyers of supplier -> hop2: count their notices", int(len(sub)))
    return rows, failures


# --------------------------------------------------------------------------- validation

def validation_summary(subset, rows, failures, n_target, filename) -> dict:
    answerable = [r for r in rows if r["expected_status"] == "answerable"]
    with_ev = sum(1 for r in answerable if r["evidence_ids"] or r.get("evidence_count") is not None)
    ids = [r["id"] for r in rows]
    qs = [r["question"] for r in rows]
    return {
        "subset": subset, "file": filename, "n_target": n_target,
        "passed": len(rows), "failed": len(failures),
        "failure_reason": dict(Counter(f["reason"] for f in failures)),
        "answer_type": dict(Counter(r["answer_type"] for r in rows)),
        "expected_status": dict(Counter(r["expected_status"] for r in rows)),
        "executor_support": dict(Counter(r["executor_support"] for r in rows)),
        "evidence_coverage": f"{with_ev}/{len(answerable)}",
        "rewrite_source": dict(Counter(r.get("rewrite_source", "n/a") for r in rows)) if subset == "naturalized" else {},
        "duplicate_ids": len(ids) - len(set(ids)),
        "duplicate_questions": len(qs) - len(set(qs)),
    }


def _build(subset, args, bench, backend, df, afield, chat, verifier):
    n = args.n or DEFAULT_N[subset]
    kw = dict(n=n, limit=args.limit)
    if subset == "naturalized":
        rows, failures = build_naturalized(bench, backend, afield, chat=chat, verifier=verifier,
                                           model=args.model, retries=args.retries, **kw)
    elif subset == "coverage_fixed":
        rows, failures = build_coverage_fixed(bench, backend, **kw)
    elif subset == "unanswerable":
        rows, failures = build_unanswerable(bench, backend, df, **kw)
    elif subset == "extended_ops":
        rows, failures = build_extended_ops(df, backend, **kw)
    else:
        rows, failures = build_bridge_join(df, backend, **kw)
    return rows, failures, n


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", choices=["all", *DEFAULT_N], default="all")
    ap.add_argument("--n", type=int, default=0, help="target rows (0 = per-subset default)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows for a quick test (0 = full n)")
    ap.add_argument("--llm", choices=["on", "off"], default="on", help="off = naturalized uses template + dry-run verifier")
    ap.add_argument("--model", default="gpt-5.4-nano", help="naturalized rewrite model")
    ap.add_argument("--verify-model", default="grok-4-1-fast-non-reasoning", help="Gate-B faithfulness model")
    ap.add_argument("--retries", type=int, default=2, help="nano rewrite attempts before template fallback")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    bench = P.read_jsonl(GEN / "benchmark.jsonl")
    afield = {r["spec"]["spec_id"]: r["spec"].get("answer_field", "") for r in P.read_jsonl(GEN / "answer_specs.jsonl")}
    print("[v2] loading KG ...", flush=True)
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    df = backend._backend.records_df

    chat = verifier = None
    subsets = list(DEFAULT_N) if args.subset == "all" else [args.subset]
    if "naturalized" in subsets and args.llm == "on":
        from procurement_graph.qa.benchmark.chat import ChatClient
        from procurement_graph.qa.benchmark.gate_b import LLMGateBVerifier
        chat = ChatClient.from_env()
        verifier = LLMGateBVerifier(client=chat, model=args.verify_model)
    elif "naturalized" in subsets:
        from procurement_graph.qa.benchmark.gate_b import DryRunGateBVerifier
        verifier = DryRunGateBVerifier()  # chat stays None -> template path

    summaries = {}
    for subset in subsets:
        rows, failures, n = _build(subset, args, bench, backend, df, afield, chat, verifier)
        filename = f"{subset}_{len(rows)}.jsonl"  # name = ACTUAL row count (never misleading)
        (out / filename).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        s = validation_summary(subset, rows, failures, n, filename)
        (out / f"{subset}.validation.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries[subset] = s
        print(f"  {filename}: passed={s['passed']} failed={s['failed']} types={s['answer_type']} "
              f"evidence={s['evidence_coverage']} dup_id={s['duplicate_ids']} dup_q={s['duplicate_questions']}", flush=True)
    (out / "validation_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(subsets)} subset(s) + validation_summary.json to {out}")


if __name__ == "__main__":
    main()
