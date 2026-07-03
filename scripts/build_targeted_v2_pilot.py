#!/usr/bin/env python3
"""Build the targeted-v2 benchmark PILOT (50 rows/subset) for human review before scaling to 2k.

Read-only over v1 (`data/qa/generated/`) and the real KG (`data/kg/`); writes only under
`data/qa/targeted_v2/`. Deterministic, no LLM calls. Every row is verified against the KG at build
time (verification-first): naturalized re-executes to confirm the preserved oracle; unanswerable
proves 0-match / non-KG-field / >1-match; coverage_fixed recomputes the guard-free count; bridge_join
computes ground-truth oracles offline but marks them executor-unsupported (the executor frontier).

See docs/targeted_v2_benchmark_design.md for the schema and rationale.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
from procurement_graph.reasoning.models import QueryConstraint as QC

GEN = ROOT / "data" / "qa" / "generated"
OUT = ROOT / "data" / "qa" / "targeted_v2"
N = 50

# concepts that are NOT KG v0.1 fields -> asking for them is verifiably `unsupported`
NON_KG_CONCEPTS = [
    "how many bidders submitted a tender",
    "the social value score",
    "the payment terms",
    "the named subcontractor",
    "the framework agreement reference",
    "whether the supplier is an SME",
    "the named contract manager",
    "the evaluation criteria weighting",
]
_FIELD_PHRASE = {
    "buyer_name": "buyer (contracting authority)",
    "supplier_name": "awarded supplier",
    "tender_category": "procurement category",
    "award_date_signed": "award signed date",
    "tender_cpv_id": "CPV code",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def cons_of(items: list[dict[str, Any]]) -> tuple[QC, ...]:
    return tuple(QC(c["field"], c["op"], c["value"]) for c in items)


def visible(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    internal = {"supplier_count", "buyer_count", "value_is_additive"}
    return [c for c in items if c.get("field") not in internal]


def _cval(items: list[dict[str, Any]], field: str) -> Any:
    return next((c["value"] for c in items if c["field"] == field), None)


def _year_phrase(y: Any) -> str:
    return f"published in {y}" if y is not None else ""


def _cpv_phrase(cpv: Any) -> str:
    return f"under CPV {cpv}" if cpv is not None else ""


# ---------------- naturalized (oracle-preserving paraphrase) ----------------

def _render_count(cons: list[dict[str, Any]], variant: int) -> str:
    cat = _cval(cons, "tender_category"); y = _cval(cons, "release_year"); cpv = _cval(cons, "tender_cpv_id")
    catw = f"{cat} " if cat else ""
    templates = [
        f"Count the {catw}contract notices {_cpv_phrase(cpv)} {_year_phrase(y)}.",
        f"What is the total number of {catw}contract notices {_cpv_phrase(cpv)} {_year_phrase(y)}?",
        f"How many {catw}notices are recorded {_cpv_phrase(cpv)} {_year_phrase(y)}?",
    ]
    return re.sub(r"\s+", " ", templates[variant % len(templates)]).strip()


def _render_sum(cons: list[dict[str, Any]], variant: int) -> str:
    cat = _cval(cons, "tender_category"); y = _cval(cons, "release_year"); cpv = _cval(cons, "tender_cpv_id")
    catw = f"{cat} " if cat else ""
    templates = [
        f"Sum the contract values for {catw}notices {_cpv_phrase(cpv)} {_year_phrase(y)}.",
        f"What is the combined contract value of all {catw}notices {_cpv_phrase(cpv)} {_year_phrase(y)}?",
        f"What is the aggregate value of {catw}contract notices {_cpv_phrase(cpv)} {_year_phrase(y)}?",
    ]
    return re.sub(r"\s+", " ", templates[variant % len(templates)]).strip()


def _render_factoid(buyer: str, supplier: str, year: Any, cpv: Any, field: str, variant: int) -> str:
    fp = _FIELD_PHRASE.get(field, field)
    templates = [
        f"For the contract awarded by {buyer} to {supplier} in {year} (CPV {cpv}), what is the {fp}?",
        f"Which {fp} is recorded for the {year} contract between {buyer} and {supplier} (CPV {cpv})?",
        f"The contract {supplier} delivered to {buyer} in {year} under CPV {cpv} — what is its {fp}?",
    ]
    return templates[variant % len(templates)]


def build_naturalized(bench: list[dict], backend: RuntimeKGBackend, afield: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    counts = _pick(bench, "filtered_count", 17)
    sums = _pick(bench, "additive_sum", 17)
    facts = [b for b in _pick(bench, "contract_factoid", 60) if afield.get(b["spec_id"]) != "value_source"][:16]
    seq = 0
    for b in counts:
        vc = visible(b["constraints"])
        rs = backend.query(cons_of(vc))
        if len(rs) != b["golden_answer"]:
            continue
        rows.append(_row("naturalized", seq, _render_count(vc, seq), "count", "count", "answerable",
                         vc, b["golden_answer"], b, evidence_ids=rec_ids(backend, rs), ev_count=len(rs),
                         notes=f"paraphrase of {b['spec_id']}; oracle preserved & re-verified"))
        seq += 1
    for b in sums:
        vc = visible(b["constraints"])
        rs = backend.query(cons_of(vc) + (QC("value_is_additive", "eq", True),))  # additive rows only
        rows.append(_row("naturalized", seq, _render_sum(vc, seq), "sum", "sum", "answerable",
                         vc, b["golden_answer"], b, evidence_ids=rec_ids(backend, rs), ev_count=len(rs),
                         notes=f"paraphrase of {b['spec_id']}; oracle preserved; evidence = additive rows"))
        seq += 1
    for b in facts:
        cid = _cval(b["constraints"], "contract_node_id")
        rec = _record(backend, cid)
        field = afield.get(b["spec_id"], "")
        if rec is None or not field:
            continue
        buyer, supplier = rec.get("buyer_name"), rec.get("supplier_name")
        year, cpv = rec.get("release_year"), rec.get("tender_cpv_id")
        anchor = [{"field": "buyer_name", "op": "eq", "value": buyer},
                  {"field": "supplier_name", "op": "eq", "value": supplier},
                  {"field": "release_year", "op": "eq", "value": year},
                  {"field": "tender_cpv_id", "op": "eq", "value": cpv}]
        matches = backend.query(cons_of(anchor))
        if len({str(r.get(field)) for r in matches}) != 1:  # anchor must resolve to one consistent value
            continue
        rows.append(_row("naturalized", seq, _render_factoid(buyer, supplier, year, cpv, field, seq),
                         "factoid", "select_unique", "answerable", anchor, rec.get(field), b,
                         evidence_ids=rec_ids(backend, matches) or [str(cid)], ev_count=len(matches),
                         notes=f"paraphrase of {b['spec_id']}; anchor resolves to 1 value; src contract {cid}"))
        seq += 1
    return rows


# ---------------- coverage_fixed (drop hidden guards, recompute) ----------------

def build_coverage_fixed(bench: list[dict], backend: RuntimeKGBackend) -> list[dict]:
    # prefer high-evidence conjunctions: the guard is only answer-changing when the slice contains
    # contracts with no resolved supplier/buyer, which is far likelier in a large slice.
    pool = sorted((b for b in bench if b["operation_family"] == "conjunction"),
                  key=lambda b: (-int(b.get("evidence_count") or 0), b["spec_id"]))[: N * 4]
    stride = max(1, len(pool) // N)
    rows: list[dict] = []
    for seq, b in enumerate(pool[::stride][:N]):
        vc = visible(b["constraints"])
        removed = [c for c in b["constraints"] if c not in vc]
        rs = backend.query(cons_of(vc))
        fixed_oracle = len(rs)
        rows.append(_row("coverage_fixed", seq, b["question"], "count", "count", "answerable",
                         vc, fixed_oracle, b, evidence_ids=rec_ids(backend, rs), ev_count=fixed_oracle,
                         notes=(f"from {b['spec_id']}; removed guards {[c['field'] for c in removed]}; "
                                f"v1_oracle={b['golden_answer']} -> recomputed={fixed_oracle}"
                                f"{' (changed)' if fixed_oracle != b['golden_answer'] else ''}")))
    return rows


# ---------------- unanswerable (verified) ----------------

def build_unanswerable(bench: list[dict], backend: RuntimeKGBackend, df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    facts = _pick(bench, "contract_factoid", 200)
    seq = 0
    # (a) unsupported: real anchor, non-KG field
    for i, b in enumerate(facts[:17]):
        rec = _record(backend, _cval(b["constraints"], "contract_node_id"))
        if rec is None:
            continue
        concept = NON_KG_CONCEPTS[i % len(NON_KG_CONCEPTS)]
        q = (f"For the contract awarded by {rec.get('buyer_name')} to {rec.get('supplier_name')} in "
             f"{rec.get('release_year')}, {concept}?")
        rows.append(_row("unanswerable", seq, q, "unsupported", "none", "unsupported", [], None, b,
                         support="n/a", notes=f"asks '{concept}': not a KG v0.1 field"))
        seq += 1
    # (b) ambiguous: buyer with many contracts, ask a field that varies
    big_buyers = [str(x) for x in df["buyer_name"].value_counts().head(60).index if str(x).strip()]
    for buyer in big_buyers:
        rs = backend.query((QC("buyer_name", "eq", buyer),))
        cpvs = {str(r.get("tender_cpv_id")) for r in rs}
        if len(cpvs) < 3:
            continue
        q = f"What is the CPV code of the contract awarded by {buyer}?"
        rows.append(_row("unanswerable", seq, q, "ambiguous", "none", "ambiguous",
                         [{"field": "buyer_name", "op": "eq", "value": buyer}], None, None,
                         support="n/a", notes=f"anchor matches {len(rs)} contracts / {len(cpvs)} distinct CPVs"))
        seq += 1
        if seq >= 34:
            break
    # (c) no_results: real buyer + a CPV it never used
    all_cpvs = [str(x) for x in df["tender_cpv_id"].dropna().unique()]
    for buyer in big_buyers:
        used = {str(r.get("tender_cpv_id")) for r in backend.query((QC("buyer_name", "eq", buyer),))}
        unused = next((c for c in all_cpvs if c not in used), None)
        if unused is None:
            continue
        empty = backend.query((QC("buyer_name", "eq", buyer), QC("tender_cpv_id", "eq", unused)))
        if empty:
            continue
        q = f"Who was the awarded supplier for the contract by {buyer} under CPV {unused}?"
        rows.append(_row("unanswerable", seq, q, "factoid", "select_unique", "no_results",
                         [{"field": "buyer_name", "op": "eq", "value": buyer},
                          {"field": "tender_cpv_id", "op": "eq", "value": unused}], None, None,
                         support="supported", notes=f"buyer never used CPV {unused}: 0 matches (verified)"))
        seq += 1
        if seq >= N:
            break
    return rows


# ---------------- extended_ops (single-hop, needs a new reduction op; NOT decomposition) ----------------

def build_extended_ops(df: pd.DataFrame, backend: RuntimeKGBackend) -> list[dict]:
    rows: list[dict] = []
    d = df.copy()
    d["_val"] = pd.to_numeric(d.get("value_amount"), errors="coerce")
    seq = 0

    def add(q, atype, aop, cons, oracle, ev, support, notes, evc=None, extra=None):
        nonlocal seq
        # extended_ops are single query + a new reduction op -> requires_decomposition is FALSE.
        rows.append(_row("extended_ops", seq, q, atype, aop, "answerable", cons, oracle, None,
                         support=support, decomp=False, notes=notes, evidence_ids=ev, ev_count=evc, extra=extra))
        seq += 1

    # min_max: single-hop argmax over a filtered slice
    valid = d[d["_val"].notna()
              & d["tender_category"].notna() & (d["tender_category"].astype(str).str.strip() != "")
              & d["tender_cpv_id"].notna() & (d["tender_cpv_id"].astype(str).str.strip() != "")]
    for cat, cpv in valid.groupby(["tender_category", "tender_cpv_id"]).size().sort_values(ascending=False).head(14).index:
        sub = valid[(valid["tender_category"] == cat) & (valid["tender_cpv_id"] == cpv)]
        top = sub.loc[sub["_val"].idxmax()].to_dict()
        cid = str(backend.record_id(top))
        add(f"Which contract has the highest value among {cat} notices under CPV {cpv}?",
            "min_max", "argmax",
            [{"field": "tender_category", "op": "eq", "value": cat}, {"field": "tender_cpv_id", "op": "eq", "value": cpv}],
            cid, [cid], "needs_op:argmax", "argmax over value_amount on a filtered slice", evc=len(sub))

    # top_k: group + rank + head-k. Evidence is the group-by, recomputable -> record the recipe.
    for cat in ["services", "goods", "works"]:
        vc = d[d["tender_category"] == cat]["buyer_name"].value_counts().head(3)
        cat_size = int((d["tender_category"] == cat).sum())
        add(f"What are the top 3 buyers by number of {cat} contract notices?",
            "top_k", "rank_top_k", [{"field": "tender_category", "op": "eq", "value": cat}],
            [[str(b), int(n)] for b, n in vc.items()], [], "needs_op:top_k",
            "group by buyer_name, count, rank, head-3", evc=cat_size,
            extra={"group_by_field": "buyer_name", "metric": "count", "k": 3,
                   "evidence_kind": "aggregate_recomputable"})

    # set_list: distinct field over the match set
    for buyer in [str(x) for x in df["buyer_name"].value_counts().head(14).index]:
        rs = backend.query((QC("buyer_name", "eq", buyer), QC("tender_category", "eq", "works")))
        sup = sorted({str(r.get("supplier_name")) for r in rs if r.get("supplier_name")})
        if not sup:
            continue
        add(f"List the distinct suppliers awarded works contracts by {buyer}.",
            "set_list", "distinct_set",
            [{"field": "buyer_name", "op": "eq", "value": buyer}, {"field": "tender_category", "op": "eq", "value": "works"}],
            sup[:50], rec_ids(backend, rs), "needs_op:distinct_set", "distinct supplier_name over the match set", evc=len(rs))

    # comparison: two INDEPENDENT counts + comparator (NOT a bridge -> decomposition=False)
    top_buyers = [str(x) for x in df["buyer_name"].value_counts().head(16).index]
    for i in range(0, 12, 2):
        bx, by = top_buyers[i], top_buyers[i + 1]
        nx = len(backend.query((QC("buyer_name", "eq", bx), QC("release_year", "eq", 2024))))
        ny = len(backend.query((QC("buyer_name", "eq", by), QC("release_year", "eq", 2024))))
        add(f"Did {bx} publish more contract notices than {by} in 2024?",
            "comparison", "compare", [{"field": "release_year", "op": "eq", "value": 2024}],
            {"answer": bool(nx > ny), bx: nx, by: ny}, [], "needs_op:compare",
            "two INDEPENDENT sub-counts + comparator (parallel, not a bridge)", evc=nx + ny,
            extra={"comparison_breakdown": {bx: {"count": nx}, by: {"count": ny}},
                   "evidence_kind": "two_subcounts_recomputable"})
    return rows


# ---------------- bridge_join (PURE multi-hop: sub-answer set binds the next query) ----------------

def build_bridge_join(df: pd.DataFrame, backend: RuntimeKGBackend) -> list[dict]:
    """Only genuine semijoins: hop-1 resolves an entity SET, hop-2 filters on membership in it
    (`in_subquery`). requires_decomposition=True. Quality over quantity — no padding."""
    rows: list[dict] = []
    d = df.copy()
    d["_val"] = pd.to_numeric(d.get("value_amount"), errors="coerce")
    seq = 0

    def add(q, atype, aop, cons, oracle, ev, notes, evc=None):
        nonlocal seq
        rows.append(_row("bridge_join", seq, q, atype, aop, "answerable", cons, oracle, None,
                         support="needs_op:in_subquery", decomp=True, notes=notes, evidence_ids=ev, ev_count=evc))
        seq += 1

    # hop1: suppliers of buyer -> hop2: sum value over those suppliers' contracts
    for buyer in [str(x) for x in df["buyer_name"].value_counts().head(20).index]:
        suppliers = {str(r.get("supplier_name")) for r in backend.query((QC("buyer_name", "eq", buyer),))
                     if r.get("supplier_name")}
        sub = d[d["supplier_name"].astype(str).isin(suppliers) & d["_val"].notna()]
        if not suppliers or sub.empty:
            continue
        add(f"What is the total value of contracts awarded to suppliers who also won a contract from {buyer}?",
            "sum", "sum",
            [{"field": "supplier_name", "op": "in_subquery", "value": {"resolve": "suppliers_of_buyer", "buyer": buyer}}],
            round(float(sub["_val"].sum()), 2), rec_ids(backend, sub.head(50).to_dict("records")),
            "hop1: suppliers of buyer -> hop2: sum value over their contracts", evc=int(len(sub)))
        if seq >= N:
            return rows

    # hop1: buyers of supplier -> hop2: count their notices (other join direction)
    for supplier in [str(x) for x in df["supplier_name"].value_counts().head(20).index if str(x).strip()]:
        buyers = {str(r.get("buyer_name")) for r in backend.query((QC("supplier_name", "eq", supplier),))
                  if r.get("buyer_name")}
        if len(buyers) < 2:
            continue
        sub = d[d["buyer_name"].astype(str).isin(buyers)]
        add(f"How many contract notices were published by buyers who have awarded a contract to {supplier}?",
            "count", "count",
            [{"field": "buyer_name", "op": "in_subquery", "value": {"resolve": "buyers_of_supplier", "supplier": supplier}}],
            int(len(sub)), rec_ids(backend, sub.head(50).to_dict("records")),
            "hop1: buyers of supplier -> hop2: count their notices", evc=int(len(sub)))
        if seq >= N:
            return rows
    return rows


# ---------------- shared helpers ----------------

def _pick(bench: list[dict], family: str, n: int) -> list[dict]:
    pool = sorted((b for b in bench if b["operation_family"] == family), key=lambda b: b["spec_id"])
    if len(pool) <= n:
        return pool
    stride = len(pool) / n
    return [pool[int(i * stride)] for i in range(n)]


def backend_count(backend: RuntimeKGBackend, cons: list[dict]) -> int:
    return len(backend.query(cons_of(cons)))


def _record(backend: RuntimeKGBackend, cid: Any) -> dict | None:
    if not cid:
        return None
    rs = backend.query((QC("contract_node_id", "eq", cid),))
    return rs[0] if rs else None


def _row(subset: str, seq: int, question: str, atype: str, aop: str, status: str,
         cons: list[dict], oracle: Any, src: dict | None, *, support: str = "supported",
         decomp: bool = False, notes: str = "", evidence_ids: list | None = None,
         ev_count: int | None = None, extra: dict | None = None) -> dict:
    row = {
        "id": f"{subset}_{seq:04d}",
        "subset": subset,
        "question": re.sub(r"\s+", " ", question).strip(),
        "answer_type": atype,
        "answer_operation": aop,
        "expected_status": status,
        "constraints": cons,
        "oracle_answer": oracle,
        "evidence_ids": (evidence_ids or [])[:50],
        "evidence_count": ev_count,
        "difficulty_reason": _reason(subset, atype, status),
        "requires_decomposition": decomp,
        "executor_support": support,
        "generation_notes": notes,
    }
    if extra:
        row.update(extra)
    if src is not None:
        row["source_spec_id"] = src.get("spec_id")
        row["difficulty"] = src.get("difficulty")
        row["generalization_class"] = src.get("generalization_class")
    return row


def rec_ids(backend: RuntimeKGBackend, rows: list[dict]) -> list[str]:
    return [str(backend.record_id(r)) for r in rows]


def _reason(subset: str, atype: str, status: str) -> str:
    if subset == "naturalized":
        return f"{atype} with re-phrased surface form; same oracle — tests planner phrasing robustness"
    if subset == "coverage_fixed":
        return "count with hidden coverage guards removed; oracle = faithful count over visible predicates"
    if subset == "unanswerable":
        return {"unsupported": "asks for a non-KG field — must abstain (mark_unsupported)",
                "ambiguous": "under-specified anchor matches multiple contracts — must ask clarifying",
                "no_results": "well-formed query with 0 matches — must report no matching evidence"}[status]
    if subset == "bridge_join":
        return f"{atype}: true multi-hop — a sub-answer entity SET binds the next query (in_subquery)"
    return f"{atype}: single-hop but needs a new executor reduction op (see executor_support); no decomposition"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = read_jsonl(GEN / "benchmark.jsonl")
    afield = {r["spec"]["spec_id"]: r["spec"].get("answer_field", "")
              for r in read_jsonl(GEN / "answer_specs.jsonl")}
    print("[v2] loading KG ...", flush=True)
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    df = backend._backend.records_df

    subsets = {
        "naturalized": build_naturalized(bench, backend, afield),
        "unanswerable": build_unanswerable(bench, backend, df),
        "coverage_fixed": build_coverage_fixed(bench, backend),
        "extended_ops": build_extended_ops(df, backend),
        "bridge_join": build_bridge_join(df, backend),
    }
    from collections import Counter
    # remove stale fixed-name files from earlier runs so naming never lies about the row count
    for stale in OUT.glob("*_50.jsonl"):
        stale.unlink()
    summary: dict[str, Any] = {"per_subset": {}}
    for name, rows in subsets.items():
        # name reflects the ACTUAL row count: exactly-N subsets are "_50"; quality-first frontier
        # subsets that fall short are "_pilot" (never a misleading _50).
        filename = f"{name}_{N}.jsonl" if len(rows) == N else f"{name}_pilot.jsonl"
        (OUT / filename).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        answerable = [r for r in rows if r["expected_status"] == "answerable"]
        with_ev = sum(1 for r in answerable if r["evidence_ids"] or r.get("evidence_count") is not None)
        summary["per_subset"][name] = {
            "file": filename,
            "rows": len(rows),
            "answer_type": dict(Counter(r["answer_type"] for r in rows)),
            "expected_status": dict(Counter(r["expected_status"] for r in rows)),
            "executor_support": dict(Counter(r["executor_support"] for r in rows)),
            "requires_decomposition": sum(1 for r in rows if r["requires_decomposition"]),
            "answerable_with_evidence": f"{with_ev}/{len(answerable)}",
        }
        print(f"  {filename}: {len(rows)} rows | types={summary['per_subset'][name]['answer_type']} | "
              f"evidence {summary['per_subset'][name]['answerable_with_evidence']}")
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote pilots + summary to {OUT}")


if __name__ == "__main__":
    main()
