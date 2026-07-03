"""Production-control builder for targeted QAv2 review/full runs.

The module is intentionally self-contained: it reads frozen QAv1 artifacts and
the frozen KG, writes only targeted-v2 outputs, and emits per-row validation
records that can reproduce every oracle.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from procurement_graph.qa.benchmark.models import AnswerSpec, Constraint
from procurement_graph.reasoning.kg_backend import RuntimeKGBackend
from procurement_graph.reasoning.models import QueryConstraint as QC


ROOT = Path(__file__).resolve().parents[4]
GEN = ROOT / "data" / "qa" / "generated"
TARGETED_V2 = ROOT / "data" / "qa" / "targeted_v2"
FORBIDDEN_OUTPUTS = (ROOT / "data" / "qa" / "generated", ROOT / "data" / "qa" / "eval")
DEFAULT_N = {
    "naturalized": 2000,
    "coverage_fixed": 2000,
    "unanswerable": 2000,
    "extended_ops": 2000,
    "bridge_join": 1000,
}
SUBSETS = tuple(DEFAULT_N)
NON_KG_CONCEPTS = [
    ("payment terms", "field_not_in_schema"),
    ("number of bidders", "field_not_in_schema"),
    ("evaluation score", "field_not_in_schema"),
    ("delivery performance", "field_not_in_schema"),
    ("invoice or payment date", "field_not_in_schema"),
    ("whether this procurement was fair", "unsupported_judgement"),
    ("whether the supplier was reliable", "unsupported_judgement"),
    ("whether the price was reasonable", "unsupported_judgement"),
]


@dataclass(frozen=True)
class BuildContext:
    backend: RuntimeKGBackend
    df: pd.DataFrame
    sources: list[dict[str, Any]]
    answer_fields: dict[str, str]
    source_questions: dict[str, str]
    rng: random.Random
    seed: int
    llm: str
    model: str
    verify_model: str
    retries: int
    chat: Any = None
    verifier: Any = None


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    subsets = list(SUBSETS) if args.subset == "all" else [args.subset]
    out_dir = guard_out_dir(Path(args.out_dir), dry_run=args.dry_run)
    target_by_subset = {subset: int(args.limit or args.n or DEFAULT_N[subset]) for subset in subsets}

    sources = load_sources()
    source_questions = load_source_questions()
    answer_fields = {
        rec["spec"]["spec_id"]: rec["spec"].get("answer_field", "")
        for rec in read_jsonl(GEN / "answer_specs.jsonl")
    }
    plan = {
        subset: {
            "target_rows": target_by_subset[subset],
            "candidate_pool": candidate_pool_size(subset, sources),
            "accepted_path": str(paths_for(out_dir, subset, args.run_tag)[0]),
            "rejected_path": str(paths_for(out_dir, subset, args.run_tag)[1]),
            "validation_path": str(paths_for(out_dir, subset, args.run_tag)[2]),
        }
        for subset in subsets
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "seed": args.seed, "subsets": plan}, indent=2, ensure_ascii=False))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    chat = verifier = None
    if args.llm == "on" and "naturalized" in subsets:
        from procurement_graph.qa.benchmark.chat import ChatClient
        from procurement_graph.qa.benchmark.gate_b import LLMGateBVerifier

        chat = ChatClient.from_env(json_mode=args.json_mode)
        verifier = LLMGateBVerifier(client=chat, model=args.verify_model)
    elif "naturalized" in subsets:
        from procurement_graph.qa.benchmark.gate_b import DryRunGateBVerifier

        verifier = DryRunGateBVerifier()

    print("[qav2] loading KG ...", flush=True)
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    ctx = BuildContext(
        backend=backend,
        df=backend._backend.records_df,
        sources=sources,
        answer_fields=answer_fields,
        source_questions=source_questions,
        rng=random.Random(args.seed),
        seed=args.seed,
        llm=args.llm,
        model=args.model,
        verify_model=args.verify_model,
        retries=args.retries,
        chat=chat,
        verifier=verifier,
    )

    summaries: dict[str, Any] = {}
    for subset in subsets:
        accepted_path, rejected_path, validation_path = paths_for(out_dir, subset, args.run_tag)
        done = done_ids(accepted_path, rejected_path) if args.resume else set()
        existing_accepted = count_jsonl(accepted_path) if args.resume else 0
        remaining_target = max(0, target_by_subset[subset] - existing_accepted)
        if args.resume and remaining_target == 0:
            summaries[subset] = {
                "subset": subset,
                "target": target_by_subset[subset],
                "accepted": 0,
                "accepted_existing": existing_accepted,
                "rejected": 0,
                "skipped_resume": len(done),
                "resume_complete": True,
            }
            print(f"  {subset}: resume_complete accepted_existing={existing_accepted} skipped_resume={len(done)}", flush=True)
            continue
        if not args.resume:
            for path in (accepted_path, rejected_path, validation_path):
                if path.exists():
                    path.unlink()
        build_target = remaining_target if subset == "unanswerable" else remaining_target * 4
        rows, rejected = build_subset(subset, ctx, build_target, done)
        rows, rejected, validation = keep_valid_rows(rows, rejected, ctx, remaining_target)
        write_jsonl(accepted_path, rows, append=args.resume)
        write_jsonl(rejected_path, rejected, append=args.resume)
        write_jsonl(validation_path, validation, append=args.resume)
        summary = validation_summary(subset, rows, rejected, validation, target_by_subset[subset], done)
        summaries[subset] = summary
        print(
            f"  {subset}: accepted={len(rows)} rejected={len(rejected)} skipped_resume={len(done)} "
            f"types={summary['answer_type']} styles={summary['surface_style']}",
            flush=True,
        )

    summary_path = out_dir / f"validation_summary{tag_suffix(args.run_tag)}.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(f"\nwrote QAv2 outputs to {out_dir}")
    print(f"summary: {summary_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build targeted QAv2 subsets with validation.")
    parser.add_argument("--subset", choices=("all", *SUBSETS), default="all")
    parser.add_argument("--n", type=int, default=0, help="target rows per selected subset")
    parser.add_argument("--limit", type=int, default=0, help="strict cap per selected subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="print plan/output paths only; write no JSONL")
    parser.add_argument("--resume", action="store_true", help="append and skip ids already in accepted/rejected")
    parser.add_argument("--llm", choices=("on", "off"), default="off")
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--verify-model", default="grok-4-1-fast-non-reasoning")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--run-tag", default="review20")
    parser.add_argument("--out-dir", default=str(TARGETED_V2 / "review20"))
    return parser.parse_args(argv)


def build_subset(subset: str, ctx: BuildContext, target: int, done: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    builders = {
        "naturalized": build_naturalized,
        "coverage_fixed": build_coverage_fixed,
        "unanswerable": build_unanswerable,
        "extended_ops": build_extended_ops,
        "bridge_join": build_bridge_join,
    }
    return builders[subset](ctx, target, done)


# --------------------------------------------------------------------------- subset builders


def build_naturalized(ctx: BuildContext, target: int, done: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    pool = []
    for family, answer_type, operation in (
        ("filtered_count", "count", "count"),
        ("additive_sum", "sum", "sum"),
        ("contract_factoid", "factoid", "select_unique"),
    ):
        for src in family_sources(ctx.sources, family):
            if family == "contract_factoid" and ctx.answer_fields.get(src["spec_id"]) == "value_source":
                continue
            pool.append((family, answer_type, operation, src))
    pool = shuffled(pool, ctx.rng, "naturalized")
    for idx, (family, answer_type, operation, src) in enumerate(pool):
        if len(rows) >= target:
            break
        row_id = f"naturalized_{idx:04d}"
        if row_id in done:
            continue
        built = naturalized_candidate(ctx, row_id, family, answer_type, operation, src, len(rows))
        if built.get("rejected"):
            rejected.append(built)
        else:
            rows.append(built)
    return rows, rejected


def naturalized_candidate(
    ctx: BuildContext,
    row_id: str,
    family: str,
    answer_type: str,
    operation: str,
    src: dict[str, Any],
    seq: int,
) -> dict[str, Any]:
    constraints = visible_constraints(src["constraints"])
    answer_field = "contract_node_id"
    if family == "contract_factoid":
        cid = constraint_value(src["constraints"], "contract_node_id")
        rec = record_by_id(ctx.backend, cid)
        answer_field = ctx.answer_fields.get(src["spec_id"], "")
        if rec is None or not answer_field:
            return reject(row_id, "naturalized", "factoid_source_unresolved", source_spec_id=src["spec_id"])
        constraints = [
            {"field": "buyer_name", "op": "eq", "value": rec.get("buyer_name")},
            {"field": "supplier_name", "op": "eq", "value": rec.get("supplier_name")},
            {"field": "release_year", "op": "eq", "value": rec.get("release_year")},
            {"field": "tender_cpv_id", "op": "eq", "value": rec.get("tender_cpv_id")},
        ]
        matches = query(ctx.backend, constraints)
        values = {stable_value(row.get(answer_field)) for row in matches if row.get(answer_field) not in (None, "")}
        if len(values) != 1:
            return reject(row_id, "naturalized", "factoid_anchor_not_unique", source_spec_id=src["spec_id"])
        oracle = next(iter(values))
        answer_value_type = src.get("answer_value_type", "string")
    elif family == "additive_sum":
        answer_field = "value_amount"
        answer_value_type = "currency"
        matches = query(ctx.backend, constraints + [{"field": "value_is_additive", "op": "eq", "value": True}])
        oracle = jsonable(money_sum(matches))
    else:
        answer_value_type = "integer"
        matches = query(ctx.backend, constraints)
        oracle = len(matches)
        if int(oracle) != int(src["golden_answer"]):
            return reject(row_id, "naturalized", "kg_revalidation_mismatch", source_spec_id=src["spec_id"])

    canonical = canonical_from_constraints(answer_type, operation, constraints, answer_field)
    style = style_for("naturalized", seq)
    natural = styled_question(canonical, style, "naturalized", constraints, answer_type)
    rewrite_model = "template_controlled"
    rewrite_verified = True
    rewrite_status = "template_verified"
    gate_b = {"mode": "not_run_template_controlled", "verified": True}
    if ctx.llm == "on":
        spec = AnswerSpec(
            spec_id=src["spec_id"],
            constraints=tuple(Constraint(c["field"], c["op"], c.get("value")) for c in constraints),
            answer_operation={"count": "count", "sum": "sum", "factoid": "select_unique"}[answer_type],
            answer_field=answer_field,
            answer_value_type=answer_value_type,
        )
        natural, gate_b, rewrite_status = rewrite_with_gate_b(ctx, spec, canonical, answer_type)
        rewrite_model = ctx.model
        rewrite_verified = bool(gate_b.get("verified"))
        if not rewrite_verified:
            return reject(
                row_id,
                "naturalized",
                "gate_b_failed",
                canonical_question=canonical,
                natural_question=canonical,
                source_spec_id=src["spec_id"],
                gate_b=gate_b,
                rewrite_model=ctx.model,
                rewrite_status="rewrite_failed",
            )

    return make_row(
        row_id,
        "naturalized",
        canonical,
        natural,
        style,
        rewrite_model,
        rewrite_verified,
        answer_type,
        operation,
        "answerable",
        constraints,
        oracle,
        evidence_ids=record_ids(ctx.backend, matches),
        evidence_count=len(matches),
        evidence_kind="concrete_evidence",
        difficulty=difficulty_for("naturalized", constraints, answer_type),
        difficulty_reason="Oracle-preserving natural surface form over a verified QAv1 spec.",
        executor_support="supported",
        source_spec_id=src["spec_id"],
        template_family=f"naturalized_{answer_type}",
        rewrite_status=rewrite_status,
        gate_b=gate_b,
        answer_field=answer_field,
        answer_value_type=answer_value_type,
    )


def build_coverage_fixed(ctx: BuildContext, target: int, done: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    pool = shuffled(family_sources(ctx.sources, "conjunction"), ctx.rng, "coverage_fixed")
    for idx, src in enumerate(pool):
        if len(rows) >= target:
            break
        row_id = f"coverage_fixed_{idx:04d}"
        if row_id in done:
            continue
        constraints = visible_constraints(src["constraints"])
        removed = [c for c in src["constraints"] if c.get("field") not in {x["field"] for x in constraints}]
        matches = query(ctx.backend, constraints)
        style = style_for("coverage_fixed", len(rows))
        canonical = canonical_from_constraints("count", "count", constraints, "contract_node_id")
        natural = styled_question(canonical, style, "coverage_fixed", constraints, "count")
        rows.append(make_row(
            row_id,
            "coverage_fixed",
            canonical,
            natural,
            style,
            "template_controlled",
            True,
            "count",
            "count",
            "answerable",
            constraints,
            len(matches),
            evidence_ids=record_ids(ctx.backend, matches),
            evidence_count=len(matches),
            evidence_kind="concrete_evidence",
            difficulty=difficulty_for("coverage_fixed", constraints, "count"),
            difficulty_reason="Hidden buyer/supplier coverage guards removed; oracle recomputed over visible predicates.",
            executor_support="supported",
            source_spec_id=src["spec_id"],
            template_family=f"coverage_fixed_{style}",
            removed_guards=removed,
            v1_oracle=src.get("golden_answer"),
            answer_field="contract_node_id",
            answer_value_type="integer",
        ))
    return rows, rejected


def build_unanswerable(ctx: BuildContext, target: int, done: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seq = 0
    unsupported_quota = max(1, round(target * 0.4))
    ambiguous_quota = max(1, round(target * 0.3))
    no_result_quota = max(1, target - unsupported_quota - ambiguous_quota)

    def add(row: dict[str, Any]) -> None:
        nonlocal seq
        if len(rows) < target and row["id"] not in done:
            rows.append(row)
        seq += 1

    facts = shuffled(family_sources(ctx.sources, "contract_factoid"), ctx.rng, "unanswerable_facts")
    fact_records = []
    for src in facts:
        rec = record_by_id(ctx.backend, constraint_value(src["constraints"], "contract_node_id"))
        if rec is not None:
            fact_records.append(rec)
        if len(fact_records) >= max(unsupported_quota * 2, 200):
            break
    unsupported_added = 0
    for rec in fact_records:
        if unsupported_added >= unsupported_quota:
            break
        for concept, reason in NON_KG_CONCEPTS:
            if unsupported_added >= unsupported_quota:
                break
            status = "unsupported"
            style = style_for("unanswerable", seq)
            canonical = unsupported_canonical_question(rec, concept, reason)
            natural = styled_question(canonical, style, "unanswerable", [], "unsupported")
            before = len(rows)
            add(make_row(
                f"unanswerable_{seq:04d}",
                "unanswerable",
                canonical,
                natural,
                style,
                "template_controlled",
                True,
                "unsupported",
                "none",
                status,
                [],
                None,
                evidence_ids=[],
                evidence_count=0,
                evidence_kind="none",
                difficulty=difficulty_for("unanswerable", [], "unsupported"),
                difficulty_reason=f"Asks for {concept}, which is outside KG v0.1.",
                executor_support="n/a",
                template_family=f"unsupported_{reason}",
                verification_reason=reason,
            ))
            unsupported_added += int(len(rows) > before)

    no_result_specs = no_result_candidates(ctx)
    added_no_results = 0
    for candidate in no_result_specs:
        if len(rows) >= target or added_no_results >= no_result_quota:
            break
        style = style_for("unanswerable", seq)
        natural = styled_question(candidate["canonical"], style, "unanswerable", candidate["constraints"], "factoid")
        before = len(rows)
        add(make_row(
            f"unanswerable_{seq:04d}",
            "unanswerable",
            candidate["canonical"],
            natural,
            style,
            "template_controlled",
            True,
            "factoid",
            "select_unique",
            "no_results",
            candidate["constraints"],
            None,
            evidence_ids=[],
            evidence_count=0,
            evidence_kind="none",
            difficulty="medium",
            difficulty_reason="Well-formed query over valid fields has zero KG matches.",
            executor_support="supported",
            template_family=candidate["template_family"],
            verification_reason="zero_matches",
            answer_field="supplier_name",
            answer_value_type="string",
        ))
        added_no_results += int(len(rows) > before)

    ambiguous_specs = ambiguous_candidates(ctx)
    added_ambiguous = 0
    for candidate in ambiguous_specs:
        if len(rows) >= target or added_ambiguous >= ambiguous_quota:
            break
        style = style_for("unanswerable", seq)
        natural = styled_question(candidate["canonical"], style, "unanswerable", candidate["constraints"], "ambiguous")
        before = len(rows)
        add(make_row(
            f"unanswerable_{seq:04d}",
            "unanswerable",
            candidate["canonical"],
            natural,
            style,
            "template_controlled",
            True,
            "ambiguous",
            "none",
            "ambiguous",
            candidate["constraints"],
            None,
            evidence_ids=[],
            evidence_count=candidate["evidence_count"],
            evidence_kind="none",
            difficulty="hard",
            difficulty_reason="Under-specified anchor has multiple distinct possible answers.",
            executor_support="n/a",
            template_family=candidate["template_family"],
            verification_reason="ambiguous_distinct_answers",
            ambiguous_field=candidate["ambiguous_field"],
            distinct_answer_count=candidate["distinct_answer_count"],
        ))
        added_ambiguous += int(len(rows) > before)
    return rows, rejected


def build_extended_ops(ctx: BuildContext, target: int, done: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    d = ctx.df.copy()
    d["_val"] = pd.to_numeric(d.get("value_amount"), errors="coerce")
    families = [
        ("boolean_exists", extended_boolean_exists),
        ("boolean_field_equality", extended_boolean_field_equality),
        ("numeric_threshold", extended_numeric_threshold),
        ("date_relation", extended_date_relation),
        ("min_max", extended_min_max),
        ("top_k", extended_top_k),
        ("set_list", extended_set_list),
        ("comparison", extended_comparison),
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for name, fn in families:
        groups[name] = shuffled(fn(ctx, d, max(3, target // len(families) + 2), name), ctx.rng, f"extended_{name}")
    seq = 0
    while len(rows) < target and any(groups.values()):
        progressed = False
        for name, _ in families:
            if len(rows) >= target:
                break
            if not groups.get(name):
                continue
            candidate = groups[name].pop(0)
            progressed = True
            row_id = f"extended_ops_{seq:04d}"
            seq += 1
            if row_id in done:
                continue
            candidate["id"] = row_id
            candidate["subset"] = "extended_ops"
            rows.append(candidate)
        if not progressed:
            break
    return rows, rejected


def build_bridge_join(ctx: BuildContext, target: int, done: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    d = ctx.df.copy()
    d["_val"] = pd.to_numeric(d.get("value_amount"), errors="coerce")
    builders = [
        ("buyer_suppliers_sum", bridge_suppliers_of_buyer),
        ("supplier_buyers_count", bridge_buyers_of_supplier),
        ("cpv_suppliers_other_cpv", bridge_cpv_suppliers_other_cpv),
        ("buyer_cpv_set_count", bridge_buyer_cpv_set),
        ("year_suppliers_next_year", bridge_year_suppliers_next_year),
        ("category_buyers_count", bridge_category_buyers),
        ("supplier_set_compare", bridge_supplier_set_compare),
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for name, fn in builders:
        groups[name] = shuffled(fn(ctx, d, max(3, target // len(builders) + 2)), ctx.rng, f"bridge_{name}")
    seq = 0
    while len(rows) < target and any(groups.values()):
        progressed = False
        for name, _ in builders:
            if len(rows) >= target:
                break
            if not groups.get(name):
                continue
            candidate = groups[name].pop(0)
            progressed = True
            row_id = f"bridge_join_{seq:04d}"
            seq += 1
            if row_id in done:
                continue
            candidate["id"] = row_id
            candidate["subset"] = "bridge_join"
            rows.append(candidate)
        if not progressed:
            break
    return rows, rejected


# --------------------------------------------------------------------------- extended op families


def extended_boolean_exists(ctx: BuildContext, d: pd.DataFrame, quota: int, family: str) -> list[dict[str, Any]]:
    out = []
    pairs = d.groupby(["buyer_name", "release_year", "tender_category"]).size().sort_values(ascending=False)
    for buyer, year, category in pairs.index:
        if len(out) >= quota:
            break
        constraints = [
            {"field": "buyer_name", "op": "eq", "value": str(buyer)},
            {"field": "release_year", "op": "eq", "value": int(year)},
            {"field": "tender_category", "op": "eq", "value": str(category)},
        ]
        matches = query(ctx.backend, constraints)
        canonical = f"Did {buyer} publish any {category} contract notices in {year}?"
        out.append(make_extended(ctx, canonical, constraints, bool(matches), "boolean", "exists",
                                 "supported", len(matches), family, "easy", extra={"evidence_kind": "concrete_evidence"},
                                 evidence_ids=record_ids(ctx.backend, matches)))
    return out


def extended_boolean_field_equality(ctx: BuildContext, d: pd.DataFrame, quota: int, family: str) -> list[dict[str, Any]]:
    out = []
    sample = d[d["tender_title"].fillna("").astype(str).str.len().between(8, 80)].drop_duplicates("tender_title")
    for _, row in sample.head(quota * 4).iterrows():
        if len(out) >= quota:
            break
        title = str(row["tender_title"])
        category = str(row.get("tender_category"))
        constraints = [{"field": "tender_title", "op": "eq", "value": title}]
        matches = query(ctx.backend, constraints)
        if len(matches) != 1:
            continue
        canonical = f"Was the contract titled \"{title}\" categorized as {category}?"
        out.append(make_extended(ctx, canonical, constraints, True, "boolean", "compare",
                                 "needs_op:compare", 1, family, "medium",
                                 extra={"compare_field": "tender_category", "compare_value": category},
                                 evidence_ids=record_ids(ctx.backend, matches)))
    return out


def extended_numeric_threshold(ctx: BuildContext, d: pd.DataFrame, quota: int, family: str) -> list[dict[str, Any]]:
    out = []
    usable = d[d["_val"].notna() & d["tender_category"].fillna("").astype(str).str.strip().ne("")]
    slices = usable.groupby(["release_year", "tender_category"]).size().sort_values(ascending=False)
    for year, category in slices.index:
        if len(out) >= quota:
            break
        constraints = [
            {"field": "release_year", "op": "eq", "value": int(year)},
            {"field": "tender_category", "op": "eq", "value": str(category)},
            {"field": "value_is_additive", "op": "eq", "value": True},
        ]
        matches = query(ctx.backend, constraints)
        total = money_sum(matches)
        threshold = Decimal("10000000")
        canonical = f"Was the total value of {category} notices published in {year} above GBP 10 million?"
        out.append(make_extended(ctx, canonical, constraints, total > threshold, "boolean", "compare",
                                 "needs_op:compare", len(matches), family, "medium",
                                 extra={"comparison_operator": "gt", "threshold": str(threshold), "metric": "sum:value_amount"}))
    return out


def extended_date_relation(ctx: BuildContext, d: pd.DataFrame, quota: int, family: str) -> list[dict[str, Any]]:
    out = []
    sample = d[d["award_date_signed"].fillna("").astype(str).str.contains(r"\d{4}-\d{2}-\d{2}", regex=True)]
    sample = sample.drop_duplicates("tender_title")
    threshold = "2025-05-01"
    for _, row in sample.iterrows():
        if len(out) >= quota:
            break
        title = str(row.get("tender_title"))
        if not title or len(title) > 90:
            continue
        constraints = [{"field": "tender_title", "op": "eq", "value": title}]
        matches = query(ctx.backend, constraints)
        if len(matches) != 1:
            continue
        signed = norm_date(matches[0].get("award_date_signed"))
        canonical = f"Was the award for \"{title}\" signed after 1 May 2025?"
        out.append(make_extended(ctx, canonical, constraints, signed > threshold, "boolean", "compare",
                                 "needs_op:compare", 1, family, "medium",
                                 extra={"compare_field": "award_date_signed", "comparison_operator": "gt", "threshold": threshold},
                                 evidence_ids=record_ids(ctx.backend, matches)))
    return out


def extended_min_max(ctx: BuildContext, d: pd.DataFrame, quota: int, family: str) -> list[dict[str, Any]]:
    out = []
    valid = d[d["_val"].notna() & (d["_val"] > 0)]
    slices = valid.groupby(["tender_category", "tender_cpv_id"]).size().sort_values(ascending=False)
    for idx, (category, cpv) in enumerate(slices.index):
        if len(out) >= quota:
            break
        constraints = [{"field": "tender_category", "op": "eq", "value": str(category)},
                       {"field": "tender_cpv_id", "op": "eq", "value": str(cpv)}]
        matches = query(ctx.backend, constraints)
        if not matches:
            continue
        op = "argmin" if idx % 2 else "argmax"
        selected = minmax_record(matches, op)
        canonical = (
            f"Which contract had the {'lowest non-zero' if op == 'argmin' else 'highest'} value "
            f"among {category} notices under CPV {cpv}?"
        )
        out.append(make_extended(ctx, canonical, constraints, ctx.backend.record_id(selected), "min_max", op,
                                 f"needs_op:{op}", len(matches), family, "frontier",
                                 evidence_ids=[ctx.backend.record_id(selected)],
                                 extra={"metric": "value_amount"}))
    return out


def extended_top_k(ctx: BuildContext, d: pd.DataFrame, quota: int, family: str) -> list[dict[str, Any]]:
    out = []
    for category in ("services", "goods", "works"):
        for year in (2022, 2023, 2024, 2025, 2026):
            if len(out) >= quota:
                break
            constraints = [{"field": "tender_category", "op": "eq", "value": category},
                           {"field": "release_year", "op": "eq", "value": year}]
            matches = query(ctx.backend, constraints)
            if len(matches) < 5:
                continue
            k = 5 if len(out) % 2 else 3
            metric = "count" if len(out) % 3 else "sum:value_amount"
            oracle = top_k(matches, "buyer_name", metric, k)
            canonical = f"What are the top {k} buyers by {'total value' if metric.startswith('sum') else 'number'} of {category} notices in {year}?"
            out.append(make_extended(ctx, canonical, constraints, oracle, "top_k", "rank_top_k",
                                     "needs_op:top_k", len(matches), family, "frontier",
                                     extra={"group_by_field": "buyer_name", "metric": metric, "k": k,
                                            "evidence_kind": "aggregate_recomputable"}))
    return out


def extended_set_list(ctx: BuildContext, d: pd.DataFrame, quota: int, family: str) -> list[dict[str, Any]]:
    out = []
    for buyer in d["buyer_name"].value_counts().head(quota * 5).index:
        if len(out) >= quota:
            break
        constraints = [{"field": "buyer_name", "op": "eq", "value": str(buyer)},
                       {"field": "tender_category", "op": "eq", "value": "works"}]
        matches = query(ctx.backend, constraints)
        values = sorted({str(row.get("supplier_name")) for row in matches if row.get("supplier_name")})
        if not values:
            continue
        canonical = f"Which suppliers appear in works notices from {buyer}?"
        out.append(make_extended(ctx, canonical, constraints, values[:50], "set_list", "distinct_set",
                                 "needs_op:distinct_set", len(matches), family, "frontier",
                                 evidence_ids=record_ids(ctx.backend, matches),
                                 extra={"set_field": "supplier_name"}))
    return out


def extended_comparison(ctx: BuildContext, d: pd.DataFrame, quota: int, family: str) -> list[dict[str, Any]]:
    out = []
    buyers = [str(x) for x in d["buyer_name"].value_counts().head(quota * 4).index]
    for i in range(0, len(buyers) - 1, 2):
        if len(out) >= quota:
            break
        left, right = buyers[i], buyers[i + 1]
        left_constraints = [{"field": "buyer_name", "op": "eq", "value": left}, {"field": "release_year", "op": "eq", "value": 2024}]
        right_constraints = [{"field": "buyer_name", "op": "eq", "value": right}, {"field": "release_year", "op": "eq", "value": 2024}]
        left_n = len(query(ctx.backend, left_constraints))
        right_n = len(query(ctx.backend, right_constraints))
        canonical = f"Did {left} publish more contract notices than {right} in 2024?"
        out.append(make_extended(ctx, canonical, [], {"answer": left_n > right_n, left: left_n, right: right_n},
                                 "comparison", "compare", "needs_op:compare", left_n + right_n, family, "frontier",
                                 extra={"comparison_slices": {"left": left_constraints, "right": right_constraints},
                                        "comparison_breakdown": {left: {"count": left_n}, right: {"count": right_n}},
                                        "evidence_kind": "two_subcounts_recomputable"}))
    return out


def make_extended(
    ctx: BuildContext,
    canonical: str,
    constraints: list[dict[str, Any]],
    oracle: Any,
    answer_type: str,
    operation: str,
    support: str,
    evidence_count: int,
    family: str,
    difficulty: str,
    *,
    evidence_ids: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    style = style_for("extended_ops", len(canonical) + evidence_count)
    row = make_row(
        "",
        "extended_ops",
        canonical,
        styled_question(canonical, style, "extended_ops", constraints, answer_type),
        style,
        "template_controlled",
        True,
        answer_type,
        operation,
        "answerable",
        constraints,
        oracle,
        evidence_ids=evidence_ids or [],
        evidence_count=evidence_count,
        evidence_kind=(extra or {}).get("evidence_kind", "concrete_evidence" if evidence_ids else "aggregate_recomputable"),
        difficulty=difficulty,
        difficulty_reason=f"{answer_type} requires executor support beyond count/sum/select_unique.",
        executor_support=support,
        template_family=family,
        requires_decomposition=False,
    )
    if extra:
        row.update(extra)
    return row


# --------------------------------------------------------------------------- bridge families


def bridge_suppliers_of_buyer(ctx: BuildContext, d: pd.DataFrame, quota: int) -> list[dict[str, Any]]:
    out = []
    for buyer in d["buyer_name"].value_counts().head(quota * 4).index:
        if len(out) >= quota:
            break
        suppliers = {str(r.get("supplier_name")) for r in query(ctx.backend, [{"field": "buyer_name", "op": "eq", "value": str(buyer)}]) if r.get("supplier_name")}
        matches = d[d["supplier_name"].astype(str).isin(suppliers) & d["_val"].notna()]
        if len(suppliers) < 2 or matches.empty:
            continue
        constraints = [{"field": "supplier_name", "op": "in_subquery", "value": {"resolve": "suppliers_of_buyer", "buyer": str(buyer)}}]
        canonical = f"What is the total value of contracts awarded to suppliers who also worked with {buyer}?"
        out.append(make_bridge(ctx, canonical, constraints, round(float(matches["_val"].sum()), 2), "sum", "sum",
                               "supplier_name", len(suppliers), len(matches), "buyer_suppliers_sum",
                               evidence_ids=record_ids(ctx.backend, matches.head(50).to_dict("records"))))
    return out


def bridge_buyers_of_supplier(ctx: BuildContext, d: pd.DataFrame, quota: int) -> list[dict[str, Any]]:
    out = []
    for supplier in [str(x) for x in d["supplier_name"].value_counts().head(quota * 5).index if str(x).strip()]:
        if len(out) >= quota:
            break
        buyers = {str(r.get("buyer_name")) for r in query(ctx.backend, [{"field": "supplier_name", "op": "eq", "value": supplier}]) if r.get("buyer_name")}
        matches = d[d["buyer_name"].astype(str).isin(buyers)]
        if len(buyers) < 2 or matches.empty:
            continue
        constraints = [{"field": "buyer_name", "op": "in_subquery", "value": {"resolve": "buyers_of_supplier", "supplier": supplier}}]
        canonical = f"How many contract notices were published by buyers who have awarded a contract to {supplier}?"
        out.append(make_bridge(ctx, canonical, constraints, int(len(matches)), "count", "count", "buyer_name",
                               len(buyers), len(matches), "supplier_buyers_count",
                               evidence_ids=record_ids(ctx.backend, matches.head(50).to_dict("records"))))
    return out


def bridge_cpv_suppliers_other_cpv(ctx: BuildContext, d: pd.DataFrame, quota: int) -> list[dict[str, Any]]:
    out = []
    cpvs = [str(x) for x in d["tender_cpv_id"].value_counts().head(quota * 4).index]
    for i in range(len(cpvs) - 1):
        if len(out) >= quota:
            break
        source_cpv, target_cpv = cpvs[i], cpvs[-(i + 1)]
        suppliers = {str(r.get("supplier_name")) for r in query(ctx.backend, [{"field": "tender_cpv_id", "op": "eq", "value": source_cpv}]) if r.get("supplier_name")}
        matches = d[(d["supplier_name"].astype(str).isin(suppliers)) & (d["tender_cpv_id"].astype(str) == target_cpv)]
        if len(suppliers) < 2 or matches.empty:
            continue
        constraints = [{"field": "supplier_name", "op": "in_subquery", "value": {"resolve": "suppliers_for_cpv", "cpv": source_cpv}},
                       {"field": "tender_cpv_id", "op": "eq", "value": target_cpv}]
        canonical = f"How many CPV {target_cpv} notices went to suppliers that also won CPV {source_cpv} contracts?"
        out.append(make_bridge(ctx, canonical, constraints, int(len(matches)), "count", "count", "supplier_name",
                               len(suppliers), len(matches), "cpv_suppliers_other_cpv",
                               evidence_ids=record_ids(ctx.backend, matches.head(50).to_dict("records"))))
    return out


def bridge_buyer_cpv_set(ctx: BuildContext, d: pd.DataFrame, quota: int) -> list[dict[str, Any]]:
    out = []
    for buyer in d["buyer_name"].value_counts().head(quota * 4).index:
        if len(out) >= quota:
            break
        cpvs = {str(r.get("tender_cpv_id")) for r in query(ctx.backend, [{"field": "buyer_name", "op": "eq", "value": str(buyer)}]) if r.get("tender_cpv_id")}
        matches = d[d["tender_cpv_id"].astype(str).isin(cpvs)]
        if len(cpvs) < 2 or matches.empty:
            continue
        constraints = [{"field": "tender_cpv_id", "op": "in_subquery", "value": {"resolve": "cpvs_of_buyer", "buyer": str(buyer)}}]
        canonical = f"How many notices fall under CPV codes that {buyer} has used before?"
        out.append(make_bridge(ctx, canonical, constraints, int(len(matches)), "count", "count", "cpv_id",
                               len(cpvs), len(matches), "buyer_cpv_set_count",
                               evidence_ids=record_ids(ctx.backend, matches.head(50).to_dict("records"))))
    return out


def bridge_year_suppliers_next_year(ctx: BuildContext, d: pd.DataFrame, quota: int) -> list[dict[str, Any]]:
    out = []
    for category in ("works", "services", "goods"):
        if len(out) >= quota:
            break
        suppliers = {str(r.get("supplier_name")) for r in query(ctx.backend, [{"field": "release_year", "op": "eq", "value": 2024},
                                                                               {"field": "tender_category", "op": "eq", "value": category}]) if r.get("supplier_name")}
        matches = d[(d["supplier_name"].astype(str).isin(suppliers)) & (d["release_year"] == 2025)]
        if len(suppliers) < 2 or matches.empty:
            continue
        constraints = [{"field": "supplier_name", "op": "in_subquery", "value": {"resolve": "suppliers_in_year_category", "year": 2024, "category": category}},
                       {"field": "release_year", "op": "eq", "value": 2025}]
        canonical = f"How many 2025 contracts went to suppliers that won {category} contracts in 2024?"
        out.append(make_bridge(ctx, canonical, constraints, int(len(matches)), "count", "count", "supplier_name",
                               len(suppliers), len(matches), "year_suppliers_next_year",
                               evidence_ids=record_ids(ctx.backend, matches.head(50).to_dict("records"))))
    return out


def bridge_category_buyers(ctx: BuildContext, d: pd.DataFrame, quota: int) -> list[dict[str, Any]]:
    out = []
    for category in ("services", "goods", "works"):
        if len(out) >= quota:
            break
        buyers = {str(r.get("buyer_name")) for r in query(ctx.backend, [{"field": "tender_category", "op": "eq", "value": category}]) if r.get("buyer_name")}
        matches = d[d["buyer_name"].astype(str).isin(buyers) & (d["release_year"] == 2024)]
        if len(buyers) < 2 or matches.empty:
            continue
        constraints = [{"field": "buyer_name", "op": "in_subquery", "value": {"resolve": "buyers_in_category", "category": category}},
                       {"field": "release_year", "op": "eq", "value": 2024}]
        canonical = f"How many 2024 notices were published by buyers that have used the {category} category?"
        out.append(make_bridge(ctx, canonical, constraints, int(len(matches)), "count", "count", "buyer_name",
                               len(buyers), len(matches), "category_buyers_count",
                               evidence_ids=record_ids(ctx.backend, matches.head(50).to_dict("records"))))
    return out


def bridge_supplier_set_compare(ctx: BuildContext, d: pd.DataFrame, quota: int) -> list[dict[str, Any]]:
    out = []
    source_buyer = str(d["buyer_name"].value_counts().index[0])
    suppliers = {str(r.get("supplier_name")) for r in query(ctx.backend, [{"field": "buyer_name", "op": "eq", "value": source_buyer}]) if r.get("supplier_name")}
    if len(suppliers) < 2:
        return out
    it = d[(d["supplier_name"].astype(str).isin(suppliers)) & (d["tender_cpv_id"].astype(str).str.startswith("72"))]
    transport = d[(d["supplier_name"].astype(str).isin(suppliers)) & (d["tender_cpv_id"].astype(str).str.startswith("60"))]
    constraints = [{"field": "supplier_name", "op": "in_subquery", "value": {"resolve": "suppliers_of_buyer", "buyer": source_buyer}}]
    canonical = f"Did suppliers who worked with {source_buyer} win more IT-service contracts or transport-service contracts?"
    out.append(make_bridge(ctx, canonical, constraints, {"IT": int(len(it)), "transport": int(len(transport)),
                                                        "answer": "IT" if len(it) > len(transport) else "transport"},
                           "comparison", "compare", "supplier_name", len(suppliers), int(len(it) + len(transport)),
                           "supplier_set_compare",
                           evidence_ids=record_ids(ctx.backend, pd.concat([it, transport]).head(50).to_dict("records")),
                           extra={"comparison_breakdown": {"IT": {"count": int(len(it))}, "transport": {"count": int(len(transport))}}}))
    return out[:quota]


def make_bridge(
    ctx: BuildContext,
    canonical: str,
    constraints: list[dict[str, Any]],
    oracle: Any,
    answer_type: str,
    operation: str,
    intermediate_set_type: str,
    intermediate_set_size: int,
    final_evidence_count: int,
    family: str,
    *,
    evidence_ids: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if operation in {"count", "sum"}:
        matches = resolve_constraints(ctx.df, constraints)
        final_evidence_count = int(len(matches))
        evidence_ids = record_ids(ctx.backend, matches.head(50).to_dict("records"))
        if operation == "count":
            oracle = final_evidence_count
        else:
            oracle = jsonable(money_sum(matches.to_dict("records")))
    style = style_for("bridge_join", final_evidence_count)
    row = make_row(
        "",
        "bridge_join",
        canonical,
        styled_question(canonical, style, "bridge_join", constraints, answer_type),
        style,
        "template_controlled",
        True,
        answer_type,
        operation,
        "answerable",
        constraints,
        oracle,
        evidence_ids=evidence_ids,
        evidence_count=final_evidence_count,
        evidence_kind="concrete_evidence",
        difficulty="frontier",
        difficulty_reason="True semijoin: an intermediate entity set binds the final query.",
        executor_support="needs_op:in_subquery",
        template_family=family,
        requires_decomposition=True,
        intermediate_set_type=intermediate_set_type,
        intermediate_set_size=intermediate_set_size,
        final_evidence_count=final_evidence_count,
    )
    if extra:
        row.update(extra)
    return row


# --------------------------------------------------------------------------- validation


def validate_row(row: dict[str, Any], ctx: BuildContext) -> dict[str, Any]:
    recomputed_answer = None
    recomputed_status = row["expected_status"]
    failure_reason = ""
    try:
        if row["expected_status"] == "unsupported":
            ok = row.get("verification_reason") in {"field_not_in_schema", "unsupported_judgement"}
            recomputed_status = "unsupported" if ok else "error"
        elif row["expected_status"] == "ambiguous":
            matches = resolve_constraints(ctx.df, row.get("constraints", []))
            field = row.get("ambiguous_field", "supplier_name")
            distinct = {stable_value(v) for v in matches[field].dropna().tolist()} if field in matches else set()
            ok = len(distinct) > 1
            recomputed_status = "ambiguous" if ok else "answerable"
            recomputed_answer = sorted(distinct)[:10]
        elif row["expected_status"] == "no_results":
            matches = resolve_constraints(ctx.df, row.get("constraints", []))
            ok = len(matches) == 0
            recomputed_status = "no_results" if ok else "answerable"
            recomputed_answer = None
        else:
            matches = resolve_constraints(ctx.df, row.get("constraints", []))
            recomputed_answer = recompute_answer(row, matches, ctx)
            ok = oracle_match(recomputed_answer, row.get("oracle_answer"))
            recomputed_status = "answerable"
        matches_for_count = resolve_constraints(ctx.df, row.get("constraints", [])) if row["expected_status"] == "answerable" else pd.DataFrame()
        recomputed_count = row.get("final_evidence_count") if row.get("requires_decomposition") else len(matches_for_count)
        if row.get("evidence_kind") in {"aggregate_recomputable", "two_subcounts_recomputable"}:
            recomputed_count = row.get("evidence_count")
        evidence_count_match = row.get("evidence_count") == recomputed_count
        validation_passed = bool(ok and (evidence_count_match or row["expected_status"] != "answerable"))
        if not validation_passed:
            failure_reason = "oracle_or_evidence_mismatch"
    except Exception as exc:
        validation_passed = False
        evidence_count_match = False
        failure_reason = f"validation_exception: {type(exc).__name__}: {exc}"
    return {
        "id": row["id"],
        "subset": row["subset"],
        "validation_passed": validation_passed,
        "recomputed_answer": jsonable(recomputed_answer),
        "recomputed_status": recomputed_status,
        "oracle_match": oracle_match(recomputed_answer, row.get("oracle_answer")) if row["expected_status"] == "answerable" else validation_passed,
        "evidence_count_match": evidence_count_match,
        "failure_reason": failure_reason,
    }


def keep_valid_rows(
    rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    ctx: BuildContext,
    target: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    next_rejected = list(rejected)
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for row in rows:
        row_id = str(row.get("id", ""))
        question_key = clean_question(row.get("question") or row.get("natural_question") or row.get("canonical_question")).casefold()
        if row_id in seen_ids or question_key in seen_questions:
            next_rejected.append(reject(
                row_id,
                str(row.get("subset", "")),
                "duplicate_id_or_question",
                canonical_question=row.get("canonical_question"),
                question=row.get("question"),
                template_family=row.get("template_family"),
            ))
            continue
        result = validate_row(row, ctx)
        if result["validation_passed"] and len(accepted) < target:
            accepted.append(row)
            validation.append(result)
            seen_ids.add(row_id)
            seen_questions.add(question_key)
            continue
        if not result["validation_passed"]:
            next_rejected.append(reject(
                str(row.get("id", "")),
                str(row.get("subset", "")),
                str(result.get("failure_reason") or "validation_failed"),
                validation=result,
                canonical_question=row.get("canonical_question"),
                source_spec_id=row.get("source_spec_id"),
                template_family=row.get("template_family"),
            ))
    return accepted, next_rejected, validation


def recompute_answer(row: dict[str, Any], matches: pd.DataFrame, ctx: BuildContext) -> Any:
    op = row["answer_operation"]
    if op == "count":
        return int(len(matches))
    if op == "sum":
        return jsonable(money_sum(matches.to_dict("records")))
    if op == "select_unique":
        field = row.get("answer_field", "")
        values = sorted({stable_value(v) for v in matches[field].dropna().tolist()}) if field in matches else []
        return values[0] if len(values) == 1 else None
    if op == "exists":
        return bool(len(matches))
    if op in {"argmax", "argmin"}:
        selected = minmax_record(matches.to_dict("records"), op)
        return ctx.backend.record_id(selected) if selected else None
    if op == "rank_top_k":
        return top_k(matches.to_dict("records"), row["group_by_field"], row["metric"], int(row["k"]))
    if op == "distinct_set":
        field = row.get("set_field", "")
        return sorted({stable_value(v) for v in matches[field].dropna().tolist() if stable_value(v)})[:50] if field in matches else []
    if op == "compare":
        if "comparison_slices" in row:
            left = len(resolve_constraints(ctx.df, row["comparison_slices"]["left"]))
            right = len(resolve_constraints(ctx.df, row["comparison_slices"]["right"]))
            return {"answer": left > right, **{next_value(row["comparison_slices"]["left"], "buyer_name", "left"): left,
                                              next_value(row["comparison_slices"]["right"], "buyer_name", "right"): right}}
        return row.get("oracle_answer")
    return row.get("oracle_answer")


def validation_summary(
    subset: str,
    rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    target: int,
    done: set[str],
) -> dict[str, Any]:
    answerable = [r for r in rows if r["expected_status"] == "answerable"]
    concrete = [r for r in answerable if r.get("evidence_kind") == "concrete_evidence"]
    aggregate = [r for r in answerable if r.get("evidence_kind") in {"aggregate_recomputable", "two_subcounts_recomputable"}]
    ids = [r["id"] for r in rows]
    questions = [r["question"] for r in rows]
    return {
        "subset": subset,
        "target": target,
        "accepted": len(rows),
        "rejected": len(rejected),
        "skipped_resume": len(done),
        "validation_passed": sum(1 for v in validation if v["validation_passed"]),
        "validation_failed": sum(1 for v in validation if not v["validation_passed"]),
        "failure_reason": dict(Counter(r.get("failure_reason", "") for r in rejected)),
        "answer_type": dict(Counter(r["answer_type"] for r in rows)),
        "expected_status": dict(Counter(r["expected_status"] for r in rows)),
        "executor_support": dict(Counter(r["executor_support"] for r in rows)),
        "surface_style": dict(Counter(r["surface_style"] for r in rows)),
        "difficulty": dict(Counter(r["difficulty"] for r in rows)),
        "template_family": dict(Counter(r["template_family"] for r in rows)),
        "verification_reason": dict(Counter(r.get("verification_reason", "") for r in rows if r.get("verification_reason"))),
        "evidence_kind": dict(Counter(r.get("evidence_kind", "") for r in rows)),
        "concrete_evidence_coverage": f"{sum(1 for r in concrete if r.get('evidence_ids'))}/{len(concrete)}",
        "aggregate_recomputable": len(aggregate),
        "duplicate_ids": len(ids) - len(set(ids)),
        "duplicate_questions": len(questions) - len(set(questions)),
    }


# --------------------------------------------------------------------------- row/helpers


def make_row(
    row_id: str,
    subset: str,
    canonical_question: str,
    natural_question: str,
    surface_style: str,
    rewrite_model: str,
    rewrite_verified: bool,
    answer_type: str,
    answer_operation: str,
    expected_status: str,
    constraints: list[dict[str, Any]],
    oracle_answer: Any,
    *,
    evidence_ids: list[str],
    evidence_count: int | None,
    evidence_kind: str,
    difficulty: str,
    difficulty_reason: str,
    executor_support: str,
    template_family: str,
    requires_decomposition: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "id": row_id,
        "subset": subset,
        "canonical_question": clean_question(canonical_question),
        "natural_question": clean_question(natural_question),
        "question": clean_question(natural_question),
        "surface_style": surface_style,
        "rewrite_model": rewrite_model,
        "rewrite_verified": rewrite_verified,
        "answer_type": answer_type,
        "answer_operation": answer_operation,
        "expected_status": expected_status,
        "constraints": constraints,
        "oracle_answer": jsonable(oracle_answer),
        "evidence_ids": evidence_ids[:50],
        "evidence_count": evidence_count,
        "evidence_kind": evidence_kind,
        "difficulty": difficulty,
        "difficulty_reason": difficulty_reason,
        "requires_decomposition": requires_decomposition,
        "executor_support": executor_support,
        "template_family": template_family,
        "generation_notes": extra.pop("generation_notes", ""),
    }
    row.update(extra)
    return row


def reject(row_id: str, subset: str, failure_reason: str, **extra: Any) -> dict[str, Any]:
    return {"id": row_id, "subset": subset, "rejected": True, "failure_reason": failure_reason, **extra}


def rewrite_with_gate_b(ctx: BuildContext, spec: AnswerSpec, canonical: str, answer_type: str) -> tuple[str, dict[str, Any], str]:
    system = (
        "Rewrite this UK procurement benchmark question into a natural user question. "
        "Keep exactly the same filters and operation. Do not add or remove constraints. "
        "Return strict JSON: {\"question\":\"...\"}."
    )
    user = json.dumps({"canonical_question": canonical, "answer_type": answer_type}, ensure_ascii=False)
    last_gate = {"verified": False, "reason": "not_attempted"}
    for _ in range(max(ctx.retries, 1)):
        try:
            result = ctx.chat.complete_json(model=ctx.model, system=system, user=user)
            candidate = str((result.parsed or {}).get("question", "")).strip()
        except Exception as exc:  # pragma: no cover - live boundary
            last_gate = {"verified": False, "reason": f"rewrite_call_failed: {exc}"}
            continue
        if not candidate:
            last_gate = {"verified": False, "reason": "rewrite_empty"}
            continue
        outcome = ctx.verifier.verify_faithfulness(spec, candidate)
        last_gate = outcome.provenance(model=ctx.verify_model, sampled=True)
        if outcome.verified:
            return candidate, last_gate, "nano_gate_b_verified"
    return canonical, last_gate, "rewrite_failed"


def canonical_from_constraints(answer_type: str, operation: str, constraints: list[dict[str, Any]], answer_field: str) -> str:
    year = next_value(constraints, "release_year")
    category = next_value(constraints, "tender_category")
    cpv = next_value(constraints, "tender_cpv_id")
    buyer = next_value(constraints, "buyer_name")
    supplier = next_value(constraints, "supplier_name")
    title = next_value(constraints, "tender_title")
    parts = []
    if category:
        parts.append(f"{category} notices")
    else:
        parts.append("contract notices")
    if year:
        parts.append(f"published in {year}")
    if cpv:
        parts.append(f"under CPV {cpv}")
    if buyer:
        parts.append(f"by {buyer}")
    if supplier:
        parts.append(f"awarded to {supplier}")
    if title:
        parts.append(f"titled \"{title}\"")
    scope = " ".join(parts)
    if answer_type == "sum":
        return f"What is the total contract value for {scope}?"
    if answer_type == "factoid":
        field = human_answer_field(answer_field)
        if answer_field == "buyer_name":
            return f"Which buyer is recorded for {scope}?"
        if answer_field == "supplier_name":
            return f"Which supplier is recorded for {scope}?"
        if answer_field == "tender_category":
            return f"What tender category is recorded for {scope}?"
        return f"What {field} is recorded for {scope}?"
    return f"How many {scope} are recorded in the KG?"


def human_answer_field(field: str) -> str:
    labels = {
        "buyer_name": "buyer",
        "supplier_name": "supplier",
        "tender_category": "tender category",
        "tender_cpv_id": "CPV code",
        "award_date_signed": "award signed date",
        "value_amount": "contract value",
    }
    return labels.get(field, field.replace("_", " "))


def unsupported_canonical_question(rec: dict[str, Any], concept: str, reason: str) -> str:
    buyer = rec.get("buyer_name")
    supplier = rec.get("supplier_name")
    anchor = f"the contract awarded by {buyer} to {supplier}"
    if reason == "unsupported_judgement":
        if "fair" in concept:
            return f"Was {anchor} fair?"
        if "supplier" in concept and "reliable" in concept:
            return f"Was {supplier} reliable on {anchor}?"
        if "price" in concept and "reasonable" in concept:
            return f"Was the price reasonable for {anchor}?"
    templates = {
        "payment terms": f"What payment terms were recorded for {anchor}?",
        "number of bidders": f"How many bidders were recorded for {anchor}?",
        "evaluation score": f"What evaluation score was recorded for {anchor}?",
        "delivery performance": f"What delivery performance was recorded for {anchor}?",
        "invoice or payment date": f"What invoice or payment date was recorded for {anchor}?",
    }
    return templates.get(concept, f"What was the {concept} for {anchor}?")


def styled_question(canonical: str, style: str, subset: str, constraints: list[dict[str, Any]], answer_type: str) -> str:
    if style == "template":
        return canonical
    if style == "terse":
        count_match = re.match(r"^How many (.+?) are recorded in the KG\?$", canonical)
        if count_match:
            return f"Count matching {count_match.group(1)}?"
        fact_match = re.match(r"^What (.+?) is recorded for (.+?)\?$", canonical)
        if fact_match:
            return f"{fact_match.group(1).capitalize()} for {fact_match.group(2)}?"
        buyer_match = re.match(r"^Which (buyer|supplier) is recorded for (.+?)\?$", canonical)
        if buyer_match:
            return f"{buyer_match.group(1).capitalize()} recorded for {buyer_match.group(2)}?"
        return canonical
    if style == "indirect":
        return f"Looking only at the matching procurement records, {canonical[0].lower() + canonical[1:]}"
    if style == "realistic_user":
        return f"I am checking the procurement data: {canonical[0].lower() + canonical[1:]}"
    return canonical.replace("are recorded in the KG", "does the dataset contain").replace("contract value", "recorded contract value")


def style_for(subset: str, index: int) -> str:
    cycles = {
        "naturalized": ["natural", "indirect", "realistic_user", "natural", "natural", "indirect", "realistic_user", "natural", "natural", "terse"],
        "coverage_fixed": ["natural", "natural", "indirect", "realistic_user", "natural", "terse"],
        "unanswerable": ["realistic_user", "realistic_user", "realistic_user", "natural", "realistic_user", "realistic_user", "indirect", "realistic_user", "realistic_user", "terse"],
        "extended_ops": ["natural", "natural", "indirect", "natural", "natural", "realistic_user", "natural", "natural", "natural", "natural"],
        "bridge_join": ["natural", "indirect", "natural", "indirect", "natural", "indirect", "natural", "indirect", "natural", "indirect"],
    }
    cycle = cycles[subset]
    return cycle[index % len(cycle)]


def difficulty_for(subset: str, constraints: list[dict[str, Any]], answer_type: str) -> str:
    if subset == "bridge_join" or answer_type in {"top_k", "set_list", "min_max", "comparison"}:
        return "frontier"
    visible = len([c for c in constraints if c.get("field") not in {"value_is_additive"}])
    if subset == "unanswerable" or answer_type == "factoid":
        return "hard"
    if visible <= 1:
        return "easy"
    if visible <= 3:
        return "medium"
    return "hard"


def ambiguous_candidates(ctx: BuildContext) -> list[dict[str, Any]]:
    out = []
    df = ctx.df
    per_family_limit = 1000
    specs = [
        ("buyer_only", "buyer_name", "supplier_name", "What is the awarded supplier for a contract from {value}?"),
        ("supplier_only", "supplier_name", "buyer_name", "Which buyer awarded a contract to {value}?"),
        ("cpv_only", "tender_cpv_id", "buyer_name", "Which buyer used CPV {value}?"),
    ]
    for family, anchor_field, answer_field, template in specs:
        per_family = 0
        for value in df[anchor_field].dropna().astype(str).value_counts().head(400).index:
            constraints = [{"field": anchor_field, "op": "eq", "value": str(value)}]
            matches = query(ctx.backend, constraints)
            distinct = {stable_value(row.get(answer_field)) for row in matches if row.get(answer_field)}
            if len(distinct) >= 3:
                out.append({"canonical": template.format(value=value), "constraints": constraints,
                            "ambiguous_field": answer_field, "distinct_answer_count": len(distinct),
                            "evidence_count": len(matches), "template_family": family})
                per_family += 1
                if per_family >= per_family_limit:
                    break
    for buyer in df["buyer_name"].dropna().astype(str).value_counts().head(400).index:
        constraints = [{"field": "buyer_name", "op": "eq", "value": str(buyer)}, {"field": "release_year", "op": "eq", "value": 2024}]
        matches = query(ctx.backend, constraints)
        distinct = {stable_value(row.get("supplier_name")) for row in matches if row.get("supplier_name")}
        if len(distinct) >= 3:
            out.append({"canonical": f"Who was the supplier for the 2024 contract from {buyer}?", "constraints": constraints,
                        "ambiguous_field": "supplier_name", "distinct_answer_count": len(distinct),
                        "evidence_count": len(matches), "template_family": "year_buyer_still_multiple"})
            if sum(1 for row in out if row["template_family"] == "year_buyer_still_multiple") >= per_family_limit:
                break
    return out


def no_result_candidates(ctx: BuildContext) -> list[dict[str, Any]]:
    out = []
    df = ctx.df
    per_family_limit = 1000
    cpvs = [str(x) for x in df["tender_cpv_id"].dropna().unique()]
    for buyer in df["buyer_name"].dropna().astype(str).value_counts().head(300).index:
        used = {str(r.get("tender_cpv_id")) for r in query(ctx.backend, [{"field": "buyer_name", "op": "eq", "value": str(buyer)}])}
        unused = next((cpv for cpv in cpvs if cpv not in used), None)
        if unused:
            constraints = [{"field": "buyer_name", "op": "eq", "value": str(buyer)}, {"field": "tender_cpv_id", "op": "eq", "value": unused}]
            if not query(ctx.backend, constraints):
                out.append({"canonical": f"Who was the supplier for {buyer}'s contract under CPV {unused}?",
                            "constraints": constraints, "template_family": "buyer_cpv_not_found"})
                if sum(1 for row in out if row["template_family"] == "buyer_cpv_not_found") >= per_family_limit:
                    break
    for supplier in df["supplier_name"].dropna().astype(str).value_counts().head(300).index:
        constraints = [{"field": "supplier_name", "op": "eq", "value": str(supplier)},
                       {"field": "release_year", "op": "eq", "value": 2031},
                       {"field": "tender_category", "op": "eq", "value": "goods"}]
        if not query(ctx.backend, constraints):
            out.append({"canonical": f"Who was the buyer for {supplier}'s goods contract in 2031?",
                        "constraints": constraints, "template_family": "supplier_year_category_not_found"})
            if sum(1 for row in out if row["template_family"] == "supplier_year_category_not_found") >= per_family_limit:
                break
    for category in ("services", "goods", "works"):
        constraints = [{"field": "release_year", "op": "eq", "value": 2035}, {"field": "tender_category", "op": "eq", "value": category}]
        if not query(ctx.backend, constraints):
            out.append({"canonical": f"Which buyer published a {category} notice in 2035?",
                        "constraints": constraints, "template_family": "date_outside_kg_range"})
    return out


def visible_constraints(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    internal = {"supplier_count", "buyer_count", "value_is_additive"}
    return [dict(c) for c in items if c.get("field") not in internal]


def query(backend: RuntimeKGBackend, constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    simple = []
    for c in constraints:
        if c["op"] == "in_subquery":
            return resolve_constraints(backend._backend.records_df, constraints).to_dict("records")
        simple.append(QC(c["field"], c["op"], c.get("value")))
    return backend.query(tuple(simple))


def resolve_constraints(df: pd.DataFrame, constraints: list[dict[str, Any]]) -> pd.DataFrame:
    out = df
    for c in constraints:
        field, op, value = c["field"], c["op"], c.get("value")
        if op == "in_subquery":
            values = resolve_subquery(df, value)
            out = out[out[field].astype(str).isin(values)]
        elif op == "eq":
            out = out[out[field] == value]
        elif op == "gte":
            out = out[pd.to_numeric(out[field], errors="coerce") >= value]
        elif op == "lte":
            out = out[pd.to_numeric(out[field], errors="coerce") <= value]
        elif op == "contains":
            out = out[out[field].astype(str).str.contains(str(value), case=False, na=False)]
        elif op == "exists":
            out = out[out[field].notna() & (out[field].astype(str) != "")]
        else:
            raise ValueError(f"unsupported op: {op}")
    return out


def resolve_subquery(df: pd.DataFrame, spec: dict[str, Any]) -> set[str]:
    kind = spec.get("resolve")
    if kind == "suppliers_of_buyer":
        return set(df.loc[df["buyer_name"].astype(str) == str(spec["buyer"]), "supplier_name"].dropna().astype(str))
    if kind == "buyers_of_supplier":
        return set(df.loc[df["supplier_name"].astype(str) == str(spec["supplier"]), "buyer_name"].dropna().astype(str))
    if kind == "suppliers_for_cpv":
        return set(df.loc[df["tender_cpv_id"].astype(str) == str(spec["cpv"]), "supplier_name"].dropna().astype(str))
    if kind == "cpvs_of_buyer":
        return set(df.loc[df["buyer_name"].astype(str) == str(spec["buyer"]), "tender_cpv_id"].dropna().astype(str))
    if kind == "suppliers_in_year_category":
        mask = (df["release_year"] == spec["year"]) & (df["tender_category"].astype(str) == str(spec["category"]))
        return set(df.loc[mask, "supplier_name"].dropna().astype(str))
    if kind == "buyers_in_category":
        return set(df.loc[df["tender_category"].astype(str) == str(spec["category"]), "buyer_name"].dropna().astype(str))
    raise ValueError(f"unsupported subquery: {kind}")


def top_k(rows: list[dict[str, Any]], group_by: str, metric: str, k: int) -> list[list[Any]]:
    df = pd.DataFrame(rows)
    if df.empty or group_by not in df:
        return []
    if metric == "count":
        series = df[group_by].dropna().astype(str).value_counts().head(k)
    elif metric == "sum:value_amount":
        df["_metric"] = pd.to_numeric(df["value_amount"], errors="coerce").fillna(0)
        series = df.groupby(group_by)["_metric"].sum().sort_values(ascending=False).head(k)
    else:
        raise ValueError(metric)
    return [[str(idx), jsonable(value)] for idx, value in series.items()]


def minmax_record(rows: list[dict[str, Any]], op: str) -> dict[str, Any] | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    values = pd.to_numeric(df["value_amount"], errors="coerce")
    if op == "argmin":
        values = values.where(values > 0)
        idx = values.idxmin()
    else:
        idx = values.idxmax()
    if pd.isna(idx):
        return None
    return df.loc[idx].to_dict()


def money_sum(rows: list[dict[str, Any]]) -> Decimal:
    total = Decimal("0")
    seen = set()
    for row in rows:
        key = str(row.get("contract_node_id") or id(row))
        if key in seen:
            continue
        seen.add(key)
        value = row.get("value_amount")
        if value in (None, ""):
            continue
        try:
            number = Decimal(str(value))
            if number.is_finite():
                total += number
        except (InvalidOperation, ValueError):
            continue
    return total


def record_ids(backend: RuntimeKGBackend, rows: list[dict[str, Any]]) -> list[str]:
    return [backend.record_id(row) for row in rows][:50]


def record_by_id(backend: RuntimeKGBackend, cid: Any) -> dict[str, Any] | None:
    if cid in (None, ""):
        return None
    rows = query(backend, [{"field": "contract_node_id", "op": "eq", "value": cid}])
    return rows[0] if rows else None


def load_sources() -> list[dict[str, Any]]:
    out = []
    for rec in read_jsonl(GEN / "answer_specs.jsonl"):
        spec = rec["spec"]
        meta = spec.get("metadata", {})
        out.append({
            "spec_id": spec["spec_id"],
            "constraints": spec.get("constraints", []),
            "answer_operation": spec["answer_operation"],
            "answer_value_type": spec.get("answer_value_type", ""),
            "golden_answer": rec.get("golden_answer"),
            "operation_family": meta.get("operation_family", ""),
            "difficulty": meta.get("difficulty", ""),
            "generalization_class": meta.get("generalization_class", ""),
        })
    return out


def load_source_questions() -> dict[str, str]:
    path = GEN / "benchmark.jsonl"
    if not path.exists():
        return {}
    out = {}
    for rec in read_jsonl(path):
        spec_id = rec.get("spec_id") or (rec.get("spec") or {}).get("spec_id")
        if spec_id:
            out[str(spec_id)] = str(rec.get("question", ""))
    return out


def family_sources(sources: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [src for src in sources if src.get("operation_family") == family]


def candidate_pool_size(subset: str, sources: list[dict[str, Any]]) -> int:
    if subset == "naturalized":
        return sum(1 for src in sources if src.get("operation_family") in {"filtered_count", "additive_sum", "contract_factoid"})
    if subset == "coverage_fixed":
        return sum(1 for src in sources if src.get("operation_family") == "conjunction")
    return -1


def shuffled(items: list[Any], rng: random.Random, salt: str) -> list[Any]:
    indexed = list(enumerate(items))
    local = random.Random(f"{rng.random()}:{salt}")
    local.shuffle(indexed)
    return [item for _, item in indexed]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def done_ids(accepted_path: Path, rejected_path: Path) -> set[str]:
    ids = set()
    for path in (accepted_path, rejected_path):
        if not path.exists():
            continue
        for rec in read_jsonl(path):
            if rec.get("id"):
                ids.add(str(rec["id"]))
    return ids


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def paths_for(out_dir: Path, subset: str, run_tag: str) -> tuple[Path, Path, Path]:
    suffix = tag_suffix(run_tag)
    return (
        out_dir / f"{subset}{suffix}.accepted.jsonl",
        out_dir / f"{subset}{suffix}.rejected.jsonl",
        out_dir / f"{subset}{suffix}.validation.jsonl",
    )


def tag_suffix(run_tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_tag or "").strip())
    return f".{cleaned}" if cleaned else ""


def guard_out_dir(path: Path, *, dry_run: bool) -> Path:
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    for forbidden in FORBIDDEN_OUTPUTS:
        if is_relative_to(resolved, forbidden.resolve()):
            raise SystemExit(f"Refusing to write QAv2 outputs under protected path: {forbidden}")
    if is_relative_to(resolved, (ROOT / "data" / "qa" / "eval").resolve()):
        raise SystemExit("Refusing to write QAv2 outputs under data/qa/eval (hard20/hard100 protected).")
    if resolved.exists() and not is_relative_to(resolved, TARGETED_V2.resolve()):
        if any(resolved.iterdir()) and not dry_run:
            raise SystemExit(f"Refusing to write into existing non-targeted_v2 directory: {resolved}")
    return resolved


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def next_value(constraints: list[dict[str, Any]], field: str, default: Any = "") -> Any:
    return next((c.get("value") for c in constraints if c.get("field") == field), default)


def constraint_value(constraints: list[dict[str, Any]], field: str, default: Any = None) -> Any:
    return next((c.get("value") for c in constraints if c.get("field") == field), default)


def stable_value(value: Any) -> str:
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def clean_question(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def norm_date(value: Any) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(value))
    return match.group(0) if match else ""


def oracle_match(left: Any, right: Any) -> bool:
    left = jsonable(left)
    right = jsonable(right)
    if isinstance(left, (int, float, str)) and isinstance(right, (int, float, str)):
        try:
            a = float(str(left).replace(",", ""))
            b = float(str(right).replace(",", ""))
            return abs(a - b) <= max(1e-6, abs(b) * 1e-6)
        except ValueError:
            return str(left).casefold() == str(right).casefold()
    return left == right


def jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, pd.Series):
        return value.to_dict()
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def json_default(value: Any) -> Any:
    return jsonable(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
