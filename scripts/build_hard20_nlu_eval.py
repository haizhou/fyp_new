#!/usr/bin/env python3
"""Build a natural-language-understanding-biased hard-20 runtime eval set.

This set complements the earlier aggregation-heavy hard20. It stresses planner
understanding: title anchors instead of internal ids, buyer/supplier aliases,
omitted field names, explicit abstention, and bounded decomposition questions.
It writes only eval fixtures under data/qa/eval.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.reasoning.kg_backend import RuntimeKGBackend


OUT_DIR = ROOT / "data" / "qa" / "eval"
QUESTIONS_OUT = OUT_DIR / "hard20_nlu_questions_only.jsonl"
KEY_OUT = OUT_DIR / "hard20_nlu_answer_key.jsonl"
SUMMARY_OUT = OUT_DIR / "hard20_nlu_summary.json"


def main() -> None:
    backend = RuntimeKGBackend.from_directory(ROOT / "data" / "kg")
    df = backend._backend.records_df.copy()
    rows = _build_rows(df, backend)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(QUESTIONS_OUT, [_question_row(row) for row in rows])
    _write_jsonl(KEY_OUT, rows)
    SUMMARY_OUT.write_text(
        json.dumps(_summary(rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps({"questions": str(QUESTIONS_OUT), "answer_key": str(KEY_OUT),
                      "summary": str(SUMMARY_OUT), **_summary(rows)}, indent=2, ensure_ascii=False))


def _build_rows(df: Any, backend: RuntimeKGBackend) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # More factoids, anchored by quoted natural notice/award titles rather than internal ids.
    rows.append(_factoid_by_title(df, backend, "nlu_factoid_01_buyer",
                                  "BATHYTHERMOGRAPHIC PROBES",
                                  "Who was the buyer for the notice titled \"BATHYTHERMOGRAPHIC PROBES\"?",
                                  "buyer_name", "string", "factoid_title_anchor"))
    rows.append(_factoid_by_title(df, backend, "nlu_factoid_02_supplier",
                                  "Housing Disabled Adaptations",
                                  "Who supplied the contract called \"Housing Disabled Adaptations\"?",
                                  "supplier_name", "string", "factoid_title_anchor"))
    rows.append(_factoid_by_title(df, backend, "nlu_factoid_03_signed_date",
                                  "Supply of Suspension Elements",
                                  "When was the award for \"Supply of Suspension Elements\" signed?",
                                  "award_date_signed", "date", "factoid_title_anchor"))
    rows.append(_factoid_by_title(df, backend, "nlu_factoid_04_value_source",
                                  "Miscanspeed sequencing procurement",
                                  "For \"Miscanspeed sequencing procurement\", is the recorded value from the contract or another source?",
                                  "value_source", "string", "factoid_title_anchor"))
    rows.append(_factoid_by_title(df, backend, "nlu_factoid_05_category",
                                  "Repair and Maintenance of Street Furniture and Park Infrastructure",
                                  "Was \"Repair and Maintenance of Street Furniture and Park Infrastructure\" a goods, works, or services notice?",
                                  "tender_category", "string", "factoid_title_anchor"))

    # Buyer/supplier aliases and short names. Oracle uses the canonical KG value.
    rows.append(_count(df, backend, "nlu_alias_01_mod_count",
                       "How many goods notices did MoD publish in 2022?",
                       [{"field": "buyer_name", "op": "eq", "value": "Ministry of Defence"},
                        {"field": "release_year", "op": "eq", "value": 2022},
                        {"field": "tender_category", "op": "eq", "value": "goods"}],
                       "buyer_supplier_alias"))
    rows.append(_count(df, backend, "nlu_alias_02_nss_count",
                       "How many pharmaceutical product awards came from NSS in 2022?",
                       [{"field": "buyer_name", "op": "eq",
                         "value": "The Common Services Agency (more commonly known as NHS National Services Scotland) (\"NSS\")"},
                        {"field": "release_year", "op": "eq", "value": 2022},
                        {"field": "tender_cpv_id", "op": "eq", "value": "33600000"}],
                       "buyer_supplier_alias"))
    rows.append(_count(df, backend, "nlu_alias_03_powys_count",
                       "How many 2022 social care service contracts did Powys Council publish?",
                       [{"field": "buyer_name", "op": "eq", "value": "Powys County Council"},
                        {"field": "release_year", "op": "eq", "value": 2022},
                        {"field": "tender_cpv_id", "op": "eq", "value": "85000000"}],
                       "buyer_supplier_alias"))
    rows.append(_count(df, backend, "nlu_alias_04_lockheed_supplier",
                       "How many 2022 goods contracts list Lockheed Martin Sippican as supplier?",
                       [{"field": "supplier_name", "op": "eq", "value": "Lockheed Martin Sippican"},
                        {"field": "release_year", "op": "eq", "value": 2022},
                        {"field": "tender_category", "op": "eq", "value": "goods"}],
                       "buyer_supplier_alias"))

    # Omitted field names / natural CPV wording.
    rows.append(_count(df, backend, "nlu_omitted_01_road_transport",
                       "In 2024, how many notices were for road transport services?",
                       [{"field": "release_year", "op": "eq", "value": 2024},
                        {"field": "tender_cpv_id", "op": "eq", "value": "60100000"}],
                       "omitted_field_natural_cpv"))
    rows.append(_sum(df, backend, "nlu_omitted_02_social_work_total",
                     "What was the total recorded value for social work and related services in 2023?",
                     [{"field": "release_year", "op": "eq", "value": 2023},
                      {"field": "tender_cpv_id", "op": "eq", "value": "85300000"}],
                     "omitted_field_natural_cpv"))
    rows.append(_count(df, backend, "nlu_omitted_03_signed_reagents",
                       "How many signed goods awards involved laboratory reagents?",
                       [{"field": "tender_category", "op": "eq", "value": "goods"},
                        {"field": "tender_cpv_id", "op": "eq", "value": "33696500"},
                        {"field": "has_award_signed_date", "op": "eq", "value": True}],
                       "omitted_field_natural_cpv"))

    # Explicitly unanswerable / ambiguous: correct behaviour is abstention.
    rows.extend([
        _unanswerable("nlu_unanswerable_01_carbon",
                      "Which 2024 supplier had the strongest carbon reduction commitments?",
                      "carbon-reduction clause quality is not in KG v0.1"),
        _unanswerable("nlu_unanswerable_02_best_value",
                      "Which supplier offered the best value for money on health contracts?",
                      "best-value judgement is not a structured KG field"),
        _unanswerable("nlu_unanswerable_03_average",
                      "What was the average contract value for services in 2024?",
                      "average is not supported by the deterministic runtime yet"),
        _unanswerable("nlu_ambiguous_01_the_care_contract",
                      "Who won the care contract?",
                      "the question does not identify a unique contract or filter set"),
    ])

    # Bounded decomposition: oracle exists, but runtime should expose whether it can decompose.
    rows.append(_decomp_count(df, backend, "nlu_decomp_01_buyer_services",
                              "For the buyer of \"BATHYTHERMOGRAPHIC PROBES\", how many services contracts did they publish in 2022?",
                              anchor_title="BATHYTHERMOGRAPHIC PROBES",
                              role_field="buyer_name",
                              followup=[{"field": "release_year", "op": "eq", "value": 2022},
                                        {"field": "tender_category", "op": "eq", "value": "services"}]))
    rows.append(_decomp_count(df, backend, "nlu_decomp_02_supplier_services",
                              "For the supplier that won \"Housing Disabled Adaptations\", how many services contracts did they supply in 2022?",
                              anchor_title="Housing Disabled Adaptations",
                              role_field="supplier_name",
                              followup=[{"field": "release_year", "op": "eq", "value": 2022},
                                        {"field": "tender_category", "op": "eq", "value": "services"}]))
    rows.append(_decomp_sum(df, backend, "nlu_decomp_03_buyer_goods_total",
                            "For the buyer behind \"Supply of Suspension Elements\", what was the total value of their 2022 goods notices?",
                            anchor_title="Supply of Suspension Elements",
                            role_field="buyer_name",
                            followup=[{"field": "release_year", "op": "eq", "value": 2022},
                                      {"field": "tender_category", "op": "eq", "value": "goods"}]))
    rows.append(_decomp_count(df, backend, "nlu_decomp_04_supplier_signed",
                              "For the supplier on \"Miscanspeed sequencing procurement\", how many of their contracts have a signed award date?",
                              anchor_title="Miscanspeed sequencing procurement",
                              role_field="supplier_name",
                              followup=[{"field": "has_award_signed_date", "op": "eq", "value": True}]))
    return rows


def _factoid_by_title(df: Any, backend: RuntimeKGBackend, qid: str, title: str, question: str,
                      answer_field: str, value_type: str, category: str) -> dict[str, Any]:
    constraints = [{"field": "tender_title", "op": "eq", "value": title}]
    rows = _query(df, constraints)
    values = sorted({_jsonable(row.get(answer_field)) for _, row in rows.iterrows() if row.get(answer_field) not in (None, "")})
    if len(values) != 1:
        raise ValueError(f"{qid} does not have a unique {answer_field}: {values[:10]}")
    return _answerable(qid, question, "factoid", category, "select_unique", answer_field, value_type,
                       values[0], constraints, rows, backend,
                       difficulty_reason="natural title anchor; no internal contract id exposed")


def _count(df: Any, backend: RuntimeKGBackend, qid: str, question: str,
           constraints: list[dict[str, Any]], category: str) -> dict[str, Any]:
    rows = _query(df, constraints)
    return _answerable(qid, question, "aggregation_count", category, "count", "contract_node_id", "integer",
                       int(len(rows)), constraints, rows, backend,
                       difficulty_reason="natural wording must map to structured filters")


def _sum(df: Any, backend: RuntimeKGBackend, qid: str, question: str,
         constraints: list[dict[str, Any]], category: str) -> dict[str, Any]:
    all_constraints = constraints + [{"field": "value_is_additive", "op": "eq", "value": True}]
    rows = _query(df, all_constraints)
    total = sum(Decimal(str(value)) for value in rows["value_amount"].dropna().tolist())
    return _answerable(qid, question, "aggregation_sum", category, "sum", "value_amount", "currency",
                       str(total), all_constraints, rows, backend,
                       difficulty_reason="natural value wording must trigger additive sum guard")


def _decomp_count(df: Any, backend: RuntimeKGBackend, qid: str, question: str, *,
                  anchor_title: str, role_field: str, followup: list[dict[str, Any]]) -> dict[str, Any]:
    anchor = _query(df, [{"field": "tender_title", "op": "eq", "value": anchor_title}])
    values = sorted({str(v) for v in anchor[role_field].dropna().unique()})
    if len(values) != 1:
        raise ValueError(f"{qid} anchor not unique for {role_field}: {values[:10]}")
    constraints = [{"field": role_field, "op": "eq", "value": values[0]}] + followup
    rows = _query(df, constraints)
    result = _answerable(qid, question, "role_path", "bounded_decomposition", "count", "contract_node_id", "integer",
                         int(len(rows)), constraints, rows, backend,
                         difficulty_reason="requires title -> role entity -> filtered count decomposition")
    result["decomposition"] = [
        {"step": 1, "anchor_title": anchor_title, "extract_field": role_field, "intermediate_answer": values[0]},
        {"step": 2, "operation": "count", "constraints": constraints},
    ]
    return result


def _decomp_sum(df: Any, backend: RuntimeKGBackend, qid: str, question: str, *,
                anchor_title: str, role_field: str, followup: list[dict[str, Any]]) -> dict[str, Any]:
    anchor = _query(df, [{"field": "tender_title", "op": "eq", "value": anchor_title}])
    values = sorted({str(v) for v in anchor[role_field].dropna().unique()})
    if len(values) != 1:
        raise ValueError(f"{qid} anchor not unique for {role_field}: {values[:10]}")
    constraints = [{"field": role_field, "op": "eq", "value": values[0]}] + followup
    all_constraints = constraints + [{"field": "value_is_additive", "op": "eq", "value": True}]
    rows = _query(df, all_constraints)
    total = sum(Decimal(str(value)) for value in rows["value_amount"].dropna().tolist())
    result = _answerable(qid, question, "role_path", "bounded_decomposition", "sum", "value_amount", "currency",
                         str(total), all_constraints, rows, backend,
                         difficulty_reason="requires title -> role entity -> filtered additive sum decomposition")
    result["decomposition"] = [
        {"step": 1, "anchor_title": anchor_title, "extract_field": role_field, "intermediate_answer": values[0]},
        {"step": 2, "operation": "sum", "constraints": all_constraints},
    ]
    return result


def _unanswerable(qid: str, question: str, reason: str) -> dict[str, Any]:
    return {
        "id": qid,
        "category": "unanswerable_or_ambiguous",
        "question": question,
        "question_type": "unanswerable",
        "expected_answerable": False,
        "oracle_answer": None,
        "answer_operation": "unsupported",
        "answer_field": "",
        "answer_value_type": "",
        "constraints": [],
        "logic_chain": ["abstain"],
        "difficulty": "hard",
        "difficulty_reason": reason,
        "evidence_count": 0,
        "evidence_ids": [],
    }


def _answerable(qid: str, question: str, question_type: str, category: str, operation: str, answer_field: str,
                value_type: str, answer: Any, constraints: list[dict[str, Any]], rows: Any,
                backend: RuntimeKGBackend, *, difficulty_reason: str) -> dict[str, Any]:
    evidence_ids = [str(backend.record_id(row)) for row in rows.to_dict("records")]
    return {
        "id": qid,
        "category": category,
        "question": question,
        "question_type": question_type,
        "expected_answerable": True,
        "oracle_answer": _jsonable(answer),
        "answer_operation": operation,
        "answer_field": answer_field,
        "answer_value_type": value_type,
        "constraints": constraints,
        "logic_chain": [f"{c['field']}{c['op']}{c['value']}" for c in constraints] + [f"{operation} {answer_field}"],
        "difficulty": "hard" if question_type == "role_path" else "medium",
        "difficulty_reason": difficulty_reason,
        "evidence_count": int(len(rows)),
        "evidence_ids": evidence_ids[:50],
    }


def _query(df: Any, constraints: list[dict[str, Any]]) -> Any:
    out = df
    for constraint in constraints:
        field, op, value = constraint["field"], constraint["op"], constraint.get("value")
        if op == "eq":
            out = out[out[field] == value]
        else:
            raise ValueError(f"unsupported fixture op: {op}")
    return out


def _question_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "question": row["question"],
        "question_type": row["question_type"],
        "category": row["category"],
        "difficulty_reason": row["difficulty_reason"],
        "expected_answerable": row["expected_answerable"],
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return {
        "total": len(rows),
        "expected_answerable": sum(1 for row in rows if row["expected_answerable"]),
        "expected_abstention": sum(1 for row in rows if not row["expected_answerable"]),
        "by_category": counts,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


if __name__ == "__main__":
    main()
