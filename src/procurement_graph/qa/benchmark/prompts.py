"""Prompt builders for QA benchmark generation and semantic validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from .models import AnswerSpec, Constraint


def build_question_generation_prompt(
    spec: AnswerSpec,
    evidence_records: list[dict[str, Any]],
    golden_answer: Any,
) -> str:
    payload = {
        "task": "Write one natural-language question answerable from the KG evidence.",
        "requirements": [
            "Ask for exactly the value described by answer_spec.",
            "Do not reveal the answer in the question.",
            "Do not introduce entities, dates, amounts, or filters absent from the evidence.",
            "Prefer concise procurement-domain wording.",
        ],
        "answer_spec": _spec_payload(spec),
        "golden_answer": str(golden_answer),
        "evidence_records": evidence_records,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_semantic_verification_prompt(
    question: str,
    evidence_records: list[dict[str, Any]],
) -> str:
    payload = {
        "task": "Answer the question using only the supplied KG evidence.",
        "requirements": [
            "Return the answer and a short reason.",
            "If the question is ambiguous or unsupported, return uncertain.",
            "Do not use outside knowledge.",
        ],
        "question": question,
        "evidence_records": evidence_records,
        "output_schema": {
            "answer": "string | number | boolean | uncertain",
            "reason": "string",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _spec_payload(spec: AnswerSpec) -> dict[str, Any]:
    return {
        "spec_id": spec.spec_id,
        "logic_chain": list(spec.logic_chain),
        "constraints": [constraint.__dict__ for constraint in spec.constraints],
        "answer_operation": spec.answer_operation,
        "answer_field": spec.answer_field,
        "answer_value_type": spec.answer_value_type,
        "dedupe_key": spec.dedupe_key,
    }


# --- Stage 2 versioned prompts -------------------------------------------------------

GENERATION_PROMPT_VERSION = "qa_gen_v3"
GENERATION_SCHEMA_VERSION = "qa_question_schema_v1"
VERIFY_PROMPT_VERSION = "qa_verify_v2"
VERIFY_SCHEMA_VERSION = "qa_verify_schema_v1"
GenerationPromptVariant = Literal["current", "strict_filters", "natural_procurement"]

GENERATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "question": "string",
    "names_entity": "boolean",
    "disambiguators": ["string"],
}
VERIFY_RECOMPUTE_OUTPUT_SCHEMA: dict[str, Any] = {
    "answer": "string | number | 'uncertain'",
    "reason": "string",
}
VERIFY_FAITHFULNESS_OUTPUT_SCHEMA: dict[str, Any] = {
    "operation": "count | sum | select",
    "filters": [{"field": "string", "value": "string"}],
    "ambiguous": "boolean",
    "reason": "string",
}

# Fields a natural-language question may legitimately encode. Used as the verifier's
# extraction vocabulary in faithfulness mode (the spec values are never shown to it).
# value_source is deliberately excluded: it is a data-provenance attribute (which OCDS
# record the value came from), not a procurement concept a user would filter on, so it is
# kept as an evidence/answer field only and never used as a question filter.
FIELD_VOCABULARY: dict[str, str] = {
    "release_year": "calendar year the contract was published (integer 2022-2026)",
    "tender_category": "procurement category: goods, services, or works",
    "tender_cpv_id": "CPV classification code (8-digit string, e.g. 79623000)",
    "has_award_signed_date": "whether the award has a signed date (boolean)",
}

# Internal scoping/coverage constraints that the natural-language question does not
# express, so they are excluded from question generation and faithfulness comparison.
INTERNAL_FIELDS: frozenset[str] = frozenset({"value_is_additive", "supplier_count", "buyer_count"})

# Plain-language meaning of each filter field, so the generator phrases it faithfully
# (e.g. release_year is the publication year, not the award/signature year; and
# has_award_signed_date is an existence flag, not a date filter).
FIELD_MEANINGS: dict[str, str] = {
    "release_year": "the calendar year the contract notice was PUBLISHED (not the award or signature year)",
    "tender_category": "high-level procurement category: goods, services, or works",
    "tender_cpv_id": "the CPV classification code of the procurement",
    "value_source": "which record the contract value was taken from: award, contract, or tender",
    "has_award_signed_date": "the award simply HAS a signed date on record (an existence flag, not a date filter)",
    "contract_node_id": "the unique identifier of a single contract",
    "supplier_count": "number of suppliers on the contract",
    "buyer_count": "number of buyers on the contract",
}


def schema_hash(schema: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def generation_prompt_version(variant: str = "current") -> str:
    return GENERATION_PROMPT_VERSION if variant == "current" else f"{GENERATION_PROMPT_VERSION}_{variant}"


def describe_target(spec: AnswerSpec) -> str:
    if spec.answer_operation == "count":
        return "the number of matching contracts"
    if spec.answer_operation == "sum":
        return "the total contract value"
    return f"the {spec.answer_field.replace('_', ' ')}"


def generation_filters(spec: AnswerSpec) -> list[dict[str, Any]]:
    """Eq constraints the question should encode (drops internal scoping fields)."""
    return [
        {"field": c.field, "value": c.value}
        for c in spec.constraints
        if c.op == "eq" and c.field not in INTERNAL_FIELDS
    ]


def semantic_filters(spec: AnswerSpec) -> list[tuple[str, str]]:
    """Normalised (field, value) eq filters restricted to the question vocabulary."""
    return sorted(
        (c.field, str(c.value))
        for c in spec.constraints
        if c.op == "eq" and c.field in FIELD_VOCABULARY
    )


def _feedback_block(feedback: str) -> str:
    if not feedback:
        return ""
    return (
        "\nA previous attempt at this question was REJECTED by an independent checker for this "
        f"reason: {feedback}\nWrite a corrected question that fixes exactly that problem while "
        "still encoding every listed filter and nothing more.\n"
    )


def build_generation_messages(
    spec: AnswerSpec,
    *,
    cpv_description: str = "",
    variant: str = "current",
    feedback: str = "",
) -> tuple[str, str]:
    """(system, user) for question generation. The golden answer and answer-bearing
    evidence are deliberately NOT included, so the model cannot leak the answer."""
    system = (
        "You are a careful question writer for a UK public-procurement knowledge-graph QA "
        "benchmark. Given a structured query specification, write exactly ONE natural, "
        "unambiguous English question whose unique correct answer is what the specification "
        "selects.\n"
        "Rules:\n"
        "1. Encode every listed filter, using its stated meaning, so the answer is uniquely determined.\n"
        "2. Never state or hint at the answer value itself.\n"
        "3. Never introduce any entity, date, amount, code, or filter not listed.\n"
        "4. Use natural procurement wording (e.g. 'published in 2024', 'under CPV 79623000').\n"
        "5. release_year is the PUBLICATION year — phrase it as 'published/released in <year>', "
        "never as the award or signing year. If NO release_year is listed, do not mention a year, "
        "publication, or 'the dataset' at all; describe only the listed filters.\n"
        "6. A boolean existence filter such as has_award_signed_date=true means the record simply "
        "HAS that attribute — phrase it as 'that have a signed award date', never as a specific date "
        "or year, and never merge it with release_year.\n"
        "7. Only name an organisation if its name is unambiguous; otherwise add a disambiguator "
        "(region/year/category) or omit it.\n"
        f"{_variant_rules(variant)}"
        f"{_feedback_block(feedback)}"
        "Respond with strict JSON only, matching the output schema."
    )
    filters = generation_filters(spec)
    payload = {
        "target": describe_target(spec),
        "answer_operation": spec.answer_operation,
        "filters": filters,
        "filter_meanings": {item["field"]: FIELD_MEANINGS.get(item["field"], "") for item in filters},
        "cpv_description": cpv_description,
        "prompt_variant": variant,
        "output_schema": GENERATION_OUTPUT_SCHEMA,
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


_FACTOID_ASK = {
    "buyer_name": "the buyer (contracting authority)",
    "supplier_name": "the supplier (the awarded company)",
    "tender_category": "the procurement category (goods, services, or works)",
    "value_source": "which record the contract value was taken from (award, contract, or tender)",
    "award_date_signed": "the date the award was signed",
}


def factoid_ask(answer_field: str) -> str:
    return _FACTOID_ASK.get(answer_field, answer_field.replace("_", " "))


def build_factoid_generation_messages(
    answer_field: str,
    anchor: dict[str, Any],
    *,
    variant: str = "current",
    feedback: str = "",
) -> tuple[str, str]:
    """(system, user) for a factoid question anchored on a contract's natural attributes
    (buyer/supplier/CPV/year/category) instead of its internal id. The asked field is
    excluded from the anchor, so the answer cannot leak."""
    system = (
        "You are a careful question writer for a UK public-procurement knowledge-graph QA "
        "benchmark. You are given ONE specific contract, described by its attributes, and a "
        "field to ask about. Write exactly ONE natural English question asking for that field.\n"
        "Rules:\n"
        "1. Identify the contract naturally by the given attributes (buyer, supplier, CPV, year, "
        "category). NEVER use or invent an internal id or reference number.\n"
        "2. Never state or hint at the value of the asked field.\n"
        "3. Use only the given attributes; do not add new ones.\n"
        "4. Phrase CPV as 'CPV <code> (<description>)' and year as 'in <year>'.\n"
        f"{_factoid_variant_rules(variant)}"
        f"{_feedback_block(feedback)}"
        "Respond with strict JSON only, matching the output schema."
    )
    payload = {
        "ask_for": factoid_ask(answer_field),
        "contract": anchor,
        "prompt_variant": variant,
        "output_schema": GENERATION_OUTPUT_SCHEMA,
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


def _variant_rules(variant: str) -> str:
    if variant == "strict_filters":
        return (
            "8. Every item in the filters list is a semantic filter and MUST be explicitly verbalised "
            "in the question. Do not omit a filter because it feels technical.\n"
            "9. If a CPV code is present, include the exact CPV code; include the description only as "
            "supporting wording.\n"
        )
    if variant == "natural_procurement":
        return (
            "8. Prefer fluent UK procurement wording such as 'contract notices', 'published in', "
            "'under CPV', and 'where the value was taken from'.\n"
            "9. Keep the question natural, but never trade away precision: every listed filter must "
            "still be recoverable from the wording.\n"
        )
    return ""


def _factoid_variant_rules(variant: str) -> str:
    if variant == "strict_filters":
        return (
            "5. Use every provided contract attribute needed to identify the contract; do not drop "
            "CPV, year, buyer, supplier, or category when present.\n"
        )
    if variant == "natural_procurement":
        return (
            "5. Prefer a concise procurement-style sentence over a database-style lookup, while "
            "preserving all identifying attributes.\n"
        )
    return ""


def build_verify_recompute_messages(
    question: str,
    evidence_sample: list[dict[str, Any]],
    *,
    answer_value_type: str,
) -> tuple[str, str]:
    """(system, user) for recompute-mode Gate B: answer from evidence only."""
    system = (
        "You answer questions about UK public-procurement contracts using ONLY the supplied "
        "evidence records. Do not use outside knowledge. If the question is ambiguous or the "
        "evidence is insufficient, answer exactly 'uncertain'. For counting questions return "
        "the integer count of matching evidence records. Respond with strict JSON only."
    )
    payload = {
        "question": question,
        "expected_answer_type": answer_value_type,
        "evidence_records": evidence_sample,
        "output_schema": VERIFY_RECOMPUTE_OUTPUT_SCHEMA,
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


def build_verify_faithfulness_messages(question: str) -> tuple[str, str]:
    """(system, user) for faithfulness-mode Gate B: extract the structured query the
    question encodes, using only the allowed field vocabulary (never the spec)."""
    system = (
        "You convert a natural-language procurement question into the structured database query "
        "it asks for, to check the question is unambiguous.\n"
        "- Report the aggregation OPERATION: 'count' = number of matching contracts; "
        "'sum' = total monetary value of the matching contracts; 'select' = a single field value.\n"
        "- The value being summed is ALWAYS the contract's total value. It is implicit, it is NOT a "
        "filter, and it is NOT expected to appear in filter_fields. Never treat its absence as a problem.\n"
        "- Report the FILTERS the question applies, using ONLY filter_fields (these cover filters only).\n"
        "- Set ambiguous=true ONLY when the question is genuinely unclear about which operation to use "
        "or which filter values to apply. Never set ambiguous merely because the summed value, or any "
        "concept, is not in filter_fields.\n"
        "Respond with strict JSON only."
    )
    payload = {
        "question": question,
        "filter_fields": FIELD_VOCABULARY,
        "operations": {
            "count": "number of matching contracts",
            "sum": "total contract value of the matches (value is implicit, not a filter)",
            "select": "one field value",
        },
        "output_schema": VERIFY_FAITHFULNESS_OUTPUT_SCHEMA,
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "build_question_generation_prompt",
    "build_semantic_verification_prompt",
    "build_generation_messages",
    "build_verify_recompute_messages",
    "build_verify_faithfulness_messages",
    "describe_target",
    "generation_filters",
    "semantic_filters",
    "schema_hash",
    "generation_prompt_version",
    "GENERATION_PROMPT_VERSION",
    "GenerationPromptVariant",
    "GENERATION_SCHEMA_VERSION",
    "GENERATION_OUTPUT_SCHEMA",
    "VERIFY_PROMPT_VERSION",
    "VERIFY_SCHEMA_VERSION",
    "VERIFY_RECOMPUTE_OUTPUT_SCHEMA",
    "VERIFY_FAITHFULNESS_OUTPUT_SCHEMA",
    "FIELD_VOCABULARY",
    "INTERNAL_FIELDS",
]
