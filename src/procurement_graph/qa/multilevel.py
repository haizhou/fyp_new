"""Multi-level QA surface generation and validation (plan-first).

The template-controlled benchmark conflates two capabilities: *executor validation* (given the
right plan, is the computed answer right?) and *language-to-plan generalization* (given diverse
natural language, can a planner recover the plan?). This module separates them:

- **Level 0 — plan**: the executor-validated (constraints, operation, oracle) row itself; measured
  with `eval --mode executor`. No language involved.
- **Level 1 — template surface**: the source row's question, exactly as generated. A rule planner
  built against the same templates SHOULD saturate this level; that is the control, not the result.
- **Level 2 — paraphrase surface**: an LLM rewrite that preserves the plan semantics but varies
  syntax, register and word order. Measures genuine language-to-plan generalization.
- **Level 3 — adversarial surface**: an LLM rewrite that additionally embeds the question in
  distracting context (irrelevant preamble, split sentences, indirection) WITHOUT introducing any
  new entity, number or filter. Measures robustness of the plan layer.

All surfaces of one plan share ONE oracle, so accuracy differences across levels are attributable
to language alone. LLM rewrites are accepted only through deterministic gates (`check_surface`):
every plan-bearing atom must survive verbatim, no new numbers may appear, no new KG organisation
may be mentioned, and unanswerable items must keep their unanswerable trigger phrase. The gate is
generate-then-validate — the LLM can be wrong, but a wrong rewrite is rejected, never graded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..reasoning.linking import UNSUPPORTED_TERMS, _JUDGEMENT_TERMS

LEVELS = (1, 2, 3)
# standalone numbers only: digits embedded in alphanumeric tokens ("B1", "4Delivery Ltd") are
# parts of names, not quantities, and must not trip the new-number gate.
_NUM_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])")
_ORG_FIELDS = ("buyer_name", "supplier_name")
_WORDY = re.compile(r"[A-Za-z]")
_TEMPORAL_RELATION_RE = re.compile(r"\b(after|before|later|subsequently|previously|prior to|then)\b", re.I)
_BUYERS_OF_SUPPLIER_DRIFT_RE = re.compile(
    r"\bwhere\s+(?:the\s+)?(?:contract|notice|contract\s+notice)s?\s+"
    r"(?:was|were|is|are)?\s*awarded\s+to\b",
    re.I,
)


# ---------------------------------------------------------------------------- atoms
@dataclass(frozen=True)
class SurfaceAtoms:
    """Plan-bearing fragments that any surface realisation must preserve verbatim."""

    required_texts: tuple[str, ...] = ()      # org names, categories, quoted titles, dates
    required_numbers: tuple[str, ...] = ()    # years, CPV codes, thresholds (digit strings)
    unanswerable_trigger: str = ""            # e.g. "social value" for unsupported items
    org_texts: tuple[str, ...] = ()            # orgs may be abbreviated at L2 if the checker agrees


def required_atoms(row: dict[str, Any]) -> SurfaceAtoms:
    texts: list[str] = []
    numbers: list[str] = []
    orgs: list[str] = []
    for constraint in row.get("constraints", ()):  # hidden guards carry no surface text
        if constraint.get("visible_to_user") is False:
            continue
        value = constraint.get("value")
        field_name = str(constraint.get("field", ""))
        _collect_atoms(field_name, value, texts, numbers)
        _collect_org_atoms(field_name, value, orgs)
        if field_name == "tender_cpv_id" and isinstance(value, (str, int)):
            pass  # already captured as a number above
    trigger = _unanswerable_trigger(row)
    return SurfaceAtoms(
        required_texts=tuple(dict.fromkeys(texts)),
        required_numbers=tuple(dict.fromkeys(numbers)),
        unanswerable_trigger=trigger,
        org_texts=tuple(dict.fromkeys(orgs)),
    )


def _collect_atoms(field_name: str, value: Any, texts: list[str], numbers: list[str]) -> None:
    if isinstance(value, bool) or value in (None, ""):
        return
    if isinstance(value, (int, float)):
        numbers.append(_digits(value))
    elif isinstance(value, str) and value.isdigit():
        numbers.append(value)
    elif isinstance(value, str):
        date = re.match(r"^(\d{4})-(\d{2})-(\d{2})", value)
        if date:
            numbers.extend(g.lstrip("0") or "0" for g in date.groups())
        elif _looks_surface_atom(field_name, value):
            texts.append(value)
    elif isinstance(value, dict):
        for key, nested in value.items():
            if key in {"resolve", "op", "field"}:
                continue
            _collect_atoms(str(key), nested, texts, numbers)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _collect_atoms(field_name, nested, texts, numbers)


def _collect_org_atoms(field_name: str, value: Any, orgs: list[str]) -> None:
    if field_name in {"buyer", "buyer_name", "supplier", "supplier_name"} and isinstance(value, str) and value:
        orgs.append(value)
    elif isinstance(value, dict):
        for key, nested in value.items():
            if key in {"buyer", "supplier", "buyer_name", "supplier_name"} and isinstance(nested, str) and nested:
                orgs.append(nested)
            else:
                _collect_org_atoms(str(key), nested, orgs)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _collect_org_atoms(field_name, nested, orgs)


def _looks_surface_atom(field_name: str, text: str) -> bool:
    if field_name in {"buyer", "buyer_name", "supplier", "supplier_name", "tender_title", "award_title"}:
        return True
    return bool(re.search(r"\s", text)) or text in {"goods", "services", "works"}


def _digits(value: Any) -> str:
    text = f"{value}"
    return text[:-2] if text.endswith(".0") else text


def _unanswerable_trigger(row: dict[str, Any]) -> str:
    if row.get("expected_status", "answerable") == "answerable":
        return ""
    question = str(row.get("question", "")).casefold()
    for term in UNSUPPORTED_TERMS:
        if term in question:
            return term
    for term in _JUDGEMENT_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", question):
            return term
    return ""


# ---------------------------------------------------------------------------- gate
@dataclass(frozen=True)
class SurfaceVerdict:
    ok: bool
    reasons: tuple[str, ...] = ()
    checks: dict[str, Any] = field(default_factory=dict)


def check_surface(surface: str, atoms: SurfaceAtoms, source_question: str, *,
                  level: int, org_resolver: Any = None,
                  known_org_names: frozenset[str] | None = None,
                  relax_org_texts: bool = False) -> SurfaceVerdict:
    """Deterministic accept/reject for an LLM-produced surface realisation."""
    reasons: list[str] = []
    text = " ".join(str(surface or "").split())
    lowered = text.casefold()
    source_lowered = " ".join(source_question.casefold().split())

    if len(text) < 20 or not _WORDY.search(text):
        reasons.append("too_short")
    if text.count("?") != 1 or not text.rstrip().endswith("?"):
        reasons.append("not_a_single_question")

    # L2 hard gate is intentionally numeric-only. Entity names, categories, role direction,
    # unsupported concepts, and other semantic constraints are judged by the Checker LLM against
    # the reference reasoning path. L3 keeps the older stricter atom-preservation gate.
    if level != 2:
        relaxed_orgs = {t.casefold() for t in atoms.org_texts} if relax_org_texts else set()
        for required in atoms.required_texts:
            if required.casefold() not in lowered:
                if required.casefold() in relaxed_orgs:
                    continue
                reasons.append(f"missing_text:{required}")
    for number in atoms.required_numbers:
        if number not in _NUM_RE.findall(text):
            reasons.append(f"missing_number:{number}")
    if level != 2 and atoms.unanswerable_trigger and atoms.unanswerable_trigger not in lowered:
        reasons.append(f"missing_unanswerable_trigger:{atoms.unanswerable_trigger}")

    allowed_numbers = set(atoms.required_numbers) | set(_NUM_RE.findall(source_question))
    foreign = [n for n in _NUM_RE.findall(text) if n not in allowed_numbers]
    if foreign:
        reasons.append(f"new_numbers:{','.join(sorted(set(foreign)))[:60]}")
    new_temporal = _new_temporal_relations(text, source_question)
    if new_temporal:
        reasons.append(f"new_temporal_relation:{','.join(new_temporal)}")

    if level == 3 and _too_similar(lowered, source_lowered):
        reasons.append("not_actually_rewritten")
    # L2 is verbalized from the plan without seeing the template, so coincidental similarity is
    # legitimate evidence, not laziness — the similarity gate applies only to L3 paraphrases.

    if level != 2:
        foreign_org = _foreign_org(text, atoms, org_resolver=org_resolver, known=known_org_names)
        if foreign_org:
            reasons.append(f"new_org_mention:{foreign_org[:60]}")

    return SurfaceVerdict(ok=not reasons, reasons=tuple(reasons),
                          checks={"length": len(text), "level": level})


def _too_similar(a: str, b: str) -> bool:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return True
    return len(ta & tb) / len(ta | tb) > 0.9


def _new_temporal_relations(surface: str, source_question: str) -> tuple[str, ...]:
    """Reject invented ordering language such as "later awarded" on bridge questions."""
    source_terms = {m.group(1).casefold() for m in _TEMPORAL_RELATION_RE.finditer(source_question)}
    surface_terms = {m.group(1).casefold() for m in _TEMPORAL_RELATION_RE.finditer(surface)}
    return tuple(sorted(surface_terms - source_terms))


def bridge_drift_reasons(surface: str, row: dict[str, Any]) -> tuple[str, ...]:
    """Plan-aware L2 guards for bridge rewrites that can fool a generic checker."""
    reasons: list[str] = []
    for constraint in row.get("constraints", ()):
        value = constraint.get("value")
        if (
            constraint.get("field") == "buyer_name"
            and constraint.get("op") == "in_subquery"
            and isinstance(value, dict)
            and value.get("resolve") == "buyers_of_supplier"
            and _BUYERS_OF_SUPPLIER_DRIFT_RE.search(surface)
        ):
            reasons.append("bridge_relation_drift:buyers_of_supplier_as_direct_supplier_filter")
    return tuple(reasons)


def _foreign_org(text: str, atoms: SurfaceAtoms, *, org_resolver: Any = None,
                 known: frozenset[str] | None = None) -> str:
    """A capitalised span that resolves to a KG organisation NOT in the plan is a new reading."""
    # containment is only meaningful against org-length atoms; short atoms like the category
    # word 'services' must not whitelist every org name that happens to contain them.
    allowed = {t.casefold() for t in atoms.required_texts if len(t) >= 10}
    spans = re.findall(r"(?:[A-Z][\w&.'()\-]*\s+){1,6}(?:Ltd\.?|Limited|LLP|Council|Trust|Board|"
                       r"University|NHS|Group|Authority|Agency)", text)
    for span in spans:
        candidate = span.strip(" .,")
        lowered = candidate.casefold()
        if any(lowered in a or a in lowered for a in allowed):
            continue
        if known is not None and lowered in known:
            return candidate
        if org_resolver is not None:
            hits = org_resolver.resolve(candidate)
            if hits and hits[0].source == "records_exact":
                return candidate
    return ""


# ---------------------------------------------------------------------------- prompts
def rewrite_messages(row: dict[str, Any], atoms: SurfaceAtoms, *, level: int,
                     n_variants: int = 1) -> tuple[str, str]:
    """(system, user) messages asking the LLM for level-2/3 surfaces. JSON out, no answering."""
    base_rules = [
        "Rewrite the question so the MEANING is identical: same filters, same asked-for quantity.",
        "Keep every organisation name, CPV code, year, amount and quoted title EXACTLY verbatim.",
        "Do NOT introduce any other organisation, number, year, CPV code or place.",
        "Do NOT answer the question.",
        "Each variant must end with a question mark and contain exactly one question.",
    ]
    if level == 2:
        style = ("Vary the syntax, register and word order substantially (formal report style, "
                 "casual analyst style, passive voice, clause reordering). No added context.")
    else:
        style = ("Embed the question in realistic distracting context: an irrelevant preamble "
                 "sentence, hedging, or an indirect reference back to the named organisation "
                 "(e.g. 'that organisation'). The distraction must stay content-free: no new "
                 "facts, names or figures. You may use two sentences, but only one question.")
    system = ("You rewrite UK public-procurement benchmark questions. Return strict JSON "
              '{"variants": ["...", ...]} and nothing else.')
    user = json.dumps({
        "level": level,
        "n_variants": n_variants,
        "rules": base_rules + [style],
        "must_keep_verbatim": list(atoms.required_texts) + list(atoms.required_numbers),
        "must_keep_phrase": atoms.unanswerable_trigger or None,
        "question": row["question"],
    }, ensure_ascii=False)
    return system, user


PERSONAS = {
    "citizen": "an ordinary member of the public who wants a plain-language answer",
    "policy_analyst": "a policy analyst writing a briefing note, precise and formal",
    "auditor": "a public-spending auditor checking records, terse and exact",
    "journalist": "an investigative journalist drafting a data query, direct and probing",
}

PERSONA_EXAMPLES = {
    "citizen": {
        "before": "How many services contract notices did Birmingham City Council publish in 2024 under CPV 85000000?",
        "after": "In 2024, how many services notices did Birmingham City Council put out under CPV 85000000?",
    },
    "policy_analyst": {
        "before": "What is the total value of works contracts awarded to suppliers who also worked with NHS England?",
        "after": "For suppliers that had also worked with NHS England, what was the total value of the relevant works contracts?",
    },
    "auditor": {
        "before": "Which supplier is recorded for contract notices published in 2023 by Kent County Council under CPV 72000000?",
        "after": "For 2023 notices from Kent County Council under CPV 72000000, which supplier is recorded?",
    },
    "journalist": {
        "before": "Did Leeds City Council publish more goods notices in 2025 than Sheffield City Council?",
        "after": "In 2025, did Leeds City Council put out more goods notices than Sheffield City Council?",
    },
}


def persona_for(plan_id: str) -> str:
    """Balanced deterministic persona assignment: ONE style per plan (diversity mechanism,
    not question multiplication)."""
    names = sorted(PERSONAS)
    return names[sum(ord(c) for c in plan_id) % len(names)]


def l2_rewrite_messages(row: dict[str, Any], atoms: SurfaceAtoms, *,
                        persona: str = "", n_variants: int = 1) -> tuple[str, str]:
    """Task / example / note prompt for L2: rewrite the L1 question, not the raw plan."""
    chosen = persona if persona in PERSONAS else persona_for(str(row.get("id", "")))
    system = ("You rewrite UK public-procurement benchmark questions. Return strict JSON "
              '{"variants": ["...", ...]} and nothing else.')
    user = json.dumps({
        "task": {
            "identity": PERSONAS[chosen],
            "instruction": (
                "Rewrite the L1 question into a natural question in this identity's voice. "
                "Keep the exact same meaning, answer target, filters, comparison direction, "
                "and unanswerable/ambiguous/no-results status if applicable."
            ),
            "n_variants": n_variants,
            "l1_question": row.get("question", ""),
        },
        "examples_by_identity": PERSONA_EXAMPLES,
        "selected_identity_example": PERSONA_EXAMPLES[chosen],
        "note": [
            "Do not answer the question.",
            "Return exactly one question per variant, ending with a question mark.",
            "All years, CPV codes, dates, counts, thresholds, and money amounts must be unchanged.",
            "Organisation names may use an obvious abbreviation, acronym, or shortened public name "
            "only when it still clearly refers to the same organisation.",
            "Do not introduce any new organisation, place, CPV code, year, amount, threshold, filter, "
            "or condition.",
            "Do not add ordering or timing relations such as later, subsequently, after, before, "
            "previously, or prior to unless the L1 question already says that.",
            "Avoid database-style wording such as release_year, buyer_name, supplier_name, "
            "tender_cpv_id, tender_category, in_subquery, matching procurement records, or asked-for field.",
            "Vary sentence structure substantially; do not just swap one or two words.",
        ],
        "must_keep_exact_numbers": list(atoms.required_numbers),
        "must_preserve_meaning_of_text": list(atoms.required_texts),
        "org_names_may_be_abbreviated": list(atoms.org_texts),
        "must_keep_unanswerable_phrase_if_present": atoms.unanswerable_trigger or None,
    }, ensure_ascii=False)
    return system, user


def checker_messages(surface: str, row: dict[str, Any]) -> tuple[str, str]:
    """Independent Checker LLM: judges whether the generated question still expresses the plan.

    The checker never sees or produces the oracle answer — it outputs its own reading of the
    question (operation / answer field / constraints) plus a match verdict, and acceptance
    requires `matches_original_plan` and `can_be_answered_by_original_plan`.
    """
    system = ("You verify benchmark questions for a procurement KGQA dataset. Read the question, "
              "state the reasoning task it expresses, and judge whether it matches the reference "
              "plan. NEVER answer the question itself. Return strict JSON with keys: "
              "intended_operation, intended_answer_field, required_constraints (object), "
              "reasoning_steps (array), matches_original_plan (bool), "
              "can_be_answered_by_original_plan (bool), mismatch_reason (string|null).")
    user = json.dumps({
        "question": surface,
        "reference_plan": {
            "operation": row.get("answer_operation"),
            "answer_type": row.get("answer_type"),
            "constraints": [c for c in row.get("constraints", ())
                            if c.get("visible_to_user") is not False],
            "expected_status": row.get("expected_status", "answerable"),
        },
        "judge": "Does the question ask EXACTLY the task the reference plan computes — same "
                 "operation, same filters, same asked-for field, buyer/supplier roles the right "
                 "way round, comparison directions preserved?",
    }, ensure_ascii=False)
    return system, user


def checker_accepts(verdict: Any) -> tuple[bool, str]:
    if not isinstance(verdict, dict):
        return False, "checker_unparseable"
    if not verdict.get("matches_original_plan"):
        return False, f"checker_mismatch:{verdict.get('mismatch_reason')}"
    if not verdict.get("can_be_answered_by_original_plan"):
        return False, f"checker_unanswerable:{verdict.get('mismatch_reason')}"
    return True, ""


def checker_messages(surface: str, row: dict[str, Any]) -> tuple[str, str]:
    """Independent Checker LLM: government-consultant semantic audit."""
    expected_status = row.get("expected_status", "answerable")
    system = (
        "You are a UK public-procurement government consultation adviser. A member of the public "
        "asks a question; we provide the intended reasoning route and the benchmark reference "
        "answer status. Your job is NOT to answer the question. Your job is to verify whether the "
        "question and the provided reasoning route are logically consistent, and whether following "
        "that route would derive the reference answer. For abstention items, 'cannot answer' is "
        "itself the reference answer. Return strict JSON only."
    )
    judge = (
        "Check semantic logic, not spelling. Organisation names may be abbreviated or shortened if "
        "they still clearly refer to the same body. Verify the answer target, filters, buyer/supplier "
        "roles, comparison direction, bridge/multi-hop relationship, and abstention status. Numeric "
        "items (years, CPV codes, dates, money amounts, thresholds) have already been hard-checked "
        "by deterministic code, but you may still mention numeric inconsistencies if they change "
        "the meaning."
    )
    user = json.dumps({
        "task": "Verify whether the rewritten public question matches the provided reasoning route.",
        "question": surface,
        "source_question": row.get("question", ""),
        "reference_reasoning_route": {
            "operation": row.get("answer_operation"),
            "answer_type": row.get("answer_type"),
            "constraints": [c for c in row.get("constraints", ())
                            if c.get("visible_to_user") is not False],
            "expected_status": expected_status,
        },
        "reference_answer": row.get("oracle_answer"),
        "examples": [
            {
                "question": "In 2024, how many services notices did Birmingham City Council publish under CPV 85000000?",
                "reasoning_route": {
                    "operation": "count",
                    "constraints": [
                        {"field": "buyer_name", "op": "eq", "value": "Birmingham City Council"},
                        {"field": "release_year", "op": "eq", "value": 2024},
                        {"field": "tender_category", "op": "eq", "value": "services"},
                        {"field": "tender_cpv_id", "op": "eq", "value": "85000000"}
                    ],
                    "expected_status": "answerable"
                },
                "reference_answer": 12,
                "verdict": {
                    "matches_original_plan": True,
                    "can_derive_reference_answer": True,
                    "mismatch_reason": None
                }
            },
            {
                "question": "In 2024, how many goods notices did Birmingham City Council publish under CPV 85000000?",
                "reasoning_route": {
                    "operation": "count",
                    "constraints": [
                        {"field": "buyer_name", "op": "eq", "value": "Birmingham City Council"},
                        {"field": "release_year", "op": "eq", "value": 2024},
                        {"field": "tender_category", "op": "eq", "value": "services"},
                        {"field": "tender_cpv_id", "op": "eq", "value": "85000000"}
                    ],
                    "expected_status": "answerable"
                },
                "reference_answer": 12,
                "verdict": {
                    "matches_original_plan": False,
                    "can_derive_reference_answer": False,
                    "mismatch_reason": "Question asks for goods, but the reasoning route filters services."
                }
            }
        ],
        "output_schema": {
            "intended_operation": "operation implied by the question",
            "intended_answer_field": "field or quantity the question asks for",
            "required_constraints": "filters and relationships implied by the question",
            "reasoning_steps": ["short logical reading of the question"],
            "matches_original_plan": "bool: question and reasoning route are semantically aligned",
            "can_be_answered_by_original_plan": "bool: old alias for can_derive_reference_answer",
            "can_derive_reference_answer": "bool: route would derive the reference answer/status",
            "same_meaning_as_source_question": "bool, especially for abstention rewrites",
            "preserves_unanswerable_status": "bool, especially for unsupported/ambiguous/no-results rows",
            "mismatch_reason": "null or concise reason"
        },
        "note": [
            "Do not solve the procurement query yourself.",
            "Do not reject solely because an organisation name is shortened, if the referent is clear.",
            "Reject if buyer/supplier roles are flipped.",
            "Reject if the asked-for quantity changes, such as count vs total value vs which supplier.",
            "Reject if comparison direction changes, such as more-than vs less-than or after vs before.",
            "Reject if a bridge relationship changes, such as suppliers-who-worked-with-X vs buyers-of-supplier-X.",
            "For buyers-of-supplier bridge questions, reject wording that makes the final notices themselves "
            "sound awarded to the supplier, such as 'where the contract was awarded to X'.",
            "For unsupported/ambiguous/no-results rows, accept only if the rewritten question preserves the same reason to abstain."
        ],
        "judge": judge,
    }, ensure_ascii=False)
    return system, user


def checker_accepts(verdict: Any, *, expected_status: str = "answerable") -> tuple[bool, str]:
    if not isinstance(verdict, dict):
        return False, "checker_unparseable"
    can_derive = verdict.get("can_derive_reference_answer", verdict.get("can_be_answered_by_original_plan"))
    if expected_status != "answerable":
        same = verdict.get("same_meaning_as_source_question", verdict.get("matches_original_plan"))
        preserves = verdict.get("preserves_unanswerable_status", True)
        if not same:
            return False, f"checker_mismatch:{verdict.get('mismatch_reason')}"
        if preserves is False:
            return False, f"checker_status_drift:{verdict.get('mismatch_reason')}"
        if can_derive is False:
            return False, f"checker_cannot_derive_reference:{verdict.get('mismatch_reason')}"
        return True, ""
    if not verdict.get("matches_original_plan"):
        return False, f"checker_mismatch:{verdict.get('mismatch_reason')}"
    if not can_derive:
        return False, f"checker_unanswerable:{verdict.get('mismatch_reason')}"
    return True, ""


_FIELD_LABELS = {
    "buyer_name": "buyer",
    "supplier_name": "supplier",
    "release_year": "publication year",
    "tender_cpv_id": "CPV code",
    "tender_category": "procurement category",
    "tender_title": "tender title",
    "award_title": "award title",
    "award_date_signed": "award signed date",
    "contract_node_id": "contract identifier",
    "value_amount": "contract value",
}


def l2_plan_ready(row: dict[str, Any]) -> tuple[bool, str]:
    """L2 rewrites L1 directly, so every source row is eligible for generation."""
    return True, ""


def answer_request(row: dict[str, Any]) -> str:
    op = row.get("answer_operation", "count")
    field = _FIELD_LABELS.get(str(row.get("answer_field", "")), str(row.get("answer_field", "")))
    if op == "count":
        return "how many matching contract notices there are"
    if op == "sum":
        return "the total awarded value of the matching contract notices"
    if op == "exists":
        return "whether any matching contract notice exists"
    if op == "select_unique":
        return f"which {field} is recorded for the matching contract notice"
    if op in {"argmax", "argmin"}:
        direction = "highest" if op == "argmax" else "lowest"
        return f"which matching contract notice has the {direction} recorded value"
    if op in {"distinct_set", "set"}:
        return f"the distinct {field or 'values'} recorded across the matching notices"
    if op in {"rank_top_k", "top_k"}:
        group = _FIELD_LABELS.get(str(row.get("group_by_field", "")), str(row.get("group_by_field", "")) or "groups")
        metric = str(row.get("metric", "count"))
        return f"the top matching {group} by {metric}"
    return f"the result of the {op} operation over the matching notices"


def filter_phrases(row: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    for constraint in row.get("constraints", ()):
        if constraint.get("visible_to_user") is False:
            continue
        field = str(constraint.get("field", ""))
        op = str(constraint.get("op", "eq"))
        value = constraint.get("value")
        phrase = _constraint_phrase(field, op, value)
        if phrase:
            phrases.append(phrase)
    return phrases


def _constraint_phrase(field: str, op: str, value: Any) -> str:
    if op == "in_subquery" and isinstance(value, dict):
        return _subquery_phrase(field, value)
    if op != "eq":
        label = _FIELD_LABELS.get(field, field)
        return f"{label} {op} {_display_value(value)}"
    if field == "buyer_name":
        return f"published by {_display_value(value)}"
    if field == "supplier_name":
        return f"awarded to {_display_value(value)}"
    if field == "release_year":
        return f"published in {_display_value(value)}"
    if field == "tender_cpv_id":
        return f"under CPV code {_display_value(value)}"
    if field == "tender_category":
        return f"in the {_display_value(value)} category"
    if field in {"tender_title", "award_title"}:
        return f"with {_FIELD_LABELS.get(field, field)} \"{_display_value(value)}\""
    if field == "award_date_signed":
        return f"with award signed date {_display_value(value)}"
    return f"{_FIELD_LABELS.get(field, field)} {_display_value(value)}"


def _subquery_phrase(field: str, value: dict[str, Any]) -> str:
    resolve = str(value.get("resolve", ""))
    if resolve == "suppliers_of_buyer":
        return f"awarded to suppliers who also worked with {_display_value(value.get('buyer'))}"
    if resolve == "buyers_of_supplier":
        return f"published by buyers who have awarded to {_display_value(value.get('supplier'))}"
    if resolve == "cpvs_of_buyer":
        return f"under CPV codes that {_display_value(value.get('buyer'))} has used"
    label = _FIELD_LABELS.get(field, field)
    return f"{label} drawn from the linked set {_display_value(value)}"


def _display_value(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_display_value(v)}" for k, v in value.items() if k != "resolve")
    return str(value)


def verbalize_messages(row: dict[str, Any], atoms: SurfaceAtoms, *,
                       n_variants: int = 1, persona: str = "") -> tuple[str, str]:
    """(system, user) messages for Level 2: verbalize the PLAN alone into a question.

    The template question is deliberately withheld — L2 surfaces carry no template DNA, so a
    planner cannot succeed by matching generation-side phrasing. Only the structured plan
    (operation + constraints) and the verbatim atoms are shown.
    """
    voice = PERSONAS.get(persona, "a procurement analyst")
    system = ("You write natural-language questions for a UK public-procurement QA benchmark. "
              'Return strict JSON {"variants": ["...", ...]} and nothing else.')
    user = json.dumps({
        "n_variants": n_variants,
        "task": f"Write a question, in the voice of {voice}, whose answer is {answer_request(row)}, "
                "restricted by ALL of the filters below.",
        "filters": filter_phrases(row),
        "rules": [
            "Every filter must be expressed; keep organisation names, CPV codes, years, amounts "
            "and quoted titles EXACTLY verbatim.",
            "Do not add any other organisation, number, year or place.",
            "Phrase it naturally; do NOT use database field names such as release_year, buyer_name, "
            "supplier_name, tender_cpv_id, tender_category, in_subquery, or asked-for field.",
            "Exactly one question, ending with a question mark.",
            "Do NOT answer it.",
        ],
        "must_keep_verbatim": list(atoms.required_texts) + list(atoms.required_numbers),
    }, ensure_ascii=False)
    return system, user


# ---------------------------------------------------------------------------- assembly
def plan_bank_row(row: dict[str, Any]) -> dict[str, Any]:
    """Level-0 record: the executor-validated plan.

    `canonical_question` is provenance for spec reconstruction (e.g. the k in top-k), not an eval
    surface; the exec-relevant plan fields (answer_field / metric / group_by_field /
    executor_support) travel with the plan so the L0 executor run rebuilds the exact spec.
    """
    bank = {
        "plan_id": row["id"],
        "source_subset": row.get("subset", ""),
        "constraints": row.get("constraints", []),
        "answer_operation": row.get("answer_operation", ""),
        "answer_type": row.get("answer_type", ""),
        "expected_status": row.get("expected_status", "answerable"),
        "oracle_answer": row.get("oracle_answer"),
        "metadata": row.get("metadata", {}),
        "template_family": row.get("template_family", ""),
        "canonical_question": row.get("canonical_question") or row.get("question", ""),
    }
    for key in ("answer_field", "metric", "group_by_field", "executor_support"):
        if key in row:
            bank[key] = row[key]
    return bank


def surface_row(row: dict[str, Any], *, level: int, question: str, origin: str,
                variant: int = 0, verdict: SurfaceVerdict | None = None) -> dict[str, Any]:
    """One flat eval row: a surface plus everything needed to score it against the shared oracle."""
    return {
        "id": f"{row['id']}#L{level}{chr(ord('a') + variant) if level > 1 else ''}",
        "plan_id": row["id"],
        "level": level,
        "surface_origin": origin,
        "subset": row.get("subset", ""),
        "question": question,
        "answer_type": row.get("answer_type", ""),
        "answer_operation": row.get("answer_operation", ""),
        "expected_status": row.get("expected_status", "answerable"),
        "constraints": row.get("constraints", []),
        "oracle_answer": row.get("oracle_answer"),
        "metadata": row.get("metadata", {}),
        "surface_checks": verdict.checks if verdict else {},
    }


__all__ = ["LEVELS", "PERSONAS", "PERSONA_EXAMPLES", "SurfaceAtoms", "SurfaceVerdict", "answer_request",
           "bridge_drift_reasons", "check_surface", "checker_accepts", "checker_messages", "filter_phrases",
           "l2_plan_ready", "l2_rewrite_messages", "persona_for", "plan_bank_row", "required_atoms",
           "rewrite_messages", "surface_row", "verbalize_messages"]
