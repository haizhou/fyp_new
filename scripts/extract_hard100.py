#!/usr/bin/env python3
"""Extract a hard-100 diagnostic eval set from the verified benchmark.

Read-only over the generated QA files (never regenerates or verifies). Selects 100 "hard but
diagnosable" questions with a reasonable, audited distribution, and writes two files drop-in
compatible with `scripts/run_hard20_nano.py`:

  data/qa/eval/hard100_answer_key.jsonl   full: oracle_answer, constraints, evidence_ids, ...
  data/qa/eval/hard100_questions_only.jsonl   id / question / question_type / difficulty_reason

Selection is deterministic (score desc, then spec_id) with a diversity cap so a category is not
dominated by one CPV/year/answer-field. "Hard" = high evidence_count (stresses exhaustive
retrieval) / many predicates (stresses anchor resolution) / compositional (OOD) generalisation.
"Diagnosable" = verified, has a clean golden and constraints; `value_source` factoids are excluded
(that provenance field is intentionally unsupported at runtime, so a miss would be a design
boundary, not a planning signal).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "qa" / "generated"
EVAL = ROOT / "data" / "qa" / "eval"
BENCHMARK = GEN / "benchmark.jsonl"
ANSWER_SPECS = GEN / "answer_specs.jsonl"
HARD20_KEY = EVAL / "hard20_answer_key.jsonl"

EVIDENCE_ID_CAP = 50  # keep the file small; evidence_count stays exact

# operation_family -> eval category. WARN value-sanity rows are re-homed to 'boundary'.
FAMILY_CATEGORY = {
    "additive_sum": "sum",
    "conjunction": "conjunction",
    "filtered_count": "count",
    "cpv_slice": "cpv_count",
    "temporal_count": "temporal_count",
    "contract_factoid": "factoid",
}

# target size per eval category (sums to 100)
TARGETS = {
    "sum": 20,
    "boundary": 8,
    "conjunction": 18,
    "count": 16,
    "cpv_count": 8,
    "temporal_count": 12,
    "factoid": 18,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_answer_fields() -> dict[str, str]:
    fields: dict[str, str] = {}
    for row in read_jsonl(ANSWER_SPECS):
        spec = row.get("spec", {})
        fields[spec.get("spec_id")] = spec.get("answer_field", "")
    return fields


def _constraint_value(constraints: list[dict[str, Any]], field: str) -> Any:
    for c in constraints:
        if c.get("field") == field:
            return c.get("value")
    return None


def _visible_predicates(constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # coverage guards (supplier_count/buyer_count gte) and the internal additive guard are not
    # question-visible predicates.
    internal = {"supplier_count", "buyer_count", "value_is_additive"}
    return [c for c in constraints if c.get("field") not in internal]


def _category(row: dict[str, Any]) -> str:
    if str(row.get("value_sanity_status")) == "WARN" and row["operation_family"] == "additive_sum":
        return "boundary"
    return FAMILY_CATEGORY[row["operation_family"]]


def _hardness(row: dict[str, Any], category: str) -> float:
    import math

    compositional = 1.0 if row.get("generalization_class") == "compositional" else 0.0
    hard = 0.5 if row.get("difficulty") == "hard" else 0.0
    if category == "factoid":
        # more attributes to resolve to one contract = harder anchor
        return len(_visible_predicates(row.get("constraints", []))) + compositional
    if category == "boundary":
        return 10.0 + compositional  # always prefer the sanity-WARN cases
    evidence = int(row.get("evidence_count") or 0)
    return math.log10(evidence + 1) + compositional + hard


def _diversity_key(row: dict[str, Any], category: str) -> tuple:
    constraints = row.get("constraints", [])
    if category == "factoid":
        return (row.get("_answer_field", ""),)
    return (
        str(row.get("domain_slice")),
        str(_constraint_value(constraints, "tender_cpv_id")),
        str(_constraint_value(constraints, "release_year")),
    )


def _select(rows: list[dict[str, Any]], category: str, target: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: (-_hardness(r, category), r["spec_id"]))
    for cap in (1, 2, 3, 4, 999):  # relax diversity cap until the quota is filled
        chosen: list[dict[str, Any]] = []
        seen: Counter = Counter()
        for row in ranked:
            key = _diversity_key(row, category)
            if seen[key] >= cap:
                continue
            seen[key] += 1
            chosen.append(row)
            if len(chosen) == target:
                return chosen
        if len(chosen) == target:
            return chosen
    return chosen


def _difficulty_reason(row: dict[str, Any], category: str) -> str:
    n = int(row.get("evidence_count") or 0)
    dom = row.get("domain_slice")
    if category == "sum":
        return f"sum over value_amount ({n} contracts, {dom}); needs additive-value guard + exhaustive retrieval"
    if category == "boundary":
        return f"value-sanity {row.get('value_sanity_status')} sum over {n} contracts; tests the placeholder/dominant-contributor gate"
    if category == "conjunction":
        preds = len(_visible_predicates(row.get("constraints", [])))
        return (f"count over {preds} conjunctive predicates (year+category+CPV); golden carries hidden "
                f"supplier/buyer coverage guards (benchmark artifact) — tests faithful multi-predicate extraction")
    if category == "count":
        return f"filtered count over {n} contracts ({dom}); needs correct filter extraction + exhaustive retrieval"
    if category == "cpv_count":
        return f"CPV-slice count over {n} contracts; tests 8-digit CPV extraction + exhaustive retrieval"
    if category == "temporal_count":
        return f"temporal count over {n} contracts with has_award_signed_date; tests the temporal filter"
    # factoid
    preds = len(_visible_predicates(row.get("constraints", [])))
    return (f"select_unique {row.get('_answer_field')} over a {preds}-attribute anchor ({dom}); "
            f"needs multi-constraint resolution to exactly one contract")


def _answer_key_row(row: dict[str, Any], category: str) -> dict[str, Any]:
    return {
        "id": row["spec_id"],
        "category": category,
        "question": row["question"],
        "question_type": row.get("question_type", ""),
        "difficulty_reason": _difficulty_reason(row, category),
        "answer_operation": row.get("answer_operation", ""),
        "answer_field": row.get("_answer_field", ""),
        "answer_value_type": row.get("answer_value_type", ""),
        "oracle_answer": row.get("golden_answer"),
        "evidence_count": row.get("evidence_count"),
        "evidence_ids": list(row.get("evidence_ids", []))[:EVIDENCE_ID_CAP],
        "constraints": row.get("constraints", []),
        "gate_b_verified": bool((row.get("gate_b") or {}).get("verified")),
        "value_sanity_warn": str(row.get("value_sanity_status")) == "WARN",
        "difficulty": row.get("difficulty"),
        "generalization_class": row.get("generalization_class"),
        "hop_class": row.get("hop_class"),
        "domain_slice": row.get("domain_slice"),
        "logic_chain": row.get("logic_chain"),
        "expected_answerable": True,
    }


def main() -> None:
    answer_fields = load_answer_fields()
    exclude = {r["id"] for r in read_jsonl(HARD20_KEY)} if HARD20_KEY.exists() else set()

    benchmark = read_jsonl(BENCHMARK)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in benchmark:
        if row["spec_id"] in exclude:
            continue
        if not (row.get("gate_b") or {}).get("verified"):
            continue
        row["_answer_field"] = answer_fields.get(row["spec_id"], "")
        # value_source factoids are intentionally unsupported at runtime -> not diagnosable
        if row["operation_family"] == "contract_factoid" and row["_answer_field"] == "value_source":
            continue
        by_cat[_category(row)].append(row)

    selected: list[dict[str, Any]] = []
    for category, target in TARGETS.items():
        pool = by_cat.get(category, [])
        picked = _select(pool, category, target)
        if len(picked) < target:
            print(f"[warn] {category}: only {len(picked)}/{target} available")
        selected.extend(_answer_key_row(row, category) for row in picked)

    EVAL.mkdir(parents=True, exist_ok=True)
    key_path = EVAL / "hard100_answer_key.jsonl"
    q_path = EVAL / "hard100_questions_only.jsonl"
    with key_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with q_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps({k: row[k] for k in ("id", "question", "question_type", "difficulty_reason")},
                                    ensure_ascii=False) + "\n")

    print(f"wrote {len(selected)} questions")
    print("by category:", dict(Counter(r["category"] for r in selected)))
    print("by difficulty:", dict(Counter(r["difficulty"] for r in selected)))
    print("by generalization:", dict(Counter(r["generalization_class"] for r in selected)))
    print("by hop_class:", dict(Counter(r["hop_class"] for r in selected)))
    print("value_sanity WARN:", sum(1 for r in selected if r["value_sanity_warn"]))
    print(f"answer key : {key_path}")
    print(f"questions  : {q_path}")


if __name__ == "__main__":
    main()
