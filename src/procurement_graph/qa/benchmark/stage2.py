"""Stage 2 orchestration: question generation + Gate B over Stage 1 answer specs.

Reads `answer_specs.jsonl` (Stage 1 accepted specs + golden answers), generates one
natural-language question per spec, runs independent Gate B verification, and writes a
benchmark plus rejection/summary logs. Supports `--limit`, resume, and a fully offline
dry-run (deterministic stand-ins) so the plumbing can be debugged with no API calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from .kg_interface import ParquetKGQueryBackend
from .models import AnswerSpec, Constraint
from .gate_b import verification_mode
from .serialization import json_default


def build_stage2(
    *,
    specs_path: Path,
    kg_dir: Path,
    output_dir: Path,
    report_dir: Path,
    generator: Any,
    verifier: Any,
    evidence_cap: int = 40,
    factoid_sample_rate: float = 0.3,
    limit: int | None = None,
    sample_per_type: int | None = None,
    with_evidence: bool = True,
    resume: bool = True,
    out_tag: str = "",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    specs = load_accepted_specs(specs_path)
    if sample_per_type is not None:
        specs = _stratified_sample(specs, sample_per_type)
    if limit is not None:
        specs = specs[:limit]

    backend = None
    cpv_descriptions: dict[str, str] = {}
    if with_evidence:
        backend = ParquetKGQueryBackend.from_directory(kg_dir, include_evidence=False)
        cpv_descriptions = _cpv_description_map(backend)

    suffix = f".{out_tag}" if out_tag else ""
    benchmark_path = output_dir / f"benchmark{suffix}.jsonl"
    rejected_path = output_dir / f"rejected_stage2{suffix}.jsonl"
    summary_path = report_dir / f"stage2_summary{suffix}.json"

    done = _resume_ids([benchmark_path, rejected_path]) if resume else set()
    mode = "a" if (resume and done) else "w"

    counters: Counter[str] = Counter()
    gate_b_modes: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()

    with benchmark_path.open(mode, encoding="utf-8") as bench_file, rejected_path.open(
        mode, encoding="utf-8"
    ) as reject_file:
        for spec, golden in specs:
            if spec.spec_id in done:
                counters["skipped_resume"] += 1
                continue
            counters["attempted"] += 1
            qtype = str(spec.metadata.get("question_type", "unknown"))

            cpv_desc = _cpv_for_spec(spec, cpv_descriptions)
            anchor = None
            if qtype == "factoid" and backend is not None:
                anchor = _factoid_anchor(backend, spec, cpv_descriptions)
                if anchor:
                    counters["factoid_natural_anchor"] += 1
                else:
                    counters["factoid_no_natural_anchor"] += 1
                    counters["rejected_generation"] += 1
                    reject_file.write(
                        json.dumps(
                            _rejected_record(
                                spec,
                                qtype,
                                stage_gate="generation",
                                reason=(
                                    "factoid_no_natural_anchor: internal contract ids are not allowed "
                                    "in benchmark questions"
                                ),
                                question="",
                            ),
                            default=json_default,
                        )
                        + "\n"
                    )
                    continue
            gen = generator.generate(spec, cpv_description=cpv_desc, anchor=anchor)
            _accumulate_usage(usage_totals, "gen", gen.usage)
            if not gen.ok:
                counters["rejected_generation"] += 1
                reject_file.write(
                    json.dumps(
                        _rejected_record(spec, qtype, stage_gate="generation", reason=gen.error, question=""),
                        default=json_default,
                    )
                    + "\n"
                )
                continue
            id_leak = _question_id_leak_reason(gen.question)
            if id_leak:
                counters["rejected_generation"] += 1
                counters["rejected_id_leak"] += 1
                reject_file.write(
                    json.dumps(
                        _rejected_record(
                            spec,
                            qtype,
                            stage_gate="generation",
                            reason=id_leak,
                            question=gen.question,
                        ),
                        default=json_default,
                    )
                    + "\n"
                )
                continue

            evidence_count = len(spec.sampled_evidence_ids)
            vmode = verification_mode(spec, evidence_count, evidence_cap)
            sampled = _should_verify(qtype, factoid_sample_rate, spec.spec_id)
            gate_b_modes[vmode if sampled else "skipped_sampling"] += 1

            if not sampled:
                verified = True
                vout = None
            elif vmode == "recompute":
                evidence = _evidence_sample(backend, spec, evidence_cap) if backend else []
                vout = verifier.verify_recompute(spec, gen.question, evidence, golden)
                verified = vout.verified
                _accumulate_usage(usage_totals, "verify", vout.usage)
            else:
                vout = verifier.verify_faithfulness(spec, gen.question)
                verified = vout.verified
                _accumulate_usage(usage_totals, "verify", vout.usage)

            if not verified:
                counters["rejected_gate_b"] += 1
                reject_file.write(
                    json.dumps(
                        _rejected_record(
                            spec,
                            qtype,
                            stage_gate="gate_b",
                            reason=vout.reason if vout else "verification failed",
                            question=gen.question,
                            gate_b=vout.provenance(model=getattr(verifier, "model", ""), sampled=sampled)
                            if vout
                            else None,
                        ),
                        default=json_default,
                    )
                    + "\n"
                )
                continue

            counters["accepted"] += 1
            accepted_type = qtype
            counters[f"accepted::{accepted_type}"] += 1
            counters[f"value_sanity::{spec.metadata.get('value_sanity_status', 'PASS')}"] += 1
            record = _benchmark_record(
                spec,
                golden,
                question=gen.question,
                generation=gen.provenance() | {"model": getattr(generator, "model", "")},
                gate_b=(
                    vout.provenance(model=getattr(verifier, "model", ""), sampled=True)
                    if vout
                    else {"model": getattr(verifier, "model", ""), "mode": "skipped_sampling", "sampled": False, "verified": True}
                ),
            )
            bench_file.write(json.dumps(record, default=json_default) + "\n")

    summary = {
        "specs_path": str(specs_path),
        "attempted": counters.get("attempted", 0),
        "accepted": counters.get("accepted", 0),
        "rejected_generation": counters.get("rejected_generation", 0),
        "rejected_gate_b": counters.get("rejected_gate_b", 0),
        "skipped_resume": counters.get("skipped_resume", 0),
        "accepted_by_type": {
            key.split("::", 1)[1]: value for key, value in sorted(counters.items()) if key.startswith("accepted::")
        },
        "accepted_by_value_sanity": {
            key.split("::", 1)[1]: value for key, value in sorted(counters.items()) if key.startswith("value_sanity::")
        },
        "gate_b_modes": dict(sorted(gate_b_modes.items())),
        "factoid_anchor": {
            "natural": counters.get("factoid_natural_anchor", 0),
            "fallback_id": 0,
            "rejected_no_natural_anchor": counters.get("factoid_no_natural_anchor", 0),
        },
        "generation_quality": {
            "rejected_id_leak": counters.get("rejected_id_leak", 0),
        },
        "token_usage": dict(sorted(usage_totals.items())),
        "config": {
            "evidence_cap": evidence_cap,
            "factoid_sample_rate": factoid_sample_rate,
            "with_evidence": with_evidence,
            "generator_model": getattr(generator, "model", ""),
            "prompt_variant": getattr(generator, "prompt_variant", "current"),
            "verifier_model": getattr(verifier, "model", ""),
        },
        "outputs": {
            "benchmark": str(benchmark_path),
            "rejected": str(rejected_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    return summary


def load_accepted_specs(path: Path) -> list[tuple[AnswerSpec, Any]]:
    out: list[tuple[AnswerSpec, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            spec = _spec_from_dict(record["spec"])
            # Carry the Stage 1 value-sanity verdict (PASS/WARN) so the final benchmark
            # stays filterable without dropping anything here.
            spec.metadata["value_sanity_status"] = record.get("value_sanity_status", "PASS")
            out.append((spec, record.get("golden_answer")))
    return out


def _stratified_sample(specs: list[tuple[AnswerSpec, Any]], per_type: int) -> list[tuple[AnswerSpec, Any]]:
    """First `per_type` specs of each question_type, preserving order (for diverse pilots)."""
    seen: Counter[str] = Counter()
    out: list[tuple[AnswerSpec, Any]] = []
    for spec, golden in specs:
        qtype = str(spec.metadata.get("question_type", ""))
        if seen[qtype] < per_type:
            seen[qtype] += 1
            out.append((spec, golden))
    return out


def _spec_from_dict(data: dict[str, Any]) -> AnswerSpec:
    constraints = tuple(Constraint(c["field"], c["op"], c.get("value")) for c in data.get("constraints", []))
    return AnswerSpec(
        spec_id=data["spec_id"],
        constraints=constraints,
        answer_operation=data["answer_operation"],
        answer_field=data["answer_field"],
        answer_value_type=data["answer_value_type"],
        dedupe_key=data.get("dedupe_key", ""),
        logic_chain=tuple(data.get("logic_chain", []) or []),
        sampled_evidence_ids=tuple(data.get("sampled_evidence_ids", []) or []),
        metadata=dict(data.get("metadata", {}) or {}),
    )


def _resume_ids(paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["spec_id"])
                except Exception:
                    continue
    return done


def _should_verify(qtype: str, factoid_rate: float, spec_id: str) -> bool:
    if qtype != "factoid":
        return True
    bucket = int(hashlib.md5(spec_id.encode("utf-8")).hexdigest(), 16) % 1000
    return bucket < factoid_rate * 1000


_FORBIDDEN_QUESTION_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcontract\s*:", re.IGNORECASE),
    re.compile(r"\bocds-[a-z0-9-]+", re.IGNORECASE),
    re.compile(r"\b(?:contract_node_id|canonical_id|ocid|release_id|award_id|tender_id)\b", re.IGNORECASE),
    re.compile(r"\b(?:GB-FTS|GB-COH|GB-CHC|GB-SC|GB-NHS|GOV-[A-Z0-9-]+)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:unique identifier|internal id|contract id|reference number|registration number|"
        r"company number|charity number|nhs ods code)\s+[A-Z0-9:/._-]+",
        re.IGNORECASE,
    ),
)


def _question_id_leak_reason(question: str) -> str:
    """Reject natural-language questions that expose identifiers other than CPV codes.

    CPV codes are allowed because they are a public procurement classification used as
    a semantic category. Internal KG ids, OCDS ids, canonical ids, registry ids, and raw
    release/award/tender ids are not acceptable question anchors for the benchmark.
    """
    for pattern in _FORBIDDEN_QUESTION_ID_PATTERNS:
        match = pattern.search(question)
        if match:
            return f"question_contains_forbidden_identifier: {match.group(0)}"
    return ""


def _cpv_description_map(backend: ParquetKGQueryBackend) -> dict[str, str]:
    df = backend.records_df
    if "tender_cpv_id" not in df.columns or "tender_cpv_description" not in df.columns:
        return {}
    pairs = df[["tender_cpv_id", "tender_cpv_description"]].dropna().astype(str)
    return dict(zip(pairs["tender_cpv_id"], pairs["tender_cpv_description"]))


def _cpv_for_spec(spec: AnswerSpec, cpv_descriptions: dict[str, str]) -> str:
    for constraint in spec.constraints:
        if constraint.field == "tender_cpv_id" and constraint.op == "eq":
            return cpv_descriptions.get(str(constraint.value), "")
    return ""


# Natural fields shown to the recompute verifier (factoid / select_unique only). Excludes
# contract_node_id and every other internal id by construction, while giving enough context to
# confirm the question's anchor and read the asked field.
_FACTOID_EVIDENCE_FIELDS = (
    "buyer_name",
    "supplier_name",
    "tender_cpv_id",
    "tender_cpv_description",
    "release_year",
    "tender_category",
    "value_source",
    "award_date_signed",
)

_FACTOID_ANCHOR_FIELDS = ("buyer_name", "supplier_name", "tender_cpv_id", "release_year", "tender_category")
_ANCHOR_LABEL = {
    "buyer_name": "buyer",
    "supplier_name": "supplier",
    "tender_cpv_id": "cpv_code",
    "release_year": "year",
    "tender_category": "category",
    "award_date_signed": "signed_date",
}


def _evidence_sample(backend: ParquetKGQueryBackend, spec: AnswerSpec, cap: int) -> list[dict[str, Any]]:
    rows = backend.query(spec.constraints)
    sample: list[dict[str, Any]] = []
    for row in rows[:cap]:
        projected: dict[str, Any] = {}
        for field in _FACTOID_EVIDENCE_FIELDS:
            if field in row:
                value = _scalar(row.get(field))
                if value not in (None, ""):
                    projected[field] = value
        sample.append(projected)
    return sample


def _factoid_anchor(
    backend: ParquetKGQueryBackend, spec: AnswerSpec, cpv_descriptions: dict[str, str]
) -> dict[str, Any] | None:
    """Natural anchor for a factoid: the contract's buyer/supplier/CPV/year/category (minus the
    asked field). Returned only when the contract has a single buyer and single supplier (so the
    named anchor is unambiguous) AND the anchor uniquely determines the asked field. Otherwise
    None, and Stage 2 rejects the factoid rather than exposing an internal contract id."""
    rows = backend.query(spec.constraints)
    if len(rows) != 1:
        return None
    row = rows[0]
    if _as_int(row.get("buyer_count")) != 1 or _as_int(row.get("supplier_count")) != 1:
        return None
    answer_field = spec.answer_field
    base = [field for field in _FACTOID_ANCHOR_FIELDS if field != answer_field]
    # Prefer the clean anchor (no signed date); add the signed date as a tie-breaker only when
    # the base anchor does not uniquely determine the asked field (common for buyer/supplier).
    candidates = [base]
    if answer_field != "award_date_signed":
        candidates.append(base + ["award_date_signed"])
    for fields in candidates:
        anchor = _try_anchor(backend, row, fields, answer_field, cpv_descriptions)
        if anchor is not None:
            return anchor
    return None


def _try_anchor(
    backend: ParquetKGQueryBackend,
    row: dict[str, Any],
    fields: list[str],
    answer_field: str,
    cpv_descriptions: dict[str, str],
) -> dict[str, Any] | None:
    constraints: list[Constraint] = []
    anchor: dict[str, Any] = {}
    for field in fields:
        value = _scalar(row.get(field))
        if value in (None, "", "nan"):
            return None
        constraints.append(Constraint(field, "eq", value))
        anchor[_ANCHOR_LABEL[field]] = _date_part(value) if field == "award_date_signed" else value
    matches = backend.query(tuple(constraints))
    if len({_scalar(match.get(answer_field)) for match in matches}) != 1:
        return None
    if "cpv_code" in anchor:
        anchor["cpv_description"] = cpv_descriptions.get(str(anchor["cpv_code"]), "")
    return anchor


def _date_part(value: Any) -> str:
    text = str(value)
    return text[:10] if len(text) >= 10 and text[4:5] == "-" else text


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _benchmark_record(
    spec: AnswerSpec,
    golden: Any,
    *,
    question: str,
    generation: dict[str, Any],
    gate_b: dict[str, Any],
) -> dict[str, Any]:
    metadata = spec.metadata or {}
    return {
        "spec_id": spec.spec_id,
        "question": question,
        "golden_answer": golden,
        "answer_value_type": spec.answer_value_type,
        "answer_operation": spec.answer_operation,
        "question_type": metadata.get("question_type", ""),
        "difficulty": metadata.get("difficulty", ""),
        "hop_class": metadata.get("hop_class", ""),
        "value_sanity_status": metadata.get("value_sanity_status", "PASS"),
        "evidence_ids": list(spec.sampled_evidence_ids),
        "evidence_count": len(spec.sampled_evidence_ids),
        "constraints": [{"field": c.field, "op": c.op, "value": c.value} for c in spec.constraints],
        "logic_chain": list(spec.logic_chain),
        "generation": generation,
        "gate_b": gate_b,
        "stage": "stage2_passed",
    }


def _rejected_record(
    spec: AnswerSpec,
    qtype: str,
    *,
    stage_gate: str,
    reason: str,
    question: str,
    gate_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "spec_id": spec.spec_id,
        "question_type": qtype,
        "stage": "stage2_rejected",
        "rejected_gate": stage_gate,
        "reason": reason,
        "question": question,
    }
    if gate_b is not None:
        record["gate_b"] = gate_b
    return record


def _accumulate_usage(totals: Counter[str], prefix: str, usage: dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key) if isinstance(usage, dict) else None
        if isinstance(value, (int, float)):
            totals[f"{prefix}_{key}"] += int(value)


def _scalar(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple)):
        return [_scalar(item) for item in value]
    return value


__all__ = ["build_stage2", "load_accepted_specs"]
