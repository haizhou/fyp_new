"""Decoupled, concurrent Stage 2: generation pass, verification pass, and assemble.

Instead of the per-spec ``generate -> verify`` loop, Stage 2 is split so the two models run
independently, each under its own rate limit:

- ``generate_pass``  (nano): answer_specs -> questions.jsonl   (high concurrency)
- ``verify_pass``    (grok): questions    -> verifications.jsonl (capped at ~50 RPM)
- ``assemble``               questions + verifications -> benchmark.jsonl + regen_queue.jsonl

Rejected questions go to a regen queue carrying the verifier's reason; re-running
``generate_pass`` on that queue feeds the reason back to the generator so it can fix the
question (needed because temperature=0 would otherwise reproduce the same rejection).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .concurrency import run_concurrent
from .gate_b import verification_mode
from .kg_interface import ParquetKGQueryBackend
from .serialization import json_default, spec_to_dict
from .stage2 import (
    _cpv_description_map,
    _cpv_for_spec,
    _evidence_sample,
    _factoid_anchor,
    _should_verify,
    _spec_from_dict,
)


def generate_pass(
    *,
    input_path: Path,
    kg_dir: Path,
    out_path: Path,
    generator: Any,
    workers: int = 24,
    rpm: float = 2400,
    resume: bool = True,
    with_anchor: bool = True,
) -> dict[str, int]:
    backend = ParquetKGQueryBackend.from_directory(kg_dir, include_evidence=False) if with_anchor else None
    cpv_descriptions = _cpv_description_map(backend) if backend is not None else {}
    items = _load_generation_inputs(input_path)
    done = _done_ids(out_path) if resume else set()
    todo = [item for item in items if item[0].spec_id not in done]
    counters: Counter[str] = Counter()

    def fn(item: Any) -> dict[str, Any]:
        spec, golden, feedback = item
        qtype = str(spec.metadata.get("question_type", ""))
        anchor = None
        if qtype == "factoid" and backend is not None:
            anchor = _factoid_anchor(backend, spec, cpv_descriptions)
            if anchor is None:
                return _question_record(spec, golden, status="rejected_no_anchor")
        gen = generator.generate(
            spec, cpv_description=_cpv_for_spec(spec, cpv_descriptions), anchor=anchor, feedback=feedback
        )
        model = getattr(generator, "model", "")
        if not gen.ok:
            return _question_record(spec, golden, status="rejected_generation", error=gen.error,
                                    generation=gen.provenance() | {"model": model})
        return _question_record(
            spec, golden, status="generated", question=gen.question, anchor=anchor,
            feedback_used=feedback, generation=gen.provenance() | {"model": model},
        )

    with out_path.open("a" if (resume and done) else "w", encoding="utf-8") as handle:
        def on_result(item: Any, result: Any) -> None:
            record = result if isinstance(result, dict) else _error_question_record(item, result)
            handle.write(json.dumps(record, default=json_default) + "\n")
            handle.flush()
            counters[record["status"]] += 1

        run_concurrent(todo, fn, workers=workers, rpm=rpm, on_result=on_result)
    counters["skipped_resume"] = len(items) - len(todo)
    return dict(counters)


def verify_pass(
    *,
    questions_path: Path,
    kg_dir: Path,
    out_path: Path,
    verifier: Any,
    evidence_cap: int = 40,
    factoid_sample_rate: float = 0.3,
    workers: int = 6,
    rpm: float = 45,
    resume: bool = True,
    with_evidence: bool = True,
) -> dict[str, int]:
    backend = ParquetKGQueryBackend.from_directory(kg_dir, include_evidence=False) if with_evidence else None
    generated = [q for q in _read_jsonl(questions_path) if q.get("status") == "generated"]
    done = _done_ids(out_path) if resume else set()
    pending = [q for q in generated if q["spec_id"] not in done]
    sampled = [q for q in pending if _should_verify(q.get("question_type", ""), factoid_sample_rate, q["spec_id"])]
    skipped = [q for q in pending if q not in sampled]
    counters: Counter[str] = Counter()

    def fn(q: dict[str, Any]) -> dict[str, Any]:
        spec = _spec_from_dict(q["spec"])
        golden = q.get("golden_answer")
        mode = verification_mode(spec, len(spec.sampled_evidence_ids), evidence_cap)
        if mode == "recompute":
            evidence = _evidence_sample(backend, spec, evidence_cap) if backend is not None else []
            vout = verifier.verify_recompute(spec, q["question"], evidence, golden)
        else:
            vout = verifier.verify_faithfulness(spec, q["question"])
        return {
            "spec_id": q["spec_id"],
            "verified": vout.verified,
            "gate_b": vout.provenance(model=getattr(verifier, "model", ""), sampled=True),
        }

    with out_path.open("a" if (resume and done) else "w", encoding="utf-8") as handle:
        for q in skipped:
            record = {
                "spec_id": q["spec_id"],
                "verified": True,
                "gate_b": {"model": getattr(verifier, "model", ""), "mode": "skipped_sampling",
                           "sampled": False, "verified": True, "reason": "factoid sampled out"},
            }
            handle.write(json.dumps(record, default=json_default) + "\n")
            counters["skipped_sampling"] += 1

        def on_result(item: Any, result: Any) -> None:
            record = result if isinstance(result, dict) else {
                "spec_id": item["spec_id"], "verified": False,
                "gate_b": {"mode": "error", "verified": False, "reason": f"exception: {result[1]}"},
            }
            handle.write(json.dumps(record, default=json_default) + "\n")
            handle.flush()
            counters["verified" if record["verified"] else "failed"] += 1

        run_concurrent(sampled, fn, workers=workers, rpm=rpm, on_result=on_result)
    counters["skipped_resume"] = len(generated) - len(pending)
    return dict(counters)


def assemble(
    *,
    questions_paths: list[Path],
    verifications_paths: list[Path],
    benchmark_path: Path,
    rejected_path: Path,
    regen_path: Path,
) -> dict[str, int]:
    # Last file wins per spec_id, so later regen rounds override earlier attempts.
    questions: dict[str, Any] = {}
    for path in questions_paths:
        for q in _read_jsonl(path):
            questions[q["spec_id"]] = q
    verifications: dict[str, Any] = {}
    for path in verifications_paths:
        for v in _read_jsonl(path):
            verifications[v["spec_id"]] = v
    counters: Counter[str] = Counter()

    with benchmark_path.open("w", encoding="utf-8") as bench, rejected_path.open(
        "w", encoding="utf-8"
    ) as reject, regen_path.open("w", encoding="utf-8") as regen:
        for spec_id, q in questions.items():
            if q.get("status") != "generated":
                reject.write(json.dumps({"spec_id": spec_id, "stage": "stage2_rejected",
                                         "rejected_gate": q.get("status"), "reason": q.get("error", ""),
                                         "question_type": q.get("question_type", "")}, default=json_default) + "\n")
                counters[q.get("status", "rejected")] += 1
                continue
            verification = verifications.get(spec_id)
            if verification is None:
                counters["pending_verification"] += 1
                continue
            if verification.get("verified"):
                bench.write(json.dumps(_benchmark_record(q, verification), default=json_default) + "\n")
                counters["accepted"] += 1
            else:
                reason = verification.get("gate_b", {}).get("reason", "verification failed")
                regen.write(json.dumps({"spec": q["spec"], "golden_answer": q.get("golden_answer"),
                                        "feedback": reason}, default=json_default) + "\n")
                counters["regen_queued"] += 1
    return dict(counters)


def _benchmark_record(question: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    spec = question["spec"]
    metadata = spec.get("metadata", {}) or {}
    return {
        "spec_id": question["spec_id"],
        "question": question["question"],
        "golden_answer": question.get("golden_answer"),
        "answer_value_type": spec.get("answer_value_type", ""),
        "answer_operation": spec.get("answer_operation", ""),
        "question_type": metadata.get("question_type", ""),
        "operation_family": metadata.get("operation_family", ""),
        "domain_slice": metadata.get("domain_slice", ""),
        "difficulty": metadata.get("difficulty", ""),
        "hop_class": metadata.get("hop_class", ""),
        "generalization_class": metadata.get("generalization_class", ""),
        "value_sanity_status": metadata.get("value_sanity_status", "PASS"),
        "evidence_ids": spec.get("sampled_evidence_ids", []),
        "evidence_count": len(spec.get("sampled_evidence_ids", []) or []),
        "constraints": spec.get("constraints", []),
        "logic_chain": spec.get("logic_chain", []),
        "generation": question.get("generation", {}),
        "gate_b": verification.get("gate_b", {}),
        "stage": "stage2_passed",
    }


def _question_record(spec: Any, golden: Any, *, status: str, question: str = "", anchor: Any = None,
                     error: str = "", feedback_used: str = "", generation: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "spec_id": spec.spec_id,
        "status": status,
        "question": question,
        "anchor": anchor,
        "error": error,
        "feedback_used": feedback_used,
        "question_type": str(spec.metadata.get("question_type", "")),
        "spec": spec_to_dict(spec),
        "golden_answer": golden,
        "generation": generation or {},
    }


def _error_question_record(item: Any, result: Any) -> dict[str, Any]:
    spec, golden, _feedback = item
    return _question_record(spec, golden, status="rejected_generation", error=f"exception: {result[1]}")


def _load_generation_inputs(path: Path) -> list[tuple[Any, Any, str]]:
    out: list[tuple[Any, Any, str]] = []
    for record in _read_jsonl(path):
        spec = _spec_from_dict(record["spec"])
        spec.metadata.setdefault("value_sanity_status", record.get("value_sanity_status", "PASS"))
        out.append((spec, record.get("golden_answer"), str(record.get("feedback", ""))))
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _done_ids(path: Path) -> set[str]:
    return {row["spec_id"] for row in _read_jsonl(path) if "spec_id" in row}


__all__ = ["generate_pass", "verify_pass", "assemble"]
