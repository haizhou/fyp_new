#!/usr/bin/env python3
"""Probe Step-1 question understanding only.

This script intentionally stops before Grok planning, grounding, execution, and reflector.
It is meant to answer one question: can the understanding model read the procurement question,
preserve explicit atoms, and describe a reverse derivation/procedure without answering?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.qa.benchmark.chat import ChatClient  # noqa: E402
from procurement_graph.reasoning.typed_planning import (  # noqa: E402
    intent_program_schema,
    question_intent_program_messages,
    question_understanding_messages,
)


DEFAULT_INPUT = Path("data/qa/cicada_merged_l1_l2_trainbalanced_v1/dev_smoke.jsonl")
NUM_RE = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?(?![A-Za-z])")
YEAR_RE = re.compile(r"\b20\d{2}\b")
CPV_RE = re.compile(r"\b\d{5,8}\b")
ROLE_WORDS = (
    "buyer",
    "supplier",
    "contracting authority",
    "award winner",
    "publisher",
    "awarded to",
    "awarded by",
    "published by",
    "issued by",
)
COMPLEX_BUCKETS = {"bridge_join", "comparison", "compare_two", "top_k", "min_max", "sum", "boolean"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalise_num(text: str) -> str:
    return text.replace(",", "")


def question_numbers(question: str) -> list[str]:
    seen: set[str] = set()
    nums: list[str] = []
    for match in NUM_RE.findall(question):
        value = normalise_num(match)
        if value not in seen:
            seen.add(value)
            nums.append(value)
    return nums


def question_years(question: str) -> list[str]:
    return sorted(set(YEAR_RE.findall(question)))


def question_cpvs(question: str) -> list[str]:
    # Treat long numeric atoms as CPV-like only when the question says CPV near them.
    if "cpv" not in question.casefold():
        return []
    return sorted({value for value in CPV_RE.findall(question) if len(value) >= 5})


def oracle_leaked(row: dict[str, Any], raw: str) -> bool:
    answer = row.get("oracle_answer")
    if answer is None or row.get("expected_status") != "answerable":
        return False
    answer_text = str(answer).strip()
    if len(answer_text) < 3:
        return False
    question = str(row.get("question") or "")
    return answer_text.casefold() in raw.casefold() and answer_text.casefold() not in question.casefold()


def ascii_sanitise_text(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u2192": "->",
        "\u2194": "<->",
        "\u2208": " in ",
        "\u2229": " intersection ",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u00a3": "GBP ",
    }
    cleaned = "".join(replacements.get(ch, ch) for ch in text)
    return cleaned.encode("ascii", "ignore").decode("ascii")


def is_complex(row: dict[str, Any]) -> bool:
    bucket = str(row.get("train_bucket") or row.get("question_type") or row.get("answer_operation") or "").casefold()
    if bucket in COMPLEX_BUCKETS:
        return True
    return str(row.get("level") or "") == "2"


def checks_for(row: dict[str, Any], raw: str) -> dict[str, Any]:
    question = str(row.get("question") or "")
    raw_cf = raw.casefold()
    nums = question_numbers(question)
    years = question_years(question)
    cpvs = question_cpvs(question)
    missing_numbers = [n for n in nums if n not in normalise_num(raw)]
    missing_years = [y for y in years if y not in raw]
    missing_cpvs = [c for c in cpvs if c not in normalise_num(raw)]
    has_final = any(term in raw_cf for term in ("answer type", "final answer", "answer required", "requires"))
    has_reverse = any(term in raw_cf for term in ("reverse tree", "reverse", "work backwards", "tree", "depends on", "need to know"))
    has_steps = "step" in raw_cf and any(
        term in raw_cf
        for term in ("resolve", "find", "count", "sum", "rank", "compare", "select", "abstain", "identify")
    )
    needs_targets = is_complex(row)
    has_targets = any(re.search(rf"\b{letter}\b", raw) for letter in ("A", "B", "C"))
    has_role = any(term in raw_cf for term in ROLE_WORDS)
    leaked = oracle_leaked(row, raw)
    non_ascii = sorted({ch for ch in raw if ord(ch) > 127})
    ok = (
        not missing_years
        and not missing_cpvs
        and not missing_numbers
        and not leaked
        and has_final
        and has_reverse
        and has_steps
        and (has_targets or not needs_targets)
    )
    return {
        "ok": ok,
        "missing_question_numbers": missing_numbers,
        "missing_years": missing_years,
        "missing_cpvs": missing_cpvs,
        "leaks_oracle_answer": leaked,
        "has_final_answer_required": has_final,
        "has_reverse_derivation": has_reverse,
        "has_operation_units": has_steps,
        "needs_intermediate_targets": needs_targets,
        "has_intermediate_targets": has_targets,
        "has_role_direction_language": has_role,
        "non_ascii_chars": non_ascii,
    }


def write_prompt_files(out_dir: Path, system: str, user: str) -> None:
    (out_dir / "prompt.system.txt").write_text(system, encoding="utf-8")
    (out_dir / "prompt.example.txt").write_text(user, encoding="utf-8")
    template = re.sub(r"Question:\n.*?\n\nTask:", "Question:\n{question}\n\nTask:", user, flags=re.S)
    (out_dir / "prompt.user.template.txt").write_text(template, encoding="utf-8")


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ids.add(str(json.loads(line).get("id")))
        except json.JSONDecodeError:
            continue
    return ids


def summarise(out_dir: Path, records: list[dict[str, Any]], total_requested: int) -> None:
    counters = Counter()
    bucket_totals: dict[str, Counter] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    for rec in records:
        checks = rec.get("checks") or {}
        bucket = str(rec.get("train_bucket") or rec.get("question_type") or "unknown")
        counters["total"] += 1
        counters["ok" if checks.get("ok") else "bad"] += 1
        bucket_totals[bucket]["total"] += 1
        bucket_totals[bucket]["ok" if checks.get("ok") else "bad"] += 1
        for key, value in checks.items():
            if key == "ok":
                continue
            if value:
                counters[key] += 1
        if not checks.get("ok") and len(examples) < 10:
            examples.append({
                "id": rec.get("id"),
                "bucket": bucket,
                "question": rec.get("question"),
                "checks": checks,
            })
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_requested": total_requested,
        "total_completed": counters["total"],
        "ok": counters["ok"],
        "bad": counters["bad"],
        "check_counts": dict(counters),
        "by_bucket": {bucket: dict(counts) for bucket, counts in sorted(bucket_totals.items())},
        "bad_examples": examples,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Step-1 Understanding Probe",
        "",
        f"- requested: {total_requested}",
        f"- completed: {counters['total']}",
        f"- ok: {counters['ok']}",
        f"- bad: {counters['bad']}",
        "",
        "## By Bucket",
    ]
    for bucket, counts in sorted(bucket_totals.items()):
        lines.append(f"- {bucket}: ok={counts['ok']} bad={counts['bad']} total={counts['total']}")
    lines.extend(["", "## First Bad Examples"])
    for ex in examples:
        lines.append(f"- {ex['id']} [{ex['bucket']}]: {ex['question']}")
        bad_keys = [k for k, v in (ex["checks"] or {}).items() if k != "ok" and v]
        lines.append(f"  checks: {', '.join(bad_keys) if bad_keys else ex['checks']}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--format", choices=["intent", "text"], default="intent",
                        help="intent = strict typed program; text = old prose scaffold")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "understanding.outputs.jsonl"
    prompt_path = args.out_dir / "understanding.prompts.jsonl"
    failure_path = args.out_dir / "understanding.failures.jsonl"
    if not args.resume:
        for path in (output_path, prompt_path, failure_path):
            if path.exists():
                path.unlink()
    done = existing_ids(output_path) if args.resume else set()
    chat = ChatClient.from_env(temperature=args.temperature)

    written_records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or idx)
        if row_id in done:
            continue
        question = str(row.get("question") or "")
        if args.format == "intent":
            system, user = question_intent_program_messages(question)
        else:
            system, user = question_understanding_messages(question)
        if idx == 1:
            write_prompt_files(args.out_dir, system, user)
        append_jsonl(prompt_path, {
            "id": row_id,
            "question": question,
            "system": system,
            "user": user,
        })
        try:
            parsed = None
            if args.format == "intent" and hasattr(chat, "complete_schema"):
                result = chat.complete_schema(model=args.model, system=system, user=user,
                                              schema=intent_program_schema())
                parsed = result.parsed
                raw = json.dumps(parsed, ensure_ascii=False)
            else:
                result = chat.complete_text(model=args.model, system=system, user=user)
                raw = result.raw_text
            ascii_raw = ascii_sanitise_text(raw)
            checks = checks_for(row, raw)
            record = {
                "id": row_id,
                "source": row.get("source"),
                "level": row.get("level"),
                "train_bucket": row.get("train_bucket"),
                "question_type": row.get("question_type"),
                "expected_status": row.get("expected_status"),
                "answer_type": row.get("answer_type"),
                "question": question,
                "raw_understanding": raw,
                "ascii_understanding": ascii_raw,
                "intent_program": parsed,
                "checks": checks,
                "usage": result.usage,
                "model": result.model,
                "attempts": result.attempts,
            }
            append_jsonl(output_path, record)
            written_records.append(record)
        except Exception as exc:  # pragma: no cover - live API path
            append_jsonl(failure_path, {
                "id": row_id,
                "question": question,
                "error": repr(exc),
            })
            print(f"[step1] failed {idx}/{len(rows)} {row_id}: {exc}", flush=True)
            continue
        if args.progress_every > 0 and idx % args.progress_every == 0:
            ok = sum(1 for r in written_records if (r.get("checks") or {}).get("ok"))
            print(f"[step1] {idx}/{len(rows)} completed, new_ok={ok}/{len(written_records)}", flush=True)

    all_records = []
    if output_path.exists():
        all_records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summarise(args.out_dir, all_records, len(rows))
    print(f"[step1] wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
