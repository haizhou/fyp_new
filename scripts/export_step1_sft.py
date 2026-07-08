#!/usr/bin/env python3
"""Export Step-1 (understanding) distillation data: go fully local, drop nano.

Same design invariants as export_llamafactory.py (the Step-2 exporter):
- TRAIN DISTRIBUTION == INFERENCE DISTRIBUTION: user messages rendered by the RUNTIME
  builder `question_intent_program_messages` (with live `retrieve_schema_context`), never a
  reimplementation. Serving sends byte-identical prompts.
- Targets: compact JSON of the stored briefing (traces carry the PARSED intent program —
  100% intent_program-shaped, verified 2026-07-06).
- Partial-verifiability filter: keep a question's briefing only if some harvest run reached a
  GOOD outcome with it (oracle_match=True: correct answer or correct abstention). The briefing
  is upstream of every outcome, so acceptance = the whole pipeline succeeded under it.
- Multiple harvests (teacher + RSFT rounds) may cover the same question: keep one row per
  question id, preferring teacher-run briefings (they are what Step-2 students trained on).

Outputs: cicada_step1_sft(.val).json + dataset_info.json + export_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.reasoning.typed_planning import (  # noqa: E402
    _is_intent_program, question_intent_program_messages,
)

COMPACT = (",", ":")


def read_traces(path: Path) -> dict[str, dict[str, Any]]:
    """id -> best trace row (oracle_match preferred) from one harvest dir."""
    out: dict[str, dict[str, Any]] = {}
    f = path / "traces.jsonl"
    if not f.exists():
        return out
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        qid = str(t.get("id"))
        if qid not in out or (t.get("oracle_match") and not out[qid].get("oracle_match")):
            out[qid] = t
    return out


def is_val(sample_id: str, val_frac: float) -> bool:
    return (int(hashlib.sha1(str(sample_id).encode()).hexdigest(), 16) % 1000) < val_frac * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-dirs", nargs="+", type=Path,
                    default=[Path("data/qa/teacher_full_v1"),
                             Path("data/qa/rsft_qwen_r1"), Path("data/qa/rsft_llama_r1")],
                    help="harvest dirs, PRIORITY ORDER: first dir wins on duplicate ids")
    ap.add_argument("--qa", type=Path, default=Path("data/qa/cicada_core_v4/train.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/training/llamafactory_step1_v1"))
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()

    questions = {str(r["id"]): str(r["question"])
                 for r in (json.loads(l) for l in args.qa.read_text(encoding="utf-8").splitlines() if l.strip())}

    # merge harvests, first-dir priority
    merged: dict[str, dict[str, Any]] = {}
    per_source: dict[str, int] = {}
    for d in args.trace_dirs:
        rows = read_traces(d)
        fresh = 0
        for qid, t in rows.items():
            if not t.get("oracle_match"):
                continue  # partial-verifiability gate
            b = t.get("briefing")
            if not isinstance(b, dict) or not _is_intent_program(b):
                continue
            if qid not in merged:
                merged[qid] = t
                fresh += 1
        per_source[str(d)] = fresh
    print(f"[step1] eligible questions: {len(merged)} (per source new: {per_source})")

    train, val, skipped = [], [], 0
    for qid, t in sorted(merged.items()):
        q = questions.get(qid)
        if not q:
            skipped += 1
            continue
        system, user = question_intent_program_messages(q)  # live schema_context, runtime render
        sample = {"conversations": [{"from": "human", "value": user},
                                    {"from": "gpt", "value": json.dumps(t["briefing"], ensure_ascii=False,
                                                                        separators=COMPACT)}],
                  "system": system}
        (val if is_val(qid, args.val_frac) else train).append(sample)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cicada_step1_sft.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "cicada_step1_sft_val.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=1), encoding="utf-8")
    info = {
        "cicada_step1_sft": {"file_name": "cicada_step1_sft.json", "formatting": "sharegpt",
                             "columns": {"messages": "conversations", "system": "system"}},
        "cicada_step1_sft_val": {"file_name": "cicada_step1_sft_val.json", "formatting": "sharegpt",
                                 "columns": {"messages": "conversations", "system": "system"}},
    }
    (args.out_dir / "dataset_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    report = {"step1_sft": {"train": len(train), "val": len(val), "skipped_no_question": skipped,
                            "sources": per_source}}
    (args.out_dir / "export_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
