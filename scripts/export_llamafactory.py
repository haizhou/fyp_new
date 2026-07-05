#!/usr/bin/env python3
"""Export teacher pools to LLaMA-Factory datasets (SFT + DPO) for local 8B students.

Design invariants:
- TRAIN DISTRIBUTION == INFERENCE DISTRIBUTION: the user message is rendered by the *runtime*
  prompt builders (`typed_plan_messages` / `typed_replan_messages`, lean variant) — never a
  reimplementation. At serving time the student receives byte-identical prompts via the same
  functions (vLLM OpenAI endpoint plugged into ChatClient).
- One serialization convention for targets: compact JSON (`separators=(',',':')`,
  ensure_ascii=False) of the compiler-NORMALISED graph plan.
- Family caps at emission (anti shape-memorisation): at most --family-cap samples per
  template family in the plan-SFT pool; overflow recorded, not silently dropped.
- A small validation slice (--val-frac) is split off deterministically by id hash.

Outputs under --out-dir:
  cicada_plan_sft.json      sharegpt-format: planning samples (verified + abstain)
  cicada_repair_sft.json    sharegpt-format: repair samples (feedback -> repaired plan)
  cicada_dpo.json           LLaMA-Factory DPO format (prompt + chosen/rejected)
  *_val.json                validation slices
  dataset_info.json         ready-to-merge LLaMA-Factory dataset registry entries
  export_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from procurement_graph.reasoning.typed_planning import (  # noqa: E402
    typed_plan_messages, typed_replan_messages,
)

TARGET_SEP = (",", ":")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=TARGET_SEP)


def is_val(sample_id: str, val_frac: float) -> bool:
    return (int(hashlib.sha1(str(sample_id).encode()).hexdigest(), 16) % 1000) < val_frac * 1000


def plan_sample(row: dict[str, Any]) -> dict[str, Any] | None:
    """(question, briefing) -> normalised graph plan, prompts rendered by the runtime builder."""
    graph = row.get("target_graph_plan")
    if not isinstance(graph, dict):
        return None
    system, user = typed_plan_messages(str(row["question"]),
                                       understanding=row.get("step1_briefing"),
                                       variant="lean")
    return {"conversations": [{"from": "human", "value": user},
                              {"from": "gpt", "value": compact(graph)}],
            "system": system}


def repair_sample(row: dict[str, Any]) -> dict[str, Any] | None:
    graph = row.get("target_graph_plan")
    feedback = row.get("failure_feedback")
    if not isinstance(graph, dict) or not isinstance(feedback, dict):
        return None
    system, user = typed_replan_messages(str(row["question"]), feedback)
    return {"conversations": [{"from": "human", "value": user},
                              {"from": "gpt", "value": compact(graph)}],
            "system": system}


def dpo_sample(row: dict[str, Any]) -> dict[str, Any] | None:
    chosen, rejected = row.get("chosen_graph_plan"), row.get("rejected_graph_plan")
    if not isinstance(chosen, dict) or not isinstance(rejected, dict):
        return None
    system, user = typed_plan_messages(str(row["question"]),
                                       understanding=row.get("step1_briefing"),
                                       variant="lean")
    return {"conversations": [{"from": "human", "value": user}],
            "system": system,
            "chosen": {"from": "gpt", "value": compact(chosen)},
            "rejected": {"from": "gpt", "value": compact(rejected)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-dir", type=Path, default=Path("data/qa/teacher_full_v1"))
    ap.add_argument("--qa", type=Path, default=Path("data/qa/cicada_core_v4/train.jsonl"),
                    help="source rows (for template_family lookup)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/training/llamafactory_v1"))
    ap.add_argument("--family-cap", type=int, default=150)
    ap.add_argument("--bucket-cap", type=int, default=400,
                    help="max plan-SFT samples per bucket, applied after the family cap. The "
                         "count bucket spans ~8 families so the family cap alone leaves it at "
                         "32%% of the pool - a 'when unsure, count' shape prior the bridge "
                         "hard-negatives show is the main failure mode. 0 disables.")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--abstain-frac", type=float, default=0.10,
                    help="cap abstain samples at this fraction of the verified plan pool "
                         "(natural QA abstain share is ~5%%; uncapped teacher pools run ~14%% "
                         "and would over-teach abstention). 0 disables the cap.")
    args = ap.parse_args()

    fam_by_id = {str(r["id"]): str(r.get("template_family"))
                 for r in read_jsonl(args.qa)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}

    # ---- plan SFT: verified + abstain, family-capped -----------------------------------
    verified = read_jsonl(args.teacher_dir / "verified_sft.jsonl")
    abstain = read_jsonl(args.teacher_dir / "abstain_sft.jsonl")
    fam_counts: Counter = Counter()
    bucket_counts: Counter = Counter()
    train, val, capped, bucket_capped = [], [], 0, 0
    ordered = sorted(verified, key=lambda r: hashlib.sha1(str(r["id"]).encode()).hexdigest())
    for row in ordered:
        fam = fam_by_id.get(str(row["id"]), "?")
        if fam_counts[fam] >= args.family_cap:
            capped += 1
            continue
        bucket = str(row.get("bucket", "?"))
        if args.bucket_cap and bucket_counts[bucket] >= args.bucket_cap:
            bucket_capped += 1
            continue
        s = plan_sample(row)
        if s is None:
            continue
        fam_counts[fam] += 1
        bucket_counts[bucket] += 1
        (val if is_val(row["id"], args.val_frac) else train).append(s)
    abstain_budget = (int(args.abstain_frac * (len(train) + len(val)))
                      if args.abstain_frac else len(abstain))
    abstain_kept = 0
    for row in sorted(abstain, key=lambda r: hashlib.sha1(str(r["id"]).encode()).hexdigest()):
        if abstain_kept >= abstain_budget:
            break
        s = plan_sample(row)
        if s is None:
            continue
        abstain_kept += 1
        (val if is_val(row["id"], args.val_frac) else train).append(s)
    (args.out_dir / "cicada_plan_sft.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "cicada_plan_sft_val.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=1), encoding="utf-8")
    report["plan_sft"] = {"train": len(train), "val": len(val), "family_capped_out": capped,
                          "bucket_capped_out": bucket_capped, "families": len(fam_counts),
                          "abstain_kept": abstain_kept, "abstain_total": len(abstain),
                          "bucket_composition": dict(bucket_counts.most_common())}

    # ---- repair SFT ---------------------------------------------------------------------
    rtrain, rval = [], []
    for row in read_jsonl(args.teacher_dir / "repair_sft.jsonl"):
        s = repair_sample(row)
        if s is not None:
            (rval if is_val(row["id"], args.val_frac) else rtrain).append(s)
    (args.out_dir / "cicada_repair_sft.json").write_text(
        json.dumps(rtrain, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "cicada_repair_sft_val.json").write_text(
        json.dumps(rval, ensure_ascii=False, indent=1), encoding="utf-8")
    report["repair_sft"] = {"train": len(rtrain), "val": len(rval)}

    # ---- DPO ----------------------------------------------------------------------------
    dpo = [s for s in (dpo_sample(r) for r in read_jsonl(args.teacher_dir / "dpo_pairs.jsonl"))
           if s is not None]
    (args.out_dir / "cicada_dpo.json").write_text(
        json.dumps(dpo, ensure_ascii=False, indent=1), encoding="utf-8")
    report["dpo"] = {"pairs": len(dpo)}

    # ---- dataset_info snippets ----------------------------------------------------------
    info = {
        "cicada_plan_sft": {"file_name": "cicada_plan_sft.json", "formatting": "sharegpt",
                            "columns": {"messages": "conversations", "system": "system"}},
        "cicada_plan_sft_val": {"file_name": "cicada_plan_sft_val.json", "formatting": "sharegpt",
                                "columns": {"messages": "conversations", "system": "system"}},
        "cicada_repair_sft": {"file_name": "cicada_repair_sft.json", "formatting": "sharegpt",
                              "columns": {"messages": "conversations", "system": "system"}},
        "cicada_dpo": {"file_name": "cicada_dpo.json", "formatting": "sharegpt", "ranking": True,
                       "columns": {"messages": "conversations", "system": "system",
                                   "chosen": "chosen", "rejected": "rejected"}},
    }
    (args.out_dir / "dataset_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    (args.out_dir / "export_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
