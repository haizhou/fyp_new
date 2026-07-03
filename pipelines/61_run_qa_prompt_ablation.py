"""
Pipeline step 61: QA Stage 2 prompt ablation.

Runs the same fixed Stage 1 spec slice through multiple question-generation prompt
variants, then records comparable metrics. Use ``--dry-run`` for no-cost plumbing checks;
omit it for live Azure calls.

Writes per variant:
    data/qa/generated/benchmark.<out-prefix>_<variant>.jsonl
    data/qa/generated/rejected_stage2.<out-prefix>_<variant>.jsonl
    reports/qa/stage2_summary.<out-prefix>_<variant>.json

Writes aggregate:
    reports/qa/prompt_ablation_<out-prefix>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_graph.qa.benchmark.gate_b import DryRunGateBVerifier, LLMGateBVerifier
from procurement_graph.qa.benchmark.question_gen import DryRunQuestionGenerator, LLMQuestionGenerator
from procurement_graph.qa.benchmark.stage2 import build_stage2


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("current", "strict_filters", "natural_procurement")


def main(args: argparse.Namespace) -> int:
    output_dir = ROOT / "data" / "qa" / "generated"
    report_dir = ROOT / "reports" / "qa"
    variants = tuple(args.variants or VARIANTS)

    client = None
    if not args.dry_run:
        from procurement_graph.qa.benchmark.chat import ChatClient

        client = ChatClient.from_env(json_mode=args.json_mode)

    summaries: list[dict[str, Any]] = []
    for variant in variants:
        if args.dry_run:
            generator = DryRunQuestionGenerator(model=args.generator_model, prompt_variant=variant)
            verifier = DryRunGateBVerifier(model=args.verifier_model)
        else:
            generator = LLMQuestionGenerator(
                client=client,
                model=args.generator_model,
                prompt_variant=variant,
            )
            verifier = LLMGateBVerifier(client=client, model=args.verifier_model)

        out_tag = f"{args.out_prefix}_{variant}"
        summary = build_stage2(
            specs_path=Path(args.specs),
            kg_dir=ROOT / "data" / "kg",
            output_dir=output_dir,
            report_dir=report_dir,
            generator=generator,
            verifier=verifier,
            evidence_cap=args.evidence_cap,
            factoid_sample_rate=args.factoid_sample_rate,
            limit=args.limit,
            sample_per_type=args.sample_per_type,
            with_evidence=not args.no_evidence,
            resume=False,
            out_tag=out_tag,
        )
        summaries.append(_ablation_row(summary, variant, output_dir / f"rejected_stage2.{out_tag}.jsonl"))

    aggregate = {
        "specs": str(Path(args.specs)),
        "variants": list(variants),
        "dry_run": args.dry_run,
        "sample_per_type": args.sample_per_type,
        "limit": args.limit,
        "rows": summaries,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = report_dir / f"prompt_ablation_{args.out_prefix}.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return 0


def _ablation_row(summary: dict[str, Any], variant: str, rejected_path: Path) -> dict[str, Any]:
    attempted = int(summary.get("attempted", 0))
    accepted = int(summary.get("accepted", 0))
    rejected_gate_b = int(summary.get("rejected_gate_b", 0))
    hidden_failures = _count_rejections(rejected_path, "hidden semantic constraint")
    id_leaks = _count_rejections(rejected_path, "question_contains_forbidden_identifier")
    return {
        "variant": variant,
        "attempted": attempted,
        "accepted": accepted,
        "accept_rate": round(accepted / attempted, 4) if attempted else 0.0,
        "rejected_generation": summary.get("rejected_generation", 0),
        "rejected_gate_b": rejected_gate_b,
        "hidden_constraint_failures": hidden_failures,
        "id_leak_failures": id_leaks,
        "fallback_id_factoids": summary.get("factoid_anchor", {}).get("fallback_id", 0),
        "rejected_no_natural_anchor_factoids": summary.get("factoid_anchor", {}).get(
            "rejected_no_natural_anchor", 0
        ),
        "natural_factoids": summary.get("factoid_anchor", {}).get("natural", 0),
        "accepted_by_type": summary.get("accepted_by_type", {}),
        "token_usage": summary.get("token_usage", {}),
        "outputs": summary.get("outputs", {}),
    }


def _count_rejections(path: Path, needle: str) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if needle in line:
                count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QA prompt ablation over a fixed Stage 1 spec slice")
    parser.add_argument("--specs", default=str(ROOT / "data" / "qa" / "generated" / "answer_specs.jsonl"))
    parser.add_argument("--variants", nargs="*", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--dry-run", action="store_true", help="Offline deterministic stand-ins; no API calls.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-per-type", type=int, default=4)
    parser.add_argument("--generator-model", default="gpt-5.4-nano")
    parser.add_argument("--verifier-model", default="grok-4-1-fast-non-reasoning")
    parser.add_argument("--evidence-cap", type=int, default=40)
    parser.add_argument("--factoid-sample-rate", type=float, default=1.0)
    parser.add_argument("--no-evidence", action="store_true")
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--out-prefix", default="prompt_ablation")
    raise SystemExit(main(parser.parse_args()))
