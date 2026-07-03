"""
Pipeline step 60: QA benchmark Stage 2 (question generation + Gate B).

Generation model:  gpt-5.4-nano        (chat.completions)
Gate B verifier:   grok-4-1-fast-non-reasoning  (independent model family)

Dry-run (--dry-run) uses deterministic offline stand-ins and makes NO API calls, so the
full pipeline can be debugged before spending tokens. Live calls read the API key from
AZURE_OPENAI_API_KEY (or OPENAI_API_KEY) and the endpoint from AZURE_OPENAI_BASE_URL.

Writes:
    data/qa/generated/benchmark.jsonl
    data/qa/generated/rejected_stage2.jsonl
    reports/qa/stage2_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_graph.qa.benchmark.gate_b import DryRunGateBVerifier, LLMGateBVerifier
from procurement_graph.qa.benchmark.question_gen import DryRunQuestionGenerator, LLMQuestionGenerator
from procurement_graph.qa.benchmark.stage2 import build_stage2

ROOT = Path(__file__).resolve().parents[1]


def main(args: argparse.Namespace) -> int:
    if args.dry_run:
        generator = DryRunQuestionGenerator(model=args.generator_model, prompt_variant=args.prompt_variant)
        verifier = DryRunGateBVerifier(model=args.verifier_model)
    else:
        from procurement_graph.qa.benchmark.chat import ChatClient

        client = ChatClient.from_env(json_mode=args.json_mode)
        generator = LLMQuestionGenerator(
            client=client,
            model=args.generator_model,
            prompt_variant=args.prompt_variant,
        )
        verifier = LLMGateBVerifier(client=client, model=args.verifier_model)

    summary = build_stage2(
        specs_path=Path(args.specs),
        kg_dir=ROOT / "data" / "kg",
        output_dir=ROOT / "data" / "qa" / "generated",
        report_dir=ROOT / "reports" / "qa",
        generator=generator,
        verifier=verifier,
        evidence_cap=args.evidence_cap,
        factoid_sample_rate=args.factoid_sample_rate,
        limit=args.limit,
        sample_per_type=args.sample_per_type,
        with_evidence=not args.no_evidence,
        resume=not args.no_resume,
        out_tag=args.out_tag,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build QA Stage 2 (questions + Gate B)")
    parser.add_argument("--specs", default=str(ROOT / "data" / "qa" / "generated" / "answer_specs.jsonl"))
    parser.add_argument("--dry-run", action="store_true", help="Offline deterministic stand-ins; no API calls.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N specs (pilots).")
    parser.add_argument(
        "--sample-per-type",
        type=int,
        default=None,
        help="Take first N specs of each question_type for a diverse pilot.",
    )
    parser.add_argument("--generator-model", default="gpt-5.4-nano")
    parser.add_argument("--verifier-model", default="grok-4-1-fast-non-reasoning")
    parser.add_argument(
        "--prompt-variant",
        choices=["current", "strict_filters", "natural_procurement"],
        default="current",
        help="Question-generation prompt variant for prompt ablation.",
    )
    parser.add_argument("--evidence-cap", type=int, default=40, help="Max evidence rows in a recompute prompt.")
    parser.add_argument("--factoid-sample-rate", type=float, default=0.3, help="Gate B sampling rate for factoids.")
    parser.add_argument("--no-evidence", action="store_true", help="Skip backend load (faithfulness-only / fast debug).")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing outputs and overwrite.")
    parser.add_argument("--json-mode", action="store_true", help="Request response_format=json_object (live only).")
    parser.add_argument("--out-tag", default="", help="Suffix outputs (e.g. 'dry', 'pilot') to keep runs separate.")
    args = parser.parse_args()
    raise SystemExit(main(args))
