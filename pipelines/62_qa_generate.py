"""
Pipeline step 62: Stage 2 GENERATION pass (nano only).

Reads Stage 1 answer specs (or a regen queue) and writes one generated question per spec to
questions.jsonl, concurrently and under the generator's rate limit. No verification here.

Dry-run uses an offline deterministic generator (no API). Live reads the API key from
AZURE_OPENAI_API_KEY. Re-run on a regen_queue.jsonl to apply the verifier's feedback.

Writes: data/qa/generated/questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_graph.qa.benchmark.batch import generate_pass
from procurement_graph.qa.benchmark.question_gen import DryRunQuestionGenerator, LLMQuestionGenerator

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "qa" / "generated"


def main(args: argparse.Namespace) -> int:
    if args.dry_run:
        generator = DryRunQuestionGenerator(model=args.generator_model, prompt_variant=args.prompt_variant)
    else:
        from procurement_graph.qa.benchmark.chat import ChatClient

        client = ChatClient.from_env(json_mode=args.json_mode)
        generator = LLMQuestionGenerator(client=client, model=args.generator_model, prompt_variant=args.prompt_variant)

    summary = generate_pass(
        input_path=Path(args.input),
        kg_dir=ROOT / "data" / "kg",
        out_path=Path(args.out),
        generator=generator,
        workers=args.workers,
        rpm=args.rpm,
        resume=not args.no_resume,
        with_anchor=not args.no_anchor,
    )
    print(json.dumps({"variant": args.prompt_variant, "out": args.out, "counts": summary}, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 generation pass (nano)")
    parser.add_argument("--input", default=str(GEN / "answer_specs.jsonl"),
                        help="answer_specs.jsonl or a regen_queue.jsonl")
    parser.add_argument("--out", default=str(GEN / "questions.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-variant", default="natural_procurement")
    parser.add_argument("--generator-model", default="gpt-5.4-nano")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--rpm", type=float, default=2400)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-anchor", action="store_true", help="Skip backend load / factoid anchoring (debug).")
    raise SystemExit(main(parser.parse_args()))
