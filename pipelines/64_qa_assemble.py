"""
Pipeline step 64: Stage 2 ASSEMBLE.

Joins questions.jsonl + verifications.jsonl by spec_id and writes:
- benchmark.jsonl       passed questions (the QA benchmark)
- rejected_stage2.jsonl generation / no-anchor rejections
- regen_queue.jsonl     verifier-failed questions, each with the verifier's reason as feedback

To run the feedback regen loop: re-run pipeline 62 with --input regen_queue.jsonl
--out questions.regen.jsonl, then 63 on it, then 64 again merging the rounds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_graph.qa.benchmark.batch import assemble

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "qa" / "generated"


def main(args: argparse.Namespace) -> int:
    summary = assemble(
        questions_paths=[Path(p) for p in args.questions],
        verifications_paths=[Path(p) for p in args.verifications],
        benchmark_path=Path(args.benchmark),
        rejected_path=Path(args.rejected),
        regen_path=Path(args.regen),
    )
    print(json.dumps({"counts": summary, "benchmark": args.benchmark, "regen_queue": args.regen}, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 assemble (join questions + verifications)")
    parser.add_argument("--questions", nargs="+", default=[str(GEN / "questions.jsonl")],
                        help="One or more question files; later files override earlier per spec_id.")
    parser.add_argument("--verifications", nargs="+", default=[str(GEN / "verifications.jsonl")],
                        help="One or more verification files; later files override earlier per spec_id.")
    parser.add_argument("--benchmark", default=str(GEN / "benchmark.jsonl"))
    parser.add_argument("--rejected", default=str(GEN / "rejected_stage2.jsonl"))
    parser.add_argument("--regen", default=str(GEN / "regen_queue.jsonl"))
    raise SystemExit(main(parser.parse_args()))
