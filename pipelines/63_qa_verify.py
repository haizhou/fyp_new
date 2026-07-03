"""
Pipeline step 63: Stage 2 VERIFICATION pass (grok only).

Reads questions.jsonl and writes one Gate B verdict per generated question to
verifications.jsonl, concurrently and capped at the verifier's rate limit (grok: 50 RPM).
Factoids are stratified-sampled (default 30%); the rest auto-pass without an API call.

Dry-run uses an offline verifier (no API). Live reads AZURE_OPENAI_API_KEY.

Writes: data/qa/generated/verifications.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from procurement_graph.qa.benchmark.batch import verify_pass
from procurement_graph.qa.benchmark.gate_b import DryRunGateBVerifier, LLMGateBVerifier

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "qa" / "generated"


def main(args: argparse.Namespace) -> int:
    if args.dry_run:
        verifier = DryRunGateBVerifier(model=args.verifier_model)
    else:
        from procurement_graph.qa.benchmark.chat import ChatClient

        client = ChatClient.from_env(json_mode=args.json_mode)
        verifier = LLMGateBVerifier(client=client, model=args.verifier_model)

    summary = verify_pass(
        questions_path=Path(args.questions),
        kg_dir=ROOT / "data" / "kg",
        out_path=Path(args.out),
        verifier=verifier,
        evidence_cap=args.evidence_cap,
        factoid_sample_rate=args.factoid_sample_rate,
        workers=args.workers,
        rpm=args.rpm,
        resume=not args.no_resume,
        with_evidence=not args.no_evidence,
    )
    print(json.dumps({"verifier": args.verifier_model, "out": args.out, "counts": summary}, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 verification pass (grok)")
    parser.add_argument("--questions", default=str(GEN / "questions.jsonl"))
    parser.add_argument("--out", default=str(GEN / "verifications.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verifier-model", default="grok-4-1-fast-non-reasoning")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rpm", type=float, default=45)
    parser.add_argument("--evidence-cap", type=int, default=40)
    parser.add_argument("--factoid-sample-rate", type=float, default=0.3)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-evidence", action="store_true")
    raise SystemExit(main(parser.parse_args()))
