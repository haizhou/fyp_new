#!/usr/bin/env python3
"""Ten-raw audit: for any gate-decided rate metric, dump 10 RANDOM (not cherry-picked, not
failures-only) decisions with the full raw output and the gate's verdict, for human eyeballing
BEFORE the number enters a table/figure.

Motivation (worklog 2026-07-07): the schema diagnostic had three consecutive opposite-signed
gate bugs, each caught only by after-the-fact spot-check. This makes the spot-check mandatory
and systematic instead of intuition-triggered.

Usage: dump_raws.py <results.jsonl or diagnostic.json> [--n 10] [--seed-token <str>]
Prints id, gate verdict fields, and the raw/predicted, so a human can confirm the gate agrees
with their own judgment on a sample that includes PASSES, not just failures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def stable_sample(items, n, token):
    # deterministic pseudo-random pick (no Date/rand): hash by index+token
    import hashlib
    keyed = sorted(range(len(items)),
                   key=lambda i: hashlib.sha1(f"{token}:{i}".encode()).hexdigest())
    return [items[i] for i in keyed[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--token", default="audit")
    ap.add_argument("--fields", nargs="*", default=None,
                    help="verdict fields to surface per record (default: auto)")
    args = ap.parse_args()

    text = args.path.read_text(encoding="utf-8")
    # jsonl (per-question results) or a diagnostic json with failures arrays
    if args.path.suffix == ".jsonl":
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
        sample = stable_sample(rows, args.n, args.token)
        for r in sample:
            fields = args.fields or [k for k in ("id", "predicted", "correct", "error",
                                                 "expected_status", "oracle_answer") if k in r]
            print("─" * 70)
            for f in fields:
                print(f"  {f}: {json.dumps(r.get(f), ensure_ascii=False)[:300]}")
    else:
        d = json.loads(text)
        for model, rep in d.items():
            fails = rep.get("failures", [])
            print("=" * 70, f"\n{model}: metrics = " + ", ".join(
                f"{k}={rep[k]}" for k in rep if isinstance(rep.get(k), int)))
            print(f"  (showing {min(args.n, len(fails))} random stored raws — these are the "
                  f"gate's NEGATIVE decisions; open them to confirm the gate is right)")
            for r in stable_sample(fails, args.n, args.token):
                print("─" * 60)
                print(f"  id={r.get('id')} kind={r.get('kind')}")
                print(f"  raw: {str(r.get('raw'))[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
