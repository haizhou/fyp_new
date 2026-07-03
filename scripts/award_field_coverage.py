"""
Award value and award date field coverage scanner.

Reads ALL raw OCDS JSONL records (full scan, not a sample) and reports
coverage for every candidate field related to award value and award date.

Run from project root (fyp_new/):
    python scripts/award_field_coverage.py
"""

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"

# ── field extractors ─────────────────────────────────────────────────────────
# Each entry: (field_path_label, extractor_fn)
# extractor_fn(release) -> list of (value_or_None,)
# Returns a list because one release can yield multiple values (e.g. per award/lot)

def _safe(v):
    return v if v is not None else None

def _awards_field(release, *keys):
    """Yield one value per award element."""
    results = []
    for aw in release.get("awards", []):
        v = aw
        for k in keys:
            if not isinstance(v, dict):
                v = None
                break
            v = v.get(k)
        results.append(v)
    return results

def _contracts_field(release, *keys):
    results = []
    for ct in release.get("contracts", []):
        v = ct
        for k in keys:
            if not isinstance(v, dict):
                v = None
                break
            v = v.get(k)
        results.append(v)
    return results

def _lots_field(release, *keys):
    results = []
    for lot in release.get("tender", {}).get("lots", []):
        v = lot
        for k in keys:
            if not isinstance(v, dict):
                v = None
                break
            v = v.get(k)
        results.append(v)
    return results

def _scalar(release, *keys):
    v = release
    for k in keys:
        if not isinstance(v, dict):
            return [None]
        v = v.get(k)
    return [v]

FIELDS = [
    # ── award value candidates ─────────────────────────────────────────────
    ("awards[].value.amount",
     lambda r: _awards_field(r, "value", "amount")),
    ("awards[].value.amountNet",
     lambda r: _awards_field(r, "value", "amountNet")),
    ("awards[].value.amountGross",
     lambda r: _awards_field(r, "value", "amountGross")),
    # grossAmount is sometimes a top-level key in awards.value
    ("awards[].value.grossAmount",
     lambda r: [aw.get("value", {}).get("grossAmount")
                for aw in r.get("awards", [])]),
    # contracts value
    ("contracts[].value.amount",
     lambda r: _contracts_field(r, "value", "amount")),
    ("contracts[].value.amountGross",
     lambda r: _contracts_field(r, "value", "amountGross")),
    # implementation value (sometimes used for actual spend)
    ("contracts[].implementation.transactions[].value.amount",
     lambda r: [t.get("value", {}).get("amount")
                for ct in r.get("contracts", [])
                for t in ct.get("implementation", {}).get("transactions", [])]),
    # tender value
    ("tender.value.amount",
     lambda r: _scalar(r, "tender", "value", "amount")),
    ("tender.value.amountGross",
     lambda r: _scalar(r, "tender", "value", "amountGross")),
    # lot value
    ("tender.lots[].value.amount",
     lambda r: _lots_field(r, "value", "amount")),
    ("tender.lots[].value.amountGross",
     lambda r: _lots_field(r, "value", "amountGross")),
    # minValue / maxValue on tender
    ("tender.minValue.amount",
     lambda r: _scalar(r, "tender", "minValue", "amount")),
    ("tender.maxValue.amount",
     lambda r: _scalar(r, "tender", "maxValue", "amount")),

    # ── award date candidates ──────────────────────────────────────────────
    ("awards[].date",
     lambda r: _awards_field(r, "date")),
    ("awards[].datePublished",
     lambda r: _awards_field(r, "datePublished")),
    ("awards[].contractPeriod.startDate",
     lambda r: _awards_field(r, "contractPeriod", "startDate")),
    ("awards[].contractPeriod.endDate",
     lambda r: _awards_field(r, "contractPeriod", "endDate")),
    # contracts signed date
    ("contracts[].dateSigned",
     lambda r: _contracts_field(r, "dateSigned")),
    ("contracts[].period.startDate",
     lambda r: _contracts_field(r, "period", "startDate")),
    ("contracts[].period.endDate",
     lambda r: _contracts_field(r, "period", "endDate")),
    # release-level date
    ("release.date",
     lambda r: [r.get("date")]),
    # tender award period
    ("tender.awardPeriod.startDate",
     lambda r: _scalar(r, "tender", "awardPeriod", "startDate")),
    ("tender.awardPeriod.endDate",
     lambda r: _scalar(r, "tender", "awardPeriod", "endDate")),
    # tender contractPeriod
    ("tender.contractPeriod.startDate",
     lambda r: _scalar(r, "tender", "contractPeriod", "startDate")),
    ("tender.contractPeriod.endDate",
     lambda r: _scalar(r, "tender", "contractPeriod", "endDate")),
    # lot contractPeriod
    ("tender.lots[].contractPeriod.startDate",
     lambda r: _lots_field(r, "contractPeriod", "startDate")),
    ("tender.lots[].contractPeriod.endDate",
     lambda r: _lots_field(r, "contractPeriod", "endDate")),
]

# ── counters ─────────────────────────────────────────────────────────────────

# Per field: {releases_with_any_nonnull, releases_total, element_nonnull, element_total, examples}
stats = {fp: {
    "releases_with_value": 0,
    "releases_total": 0,
    "element_nonnull": 0,
    "element_total": 0,
    "examples": [],
} for fp, _ in FIELDS}

# Also track: releases with >=1 award element at all
releases_with_awards = 0
releases_with_contracts = 0
releases_with_lots = 0
total_releases = 0

print("Scanning all raw JSONL files (full scan)...")
files = sorted(RAW_DIR.glob("*.jsonl.gz"))
for fpath in files:
    print(f"  {fpath.name} ...", end=" ", flush=True)
    count = 0
    with gzip.open(fpath, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                release = json.loads(line)
            except json.JSONDecodeError:
                continue

            count += 1
            total_releases += 1

            has_awards = bool(release.get("awards"))
            has_contracts = bool(release.get("contracts"))
            has_lots = bool(release.get("tender", {}).get("lots"))
            if has_awards:
                releases_with_awards += 1
            if has_contracts:
                releases_with_contracts += 1
            if has_lots:
                releases_with_lots += 1

            for fp, extractor in FIELDS:
                s = stats[fp]
                s["releases_total"] += 1
                try:
                    values = extractor(release)
                except Exception:
                    values = []
                nonnull = [v for v in values if v is not None and v != ""]
                s["element_total"] += len(values)
                s["element_nonnull"] += len(nonnull)
                if nonnull:
                    s["releases_with_value"] += 1
                    if len(s["examples"]) < 3:
                        s["examples"].append(str(nonnull[0])[:60])

    print(f"{count:,} releases")

print()

# ── report ────────────────────────────────────────────────────────────────────

def pct(a, b):
    return f"{a/b*100:5.1f}%" if b else "  N/A "

SEP = "─" * 100

print(SEP)
print(f"TOTAL RELEASES SCANNED : {total_releases:,}")
print(f"Releases with awards[] : {releases_with_awards:,}  ({pct(releases_with_awards, total_releases)})")
print(f"Releases with contracts[]: {releases_with_contracts:,}  ({pct(releases_with_contracts, total_releases)})")
print(f"Releases with lots[]   : {releases_with_lots:,}  ({pct(releases_with_lots, total_releases)})")
print(SEP)

print()
print("── AWARD VALUE CANDIDATES ──────────────────────────────────────────────────────────────────────────")
print(f"{'field_path':<52}  {'releases_pct':>12}  {'releases_w_val':>14}  {'elem_pct':>10}  example")
print("─" * 100)

value_fields = [fp for fp, _ in FIELDS if "value" in fp.lower() or "amount" in fp.lower() or "minValue" in fp.lower() or "maxValue" in fp.lower()]
for fp in value_fields:
    s = stats[fp]
    r_pct = pct(s["releases_with_value"], s["releases_total"])
    e_pct = pct(s["element_nonnull"], s["element_total"]) if s["element_total"] else "  N/A "
    ex = s["examples"][0][:40] if s["examples"] else ""
    print(f"  {fp:<50}  {r_pct}  {s['releases_with_value']:>14,}  {e_pct}  {ex}")

print()
print("── AWARD DATE CANDIDATES ───────────────────────────────────────────────────────────────────────────")
print(f"{'field_path':<52}  {'releases_pct':>12}  {'releases_w_val':>14}  {'elem_pct':>10}  example")
print("─" * 100)

date_fields = [fp for fp, _ in FIELDS if fp not in value_fields]
for fp in date_fields:
    s = stats[fp]
    r_pct = pct(s["releases_with_value"], s["releases_total"])
    e_pct = pct(s["element_nonnull"], s["element_total"]) if s["element_total"] else "  N/A "
    ex = s["examples"][0][:40] if s["examples"] else ""
    print(f"  {fp:<50}  {r_pct}  {s['releases_with_value']:>14,}  {e_pct}  {ex}")

print(SEP)
