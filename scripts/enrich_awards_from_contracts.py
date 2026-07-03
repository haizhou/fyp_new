"""Enrich extracted awards with contract signed dates and best value fields."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AWARDS_PATH = ROOT / "data" / "extracted" / "awards.parquet"
RELEASES_PATH = ROOT / "data" / "interim" / "releases.parquet"


def _first_non_empty(values: pd.Series) -> Any:
    for value in values:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def flatten_contracts(releases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in releases.itertuples(index=False):
        try:
            contracts = json.loads(getattr(row, "contracts_json") or "[]")
        except Exception:
            contracts = []
        for contract in contracts:
            if not isinstance(contract, dict):
                continue
            award_id = str(contract.get("award_id") or "")
            if not award_id:
                continue
            rows.append(
                {
                    "ocid": getattr(row, "ocid"),
                    "award_id": award_id,
                    "contract_id": str(contract.get("contract_id") or ""),
                    "contract_value_amount": contract.get("value_amount"),
                    "contract_value_currency": str(contract.get("value_currency") or ""),
                    "award_date_signed": str(contract.get("date_signed") or ""),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "ocid",
                "award_id",
                "contract_id",
                "contract_value_amount",
                "contract_value_currency",
                "award_date_signed",
                "contract_count",
            ]
        )
    contracts = pd.DataFrame(rows)
    contracts["contract_value_amount"] = pd.to_numeric(contracts["contract_value_amount"], errors="coerce")
    return (
        contracts.groupby(["ocid", "award_id"], as_index=False)
        .agg(
            contract_id=("contract_id", _first_non_empty),
            contract_value_amount=("contract_value_amount", _first_non_empty),
            contract_value_currency=("contract_value_currency", _first_non_empty),
            award_date_signed=("award_date_signed", _first_non_empty),
            contract_count=("contract_id", "nunique"),
        )
    )


def enrich_awards(awards: pd.DataFrame, releases: pd.DataFrame) -> pd.DataFrame:
    contracts = flatten_contracts(releases[["ocid", "contracts_json"]])
    tender_values = releases[["ocid", "tender_value_amount", "tender_value_currency"]].copy()
    tender_values["tender_value_amount"] = pd.to_numeric(tender_values["tender_value_amount"], errors="coerce")

    drop_cols = [
        "contract_id",
        "contract_value_amount",
        "contract_value_currency",
        "award_date_signed",
        "contract_count",
        "tender_value_amount",
        "tender_value_currency",
        "award_value_best_amount",
        "award_value_best_currency",
        "award_value_source",
    ]
    awards = awards.drop(columns=[col for col in drop_cols if col in awards.columns]).copy()
    enriched = awards.merge(contracts, on=["ocid", "award_id"], how="left")
    enriched = enriched.merge(tender_values, on="ocid", how="left")

    award_amount = pd.to_numeric(enriched["award_value_amount"], errors="coerce")
    contract_amount = pd.to_numeric(enriched["contract_value_amount"], errors="coerce")
    tender_amount = pd.to_numeric(enriched["tender_value_amount"], errors="coerce")

    enriched["award_value_best_amount"] = award_amount.combine_first(contract_amount).combine_first(tender_amount)
    enriched["award_value_source"] = ""
    enriched.loc[award_amount.notna(), "award_value_source"] = "award"
    enriched.loc[award_amount.isna() & contract_amount.notna(), "award_value_source"] = "contract"
    enriched.loc[
        award_amount.isna() & contract_amount.isna() & tender_amount.notna(),
        "award_value_source",
    ] = "tender"

    enriched["award_value_best_currency"] = ""
    enriched.loc[enriched["award_value_source"].eq("award"), "award_value_best_currency"] = enriched["award_value_currency"]
    enriched.loc[enriched["award_value_source"].eq("contract"), "award_value_best_currency"] = enriched["contract_value_currency"]
    enriched.loc[enriched["award_value_source"].eq("tender"), "award_value_best_currency"] = enriched["tender_value_currency"]

    for col in ["contract_value_currency", "award_date_signed", "contract_id", "award_value_best_currency", "award_value_source"]:
        enriched[col] = enriched[col].fillna("").astype(str)

    return enriched


def backup_awards(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = path.parent / "backups" / f"award_enrichment_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--awards", type=Path, default=AWARDS_PATH)
    parser.add_argument("--releases", type=Path, default=RELEASES_PATH)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    awards = pd.read_parquet(args.awards)
    releases = pd.read_parquet(
        args.releases,
        columns=["ocid", "contracts_json", "tender_value_amount", "tender_value_currency"],
    )
    enriched = enrich_awards(awards, releases)
    backup_dir = None if args.no_backup else backup_awards(args.awards)
    enriched.to_parquet(args.awards, index=False, compression="snappy")

    summary = {
        "input_rows": len(awards),
        "output_rows": len(enriched),
        "award_date_signed_non_null": int(enriched["award_date_signed"].replace("", pd.NA).notna().sum()),
        "award_value_amount_non_null": int(pd.to_numeric(enriched["award_value_amount"], errors="coerce").notna().sum()),
        "contract_value_amount_non_null": int(pd.to_numeric(enriched["contract_value_amount"], errors="coerce").notna().sum()),
        "award_value_best_amount_non_null": int(pd.to_numeric(enriched["award_value_best_amount"], errors="coerce").notna().sum()),
        "backup_dir": str(backup_dir) if backup_dir else "",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
