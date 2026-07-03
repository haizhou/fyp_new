"""
Fuzzy candidate report generator for unresolved GB-FTS-only entities.

Generates er_candidates.csv containing pairs of entities with high name similarity
(Jaro-Winkler >= threshold) for HUMAN REVIEW ONLY.

This module NEVER writes to alias_map or canonical_orgs.
It is a read-only analysis tool that produces a review artifact.

After human review, confirmed merges should be added to configs/gov_lookup.json
(for named government/public bodies) or handled via a future manual override file.

=== BLOCKING STRATEGY ===

To avoid O(n²) all-pairs matching across the full entity set, comparisons are
restricted to entities that share the same 3-character normalised-name prefix
(e.g. all entities whose normalised name starts with "NAT", "UNI", "SOT"…).

This blocking key catches transpositions and typos in the first characters only.
It is intentionally conservative: it will miss pairs whose names diverge in the
first 3 chars (e.g. "ST JAMES HOSPITAL" vs "SAINT JAMES HOSPITAL") but avoids
the combinatorial explosion of comparing every entity to every other.

Three hard limits prevent runaway execution even within a block:
  1. max_block_size  — blocks with more than N entities are skipped entirely
                       (logged as a warning). A block that large almost certainly
                       contains entities that are NOT the same organisation, e.g.
                       all "THE …" or "NAT …" entries.
  2. max_pairs_per_block — cap on output pairs from a single block (guards against
                           a medium-size block producing very many high-sim pairs).
  3. max_pairs_total — hard global cap on total output pairs across all blocks.
                       This is a true global cap: once reached, block iteration stops.

Worst-case comparisons = min(max_block_size, block_size) * (min(max_block_size, block_size)-1)/2
                         × number_of_blocks_under_cap

Import rules: imports ONLY from normalise.py.
"""

import logging
from pathlib import Path

import pandas as pd

from normalise import normalise_name

logger = logging.getLogger(__name__)

_EMPTY_COLS = [
    "entity_a_id", "entity_a_name", "entity_b_id", "entity_b_name",
    "similarity", "shared_region", "norm_name_a", "norm_name_b",
    "block_key",
]


def generate_candidates(
    canonical_orgs: pd.DataFrame,
    threshold: float = 0.92,
    max_pairs_total: int = 10_000,
    max_pairs_per_block: int = 500,
    max_block_size: int = 50,
    er_statuses: list[str] | None = None,
) -> pd.DataFrame:
    """Compute pairwise Jaro-Winkler similarity for unresolved/singleton entities.

    Blocking strategy: 3-character normalised-name prefix.
    All three limits (max_pairs_total, max_pairs_per_block, max_block_size) apply.
    See module docstring for rationale.

    Args:
        canonical_orgs:    full canonical_orgs DataFrame (any er_status mix)
        threshold:         minimum Jaro-Winkler similarity to include (default 0.92)
        max_pairs_total:   hard global cap on output pairs (default 10,000)
        max_pairs_per_block: cap on pairs from a single prefix block (default 500)
        max_block_size:    blocks larger than this are skipped, not compared (default 50)
        er_statuses:       which er_status values to include
                           (default: ['unresolved', 'singleton'])

    Returns:
        DataFrame with columns:
            entity_a_id, entity_a_name, entity_b_id, entity_b_name,
            similarity, shared_region, norm_name_a, norm_name_b, block_key
    """
    import jellyfish

    if er_statuses is None:
        er_statuses = ["unresolved", "singleton"]

    pool = canonical_orgs[canonical_orgs["er_status"].isin(er_statuses)].copy()

    if pool.empty:
        logger.info("No unresolved/singleton entities to compare.")
        return pd.DataFrame(columns=_EMPTY_COLS)

    pool["_norm"] = pool["canonical_name"].apply(normalise_name)
    pool = pool[pool["_norm"] != ""].copy()

    pool["_block"] = pool["_norm"].str[:3]

    blocks = pool.groupby("_block")
    n_blocks = pool["_block"].nunique()
    n_skipped_large = 0
    pairs: list[dict] = []

    for block_key, block_df in blocks:
        # Global cap: stop iterating blocks once total output limit is reached
        if len(pairs) >= max_pairs_total:
            logger.warning(
                "Global pair limit (%d) reached after processing some blocks; "
                "remaining blocks skipped.",
                max_pairs_total,
            )
            break

        if len(block_df) < 2:
            continue

        # Per-block size cap: skip oversized blocks entirely
        if len(block_df) > max_block_size:
            n_skipped_large += 1
            logger.debug(
                "Block '%s' has %d entities (> max_block_size=%d); skipped.",
                block_key, len(block_df), max_block_size,
            )
            continue

        rows = block_df.reset_index(drop=True)
        block_pairs: list[dict] = []

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a = rows.iloc[i]
                b = rows.iloc[j]
                if a["canonical_id"] == b["canonical_id"]:
                    continue
                sim = jellyfish.jaro_winkler_similarity(a["_norm"], b["_norm"])
                if sim >= threshold:
                    shared_region = (
                        bool(a.get("address_region"))
                        and a.get("address_region") == b.get("address_region")
                    )
                    block_pairs.append({
                        "entity_a_id": a["canonical_id"],
                        "entity_a_name": a["canonical_name"],
                        "entity_b_id": b["canonical_id"],
                        "entity_b_name": b["canonical_name"],
                        "similarity": round(sim, 4),
                        "shared_region": shared_region,
                        "norm_name_a": a["_norm"],
                        "norm_name_b": b["_norm"],
                        "block_key": block_key,
                    })
                    if len(block_pairs) >= max_pairs_per_block:
                        logger.debug(
                            "Block '%s': per-block pair limit (%d) reached.",
                            block_key, max_pairs_per_block,
                        )
                        break
            if len(block_pairs) >= max_pairs_per_block:
                break

        pairs.extend(block_pairs)

    if n_skipped_large:
        logger.warning(
            "%d prefix blocks exceeded max_block_size=%d and were skipped. "
            "These typically contain very common name prefixes (e.g. 'THE', 'NAT', 'UNI') "
            "where indiscriminate matching would produce too many false positives.",
            n_skipped_large, max_block_size,
        )

    if not pairs:
        return pd.DataFrame(columns=_EMPTY_COLS)

    result = (
        pd.DataFrame(pairs)
        .sort_values("similarity", ascending=False)
        .reset_index(drop=True)
    )
    # Enforce global cap on final output (in case pairs accumulated across blocks)
    if len(result) > max_pairs_total:
        result = result.head(max_pairs_total)
    return result


def write_candidates(candidates_df: pd.DataFrame, out_path: Path) -> None:
    """Write candidate pairs to CSV for human review."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(out_path, index=False)
    print(f"  Written: {out_path}  ({len(candidates_df):,} candidate pairs)")
    if len(candidates_df) == 0:
        print("  (No fuzzy candidates above threshold — this is fine.)")


def describe_limits(
    threshold: float,
    max_pairs_total: int,
    max_pairs_per_block: int,
    max_block_size: int,
) -> str:
    """Return a human-readable description of active limits for logging."""
    return (
        f"threshold={threshold}, "
        f"max_block_size={max_block_size} (blocks above this are skipped), "
        f"max_pairs_per_block={max_pairs_per_block}, "
        f"max_pairs_total={max_pairs_total}"
    )


# Compatibility re-export. New code should import from
# procurement_graph.er.candidates; old flat imports keep working.
from procurement_graph.er.candidates import (  # noqa: E402,F401
    describe_limits,
    generate_candidates,
    write_candidates,
)
