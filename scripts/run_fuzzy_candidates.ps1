<#
.SYNOPSIS
    Generates the fuzzy entity candidate report (er_candidates.csv) in isolation.

.DESCRIPTION
    Reads data/entities/canonical_orgs.parquet (must already exist from pipeline 03)
    and writes data/entities/er_candidates.csv.

    This script does NOT run ER phase 1 or phase 2. It does NOT modify canonical_orgs
    or alias_map. It is a pure read-and-report step.

    It can be run standalone after the main pipeline, or called by
    run_full_internal_pipeline.ps1 -RunFuzzyCandidates.

.PARAMETER MaxPairsTotal
    Hard global cap on total candidate pairs output. Default: 10000.

.PARAMETER MaxBlockSize
    Prefix blocks larger than this are skipped entirely. Default: 50.

.PARAMETER MaxPairsPerBlock
    Cap on pairs from a single prefix block. Default: 500.

.PARAMETER Threshold
    Minimum Jaro-Winkler similarity. Default: 0.92.

.PARAMETER Limit
    Optional: restrict to the first N entities (for quick testing).

.EXAMPLE
    # Standard run after pipeline completes
    powershell -ExecutionPolicy Bypass -File scripts\run_fuzzy_candidates.ps1

.EXAMPLE
    # Tighter limits (faster)
    powershell -ExecutionPolicy Bypass -File scripts\run_fuzzy_candidates.ps1 `
        -MaxPairsTotal 3000 -MaxBlockSize 25

.EXAMPLE
    # Quick smoke test: first 500 entities only
    powershell -ExecutionPolicy Bypass -File scripts\run_fuzzy_candidates.ps1 -Limit 500
#>

param(
    [int]   $MaxPairsTotal    = 10000,
    [int]   $MaxBlockSize     = 50,
    [int]   $MaxPairsPerBlock = 500,
    [double]$Threshold        = 0.92,
    [int]   $Limit            = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot

# ─── log ────────────────────────────────────────────────────────────────────

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir    = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$logFile   = Join-Path $logDir "fuzzy_run_$timestamp.log"

Start-Transcript -Path $logFile -Append | Out-Null

# ─── header ─────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host ("=" * 64)
Write-Host "  FUZZY CANDIDATE REPORT — er_candidates.csv"
Write-Host ("=" * 64)
Write-Host ""
Write-Host "  Project root  : $ProjectRoot"
Write-Host "  Log file      : $logFile"
Write-Host "  Started       : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""
Write-Host "  Read-only step: does NOT modify canonical_orgs or alias_map."
Write-Host "  Output: data/entities/er_candidates.csv  (human review only)"
Write-Host ""
Write-Host "  BLOCKING STRATEGY:"
Write-Host "    Method  : 3-character normalised-name prefix"
Write-Host "    Limits  :"
Write-Host "      threshold          = $Threshold"
Write-Host "      max_block_size     = $MaxBlockSize  (blocks above this skipped)"
Write-Host "      max_pairs_per_block= $MaxPairsPerBlock"
Write-Host "      max_pairs_total    = $MaxPairsTotal"
if ($Limit -gt 0) {
    Write-Host "      entity limit       = $Limit  (PARTIAL RUN — testing only)"
}

# ─── prerequisite check ─────────────────────────────────────────────────────

$canonicalOrgsFile = Join-Path $ProjectRoot "data\entities\canonical_orgs.parquet"
if (-not (Test-Path $canonicalOrgsFile)) {
    Write-Host ""
    Write-Host "FATAL: canonical_orgs.parquet not found."
    Write-Host "       Run the main pipeline first (pipelines/02 + 03)."
    Stop-Transcript | Out-Null 2>&1
    exit 1
}
Write-Host ""
Write-Host "  Prerequisite: data/entities/canonical_orgs.parquet  [OK]"

# ─── inline Python runner ────────────────────────────────────────────────────
# Call er_candidates directly, not pipeline 03, so that:
#   1. Phase 2 heuristic resolution is NOT re-run (no risk of overwriting canonical_orgs).
#   2. The fuzzy step is clearly decoupled and auditable.
# We generate a tiny Python runner script on the fly, streamed to python via stdin.
# This avoids creating a persistent helper file in the repo.

$srcDir = Join-Path $ProjectRoot "src"

$pythonScript = @"
import sys
sys.path.insert(0, r'$srcDir')

import time
import pandas as pd
from pathlib import Path
from er_candidates import generate_candidates, write_candidates, describe_limits
from er_phase1 import load_canonical_orgs

entities_dir = Path(r'$ProjectRoot') / 'data' / 'entities'
canonical_orgs = load_canonical_orgs(entities_dir)
print(f"  Loaded {len(canonical_orgs):,} canonical entities.")

limit = $Limit
if limit > 0:
    canonical_orgs = canonical_orgs.head(limit)
    print(f"  [PARTIAL] Limited to first {limit} entities.")

pool_statuses = ['unresolved', 'singleton']
pool_size = canonical_orgs['er_status'].isin(pool_statuses).sum()
print(f"  Entities in fuzzy pool (unresolved + singleton): {pool_size:,}")
print()

threshold        = $Threshold
max_pairs_total  = $MaxPairsTotal
max_block_size   = $MaxBlockSize
max_pairs_per_block = $MaxPairsPerBlock

print(f"  Active limits: {describe_limits(threshold, max_pairs_total, max_pairs_per_block, max_block_size)}")
print()

t0 = time.time()
candidates = generate_candidates(
    canonical_orgs,
    threshold=threshold,
    max_pairs_total=max_pairs_total,
    max_pairs_per_block=max_pairs_per_block,
    max_block_size=max_block_size,
)
elapsed = time.time() - t0
print(f"  Comparison complete in {elapsed:.1f}s — {len(candidates):,} candidate pairs above threshold.")

out_path = entities_dir / 'er_candidates.csv'
write_candidates(candidates, out_path)

if len(candidates):
    print()
    print("  Top 10 candidate pairs (REVIEW ONLY — not auto-merged):")
    for _, row in candidates.head(10).iterrows():
        shared = 'shared-region' if row['shared_region'] else ''
        print(f"    {row['similarity']:.3f}  [{row['block_key']}]  "
              f"{str(row['entity_a_name'])[:32]:<32} <-> {str(row['entity_b_name'])[:32]}  {shared}")
    print()
    if 'block_key' in candidates.columns:
        top_blocks = candidates['block_key'].value_counts().head(5)
        print("  Top prefix blocks by candidate count:")
        for block, cnt in top_blocks.items():
            print(f"    '{block}' : {cnt} pairs")

print()
print("FUZZY STEP DONE.")
"@

Write-Host ""
Write-Host ("─" * 64)

$t = Get-Date

# Write the Python script to the scratchpad to avoid stdin encoding issues
$scratchDir = $env:TEMP
if (-not $scratchDir) { $scratchDir = $ProjectRoot }
$pyTmpFile  = Join-Path $scratchDir "_fuzzy_runner_$timestamp.py"

$pythonScript | Out-File -FilePath $pyTmpFile -Encoding utf8

try {
    python $pyTmpFile
    $exitCode = $LASTEXITCODE
} finally {
    if (Test-Path $pyTmpFile) { Remove-Item $pyTmpFile -Force }
}

if ($exitCode -and $exitCode -ne 0) {
    Write-Host ""
    Write-Host "FATAL: Fuzzy candidate generation exited with code $exitCode."
    Stop-Transcript | Out-Null 2>&1
    exit $exitCode
}

$elapsed = [math]::Round(((Get-Date) - $t).TotalSeconds)

Write-Host ""
Write-Host ("=" * 64)
Write-Host "  FUZZY CANDIDATE REPORT COMPLETE  (${elapsed}s)"
Write-Host "  Output: data/entities/er_candidates.csv"
Write-Host "  Log   : $logFile"
Write-Host ("=" * 64)
Write-Host ""

Stop-Transcript | Out-Null
