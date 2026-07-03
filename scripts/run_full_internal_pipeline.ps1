<#
.SYNOPSIS
    Runs the full internal procurement KG pipeline (steps 02–04) with optional
    fuzzy candidate generation as a safe, isolated final step.

.DESCRIPTION
    Executes in order:
        Step 02   python pipelines/02_er_phase1.py                   Deterministic ER
        Step 03   python pipelines/03_er_phase2.py --no-candidates    Heuristic ER
        Step 04   python pipelines/04_extract.py                      Rich OCDS extraction
        Step FUZZY scripts/run_fuzzy_candidates.ps1   [OPTIONAL]      Jaro-Winkler report

    Steps 02–04 produce the canonical, deterministic outputs. All three must succeed
    before fuzzy is attempted. If fuzzy fails it does not invalidate steps 02–04.

    PREREQUISITE: data/interim/releases.parquet must exist.
    Run pipelines/01_ingest.py once before this script.

.PARAMETER RunFuzzyCandidates
    When set, runs the fuzzy candidate report after steps 02–04 complete.
    Fuzzy failure never affects the already-written deterministic outputs.

.PARAMETER FuzzyMaxPairsTotal
    Hard global cap on total candidate pairs written to CSV. Default: 10000.

.PARAMETER FuzzyMaxBlockSize
    Prefix blocks with more entities than this are skipped entirely. Default: 50.
    See BLOCKING STRATEGY note below.

.PARAMETER FuzzyMaxPairsPerBlock
    Cap on candidate pairs from a single prefix block. Default: 500.

.PARAMETER FuzzyThreshold
    Minimum Jaro-Winkler similarity (0–1). Default: 0.92.
    Lower values produce more (noisier) candidates and increase runtime.

.EXAMPLE
    # Default: deterministic pipeline only
    powershell -ExecutionPolicy Bypass -File scripts\run_full_internal_pipeline.ps1

.EXAMPLE
    # Full pipeline including fuzzy candidates
    powershell -ExecutionPolicy Bypass -File scripts\run_full_internal_pipeline.ps1 -RunFuzzyCandidates

.EXAMPLE
    # Fuzzy with tighter limits (faster, fewer candidates)
    powershell -ExecutionPolicy Bypass -File scripts\run_full_internal_pipeline.ps1 `
        -RunFuzzyCandidates -FuzzyMaxBlockSize 30 -FuzzyMaxPairsTotal 5000

.NOTES
    ═══════════════════════════════════════════════════════════════
    FUZZY BLOCKING STRATEGY (why it is safe for overnight runs)
    ═══════════════════════════════════════════════════════════════

    Naive all-pairs Jaro-Winkler across N entities is O(N²). With ~10,000–50,000
    unresolved singleton entities this is infeasible. Three safeguards are applied:

    BLOCKING KEY: 3-character normalised-name prefix
        Entities are grouped by the first 3 characters of their normalised name.
        Only entities in the same block are compared. This reduces comparisons from
        O(N²) to O(k × B²) where k = number of non-empty blocks and B = block size.

        Examples of prefix blocks:
            "NAT" → National Health Service, National Grid, National Highways, ...
            "UNI" → University of Manchester, University College London, ...
            "SOT" → Southampton City Council, Southampton NHS Trust, ...

        Limitation: pairs whose names diverge in the first 3 chars are missed by
        fuzzy (e.g. "St James Hospital" vs "Saint James Hospital"). These are
        handled earlier by Phase 2 name+region merge.

    LIMIT 1 — max_block_size (default 50)
        Blocks with more than N entities are SKIPPED entirely.
        Rationale: a block this large is a common-prefix cluster (all "THE ...",
        all "NAT ...", all "UNI ...") where most pairs are obviously different
        organisations. Comparing them all produces overwhelming false-positive noise.
        Skipped blocks are logged as warnings.

    LIMIT 2 — max_pairs_per_block (default 500)
        Cap on pairs from a single accepted block.
        Guards against a medium-size block (≤ max_block_size) generating very many
        high-similarity pairs. Block iteration stops once the per-block cap is reached.

    LIMIT 3 — max_pairs_total (default 10,000)
        Hard global cap across all blocks. Once reached, block iteration stops
        immediately. This is a true global exit, not a per-block soft limit.

    Worst-case comparisons per accepted block:
        50 entities × 49 / 2 = 1,225 comparisons per block
    With a realistic alphabet of ~2,000 active prefix blocks, maximum theoretical
    comparisons ≈ 2.5 million — well within overnight tolerance.

    The fuzzy output (data/entities/er_candidates.csv) is for HUMAN REVIEW ONLY.
    It NEVER modifies alias_map, canonical_orgs, or any other pipeline output.
    ═══════════════════════════════════════════════════════════════
#>

param(
    [switch]$RunFuzzyCandidates,
    [int]   $FuzzyMaxPairsTotal     = 10000,
    [int]   $FuzzyMaxBlockSize      = 50,
    [int]   $FuzzyMaxPairsPerBlock  = 500,
    [double]$FuzzyThreshold         = 0.92
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot

# ─── helpers ────────────────────────────────────────────────────────────────

function Write-Banner($text) {
    Write-Host ""
    Write-Host ("=" * 64)
    Write-Host "  $text"
    Write-Host ("=" * 64)
}

function Write-StepHeader($step, $text) {
    Write-Host ""
    Write-Host ("─" * 64)
    Write-Host "  STEP $step  |  $text"
    Write-Host ("─" * 64)
}

function Elapsed($start) {
    $secs = [math]::Round(((Get-Date) - $start).TotalSeconds)
    return "${secs}s"
}

function Assert-PathExists($path, $label) {
    if (-not (Test-Path $path)) {
        Write-Host ""
        Write-Host "FATAL: $label not found."
        Write-Host "       Expected path: $path"
        Stop-Transcript | Out-Null 2>&1
        exit 1
    }
}

function Invoke-Step($stepLabel, $stepBlock) {
    $t = Get-Date
    try {
        & $stepBlock
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "FATAL: Step '$stepLabel' exited with code $LASTEXITCODE."
            Write-Host "       Outputs from already-completed steps are preserved."
            Stop-Transcript | Out-Null 2>&1
            exit $LASTEXITCODE
        }
        Write-Host ""
        Write-Host "  Step '$stepLabel' OK  ($(Elapsed $t))"
    } catch {
        Write-Host ""
        Write-Host "FATAL: Step '$stepLabel' threw an exception:"
        Write-Host "       $_"
        Write-Host "       Outputs from already-completed steps are preserved."
        Stop-Transcript | Out-Null 2>&1
        exit 1
    }
}

# ─── log setup ──────────────────────────────────────────────────────────────

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir    = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$logFile   = Join-Path $logDir "pipeline_run_$timestamp.log"

Start-Transcript -Path $logFile -Append | Out-Null

# ─── header ─────────────────────────────────────────────────────────────────

Write-Banner "FULL INTERNAL PIPELINE  —  procurement KG rebuild"

Write-Host ""
Write-Host "  Project root  : $ProjectRoot"
Write-Host "  Log file      : $logFile"
Write-Host "  Started       : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""
Write-Host "  Steps planned :"
Write-Host "    02  Deterministic ER (er_phase1)"
Write-Host "    03  Heuristic ER (er_phase2, --no-candidates)"
Write-Host "    04  Rich OCDS extraction"
if ($RunFuzzyCandidates) {
    Write-Host "    FUZZY  Jaro-Winkler candidate report (ENABLED)"
    Write-Host "           Blocking : 3-char normalised-name prefix"
    Write-Host "           Limits   : threshold=$FuzzyThreshold, max_block_size=$FuzzyMaxBlockSize,"
    Write-Host "                      max_pairs_per_block=$FuzzyMaxPairsPerBlock, max_pairs_total=$FuzzyMaxPairsTotal"
} else {
    Write-Host "    FUZZY  Jaro-Winkler candidate report (SKIPPED — pass -RunFuzzyCandidates to enable)"
}

# ─── prerequisite check ─────────────────────────────────────────────────────

Assert-PathExists (Join-Path $ProjectRoot "data\interim\releases.parquet") `
    "Interim releases (data/interim/releases.parquet) — run 01_ingest.py first"

Write-Host ""
Write-Host "  Prerequisite: data/interim/releases.parquet  [OK]"

$pipelineStart = Get-Date

# ─── STEP 02 — Deterministic ER ─────────────────────────────────────────────

Write-StepHeader "02" "Deterministic entity resolution (ER Phase 1)"

Invoke-Step "02_er_phase1" {
    python "$ProjectRoot\pipelines\02_er_phase1.py"
}

Assert-PathExists (Join-Path $ProjectRoot "data\entities\canonical_orgs.parquet") `
    "ER Phase 1 output: data/entities/canonical_orgs.parquet"
Assert-PathExists (Join-Path $ProjectRoot "data\entities\alias_map.parquet") `
    "ER Phase 1 output: data/entities/alias_map.parquet"

# ─── STEP 03 — Heuristic ER (no fuzzy) ──────────────────────────────────────

Write-StepHeader "03" "Heuristic entity resolution (ER Phase 2, no fuzzy candidates)"

Invoke-Step "03_er_phase2" {
    python "$ProjectRoot\pipelines\03_er_phase2.py" --no-candidates
}

Assert-PathExists (Join-Path $ProjectRoot "data\entities\er_audit.csv") `
    "ER Phase 2 audit log: data/entities/er_audit.csv"

# ─── STEP 04 — Rich OCDS extraction ─────────────────────────────────────────

Write-StepHeader "04" "Rich OCDS field extraction"

Invoke-Step "04_extract" {
    python "$ProjectRoot\pipelines\04_extract.py"
}

Assert-PathExists (Join-Path $ProjectRoot "data\extracted\tender_core.parquet") `
    "Extraction output: data/extracted/tender_core.parquet"

# ─── Deterministic pipeline complete ────────────────────────────────────────

$detElapsed = Elapsed $pipelineStart

Write-Host ""
Write-Host ("═" * 64)
Write-Host "  DETERMINISTIC PIPELINE COMPLETE  ($detElapsed)"
Write-Host "  Steps 02, 03 (heuristic), 04 — all succeeded and verified."
Write-Host ""
Write-Host "  Output files:"
Write-Host "    data/entities/canonical_orgs.parquet"
Write-Host "    data/entities/alias_map.parquet"
Write-Host "    data/entities/er_audit.csv"
Write-Host "    data/extracted/tender_core.parquet   (and 6 other tables)"
Write-Host ("═" * 64)

# ─── OPTIONAL STEP: Fuzzy candidate report ───────────────────────────────────
#
# Design rationale:
#   - Runs AFTER steps 02–04 are already written and verified.
#   - Delegates to a separate script (run_fuzzy_candidates.ps1) so that:
#       a) this script remains clean and its exit code reflects steps 02–04 only
#       b) fuzzy can be re-run independently without re-running the pipeline
#   - The fuzzy script calls er_candidates directly (not pipeline 03).
#     This guarantees it cannot re-write canonical_orgs or alias_map.
#   - Fuzzy exit code is checked and reported, but does NOT cause this script to exit 1.
#

if ($RunFuzzyCandidates) {
    Write-StepHeader "FUZZY" "Jaro-Winkler fuzzy candidate report (isolated, read-only)"

    Write-Host ""
    Write-Host "  Delegating to: scripts\run_fuzzy_candidates.ps1"
    Write-Host "  This step cannot modify canonical_orgs or alias_map."
    Write-Host "  A failure here does NOT invalidate steps 02–04."
    Write-Host ""

    $fuzzyScript = Join-Path $ScriptRoot "run_fuzzy_candidates.ps1"
    Assert-PathExists $fuzzyScript "Fuzzy script (scripts/run_fuzzy_candidates.ps1)"

    $fuzzyStart = Get-Date
    $fuzzyFailed = $false

    try {
        powershell -ExecutionPolicy Bypass -File $fuzzyScript `
            -MaxPairsTotal    $FuzzyMaxPairsTotal `
            -MaxBlockSize     $FuzzyMaxBlockSize `
            -MaxPairsPerBlock $FuzzyMaxPairsPerBlock `
            -Threshold        $FuzzyThreshold

        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            $fuzzyFailed = $true
            Write-Host ""
            Write-Host "WARNING: Fuzzy step exited with code $LASTEXITCODE."
        }
    } catch {
        $fuzzyFailed = $true
        Write-Host ""
        Write-Host "WARNING: Fuzzy step threw an exception: $_"
    }

    $fuzzyElapsed = Elapsed $fuzzyStart

    if ($fuzzyFailed) {
        Write-Host ""
        Write-Host ("─" * 64)
        Write-Host "  FUZZY STEP FAILED  ($fuzzyElapsed)"
        Write-Host "  data/entities/er_candidates.csv may be absent or incomplete."
        Write-Host "  Steps 02–04 outputs are NOT affected — they are complete."
        Write-Host "  To retry fuzzy only:"
        Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\run_fuzzy_candidates.ps1"
        Write-Host ("─" * 64)
    } else {
        Write-Host ""
        Write-Host ("─" * 64)
        Write-Host "  FUZZY STEP COMPLETE  ($fuzzyElapsed)"
        Write-Host "  Output: data/entities/er_candidates.csv  (human review only)"
        Write-Host ("─" * 64)
    }
} else {
    Write-Host ""
    Write-Host "  Fuzzy step not requested. To run it independently:"
    Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\run_fuzzy_candidates.ps1"
    Write-Host "  Or re-run this script with -RunFuzzyCandidates."
}

# ─── final summary ───────────────────────────────────────────────────────────

$totalElapsed = Elapsed $pipelineStart

Write-Host ""
Write-Banner "PIPELINE FINISHED  —  total elapsed: $totalElapsed"
Write-Host "  Log: $logFile"
Write-Host ""

Stop-Transcript | Out-Null
