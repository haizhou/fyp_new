#!/usr/bin/env bash
# GPU-INDEPENDENT eval baselines for the compare_set_v4 matrix (RAG naive/strong + teacher).
# These hit Azure (nano/grok) + local KG only, so they run while the GPU trains students.
# Results land in outputs/eval/baselines/<name>/ for the final comparison table.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
set -a; . ./.env; set +a
export CUDA_VISIBLE_DEVICES=""     # force CPU: these must not contend for the training GPU
PY="$ROOT/.venv/bin/python"
Q="data/qa/eval/compare_set_v4.jsonl"
mkdir -p train_logs

run() { echo "=== [$(date -u +%H:%M:%S)] $1 ==="; shift; "$@"; echo; }

run "RAG naive (260)"  $PY scripts/run_compare.py --system rag --rag-mode naive \
    --questions "$Q" --workers 1 --out-dir outputs/eval/baselines/rag_naive
run "RAG strong (260)" $PY scripts/run_compare.py --system rag --rag-mode strong \
    --questions "$Q" --workers 1 --out-dir outputs/eval/baselines/rag_strong
run "Teacher nano+grok (260)" $PY scripts/run_compare.py --system cicada \
    --questions "$Q" --plan-model grok-4-1-fast-non-reasoning --workers 8 \
    --out-dir outputs/eval/baselines/teacher
echo "=== [$(date -u +%H:%M:%S)] baselines DONE ==="
