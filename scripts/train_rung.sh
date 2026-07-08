#!/usr/bin/env bash
# Launch one LLaMA-Factory training rung, pinned to a specific GPU BY UUID.
# usage: scripts/train_rung.sh <gpu_uuid_or_index> <config.yaml> <logname>
#
# GPU survey on this node (2026-07-05): ONLY physical GPU2 is usable for CUDA.
#   GPU0 (GPU-ad267678) -> CUDA init fails    GPU1 (GPU-c9004c71) -> "No CUDA GPUs available"
#   GPU2 (GPU-18355792) -> OK, ~96GB free     GPU3 (GPU-fe12a3be) -> another user's job (busy)
# Pin by UUID (not index): index ordering is ambiguous and silently sent an earlier run to the
# busy GPU3 (OOM). CUDA_VISIBLE_DEVICES accepts the full UUID unambiguously.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
GPU="$1"; CONFIG="$2"; NAME="$3"
mkdir -p "$ROOT/train_logs"
LOG="$ROOT/train_logs/${NAME}.log"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1                                 # live loss lines instead of block-buffered tee
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation on big logits upcast
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== [$(date -u +%H:%M:%S)] train rung: $NAME on GPU $GPU :: $CONFIG ===" | tee "$LOG"
"$ROOT/.venv/bin/llamafactory-cli" train "$CONFIG" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
echo "=== [$(date -u +%H:%M:%S)] rung $NAME finished exit=$STATUS ===" | tee -a "$LOG"
exit $STATUS
