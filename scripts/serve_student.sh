#!/usr/bin/env bash
# Serve a CICADA Step-2 student with vLLM (OpenAI-compatible) for RSFT self-harvest / eval.
# usage: scripts/serve_student.sh <base_model> <served_name> <gpu_uuid> [adapter_path]
#   - no adapter  -> zero-shot rung (base weights, still guided-json at request time)
#   - adapter     -> LoRA hot-plugged (no merge needed); request model=<served_name>
# Ladder-uniform format enforcement: clients send response_format=json_schema, which vLLM maps
# to guided decoding — so every rung (incl. zero-shot) is format-locked; deltas = planning skill.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BASE="$1"; NAME="$2"; GPU="$3"; ADAPTER="${4:-}"
mkdir -p "$ROOT/train_logs"
LOG="$ROOT/train_logs/serve_${NAME}.log"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

ARGS=( serve "$BASE" --port 8000 --host 127.0.0.1
       --chat-template-content-format string
       --max-model-len 8192
       --gpu-memory-utilization 0.85 )

# Qwen3 must serve in NON-thinking mode to match training (enable_thinking=false).
case "$BASE" in
  *[Qq]wen*) ARGS+=( --default-chat-template-kwargs '{"enable_thinking": false}' ) ;;
esac

if [ -n "$ADAPTER" ]; then
  ARGS+=( --enable-lora --max-lora-rank 64 --lora-modules "${NAME}=${ADAPTER}" )
else
  ARGS+=( --served-model-name "$NAME" )
fi

echo "=== [$(date -u +%H:%M:%S)] serving $NAME (base=$BASE adapter=${ADAPTER:-none}) on GPU $GPU ===" | tee "$LOG"
echo "args: ${ARGS[*]}" | tee -a "$LOG"
exec "$ROOT/.venv/bin/vllm" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
