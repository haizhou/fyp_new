#!/usr/bin/env bash
# Evaluate one base's ladder rungs on compare_set_v4, sequentially on one GPU.
# usage: scripts/eval_ladder.sh <qwen|llama> <gpu_uuid> <port> [out_root]
# Each rung: start vLLM -> wait ready -> run_compare (260) -> kill server.
# Eval protocol: run_compare --system cicada (repair budget 1, plan_samples 2), Step-1 nano/Azure.
# out_root separates pipeline versions (v1: outputs/eval/matrix, v2: outputs/eval/matrix_v2).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BASE_KEY="$1"; GPU="$2"; PORT="${3:-8000}"; OUT_ROOT="${4:-outputs/eval/matrix}"
set -a; . ./.env; set +a
Q="data/qa/eval/compare_set_v4.jsonl"
PY="$ROOT/.venv/bin/python"
mkdir -p train_logs outputs/eval/matrix

if [ "$BASE_KEY" = "qwen" ]; then
  BASE="Qwen/Qwen3-8B"
  EXTRA=( --default-chat-template-kwargs '{"enable_thinking": false}' )
  RUNGS=( "cicada-qwen3-zeroshot:"
          "cicada-qwen3-sft:outputs/qwen3_8b_cicada_sft_v1"
          "cicada-qwen3-rsft:outputs/qwen3_8b_cicada_rsft_v1"
          "cicada-qwen3-dpo:outputs/qwen3_8b_cicada_dpo_v1" )
else
  BASE="NousResearch/Meta-Llama-3.1-8B-Instruct"
  EXTRA=()
  RUNGS=( "cicada-llama31-zeroshot:"
          "cicada-llama31-sft:outputs/llama31_8b_cicada_sft_v1"
          "cicada-llama31-rsft:outputs/llama31_8b_cicada_rsft_v1"
          "cicada-llama31-dpo:outputs/llama31_8b_cicada_dpo_v1" )
fi

for RUNG in "${RUNGS[@]}"; do
  NAME="${RUNG%%:*}"; ADAPTER="${RUNG#*:}"
  OUT="${OUT_ROOT}/${NAME}"
  if [ -f "$OUT/compare_cicada.summary.json" ]; then
    echo "=== skip $NAME (summary exists) ==="; continue
  fi
  if [ -n "$ADAPTER" ] && [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
    echo "=== skip $NAME (adapter not trained yet) ==="; continue
  fi
  echo "=== [$(date -u +%H:%M:%S)] rung $NAME ==="
  ARGS=( serve "$BASE" --port "$PORT" --host 127.0.0.1
         --chat-template-content-format string --max-model-len 8192
         --gpu-memory-utilization 0.85 "${EXTRA[@]}" )
  if [ -n "$ADAPTER" ]; then
    ARGS+=( --enable-lora --max-lora-rank 64 --lora-modules "${NAME}=${ADAPTER}" )
  else
    ARGS+=( --served-model-name "$NAME" )
  fi
  # setsid: the server becomes its own process-group leader so teardown can kill the WHOLE
  # group (vLLM's EngineCore child survives a plain `kill` and once held 85GB into the next rung)
  CUDA_VISIBLE_DEVICES="$GPU" setsid "$ROOT/.venv/bin/vllm" "${ARGS[@]}" \
      > "train_logs/serve_${NAME}.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 60); do
    sleep 10
    curl -s "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "$NAME" && break
    kill -0 $SRV 2>/dev/null || { echo "!!! server died for $NAME"; break; }
  done
  if curl -s "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "$NAME"; then
    CUDA_VISIBLE_DEVICES="" $PY scripts/run_compare.py --system cicada \
      --plan-base-url "http://localhost:${PORT}/v1" --plan-model "$NAME" \
      --questions "$Q" --workers 8 --out-dir "$OUT" 2>&1 | tail -20
  else
    echo "!!! $NAME never became ready; skipping eval"
  fi
  # Kill the server's WHOLE process group (setsid above made $SRV the leader): vLLM's
  # EngineCore child survives a plain kill and once held 85GB into the next rung (2x OOM).
  kill -- -$SRV 2>/dev/null
  for i in $(seq 1 12); do
    kill -0 -- -$SRV 2>/dev/null || break
    sleep 5
    kill -9 -- -$SRV 2>/dev/null
  done
  wait $SRV 2>/dev/null
  sleep 8
done
echo "=== [$(date -u +%H:%M:%S)] ladder $BASE_KEY eval DONE ==="
