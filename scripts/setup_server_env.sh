#!/usr/bin/env bash
# One-shot server env build for the CICADA student-training ladder.
# Creates a Python 3.11 venv and installs the project + LLaMA-Factory + vLLM.
# Idempotent-ish: safe to re-run; pip skips already-satisfied deps.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
LOG="$ROOT/setup_env.log"
VENV="$ROOT/.venv"
PY=python3.11

echo "=== [$(date -u +%H:%M:%S)] setup start :: root=$ROOT ===" | tee -a "$LOG"

if [ ! -d "$VENV" ]; then
  $PY -m venv "$VENV" 2>&1 | tee -a "$LOG"
fi
PIP="$VENV/bin/pip"

$PIP install -U pip wheel setuptools 2>&1 | tee -a "$LOG"

echo "=== [$(date -u +%H:%M:%S)] project requirements ===" | tee -a "$LOG"
$PIP install -r "$ROOT/requirements.txt" 2>&1 | tee -a "$LOG"

echo "=== [$(date -u +%H:%M:%S)] vLLM (brings CUDA torch) ===" | tee -a "$LOG"
$PIP install "vllm" 2>&1 | tee -a "$LOG"

echo "=== [$(date -u +%H:%M:%S)] LLaMA-Factory + metrics ===" | tee -a "$LOG"
$PIP install "llamafactory[torch,metrics]" 2>&1 | tee -a "$LOG"

echo "=== [$(date -u +%H:%M:%S)] flash-attn (best effort, prebuilt wheel) ===" | tee -a "$LOG"
$PIP install flash-attn --no-build-isolation 2>&1 | tee -a "$LOG" \
  || echo "!!! flash-attn install FAILED (will fall back to sdpa if needed)" | tee -a "$LOG"

echo "=== [$(date -u +%H:%M:%S)] verify ===" | tee -a "$LOG"
"$VENV/bin/python" - 2>&1 <<'PYEOF' | tee -a "$LOG"
import importlib.metadata as m
def v(p):
    try: return m.version(p)
    except Exception as e: return f"MISSING ({e.__class__.__name__})"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(),
      "devs", torch.cuda.device_count())
for p in ["vllm","llamafactory","transformers","peft","trl","bitsandbytes","datasets","accelerate","flash_attn"]:
    print(p, v(p))
PYEOF
echo "=== [$(date -u +%H:%M:%S)] setup DONE ===" | tee -a "$LOG"
