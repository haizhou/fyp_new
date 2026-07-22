#!/bin/bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd /home/uceeh01/fyp_new/fyp_new
export HF_HOME=/var/tmp/cicada/hf
for i in $(seq 1 72); do
  GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2 < 39000 && $3 <= 60 {print $1; exit}')
  echo "poll $i: gpu=${GPU:-none}"
  if [ -n "$GPU" ]; then
    if dd if=/dev/zero of="$HOME/.qprobe" bs=64M count=20 conv=fsync 2>/dev/null; then
      rm -f "$HOME/.qprobe"; echo "quota probe OK (1.25G writable)"
    else
      rm -f "$HOME/.qprobe"; echo "quota probe FAILED, waiting"; sleep 600; continue
    fi
    rm -rf outputs/wtq_C_v5/checkpoint-1000 outputs/wtq_C_v5
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/llamafactory-cli train configs/training/wtq_C_v5_qlora.yaml || { echo "train failed"; sleep 600; continue; }
    .venv/bin/python - << 'PYEOF' || { echo "INTEGRITY FAIL"; exit 1; }
from safetensors import safe_open
p = 'outputs/wtq_C_v5/checkpoint-1000/adapter_model.safetensors'
with safe_open(p, framework='pt') as f:
    ks = list(f.keys())
    zero = sum(1 for k in ks if float(f.get_tensor(k).float().norm()) == 0.0)
assert zero < len(ks) * 0.05, f"zero tensors {zero}/{len(ks)}"
print(f"adapter integrity OK ({len(ks)} tensors, {zero} zero)")
PYEOF
    UTILGPU=$GPU FREE=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $GPU | awk -F', ' '{printf "%.2f", ($1-$2)*0.9/$1}')
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/vllm serve Qwen/Qwen3-8B --port 8001 --host 127.0.0.1 \
      --chat-template-content-format string --max-model-len 8192 --gpu-memory-utilization ${FREE:-0.5} \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      --enable-lora --max-lora-rank 64 --lora-modules wtq_C5=outputs/wtq_C_v5/checkpoint-1000 \
      > /tmp/claude-1847/8001_C5g.log 2>&1 &
    SPID=$!
    for j in $(seq 1 90); do curl -s http://127.0.0.1:8001/v1/models > /dev/null && break; sleep 10; done
    .venv/bin/python scripts/wtq/zero_shot.py --base-url http://127.0.0.1:8001/v1 --model wtq_C5 \
      --arm wtq_clean_C5 --split clean-eval-devfold --limit 300 --concurrency 8 --hints cells
    kill $SPID
    echo "C5 GUARDED PIPELINE DONE"
    exit 0
  fi
  sleep 600
done
echo "C5 GAVE UP after 12h"
