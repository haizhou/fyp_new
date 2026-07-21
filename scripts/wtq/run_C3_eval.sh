#!/bin/bash
set -e
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd /home/uceeh01/fyp_new/fyp_new
export HF_HOME=/var/tmp/cicada/hf
CUDA_VISIBLE_DEVICES=2 .venv/bin/vllm serve Qwen/Qwen3-8B --port 8001 --host 127.0.0.1 \
  --chat-template-content-format string --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-lora --max-lora-rank 64 --lora-modules wtq_C3=outputs/wtq_C_v3_final \
  > /tmp/claude-1847/8001_C3e.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do curl -s http://127.0.0.1:8001/v1/models > /dev/null && break; sleep 10; done
curl -s http://127.0.0.1:8001/v1/models | grep -q wtq_C3
.venv/bin/python scripts/wtq/zero_shot.py --base-url http://127.0.0.1:8001/v1 --model wtq_C3 \
  --arm wtq_clean_C3 --split clean-eval-devfold --limit 300 --concurrency 8 --hints cells
kill $SPID
echo "C3 EVAL DONE"
