#!/bin/bash
set -e
cd /home/uceeh01/fyp_new/fyp_new
export HF_HOME=/var/tmp/cicada/hf
CUDA_VISIBLE_DEVICES=2 .venv/bin/vllm serve Qwen/Qwen3-8B --port 8001 --host 127.0.0.1 \
  --chat-template-content-format string --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-lora --max-lora-rank 64 \
  --lora-modules cicada-qwen3-composev3=outputs/qwen3_8b_compose_sft_v3 \
  > /tmp/claude-1847/8001_devharvest.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do curl -s http://127.0.0.1:8001/v1/models > /dev/null && break; sleep 10; done
.venv/bin/python scripts/wtq/harvest_a.py --base-url http://127.0.0.1:8001/v1 \
  --dev-fold --start 0 --limit 3000 --concurrency 6
kill $SPID
echo "DEV HARVEST DONE"
