#!/bin/bash
set -e
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd /home/uceeh01/fyp_new/fyp_new
export HF_HOME=/var/tmp/cicada/hf
CUDA_VISIBLE_DEVICES=2 .venv/bin/llamafactory-cli train configs/training/wtq_C_v4_qlora.yaml
test -f outputs/wtq_C_v4/checkpoint-1000/adapter_model.safetensors
CUDA_VISIBLE_DEVICES=2 .venv/bin/vllm serve Qwen/Qwen3-8B --port 8001 --host 127.0.0.1 \
  --chat-template-content-format string --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-lora --max-lora-rank 64 --lora-modules wtq_C4=outputs/wtq_C_v4/checkpoint-1000 \
  > /tmp/claude-1847/8001_C4.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do curl -s http://127.0.0.1:8001/v1/models > /dev/null && break; sleep 10; done
.venv/bin/python scripts/wtq/zero_shot.py --base-url http://127.0.0.1:8001/v1 --model wtq_C4 --hints cells \
  --arm wtq_clean_C4 --split clean-eval-devfold --limit 300 --concurrency 8
kill $SPID
echo "C4 PIPELINE DONE"
