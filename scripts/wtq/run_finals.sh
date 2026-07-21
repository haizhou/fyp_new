#!/bin/bash
set -e
cd /home/uceeh01/fyp_new/fyp_new
export HF_HOME=/var/tmp/cicada/hf
for ARM in A C; do
  CUDA_VISIBLE_DEVICES=2 .venv/bin/llamafactory-cli train configs/training/wtq_${ARM}_final_qlora.yaml
  test -f outputs/wtq_${ARM}_final/checkpoint-1000/adapter_model.safetensors
done
echo "FINALS TRAINED"
