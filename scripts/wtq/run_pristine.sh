#!/bin/bash
# PRISTINE-UNSEEN FINAL RUN — one shot per frozen config. Do not edit after launch.
set -e
cd /home/uceeh01/fyp_new/fyp_new
export HF_HOME=/var/tmp/cicada/hf
echo "commit: $(git rev-parse HEAD)" | tee /tmp/claude-1847/pristine_manifest.txt
sha256sum scripts/wtq/zero_shot.py scripts/wtq/loader.py scripts/wtq/linker.py scripts/wtq/wtq_eval.py \
  src/procurement_graph/compose/algebra.py src/procurement_graph/compose/eval_runtime.py \
  src/procurement_graph/compose/schema.py | tee -a /tmp/claude-1847/pristine_manifest.txt
CUDA_VISIBLE_DEVICES=2 .venv/bin/vllm serve Qwen/Qwen3-8B --port 8001 --host 127.0.0.1 \
  --chat-template-content-format string --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-lora --max-lora-rank 64 --max-loras 3 \
  --lora-modules cicada-qwen3-composev3=outputs/qwen3_8b_compose_sft_v3 \
                 wtq_A_final=outputs/wtq_A_final/checkpoint-1000 \
                 wtq_C_final=outputs/wtq_C_final/checkpoint-1000 \
  > /tmp/claude-1847/8001_pristine.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do curl -s http://127.0.0.1:8001/v1/models > /dev/null && break; sleep 10; done
for SPEC in "Qwen/Qwen3-8B pristine_base" "cicada-qwen3-composev3 pristine_v3" \
            "wtq_A_final pristine_A" "wtq_C_final pristine_C"; do
  set -- $SPEC
  .venv/bin/python scripts/wtq/zero_shot.py --base-url http://127.0.0.1:8001/v1 --model $1 \
    --arm $2 --split pristine-unseen-tables --limit 5000 --concurrency 8 --hints cells
done
kill $SPID
for ARM in pristine_base pristine_v3 pristine_A pristine_C; do
  .venv/bin/python - << PYEOF
import json
rows = [json.loads(l) for l in open('data/qa/wtq/eval_${ARM}.jsonl')]
with open('/tmp/claude-1847/pred_${ARM}.tsv','w') as fh:
    for r in rows:
        items = r.get('answer_items', []) if r['outcome']=='answered' else []
        fh.write('\t'.join([r['id']] + [i.replace('\t',' ').replace('\n',' ') for i in items]) + '\n')
PYEOF
  echo "=== ${ARM} (official) ==="
  .venv/bin/python scripts/wtq/official_evaluator.py \
    -t /var/tmp/cicada/wtq/WikiTableQuestions/tagged/data /tmp/claude-1847/pred_${ARM}.tsv 2>&1 >/dev/null | tail -3
done
echo "PRISTINE RUN COMPLETE"
