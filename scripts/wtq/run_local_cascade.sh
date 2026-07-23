#!/bin/bash
set -e
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd /home/uceeh01/fyp_new/fyp_new
export HF_HOME=/var/tmp/cicada/hf
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/vllm serve Qwen/Qwen3-8B --port 8001 --host 127.0.0.1 \
  --chat-template-content-format string --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-lora --max-lora-rank 64 \
  --lora-modules cicada-qwen3-composev3=outputs/qwen3_8b_compose_sft_v3 wtq_C5=outputs/wtq_C_v5/checkpoint-1000 \
  > /tmp/claude-1847/8001_reharvest_v4b.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do curl -s http://127.0.0.1:8001/v1/models > /dev/null && break; sleep 10; done

# Stage 1: A-pool reharvest under v4b views (composev3 sampler, arm-pure)
for S in 0 4000 8000; do
  .venv/bin/python scripts/wtq/harvest_a.py --base-url http://127.0.0.1:8001/v1 \
    --outtag Av4b --start $S --limit 4000 --concurrency 6
done
.venv/bin/python scripts/wtq/harvest_a.py --base-url http://127.0.0.1:8001/v1 \
  --dev-fold --outtag devv4b --start 0 --limit 3000 --concurrency 6

# Stage 2: extract fresh zero-hit ids
.venv/bin/python - << 'PYEOF'
import glob, json
zero = []
for f in sorted(glob.glob('data/qa/wtq/harvest_Av4b_*.jsonl')):
    for line in open(f):
        r = json.loads(line)
        if r.get('status') == 'ok' and r.get('hits', 1) == 0:
            zero.append(r['id'])
open('/tmp/claude-1847/zerohit_v4b.ids', 'w').write('\n'.join(zero))
print(f"zero-hit after v4b reharvest: {len(zero)}")
PYEOF

# Stage 3: C-pool free growth — C-v5 self-harvest on the zero-hit residue only
.venv/bin/python scripts/wtq/harvest_a.py --base-url http://127.0.0.1:8001/v1 \
  --model wtq_C5 --ids-file /tmp/claude-1847/zerohit_v4b.ids \
  --outtag C5self --start 0 --limit 99999 --concurrency 6

# Stage 4: residue for the paid teacher stages
.venv/bin/python - << 'PYEOF'
import glob, json
c5win = set()
for f in sorted(glob.glob('data/qa/wtq/harvest_C5self_*.jsonl')):
    for line in open(f):
        r = json.loads(line)
        if r.get('status') == 'ok' and r.get('hits', 0) > 0:
            c5win.add(r['id'])
zero = [l.strip() for l in open('/tmp/claude-1847/zerohit_v4b.ids') if l.strip()]
residue = [i for i in zero if i not in c5win]
open('/tmp/claude-1847/teacher_pool.ids', 'w').write('\n'.join(residue))
print(f"C5 rescued {len(c5win)}; teacher pool (residue): {len(residue)}")
PYEOF

# Stage 5: holdout local B-arm matrix (uses the same served wtq_C5)
.venv/bin/python scripts/wtq/holdout_local_barm.py http://127.0.0.1:8001/v1 || echo "holdout barm failed (non-fatal)"

kill $SPID
echo "LOCAL CASCADE DONE"
