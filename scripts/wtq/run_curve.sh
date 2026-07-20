#!/bin/bash
# Learning-curve x init-control pipeline (GPU2). Marks: 100 200 500 1000 2000.
# Quota guard: non-mark checkpoints deleted as superseded; after each init's
# marks are evaluated, only checkpoint-2000 is kept.
set -u
cd /home/uceeh01/fyp_new/fyp_new
MARKS="100 200 500 1000 2000"
export HF_HOME=/var/tmp/cicada/hf

# free GPU2: stop the idle 8001 server if present
P=$(ss -tlnp 2>/dev/null | grep :8001 | grep -oP 'pid=\K[0-9]+' | head -1)
[ -n "${P:-}" ] && kill "$P" && sleep 25

for INIT in base v3init; do
  OUT=outputs/wtq_curve_${INIT}
  CUDA_VISIBLE_DEVICES=2 .venv/bin/llamafactory-cli train configs/training/wtq_curve_${INIT}.yaml &
  TPID=$!
  while kill -0 $TPID 2>/dev/null; do
    sleep 60
    for d in $OUT/checkpoint-*; do
      [ -d "$d" ] || continue
      n=${d##*-}
      case " $MARKS " in *" $n "*) continue;; esac
      nxt=$((n+100))
      [ -d "$OUT/checkpoint-$nxt" ] && rm -rf "$d"
    done
  done
  wait $TPID
  for d in $OUT/checkpoint-*; do
    n=${d##*-}
    case " $MARKS " in *" $n "*) ;; *) rm -rf "$d";; esac
  done

  LORAS=""
  for n in $MARKS; do
    [ -d "$OUT/checkpoint-$n" ] && LORAS="$LORAS wtq_${INIT}_$n=$OUT/checkpoint-$n"
  done
  CUDA_VISIBLE_DEVICES=2 .venv/bin/vllm serve Qwen/Qwen3-8B --port 8001 --host 127.0.0.1 \
    --chat-template-content-format string --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-lora --max-lora-rank 64 --lora-modules $LORAS \
    > /tmp/claude-1847/8001_curve_${INIT}.log 2>&1 &
  SPID=$!
  for i in $(seq 1 60); do
    curl -s http://127.0.0.1:8001/v1/models > /dev/null && break
    sleep 10
  done
  for n in $MARKS; do
    .venv/bin/python scripts/wtq/zero_shot.py --base-url http://127.0.0.1:8001/v1 \
      --model wtq_${INIT}_$n --arm wtq_curve_${INIT}_$n \
      --split clean-eval-devfold --limit 300 --concurrency 8 2>&1 | tail -2
  done
  kill $SPID; sleep 20
  for n in 100 200 500 1000; do rm -rf "$OUT/checkpoint-$n"; done
done
echo "CURVE PIPELINE DONE"
