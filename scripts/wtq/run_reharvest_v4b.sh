#!/bin/bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd /home/uceeh01/fyp_new/fyp_new
for i in $(seq 1 72); do
  GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2 < 39000 && $3 <= 60 {print $1; exit}')
  echo "poll $i: gpu=${GPU:-none}"
  if [ -n "$GPU" ]; then
    if dd if=/dev/zero of="$HOME/.qprobe" bs=64M count=20 conv=fsync 2>/dev/null; then
      rm -f "$HOME/.qprobe"; echo "quota probe OK"
    else
      rm -f "$HOME/.qprobe"; echo "quota probe FAILED, waiting"; sleep 600; continue
    fi
    export GPU
    bash /tmp/claude-1847/rh_core.sh && echo "V4B REHARVEST DONE" && exit 0
    echo "reharvest run failed, retrying later"; sleep 600
  else
    sleep 600
  fi
done
echo "GAVE UP after 12h"
