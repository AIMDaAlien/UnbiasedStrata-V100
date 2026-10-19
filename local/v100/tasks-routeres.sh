#!/bin/bash
cd /home/aim/llm-trade-bench || exit 1
U=/home/aim/opt/UnbiasedStrata/local
echo "$(date +%T) stopping production"; systemctl --user stop unbiasedstrata.service
for i in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1200 ] && break; sleep 2; done
LTB_MAX_VRAM_MB=32700 tools/guarded-run.sh v100-tasks-routeres /usr/bin/python3 $U/task_window.py $U/v100/prod-routeres-18081.json $U/benchmarks/v100-tasks-routeres v100-routeres-0.5 40 > $U/benchmarks/v100-tasks-routeres.log 2>&1
echo "$(date +%T) tasks exit=$?"; sleep 5; systemctl --user start unbiasedstrata.service; echo "$(date +%T) PROD_RESTARTED"
