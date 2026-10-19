#!/bin/bash
# 135 W soak: five back-to-back full 180K prefills (prompt cache defeated) on the production pr-all binary.
cd /home/aim/llm-trade-bench || exit 1
U=/home/aim/opt/UnbiasedStrata/local
echo "$(date +%T) stopping production"; systemctl --user stop unbiasedstrata.service
for i in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1200 ] && break; sleep 2; done
LTB_MAX_VRAM_MB=32700 tools/guarded-run.sh v100tune-soak135 /usr/bin/python3 $U/v100_tune.py --base $U/v100/base-prod-pr1401.json \
  --candidates $U/v100/cands-soak135.json --output $U/benchmarks/v100-soak135 --short 0 --max-tokens 700 \
  --long 180000 180000 180000 180000 180000 --salt-long --rounds 1 > $U/benchmarks/v100-soak135.log 2>&1
echo "$(date +%T) soak exit=$?"
sudo -n /usr/local/sbin/v100-tune pl 125 >/dev/null; sudo -n /usr/local/sbin/v100-tune show | head -2 | tail -1
sleep 5; systemctl --user start unbiasedstrata.service; echo "$(date +%T) PROD_RESTARTED"
