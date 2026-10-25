#!/bin/bash
# Power probe after the second PCIe connector (2026-10-25): 135 vs 150 vs 165 W, production pr-all config.
cd /home/aim/llm-trade-bench || exit 1
U=/home/aim/opt/UnbiasedStrata/local
echo "$(date +%T) stopping production"; systemctl --user stop unbiasedstrata.service
for i in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1200 ] && break; sleep 2; done
LTB_MAX_VRAM_MB=32700 tools/guarded-run.sh v100tune-pl150 /usr/bin/python3 $U/v100_tune.py --base $U/v100/base-prod-pr1401.json \
  --candidates $U/v100/cands-pl150.json --output $U/benchmarks/v100-pl150 --short 2 --max-tokens 400 --long 80000 --rounds 2 \
  > $U/benchmarks/v100-pl150.log 2>&1
echo "$(date +%T) probe exit=$?"
sudo -n /usr/local/sbin/v100-tune rgc >/dev/null; sudo -n /usr/local/sbin/v100-tune pl 135 >/dev/null
sudo -n /usr/local/sbin/v100-tune show | sed -n 2p
sleep 5; systemctl --user start unbiasedstrata.service; echo "$(date +%T) PROD_RESTARTED"
