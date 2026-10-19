#!/bin/bash
cd /home/aim/llm-trade-bench || exit 1
U=/home/aim/opt/UnbiasedStrata/local
echo "$(date +%T) stopping production"; systemctl --user stop unbiasedstrata.service
for i in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1200 ] && break; sleep 2; done
LTB_MAX_VRAM_MB=32700 tools/guarded-run.sh v100tune-round7 /usr/bin/python3 $U/v100_tune.py --base $U/v100/base-prod-pr1401.json \
  --candidates $U/v100/cands-round7b.json --output $U/benchmarks/v100-round7c --short 2 --max-tokens 400 --long 80000 --rounds 2 > $U/benchmarks/v100-round7c.log 2>&1
echo "$(date +%T) round7 exit=$?"
sudo -n /usr/local/sbin/v100-tune rgc >/dev/null; sudo -n /usr/local/sbin/v100-tune pl 135 >/dev/null
sleep 5; systemctl --user start unbiasedstrata.service; echo "$(date +%T) PROD_RESTARTED"
