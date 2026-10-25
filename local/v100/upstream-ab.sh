#!/bin/bash
# A/B: production (0.1.43.1 + Volta PR) vs upstream Strata 0.2.6 (engine 0.1.50) in our memory mode, 150 W, 2 rounds, 80K + 128K.
cd /home/aim/llm-trade-bench || exit 1
U=/home/aim/opt/UnbiasedStrata/local
echo "$(date +%T) stopping lilac agent + production"; systemctl --user stop lilac-node.service unbiasedstrata.service
for i in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1200 ] && break; sleep 2; done
LTB_MAX_VRAM_MB=32700 tools/guarded-run.sh v100tune-upab /usr/bin/python3 $U/v100_tune.py --base $U/v100/base-prod-pr1401.json \
  --candidates $U/v100/cands-upstream-ab.json --output $U/benchmarks/v100-upstream-ab \
  --short 1 --max-tokens 400 --long 80000 128000 --rounds 2 > $U/benchmarks/v100-upstream-ab.log 2>&1
echo "$(date +%T) ab exit=$?"
systemctl --user stop 'unbiasedstrata-long-*' 2>/dev/null   # backstop: a guard stop must not leave a test unit
sudo -n /usr/local/sbin/v100-tune rgc >/dev/null; sudo -n /usr/local/sbin/v100-tune pl 150 >/dev/null
sleep 5; systemctl --user start unbiasedstrata.service lilac-node.service; echo "$(date +%T) PROD_RESTARTED"
