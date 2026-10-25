#!/bin/bash
# Smoke: do the upstream 0.2.6 candidates load and answer under the harness? (1 short request each, no long prompt)
cd /home/aim/llm-trade-bench || exit 1
U=/home/aim/opt/UnbiasedStrata/local
echo "$(date +%T) stopping lilac agent + production"; systemctl --user stop lilac-node.service unbiasedstrata.service
for i in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1200 ] && break; sleep 2; done
LTB_MAX_VRAM_MB=32700 tools/guarded-run.sh v100tune-upsmoke2 /usr/bin/python3 $U/v100_tune.py --base $U/v100/base-prod-pr1401.json \
  --candidates $U/v100/cands-upstream.json --only up-matched --output $U/benchmarks/v100-upstream-smoke2 \
  --short 1 --max-tokens 200 --rounds 1 > $U/benchmarks/v100-upstream-smoke2.log 2>&1
echo "$(date +%T) smoke exit=$?"
systemctl --user stop 'unbiasedstrata-long-*' 2>/dev/null   # backstop: a guard stop must not leave a test unit
sudo -n /usr/local/sbin/v100-tune rgc >/dev/null; sudo -n /usr/local/sbin/v100-tune pl 150 >/dev/null
sleep 5; systemctl --user start unbiasedstrata.service lilac-node.service; echo "$(date +%T) PROD_RESTARTED"
