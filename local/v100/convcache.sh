#!/bin/bash
# Conversation cache A/B (2 rounds, off vs cache4g) in one window; production and the lilac agent are paused.
cd /home/aim/llm-trade-bench || exit 1
U=/home/aim/opt/UnbiasedStrata/local
echo "$(date +%T) stopping lilac agent + production"; systemctl --user stop lilac-node.service unbiasedstrata.service
for i in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1200 ] && break; sleep 2; done
LTB_MAX_VRAM_MB=32700 tools/guarded-run.sh v100-convcache /usr/bin/python3 $U/conv_cache_test.py $U/v100/base-prod-pr1401.json \
  $U/v100/cands-convcache.json $U/benchmarks/v100-convcache --rounds 2 > $U/benchmarks/v100-convcache.log 2>&1
echo "$(date +%T) convcache exit=$?"
systemctl --user stop 'unbiasedstrata-long-*' 2>/dev/null   # backstop: no test unit may outlive the window
sudo -n /usr/local/sbin/v100-tune rgc >/dev/null; sudo -n /usr/local/sbin/v100-tune pl 150 >/dev/null
sleep 5; systemctl --user start unbiasedstrata.service lilac-node.service; echo "$(date +%T) PROD_RESTARTED"
