#!/bin/bash
# Window 6: CPU boost-cap sweep, 135 W probe, STRATA_ROUTE_RESIDENT=0.5 — sequential, production stopped meanwhile.
cd /home/aim/llm-trade-bench || exit 1
U=/home/aim/opt/UnbiasedStrata/local
T=/usr/bin/python3
run() { # label candidates rounds
  LTB_MAX_VRAM_MB=32700 tools/guarded-run.sh "v100tune-$1" $T $U/v100_tune.py --base $U/v100/base-prod-final.json \
    --candidates $U/v100/cands-$1.json --output $U/benchmarks/v100-$1 --short 2 --max-tokens 400 --long 80000 --rounds "$2" \
    > $U/benchmarks/v100-$1.log 2>&1; echo "$(date +%T) $1 exit=$?"
}
echo "$(date +%T) stopping production"
systemctl --user stop unbiasedstrata.service
for i in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1200 ] && break; sleep 2; done
echo "$(date +%T) production stopped (${m} MiB)"
run cpufreq 2
sudo -n /usr/local/sbin/v100-tune maxfreq reset
run pl135 1
sudo -n /usr/local/sbin/v100-tune pl 125 >/dev/null
run routeres 2
sudo -n /usr/local/sbin/v100-tune maxfreq reset >/dev/null; sudo -n /usr/local/sbin/v100-tune pl 125 >/dev/null
sleep 5; systemctl --user start unbiasedstrata.service; echo "$(date +%T) PROD_RESTARTED"
