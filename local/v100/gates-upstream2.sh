#!/bin/bash
# Re-run of the upstream half (port race fixed). Adoption gates for upstream Strata 0.2.6 vs production at 150 W (2026-10-25): vision gate, then the 24-task suite.
cd /home/aim/llm-trade-bench || exit 1
U=/home/aim/opt/UnbiasedStrata/local; B=$U/benchmarks
PROD=$U/v100/base-prod-pr1401.json; UP=$U/v100/up-best-18081.json
echo "$(date +%T) stopping lilac agent + production"; systemctl --user stop lilac-node.service unbiasedstrata.service
for i in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1200 ] && break; sleep 2; done
sudo -n /usr/local/sbin/v100-tune rgc >/dev/null; sudo -n /usr/local/sbin/v100-tune pl 150 >/dev/null
step() {  # label, then the command; each step guarded on its own, the next runs whatever happened
  local label=$1; shift
  LTB_MAX_VRAM_MB=32700 tools/guarded-run.sh "$label" "$@" > "$B/$label.log" 2>&1
  echo "$(date +%T) $label exit=$?"
  systemctl --user stop 'unbiasedstrata-long-*' 2>/dev/null   # backstop: never leave a test unit behind
  for i in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 1200 ] && break; sleep 2; done
  port_free
}
port_free() {  # the next server must be able to bind 18081 (upstream's server gives up at once on EADDRINUSE)
  for i in $(seq 1 60); do
    python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 18081)); s.close()' 2>/dev/null && return 0
    sleep 2
  done
  echo "$(date +%T) port 18081 still busy after 120 s: $(ss -ltnp 2>/dev/null | grep ':18081 ')"
}
port_free
step v100-vgate-up2   /usr/bin/python3 $U/vision_gate.py $UP   $B/v100-vgate-up2
step v100-tasks-up026b  /usr/bin/python3 $U/task_window.py $UP   $B/v100-tasks-up026b  v100-up026 40
sudo -n /usr/local/sbin/v100-tune rgc >/dev/null; sudo -n /usr/local/sbin/v100-tune pl 150 >/dev/null
sleep 5; systemctl --user start unbiasedstrata.service lilac-node.service; echo "$(date +%T) PROD_RESTARTED"
