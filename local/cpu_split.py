#!/usr/bin/python3
"""Where does the CPU time go during one long prefill on the live service? Samples the engine's threads
(user vs system time from /proc) and the whole system's irq/softirq share every second while an 80K request runs."""
import json, subprocess, sys, threading, time, urllib.request
from pathlib import Path
BASE = 'http://127.0.0.1:8083'
pid = int(subprocess.check_output(['systemctl', '--user', 'show', 'unbiasedstrata.service', '-p', 'MainPID', '--value']))
kids = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
engine = int(kids[0]) if kids else pid
HZ = 100
def proc_times(p):
    u = s = 0
    for t in Path(f'/proc/{p}/task').iterdir():
        try: f = (t / 'stat').read_text().rsplit(')', 1)[1].split(); u += int(f[11]); s += int(f[12])
        except OSError: pass
    return u / HZ, s / HZ
def sys_times():
    f = Path('/proc/stat').read_text().splitlines()[0].split()[1:]
    f = [int(x) for x in f]; return {'user': f[0] + f[1], 'system': f[2], 'idle': f[3], 'iowait': f[4], 'irq': f[5], 'softirq': f[6]}
fx = json.loads(Path('/home/aim/Work/v100-20261013/long-context-fixtures/fresh-80000.json').read_text())
msgs = [dict(fx['messages'][0], content=f'Run cpusplit-{time.time_ns()}.\n' + fx['messages'][0]['content'])]
samples = []; stop = threading.Event()
def sampler():
    while not stop.is_set():
        samples.append((time.time(), proc_times(engine), sys_times())); stop.wait(1.0)
th = threading.Thread(target=sampler, daemon=True); th.start()
e0 = Path('/sys/class/powercap/intel-rapl:0/energy_uj').read_text()
t0 = time.monotonic()
req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps({'messages': msgs, 'temperature': 0, 'reasoning_effort': 'none', 'max_tokens': 64}).encode(), {'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=600) as r: out = json.load(r)
wall = time.monotonic() - t0; stop.set(); th.join()
e1 = Path('/sys/class/powercap/intel-rapl:0/energy_uj').read_text()
(u0, s0), (u1, s1) = samples[0][1], samples[-1][1]
a, b = samples[0][2], samples[-1][2]; tot = sum(b[k] - a[k] for k in a)
print(f'wall {wall:.1f}s prompt_tokens {out["usage"]["prompt_tokens"]} CPU package {(int(e1)-int(e0))/1e6/wall:.0f} W mean')
print(f'engine process: user {u1-u0:.1f} s  system {s1-s0:.1f} s  -> {(u1-u0)/wall:.2f} user cores, {(s1-s0)/wall:.2f} system cores busy')
print('system-wide shares: ' + ', '.join(f'{k} {100*(b[k]-a[k])/tot:.1f}%' for k in ('user', 'system', 'irq', 'softirq', 'iowait', 'idle')))
