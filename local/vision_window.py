#!/usr/bin/python3
"""Start a temporary service from a profile, report VRAM before/after the encoder, run test_vision.py, stop."""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from long_context import Server, BASE
profile = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
def vram(): return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
with Server(profile, out) as s:
    print('startup', round(s.startup,1), 's; VRAM used/total MiB after load:', vram(), flush=True)
    log = (out/'engine.log').read_text()
    for l in log.splitlines():
        if 'VRAM free' in l or 'vision encoder ready' in l or 'expert cache ' in l and 'slots' in l: print(' ', l[:160])
    rc = subprocess.call([sys.executable, str(Path(__file__).resolve().parent/'test_vision.py'), '--base', BASE] + sys.argv[3:])
    print('VRAM used/total MiB after tests:', vram())
    print('TEST_RC', rc, flush=True)
    (out/'engine.log.tail').write_text('\n'.join((out/'engine.log').read_text().splitlines()[-40:]))
sys.exit(rc)
