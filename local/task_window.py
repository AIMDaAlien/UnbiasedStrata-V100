#!/usr/bin/python3
"""Start a temporary service from a profile on 18081 and run the frozen 24-task coding suite against it."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from long_context import Server
import signal
# guarded-run.sh stops with SIGTERM: exit normally so Server.__exit__ stops the test unit
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
profile = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
tag = sys.argv[3]; seed = sys.argv[4]
with Server(profile, out) as s:
    print('startup', round(s.startup, 1), 's', flush=True)
    rc = subprocess.call([sys.executable, 'bench-run.py', '--targets', 'v100-flash-medium', '--packs', 'deepswe2', '--tag', tag,
                          '--seed', seed, '--thinking', 'medium', '--suite-version', 'contract-v2', '--include-fixtures'],
                         cwd='/home/aim/Work/v100-20261013/bench')
    print('BENCH_RC', rc, flush=True)
sys.exit(rc)
