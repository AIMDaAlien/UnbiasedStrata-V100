#!/usr/bin/python3
"""Conversation cache A/B: does a long chat survive side requests between its turns?

Usage: conv_cache_test.py PROFILE.json CANDIDATES.json OUTPUT_DIR [--rounds N]
CANDIDATES.json: [{"name": ..., "engine_args": [...extra flags appended to the profile's]}, ...]
Each candidate gets its own temporary service on 18081 (long_context.Server). The scenario is the companion's:
  A1  a long chat turn: one 2560x1440 screenshot plus the 80K fixture (three planted secrets)
  B   a side request: a different screenshot, a short question (a presence/screen check)
  C   a side request: text only (a chat-title job)
  A2  A's next turn: the whole history again plus a question about the first screenshot and a secret
  A3  A's next turn right away (no side request in between: the prompt cache's job, for reference)
Records wall time, the engine's own "N reused + M read" line, RAM (MemAvailable low, unit peak) and VRAM low for
every request, and whether every answer has its expected strings. Writes OUTPUT_DIR/results.json.
"""
import argparse
import base64
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from long_context import BASE, Server  # noqa: E402
import signal
# guarded-run.sh stops with SIGTERM: exit normally so Server.__exit__ stops the test unit
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'vision-assets'
FIXTURE = Path('/home/aim/Work/v100-20261013/long-context-fixtures/fresh-80000.json')
PROMPT_LINE = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms')


def chat(messages, max_tokens, timeout=1800):
    body = {'messages': messages, 'temperature': 0, 'reasoning_effort': 'none', 'max_tokens': max_tokens}
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d['choices'][0]['message'].get('content') or '', d.get('usage') or {}, time.monotonic() - t0


def image(name):
    data = (ASSETS / name).read_bytes()
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data).decode()}}


def mem_available_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024


def vram_used_mib():
    return int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True).strip())


def unit_mib(unit):
    v = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'MemoryCurrent', '--value'], text=True)
    return int(v) // 2**20 if v.strip().isdigit() else 0


class Sampler:
    """MemAvailable low, the test unit's peak and the VRAM peak, twice a second, reset per request."""
    def __init__(self, unit):
        self.unit, self.stop, self.lock = unit, threading.Event(), threading.Lock()
        self.reset()
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    def reset(self):
        with self.lock:
            self.low_avail, self.peak_unit, self.peak_vram = 10**9, 0, 0

    def run(self):
        while not self.stop.is_set():
            try:
                a, u, v = mem_available_mib(), unit_mib(self.unit), vram_used_mib()
                with self.lock:
                    self.low_avail, self.peak_unit = min(self.low_avail, a), max(self.peak_unit, u)
                    self.peak_vram = max(self.peak_vram, v)
            except (subprocess.CalledProcessError, ValueError, OSError):
                pass
            time.sleep(0.5)

    def take(self):
        with self.lock:
            out = {'ram_available_low_mib': self.low_avail, 'unit_peak_mib': self.peak_unit,
                   'vram_peak_mib': self.peak_vram}
        self.reset()
        return out


def scenario(server, out, sampler):
    log = out / 'engine.log'
    fx = json.loads(FIXTURE.read_text())
    sec = fx['secrets']
    steps = []

    def run(label, messages, max_tokens, expect):
        offset = log.stat().st_size
        sampler.take()
        content, usage, dt = chat(messages, max_tokens)
        time.sleep(1.0)   # the engine writes its prompt line as the reply finishes
        new = log.read_text(errors='replace')[offset:]
        m = PROMPT_LINE.findall(new)
        cache = [l.split('conversation cache: ', 1)[1] for l in new.splitlines() if 'conversation cache: ' in l]
        step = {'step': label, 'seconds': round(dt, 1), 'prompt_tokens': usage.get('prompt_tokens'),
                'reused': int(m[-1][1]) if m else None, 'read': int(m[-1][2]) if m else None,
                'prompt_ms': int(m[-1][3]) if m else None, 'cache_lines': cache,
                'ok': all(e in content for e in expect), 'expect': expect, 'answer': content[-400:],
                **sampler.take()}
        steps.append(step)
        print(f"  {label}: {dt:6.1f}s  {step['reused']} reused + {step['read']} read  "
              f"{'ok' if step['ok'] else 'MISS'}  RAM low {step['ram_available_low_mib']} MiB  {cache}", flush=True)
        return content

    a1_user = {'role': 'user', 'content': [image('status.png'), {'type': 'text', 'text': fx['messages'][-1]['content']}]}
    a1 = run('A1 long turn', [a1_user], 400, list(sec.values()))
    run('B side screenshot', [{'role': 'user', 'content': [
        image('code.png'), {'type': 'text', 'text': 'Transcribe the red error message at the bottom exactly.'}]}],
        100, ['ERROR 0x5A3F'])
    run('C side title', [{'role': 'user', 'content': 'Write a four-word title for a chat about retry backoff. '
                          'Reply with the title only.'}], 16, [])
    hist = [a1_user, {'role': 'assistant', 'content': a1}]
    q2 = {'role': 'user', 'content': 'In the screenshot from my first message, which node is inside the highlighted '
                                     'frame? Then repeat the alpha secret value exactly.'}
    a2 = run('A2 after side requests', hist + [q2], 80, ['unraid-gpu', sec['alpha']])
    q3 = {'role': 'user', 'content': 'Now repeat the beta secret value exactly, nothing else.'}
    run('A3 straight follow-up', hist + [q2, {'role': 'assistant', 'content': a2}, q3], 40, [sec['beta']])
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('profile'); ap.add_argument('candidates'); ap.add_argument('output')
    ap.add_argument('--rounds', type=int, default=1)
    a = ap.parse_args()
    base = json.loads(Path(a.profile).read_text())
    cands = json.loads(Path(a.candidates).read_text())
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for rnd in range(a.rounds):
        order = cands if rnd % 2 == 0 else cands[::-1]   # alternate the order so warm-up drift cancels
        for c in order:
            profile = {**base, 'engine_args': base.get('engine_args', []) + c.get('engine_args', [])}
            out = root / f"r{rnd + 1}-{c['name']}"
            print(f"round {rnd + 1} {c['name']}: {' '.join(c.get('engine_args', [])) or '(production)'}", flush=True)
            with Server(profile, out) as s:
                sampler = Sampler(s.unit)
                try:
                    steps = scenario(s, out, sampler)
                finally:
                    sampler.stop.set()
                    sampler.th.join(timeout=3)
            res = {'round': rnd + 1, 'candidate': c['name'], 'engine_args': c.get('engine_args', []),
                   'startup_seconds': round(s.startup, 1), 'steps': steps}
            (out / 'result.json').write_text(json.dumps(res, indent=1) + '\n')
            results.append(res)
            (root / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    print('DONE', root, flush=True)


if __name__ == '__main__':
    main()
