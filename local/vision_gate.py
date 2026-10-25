#!/usr/bin/python3
"""Vision gate for a launch profile: functional checks, scored perception, and the worst case that must fit.

Usage: vision_gate.py PROFILE.json OUTPUT_DIR
Starts a temporary service on 18081 from the profile (long_context.Server, like the tuning harness), samples VRAM
twice a second, and runs:
  1. test_vision.py (the functional image checks; its exit code is recorded, the run goes on)
  2. perception: the questions in vision-assets/questions.json on two 2560x1440 pictures (each over the 2048-token
     picture budget), scored by exact substrings
  3. worst case: both pictures plus the 180K long-context fixture in one request; passes when it answers with all
     three planted values and a short request afterwards still works (no out-of-memory, no stuck engine)
Writes OUTPUT_DIR/gate.json; exits 0 only when perception is complete and the worst case passes.
"""
import base64
import json
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
FIXTURE = Path('/home/aim/Work/v100-20261013/long-context-fixtures/fresh-180000.json')


def chat(messages, max_tokens, timeout):
    body = {'messages': messages, 'temperature': 0, 'reasoning_effort': 'none', 'max_tokens': max_tokens}
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d['choices'][0]['message'].get('content') or '', d.get('usage') or {}, time.monotonic() - t0


def image_part(name):
    data = (ASSETS / name).read_bytes()
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data).decode()}}


def vram_used():
    return int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True).strip())


def main():
    profile = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    total = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.total',
                                         '--format=csv,noheader,nounits'], text=True).strip())
    peak, stop = {'used': 0}, threading.Event()

    def sample():
        while not stop.is_set():
            try:
                peak['used'] = max(peak['used'], vram_used())
            except (subprocess.CalledProcessError, ValueError):
                pass
            time.sleep(0.5)

    gate = {'profile': sys.argv[1], 'vram_total_mib': total}
    with Server(profile, out) as s:
        gate['startup_seconds'] = round(s.startup, 1)
        gate['vram_after_load_mib'] = vram_used()
        th = threading.Thread(target=sample, daemon=True)
        th.start()
        try:
            r = subprocess.run([sys.executable, str(HERE / 'test_vision.py'), '--base', BASE],
                               capture_output=True, text=True, timeout=1800)
            gate['functional_rc'] = r.returncode
            (out / 'test_vision.out').write_text(r.stdout + r.stderr)
            print('functional checks rc', r.returncode, flush=True)

            answers = []
            for q in json.loads((ASSETS / 'questions.json').read_text()):
                content, usage, dt = chat([{'role': 'user', 'content': [
                    image_part(q['image']), {'type': 'text', 'text': q['question']}]}], 300, 900)
                ok = all(e in content for e in q['expect'])
                answers.append({**q, 'answer': content, 'ok': ok, 'seconds': round(dt, 1),
                                'prompt_tokens': usage.get('prompt_tokens')})
                print(f"[{'ok' if ok else 'MISS'} {dt:.1f}s {usage.get('prompt_tokens')} tok] {q['question'][:60]}"
                      f"\n   -> {content[:200]!r}", flush=True)
            gate['perception'] = answers
            gate['perception_score'] = f"{sum(a['ok'] for a in answers)}/{len(answers)}"

            fx = json.loads(FIXTURE.read_text())
            text = fx['messages'][-1]['content']
            worst = {'ok': False}
            try:
                content, usage, dt = chat([{'role': 'user', 'content': [
                    image_part('status.png'), image_part('code.png'), {'type': 'text', 'text': text}]}], 256, 1800)
                worst.update(seconds=round(dt, 1), prompt_tokens=usage.get('prompt_tokens'),
                             secrets_ok=all(str(v) in content for v in fx['secrets'].values()), tail=content[-300:])
                after, _, _ = chat([{'role': 'user', 'content': 'Reply with exactly: still alive'}], 16, 300)
                worst['after_ok'] = 'still alive' in after.lower()
                worst['ok'] = bool(worst['secrets_ok'] and worst['after_ok'])
            except Exception as e:  # an OOM or a dead engine shows up here
                worst['error'] = f'{type(e).__name__}: {e}'
            gate['worst_case'] = worst
            print('worst case:', json.dumps({k: v for k, v in worst.items() if k != 'tail'}), flush=True)
        finally:
            stop.set()
            th.join(timeout=3)
    gate['vram_peak_used_mib'] = peak['used']
    gate['vram_min_free_mib'] = total - peak['used']
    gate['pass'] = gate.get('perception_score', '').split('/')[0] == gate.get('perception_score', '/').split('/')[1] \
        and gate.get('worst_case', {}).get('ok', False)
    (out / 'gate.json').write_text(json.dumps(gate, indent=1) + '\n')
    print('GATE', 'PASS' if gate['pass'] else 'FAIL', 'perception', gate.get('perception_score'),
          'min free VRAM', gate['vram_min_free_mib'], 'MiB', flush=True)
    sys.exit(0 if gate['pass'] else 1)


if __name__ == '__main__':
    main()
