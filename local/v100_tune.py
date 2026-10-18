#!/usr/bin/python3
"""V100 tuning harness: paired A/B of launch profiles, with energy per token.

Each candidate is the base profile plus overrides (config keys, STRATA_* environment,
and optional root-helper commands for clock/power/EPP changes).  Candidates run in
alternating order per round, each in its own temporary user service (56 GiB cap, no
swap), with a warm-up request first.  Every measured request records the engine's own
prompt/decode line, wall time, GPU power/clock/temperature samples and CPU package
energy (RAPL), so the summary gives tokens per second and joules per token.

Never touches production: refuses to run while unbiasedstrata.service is active.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from long_context import BASE, Server  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURES = Path('/home/aim/Work/v100-20261013/long-context-fixtures')
RAPL = Path('/sys/class/powercap/intel-rapl:0')
MAX_GPU_C = 80
MIN_RAM_MB = 4096
PROMPT_RE = re.compile(
    r'strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), '
    r'(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)')
HIT_RE = re.compile(r'decode expert cache hit rate: ([\d.]+)%')

# Eight distinct coding-agent style requests (~600-900 prompt tokens each, medium reasoning).
TASKS = [
    ('rate limiter', 'a token-bucket rate limiter class with refill, a blocking acquire with timeout, and thread safety'),
    ('lru cache', 'an LRU cache with TTL expiry per entry, O(1) get/put, and a stats() method reporting hits and misses'),
    ('csv repair', 'a function that repairs a CSV with inconsistent quoting and column counts, logging each repaired row'),
    ('interval merge', 'a function merging overlapping closed integer intervals, plus a generator yielding the gaps'),
    ('retry decorator', 'a decorator with exponential backoff, jitter, a max-attempts limit and per-exception policies'),
    ('config layering', 'a config loader layering defaults, a TOML file and environment variables with type coercion'),
    ('event scheduler', 'a priority event scheduler supporting cancellation, recurring events and a drain() method'),
    ('log digest', 'a log digester grouping lines by error signature with first/last seen timestamps and counts'),
]
PREAMBLE = (
    'You are a careful senior Python engineer working in a repository with these conventions: '
    'type hints everywhere, docstrings in Google style, no third-party dependencies, pytest tests '
    'beside the module, functions under 40 lines, errors raised as custom exception classes, and '
    'logging through the module logger. The repository layout is src/pkg/, tests/, pyproject.toml. '
    'Existing modules: src/pkg/errors.py defines PkgError; src/pkg/log.py exposes get_logger(name). '
    'The CI runs ruff, mypy --strict and pytest -q. Keep public APIs small and documented. '
)


def short_messages(i, salt):
    name, spec = TASKS[i % len(TASKS)]
    filler = ' '.join(f'Note {k}: requirement {salt}-{i}-{k} must be preserved and tested.' for k in range(40))
    return [{'role': 'system', 'content': PREAMBLE},
            {'role': 'user', 'content': f'Ticket {salt}-{i} ({name}). {filler}\n\nImplement {spec}. '
             'Write the module and a focused pytest file. Explain design choices briefly at the end.'}]


def rapl_energy_uj():
    try:
        return int((RAPL / 'energy_uj').read_text())
    except OSError:
        return None


def rapl_range_uj():
    try:
        return int((RAPL / 'max_energy_range_uj').read_text())
    except OSError:
        return 1 << 62


def gpu_query():
    out = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-gpu=power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used,power.limit',
        '--format=csv,noheader,nounits'], text=True, timeout=5)
    w, sm, mem, t, util, vram, lim = (float(x) for x in out.strip().split(','))
    return {'watts': w, 'sm_mhz': sm, 'mem_mhz': mem, 'temp_c': t, 'util': util, 'vram_mib': vram, 'limit_w': lim}


def sampler(stop, samples, period=0.5):
    while not stop.is_set():
        try:
            s = gpu_query()
            s['time'] = time.time()
            samples.append(s)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        stop.wait(period)


def ram_available_mb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) // 1024
    return 0


def guard():
    g = gpu_query()
    if g['temp_c'] >= MAX_GPU_C:
        raise RuntimeError(f'GPU {g["temp_c"]} C >= {MAX_GPU_C} C: stopping')
    if ram_available_mb() < MIN_RAM_MB:
        raise RuntimeError('RAM available below floor: stopping')
    return g


def integrate(samples, t0, t1):
    pts = [s for s in samples if t0 <= s['time'] <= t1]
    if len(pts) < 2:
        return {}
    joules = 0.0
    for a, b in zip(pts, pts[1:]):
        joules += 0.5 * (a['watts'] + b['watts']) * (b['time'] - a['time'])
    def mean(k): return statistics.fmean(s[k] for s in pts)
    return {'gpu_joules': joules, 'gpu_watts_mean': mean('watts'), 'gpu_watts_max': max(s['watts'] for s in pts),
            'sm_mhz_mean': mean('sm_mhz'), 'sm_mhz_min': min(s['sm_mhz'] for s in pts),
            'temp_max': max(s['temp_c'] for s in pts), 'util_mean': mean('util'), 'samples': len(pts)}


def stream_request(payload, timeout=7200):
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(payload).encode(),
                                 {'Content-Type': 'application/json'})
    started = time.monotonic()
    ttft = None
    content = ''
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b'data: '):
                continue
            body = line[6:].strip()
            if body == b'[DONE]':
                break
            event = json.loads(body)
            if 'error' in event:
                raise RuntimeError(str(event['error']))
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                if delta.get('content') and ttft is None:
                    ttft = time.monotonic() - started
                content += delta.get('content') or ''
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
            if 'usage' in event:
                usage = event['usage']
    return {'wall_seconds': time.monotonic() - started, 'ttft_seconds': ttft, 'usage': usage,
            'finish_reason': finish, 'content_chars': len(content), 'content_tail': content[-200:], 'content': content}


def engine_lines(log, offset):
    data = log.read_bytes()[offset:]
    return data.decode(errors='replace').splitlines(), len(log.read_bytes())


def parse_engine(lines):
    out = {}
    for line in lines:
        m = PROMPT_RE.search(line)
        if m:
            v = m.groups()
            out.update(prompt_tokens=int(v[0]), reused=int(v[1]), read=int(v[2]), prefill_ms=int(v[3]),
                       prefill_tps=float(v[4]), generated=int(v[5]), decode_ms=int(v[6]), decode_tps=float(v[7]),
                       drafts_accepted=int(v[8]), drafts_offered=int(v[9]))
        m = HIT_RE.search(line)
        if m:
            out['hit_rate'] = float(m.group(1))
    return out


def measure(name, payload, log, offset, samples):
    guard()
    e0, t0 = rapl_energy_uj(), time.time()
    res = stream_request(payload)
    t1, e1 = time.time(), rapl_energy_uj()
    time.sleep(1.0)   # let the engine print its request summary
    lines, offset = engine_lines(log, offset)
    res['engine'] = parse_engine(lines)
    res['power'] = integrate(samples, t0, t1)
    if e0 is not None and e1 is not None:
        d = e1 - e0
        if d < 0:
            d += rapl_range_uj()
        res['cpu_joules'] = d / 1e6
    gen = res['engine'].get('generated') or (res['usage'] or {}).get('completion_tokens') or 0
    if gen and res['power']:
        res['gpu_joules_per_token'] = res['power']['gpu_joules'] / gen
        if 'cpu_joules' in res:
            res['cpu_joules_per_token'] = res['cpu_joules'] / gen
    res['name'] = name
    return res, offset


def run_helper(cmds, record):
    for c in cmds:
        argv = shlex.split(c) if isinstance(c, str) else list(c)
        r = subprocess.run(argv, text=True, capture_output=True)
        record.append({'cmd': argv, 'rc': r.returncode, 'out': r.stdout[-500:], 'err': r.stderr[-500:]})
        if r.returncode != 0:
            raise RuntimeError(f'helper failed: {argv}: {r.stderr.strip()}')


def candidate_profile(base, cand):
    p = json.loads(json.dumps(base))
    p.update(cand.get('config', {}))
    env = dict(p.get('environment', {}))
    env.update(cand.get('environment', {}))
    p['environment'] = env
    p['port'] = 18081
    return p


def run_candidate(base, cand, out, round_no, args, salt):
    name = cand['name']
    cdir = out / f'r{round_no}-{name}'
    cdir.mkdir(parents=True, exist_ok=True)
    profile = candidate_profile(base, cand)
    (cdir / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    helper_log = []
    record = {'candidate': name, 'round': round_no, 'requests': [], 'helper': helper_log, 'status': 'failed'}
    samples, stop = [], threading.Event()
    th = threading.Thread(target=sampler, args=(stop, samples), daemon=True)
    th.start()
    try:
        run_helper(cand.get('pre', []), helper_log)
        record['gpu_before'] = guard()
        with Server(profile, cdir) as server:
            record['startup_seconds'] = server.startup
            log = cdir / 'engine.log'
            offset = len(log.read_bytes())
            # warm-up: a short request, not counted
            warm, offset = measure('warmup', {'messages': short_messages(99, salt), 'temperature': 0,
                                               'reasoning_effort': 'none', 'max_tokens': 64, 'stream': True},
                                    log, offset, samples)
            record['warmup'] = {k: warm.get(k) for k in ('wall_seconds', 'engine')}
            for i in range(args.short):
                payload = {'messages': short_messages(i, salt), 'temperature': 0,
                           'reasoning_effort': args.reasoning, 'max_tokens': args.max_tokens, 'stream': True}
                res, offset = measure(f'short-{i}', payload, log, offset, samples)
                if not args.save_content: res.pop('content', None)
                record['requests'].append(res)
                print(f'  {name} short-{i}: wall {res["wall_seconds"]:.1f}s '
                      f'decode {res["engine"].get("decode_tps")} tok/s gen {res["engine"].get("generated")} '
                      f'W {res["power"].get("gpu_watts_mean", 0):.0f} sm {res["power"].get("sm_mhz_mean", 0):.0f}', flush=True)
            for li, length in enumerate(args.long):
                fx = json.loads((FIXTURES / f'fresh-{length}.json').read_text())
                msgs = fx['messages']
                if args.salt_long:   # a distinct first line defeats the prompt cache, so every long request prefills in full
                    msgs = [dict(msgs[0], content=f'Run {salt}-{li}.\n' + msgs[0]['content'])] + msgs[1:]
                payload = {'messages': msgs, 'temperature': 0, 'reasoning_effort': 'none',
                           'max_tokens': 256, 'stream': True}
                res, offset = measure(f'long-{length}-{li}', payload, log, offset, samples)
                secrets = fx.get('secrets') or {}
                if secrets:   # the fixture asks for its three planted values: a correctness gate for every candidate
                    res['secrets_ok'] = all(str(v) in res['content'] for v in secrets.values())
                if not args.save_content: res.pop('content', None)
                record['requests'].append(res)
                print(f'  {name} long-{length}: secrets_ok={res.get("secrets_ok")} wall {res["wall_seconds"]:.1f}s '
                      f'prefill {res["engine"].get("prefill_tps")} tok/s decode {res["engine"].get("decode_tps")} '
                      f'W {res["power"].get("gpu_watts_mean", 0):.0f} sm {res["power"].get("sm_mhz_mean", 0):.0f}', flush=True)
            record['status'] = 'ok'
    finally:
        stop.set()
        th.join(timeout=3)
        try:
            run_helper(cand.get('post', []), helper_log)
        except RuntimeError as exc:
            record['post_error'] = str(exc)
        (cdir / 'samples.json').write_text(json.dumps(samples))
        (cdir / 'record.json').write_text(json.dumps(record, indent=1) + '\n')
        # wait for VRAM to drain so the next candidate starts clean
        for _ in range(60):
            try:
                if gpu_query()['vram_mib'] < 1200:
                    break
            except Exception:
                pass
            time.sleep(2)
    return record


def summarize(records):
    by = {}
    for r in records:
        if r['status'] != 'ok':
            continue
        for q in r['requests']:
            kind = 'long' if q['name'].startswith('long') else 'short'
            b = by.setdefault((r['candidate'], kind), {'wall': [], 'decode': [], 'prefill': [], 'gpu_jpt': [],
                                                        'cpu_jpt': [], 'acc': [], 'sm': [], 'w': [], 'hit': [], 'n': 0})
            e = q.get('engine', {})
            b['n'] += 1
            b['wall'].append(q['wall_seconds'])
            if e.get('decode_tps'): b['decode'].append(e['decode_tps'])
            if e.get('prefill_tps') and e.get('read', 0) > 1000: b['prefill'].append(e['prefill_tps'])
            if q.get('gpu_joules_per_token'): b['gpu_jpt'].append(q['gpu_joules_per_token'])
            if q.get('cpu_joules_per_token'): b['cpu_jpt'].append(q['cpu_joules_per_token'])
            if e.get('drafts_offered'): b['acc'].append(e['drafts_accepted'] / e['drafts_offered'])
            if e.get('hit_rate') is not None: b['hit'].append(e['hit_rate'])
            if 'secrets_ok' in q: b.setdefault('ok', []).append(q['secrets_ok'])
            if q.get('power'):
                b['sm'].append(q['power'].get('sm_mhz_mean', 0)); b['w'].append(q['power'].get('gpu_watts_mean', 0))
    rows = []
    for (cand, kind), b in sorted(by.items()):
        med = lambda a: round(statistics.median(a), 3) if a else None
        rows.append({'candidate': cand, 'kind': kind, 'n': b['n'], 'wall_s': med(b['wall']),
                     'decode_tps': med(b['decode']), 'prefill_tps': med(b['prefill']),
                     'gpu_J_per_tok': med(b['gpu_jpt']), 'cpu_J_per_tok': med(b['cpu_jpt']),
                     'accept': med(b['acc']), 'hit_pct': med(b['hit']), 'sm_mhz': med(b['sm']), 'gpu_w': med(b['w']),
                     'secrets_ok': (f"{sum(b['ok'])}/{len(b['ok'])}" if b.get('ok') else None)})
    return rows


def print_table(rows):
    cols = ['candidate', 'kind', 'n', 'wall_s', 'decode_tps', 'prefill_tps', 'gpu_J_per_tok', 'cpu_J_per_tok',
            'accept', 'hit_pct', 'sm_mhz', 'gpu_w', 'secrets_ok']
    print(' | '.join(cols))
    for r in rows:
        print(' | '.join(str(r.get(c)) for c in cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True, help='base profile JSON (port is forced to 18081)')
    ap.add_argument('--candidates', type=Path, required=True,
                    help='JSON list of {name, config?, environment?, pre?, post?}')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rounds', type=int, default=1)
    ap.add_argument('--short', type=int, default=4, help='short coding requests per candidate per round')
    ap.add_argument('--long', type=int, nargs='*', default=[], help='fresh fixture lengths, e.g. 80000 128000')
    ap.add_argument('--max-tokens', type=int, default=700)
    ap.add_argument('--reasoning', default='medium')
    ap.add_argument('--only', nargs='*', help='candidate names to run (default all)')
    ap.add_argument('--save-content', action='store_true', help='keep every full answer in record.json')
    ap.add_argument('--salt-long', action='store_true', help='make every long fixture request unique (soak: no prompt-cache reuse)')
    ap.add_argument('--summary-only', action='store_true', help='re-summarize an existing output directory')
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    results = args.output / 'results.jsonl'
    if args.summary_only:
        records = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
        rows = summarize(records)
        (args.output / 'summary.json').write_text(json.dumps(rows, indent=1) + '\n')
        print_table(rows)
        return
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'unbiasedstrata.service']).returncode == 0:
        sys.exit('production unbiasedstrata.service is active; stop it first')
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'bench-campaign.service']).returncode == 0:
        sys.exit('bench-campaign.service is running; refusing to overlap')
    base = json.loads(args.base.read_text())
    cands = json.loads(args.candidates.read_text())
    if args.only:
        cands = [c for c in cands if c['name'] in args.only]
    salt = time.strftime('%H%M%S')
    (args.output / 'plan.json').write_text(json.dumps({'base': str(args.base), 'candidates': cands,
        'rounds': args.rounds, 'short': args.short, 'long': args.long, 'max_tokens': args.max_tokens,
        'reasoning': args.reasoning, 'salt': salt, 'gpu': gpu_query()}, indent=1) + '\n')
    records = []
    if results.exists():
        records = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
    for rnd in range(args.rounds):
        order = cands if rnd % 2 == 0 else list(reversed(cands))
        for cand in order:
            print(f'== round {rnd} candidate {cand["name"]}', flush=True)
            rec = run_candidate(base, cand, args.output, rnd, args, f'{salt}{rnd}')
            records.append(rec)
            with results.open('a') as f:
                f.write(json.dumps(rec) + '\n')
            if rec['status'] != 'ok':
                print(f'!! {cand["name"]} failed; see {args.output}', flush=True)
    rows = summarize(records)
    (args.output / 'summary.json').write_text(json.dumps(rows, indent=1) + '\n')
    print_table(rows)


if __name__ == '__main__':
    main()
