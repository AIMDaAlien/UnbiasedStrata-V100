#!/usr/bin/python3
"""Local long-context experiment. Never changes production configuration."""
import argparse
import ast
import hashlib
import itertools
import fcntl
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

from launch import CONFIG, ROOT, command, optimization_environment
from benchmark import monitor, cpu_seconds, processes

BASE = 'http://127.0.0.1:18081'
CHECKPOINTS = (80000, 128000, 180000)
TOOLS = [{'type': 'function', 'function': {
    'name': 'lookup_record', 'description': 'Look up the requested record.',
    'parameters': {'type': 'object', 'properties': {'key': {'type': 'string'}},
                   'required': ['key'], 'additionalProperties': False}}}]
TASK = '''Find the three AUDIT_SECRET records in the notes. Return JSON with keys
alpha, beta, gamma, and code. Copy their secret values exactly. CODE_TASK
Do not call a tool; all needed information is in the notes. Output only the JSON
object and complete every field.'''

def code_task(target):
    if target <= 80000:
        return 'merge_ranges', ('For code, implement merge_ranges(ranges): sort and merge overlapping or touching '
            'integer [start,end] ranges. Return a list of merged [start,end] lists, sorted by start. Do not mutate the input.')
    if target <= 128000:
        return 'ordered_results', ('For code, implement ordered_results(events). Each event is a dict with integer seq '
            'and status; done events also have value. Keep the last event per seq, remove entries whose last status '
            'is not done, and return their values in increasing seq order.')
    return 'batch_records', ('For code, implement batch_records(records,max_items,max_bytes). Records are [id,nbytes] '
        'lists. Preserve order and greedily pack each batch up to both positive limits, returning a list of nonempty '
        'batches of records. Return [] for empty input. Raise ValueError for negative or oversized record sizes.')


def api(path, body, timeout=7200):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def body(messages, stream=False, tools=False):
    b = {'messages': messages, 'temperature': 0, 'reasoning_effort': 'none',
         'max_tokens': 512, 'stream': stream, 'tools': TOOLS, 'tool_choice': 'auto'}
    if tools:
        b.update(tools=TOOLS, tool_choice='required', max_tokens=128)
    return b


def count(messages, tools=False):
    return api('/tokenize', body(messages, tools=tools))['count']


def fixture(messages, target, identity):
    """Binary search deterministic padding, then token-calibrate including template."""
    secrets = {k: hashlib.sha256((identity + k).encode()).hexdigest()[:16]
               for k in ('alpha', 'beta', 'gamma')}
    historical = []
    for message in messages:
        historical.extend(re.findall(r'AUDIT_SECRET ([\w.-]+)=([0-9a-f]{16})', message.get('content', '') or ''))
    records = [(identity + '.' + key, secrets[key]) for key in secrets]
    all_records = historical + records
    selected = [all_records[0], all_records[len(all_records)//2], all_records[-1]]
    secrets = dict(zip(('alpha', 'beta', 'gamma'), [value for _, value in selected]))
    task = TASK.replace('Find the three AUDIT_SECRET records in the notes.',
        'Find these exact AUDIT_SECRET record keys in the full conversation: ' +
        ', '.join(name for name, _ in selected) +
        '. Map them in that order to alpha, beta, gamma.')
    task = task.replace('CODE_TASK', code_task(target)[1])
    line = 'Reference note: preserve order, bound buffers, check results, and measure completed work.\n'
    def content(n, tail=''):
        thirds = [n // 3, n // 3, n - 2 * (n // 3)]
        text = 'Experiment ' + identity + '\n'
        for (key, value), size in zip(records, thirds):
            text += line * size + f'AUDIT_SECRET {key}={value}\n'
        return text + tail + '\n' + task
    def candidate(n, tail=''):
        return messages + [{'role': 'user', 'content': content(n, tail)}]
    lo, hi = 0, target // 8
    if count(candidate(0)) > target:
        raise ValueError('conversation already exceeds next checkpoint')
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(candidate(mid)) <= target:
            lo = mid
        else:
            hi = mid - 1
    # A repeated whitespace-separated word has one token per occurrence on this
    # vocabulary. Verify every increment rather than assuming token counts.
    tail = ''
    while True:
        current = count(candidate(lo, tail))
        if current == target:
            return candidate(lo, tail), secrets
        if current > target:
            raise RuntimeError('token padding overshot; fixture is invalid')
        for piece in (' x', ' a', '\n', ' ', 'z', '.', '0'):
            following = count(candidate(lo, tail + piece))
            if current < following <= target:
                tail += piece
                break
        else:
            raise RuntimeError('no exact token padding found')
        if len(tail) > 8192:
            raise RuntimeError('token padding failed to converge')


def quality(result, secrets, target):
    checks = {'completed': result['finish_reason'] == 'stop', 'retrieval': False, 'code': False}
    try:
        text = result['message'].get('content', '').strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0]
        answer = json.loads(text)
        checks['retrieval'] = all(answer.get(k) == v for k, v in secrets.items())
        checked = subprocess.run(['/usr/bin/python3', '-I', str(ROOT / 'local/check_generated_code.py')],
            input=json.dumps({'code':answer['code'], 'task':code_task(target)[0]}), text=True,
            capture_output=True, timeout=4)
        checks['code'] = checked.returncode == 0
    except (ValueError, TypeError, KeyError, SyntaxError, subprocess.TimeoutExpired):
        pass
    return checks


def completion(messages, pid, unit, tools=False):
    samples, stop = [], threading.Event()
    amd_samples = []
    def amd_monitor():
        cards = []
        for card in Path('/sys/class/drm').glob('card*/device'):
            try:
                if int((card / 'mem_info_vram_total').read_text()) > 12 * 1024**3:
                    cards.append(card)
            except OSError:
                pass
        while not stop.is_set():
            for card in cards:
                try:
                    amd_samples.append({'time': time.time(), 'vram_bytes': int(
                        (card / 'mem_info_vram_used').read_text()), 'gpu_percent': int(
                        (card / 'gpu_busy_percent').read_text())})
                except OSError:
                    pass
            stop.wait(0.5)
    def disk_read_bytes():
        total = 0
        for p in processes(pid):
            try:
                total += int(dict(line.split(':', 1) for line in
                    Path(f'/proc/{p}/io').read_text().splitlines())['read_bytes'])
            except OSError:
                pass
        return total
    worker = threading.Thread(target=monitor, args=(stop, samples), daemon=True)
    amd_worker = threading.Thread(target=amd_monitor, daemon=True)
    worker.start()
    amd_worker.start()
    disk_start = disk_read_bytes()
    started, cpu = time.monotonic(), cpu_seconds(pid)
    ttft = None
    message = {'role': 'assistant', 'content': ''}
    calls = {}
    final = None
    try:
        req = urllib.request.Request(BASE + '/v1/chat/completions',
            json.dumps(body(messages, stream=True, tools=tools)).encode(),
            {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=7200) as r:
            for line in r:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    break
                event = json.loads(payload)
                if 'error' in event:
                    raise RuntimeError(event['error'])
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    if delta.get('content') or delta.get('tool_calls'):
                        if ttft is None:
                            ttft = time.monotonic() - started
                    message['content'] += delta.get('content') or ''
                    for call in delta.get('tool_calls', []):
                        c = calls.setdefault(call['index'], {'id': '', 'type': 'function',
                            'function': {'name': '', 'arguments': ''}})
                        c['id'] += call.get('id', '')
                        for key in ('name', 'arguments'):
                            c['function'][key] += call.get('function', {}).get(key, '')
                    if choice.get('finish_reason'):
                        final = choice['finish_reason']
                if 'usage' in event:
                    usage, timings = event['usage'], event.get('timings', {})
        elapsed = time.monotonic() - started
        if final is None or 'usage' not in locals():
            raise RuntimeError('stream ended without completion and usage')
        if calls:
            message['tool_calls'] = [calls[i] for i in sorted(calls)]
        props = subprocess.check_output(['systemctl', '--user', 'show', unit,
            '--property=MemoryPeak', '--property=MemorySwapCurrent'], text=True)
        return {'message': message, 'finish_reason': final, 'usage': usage, 'timings': timings,
                'wall_seconds': elapsed, 'ttft_seconds': ttft,
                'cpu_cores': (cpu_seconds(pid) - cpu) / elapsed,
                'memory': props, 'gpu_samples': samples, 'amd_samples': amd_samples,
                'disk_read_bytes': disk_read_bytes() - disk_start}
    finally:
        stop.set()
        worker.join(timeout=6)
        amd_worker.join(timeout=2)


class Server:
    def __init__(self, profile, output):
        self.profile, self.output = profile, output
        self.unit = f'unbiasedstrata-long-{os.getpid()}-{time.time_ns()}'
    def __enter__(self):
        if subprocess.run(['systemctl', '--user', 'is-active', '--quiet',
                           'unbiasedstrata.service']).returncode == 0:
            raise RuntimeError('production inference was started; benchmark excluded')
        try:
            with urllib.request.urlopen(BASE + '/health', timeout=2):
                raise RuntimeError('benchmark port already has a server')
        except OSError:
            pass
        self.output.mkdir(parents=True, exist_ok=True)
        log = self.output / 'engine.log'
        env = optimization_environment(self.profile)
        self.environment = env
        (self.output / 'launch.json').write_text(json.dumps({
            'command': command(self.profile), 'environment': env,
            'binary_sha256': hashlib.sha256(Path(command(self.profile)[0]).read_bytes()).hexdigest(),
            'memory_max_bytes': 56 * 1024**3, 'memory_swap_max_bytes': 0,
            'gpu': subprocess.check_output(['nvidia-smi', '--id=0',
                '--query-gpu=name,pci.bus_id,power.limit,driver_version',
                '--format=csv,noheader'], text=True).strip(),
        }, indent=2) + '\n')
        start = time.monotonic()
        subprocess.run(['systemd-run', '--user', '--collect', f'--unit={self.unit}',
            '--property=MemoryMax=56G', '--property=MemorySwapMax=0',
            '--property=RuntimeMaxSec=3600',
            '--property=TimeoutStopSec=30', '--property=KillMode=control-group',
            f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}',
            # optional profile key: pin front end + engine to a CPU list, e.g. one CCD ("0-7,16-23"). taskset, not
            # AllowedCPUs: the user session has no cpuset controller delegated, so the unit property is silently inert.
            *(['/usr/bin/taskset', '-c', self.profile['cpu_affinity']] if self.profile.get('cpu_affinity') else []),
            '/usr/bin/env', *[f'{k}={v}' for k, v in sorted(env.items())],
            *command(self.profile)], check=True)
        try:
            while time.monotonic() - start < 900:
                try:
                    with urllib.request.urlopen(BASE + '/health', timeout=2) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                state = subprocess.check_output(['systemctl', '--user', 'show', self.unit,
                    '--property=ActiveState', '--value'], text=True).strip()
                if state not in ('active', 'activating'):
                    raise RuntimeError('server failed to load: ' + str(log))
                time.sleep(2)
            else:
                raise RuntimeError('startup timeout')
            self.startup = time.monotonic() - start
            self.pid = int(subprocess.check_output(['systemctl', '--user', 'show', self.unit,
                '--property=MainPID', '--value'], text=True))
            actual = dict(item.split('=', 1) for item in
                          Path(f'/proc/{self.pid}/environ').read_text().split('\0') if '=' in item)
            if any(actual.get(k) != v for k, v in env.items()):
                raise RuntimeError('benchmark optimization environment did not reach server')
            actual_controls = {k:v for k,v in actual.items() if k.startswith('STRATA_') or k in
                ('CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING', 'OMP_NUM_THREADS')}
            if actual_controls != env:
                raise RuntimeError('systemd injected unrecorded optimization environment')
            launched_hash=hashlib.sha256(Path(f'/proc/{self.pid}/exe').read_bytes()).hexdigest()
            expected=json.loads((self.output/'launch.json').read_text())['binary_sha256']
            if launched_hash != expected:
                raise RuntimeError('launched executable differs from recorded binary')
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise
    def __exit__(self, *args):
        subprocess.run(['systemctl', '--user', 'stop', self.unit], check=False)


def run(profile, lengths, workloads, output, fixtures=None):
    record = {'profile': profile, 'runs': [], 'status': 'failed'}
    try:
        with Server(profile, output) as server:
            record['startup_seconds'] = server.startup
            record['environment'] = server.environment
            record['launch'] = json.loads((output / 'launch.json').read_text())
            for workload in workloads:
                messages, conversation_time = [], 0
                targets = lengths if workload == 'fresh' else sorted(set(
                    [n for n in range(8192, max(lengths), 8192)] + list(lengths)))
                for target in targets:
                    identity = f'fixed-v1-{workload}-{target}'
                    fixture_file = fixtures / f'{workload}-{target}.json' if fixtures else None
                    if fixture_file and fixture_file.exists():
                        fixed = json.loads(fixture_file.read_text())
                        prompt, secrets = fixed['messages'], fixed['secrets']
                        if count(prompt) != target:
                            raise RuntimeError('saved fixture token count mismatch')
                    else:
                        prompt, secrets = fixture(messages if workload == 'growing' else [], target, identity)
                        if fixture_file:
                            fixtures.mkdir(parents=True, exist_ok=True)
                            fixture_file.write_text(json.dumps({'messages': prompt, 'secrets': secrets}))
                    request_file = output / f'{workload}-{target}.request.json'
                    request_file.write_text(json.dumps(body(prompt), indent=2))
                    log = output / 'engine.log'
                    log_position = log.stat().st_size
                    measured = completion(prompt, server.pid, server.unit)
                    with log.open() as engine_log:
                        engine_log.seek(log_position)
                        diagnostic_text = engine_log.read()
                    helper_diagnostics = re.findall(r'AMD helper: transfer_ms=([\d.]+) wait_ms=([\d.]+) bytes=(\d+) fallback_rows=(\d+) device_transfer_ms=([\d.]+) device_compute_ms=([\d.]+)', diagnostic_text)
                    measured['helper_diagnostics'] = helper_diagnostics
                    measured['prefill_diagnostics'] = [line for line in diagnostic_text.splitlines()
                        if 'prefill timing' in line or 'prefill host' in line]
                    conversation_time = (conversation_time + measured['wall_seconds']
                                         if workload == 'growing' else measured['wall_seconds'])
                    measured.update(workload=workload, target_tokens=target,
                                    conversation_wall_seconds=conversation_time,
                                    fixture_sha256=hashlib.sha256(request_file.read_bytes()).hexdigest())
                    measured['checks'] = quality(measured, secrets, target)
                    measured['checks']['token_count'] = measured['usage']['prompt_tokens'] == target
                    measured['checks']['no_swap'] = 'MemorySwapCurrent=0' in measured['memory']
                    if workload == 'fresh':
                        measured['checks']['fresh_state'] = measured['timings'].get('prompt_cached_n', -1) == 0
                    if profile.get('amd_helper'):
                        measured['checks']['helper_healthy'] = bool(helper_diagnostics) and all(int(d[3]) == 0 for d in helper_diagnostics)
                    if workload == 'growing' and messages:
                        measured['checks']['prefix_reuse'] = measured['timings'].get('prompt_cached_n', 0) >= count(messages) - 1024
                    record['runs'].append(measured)
                    print(profile['threads'], profile['spec'], profile.get('kv'), workload,
                          target, round(measured['wall_seconds'], 2), measured['checks'], flush=True)
                    (output / 'results.json').write_text(json.dumps(record, indent=2))
                    if not all(measured['checks'].values()):
                        raise RuntimeError('quality gate failed')
                    if workload == 'growing':
                        messages = prompt + [measured['message']]
                        tool_prompt = messages + [{'role': 'user', 'content':
                            'Call lookup_record with key alpha. Do not answer directly.'}]
                        tool_file = fixtures / f'growing-{target}.tool.json' if fixtures else None
                        if tool_file and tool_file.exists():
                            tool_prompt = json.loads(tool_file.read_text())
                        elif tool_file:
                            tool_file.write_text(json.dumps(tool_prompt))
                        tool_result = completion(tool_prompt, server.pid, server.unit, tools=True)
                        conversation_time += tool_result['wall_seconds']
                        calls = tool_result['message'].get('tool_calls', [])
                        valid = tool_result['finish_reason'] == 'tool_calls' and len(calls) == 1
                        if valid:
                            valid = calls[0]['function']['name'] == 'lookup_record' and json.loads(
                                calls[0]['function']['arguments']) == {'key': 'alpha'}
                        record['runs'][-1]['tool_turn'] = tool_result
                        record['runs'][-1]['checks']['tool_valid'] = valid
                        record['runs'][-1]['conversation_wall_seconds'] = conversation_time
                        if not valid:
                            raise RuntimeError('tool-call gate failed')
                        messages = tool_prompt + [tool_result['message'], {'role': 'tool',
                            'tool_call_id': calls[0]['id'], 'content': json.dumps({'alpha': secrets['alpha']})}]
            engine_log = (output / 'engine.log').read_text(errors='replace')
            record['fused_kernels_used'] = 'prompt experts on the fused int8 kernels' in engine_log
            record['resolved_settings'] = [line for line in engine_log.splitlines()
                if any(s in line for s in ('prompt chunk auto:', 'prompt path borrows',
                    'expert cache auto:', 'mapped pinned cache complement ready:',
                    'AMD helper mode:', 'disjoint experts,'))]
            if record['environment'].get('STRATA_PF_FUSED') == '1' and not record['fused_kernels_used']:
                raise RuntimeError('requested fused kernels were not used')
            if profile.get('amd_helper') and ('AMD helper: failed batch' in engine_log or
                    any(int(n)>0 for n in re.findall(r'fallback_rows=(\d+)', engine_log))):
                raise RuntimeError('AMD fallback occurred, including a tool turn; excluded from speed results')
            record['status'] = 'passed'
    except KeyboardInterrupt:
        record['status'] = 'interrupted'
        raise
    except Exception as e:
        record['error'] = str(e)
        print('Excluded profile:', e, flush=True)
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['smoke', 'screen', 'verify'], default='smoke')
    parser.add_argument('--profile', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fixtures', type=Path, help='shared canonical requests for paired comparisons')
    args = parser.parse_args()
    lock = open(ROOT / 'local/long-context.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('another long-context experiment is running')
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet',
                       'unbiasedstrata.service']).returncode == 0:
        parser.error('stop the production model service before benchmarking')
    try:
        with urllib.request.urlopen(BASE + '/health', timeout=2):
            parser.error('benchmark port is already occupied')
    except OSError:
        pass
    c = json.loads(args.profile.read_text())
    free_vram = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                            '--format=csv,noheader,nounits'], text=True))
    if free_vram < 10000:
        parser.error('3060 needs at least 10000 MiB free before starting a model copy')
    c.update(context=196608, port=18081)
    output = (args.output or ROOT / 'local/benchmarks' / time.strftime('long-%Y%m%d-%H%M%S')).resolve()
    variants = [(c['threads'], c['spec'], c.get('kv', 'int8'), c.get('kv_resident', 0))]
    if args.stage == 'screen':
        variants = [(t, s, k, w) for t, s in itertools.product((8, 12, 15, 16), (2, 4))
                    for k, w in [('int8', 0), ('k8v4', 0), ('q4_0', 0),
                                 ('int8', 32768), ('int8', 65536)]]
    records = []
    for t, s, k, w in variants:
        profile = dict(c, threads=t, spec=s, kv=k, kv_resident=w)
        records.append(run(profile, CHECKPOINTS if args.stage == 'verify' else (80000,),
                           ('fresh', 'growing') if args.stage == 'verify' else ('fresh',),
                           output / f't{t}-s{s}-{k}-w{w}', args.fixtures))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(records, indent=2) + '\n')
    if not any(r['status'] == 'passed' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
