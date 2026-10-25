#!/usr/bin/python3
"""Launch the local hardware profile without the old Python inference server."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path.home() / '.config/unbiasedstrata/config.json'


def command(config):
    # benchmark-only profile key: a complete command line (e.g. upstream Strata's own server for an A/B); production never sets it
    if config.get('argv'):
        argv = config['argv']
        if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            raise ValueError('argv must be a list of strings')
        return list(argv)
    kv = config.get('kv', 'int8')
    resident = config.get('kv_resident', 0)
    if kv not in ('int8', 'k8v4', 'q4_0'):
        raise ValueError('kv must be int8, k8v4 or q4_0')
    if resident and (kv != 'int8' or resident not in (32768, 65536)):
        raise ValueError('KV streaming requires int8 and a 32768/65536 resident window')
    cmd = [str(config.get('binary', ROOT / 'build/unbiased-strata')),
            '--model', config['model'], '--pack', config['pack'], '--mtp', config['mtp'],
            '--host', '127.0.0.1', '--port', str(config['port']),
            '--ctx', str(config['context']), '--threads', str(config['threads'])]
    if config.get('model_name'):   # the id /v1/models reports (the engine's default names the IQ3_XXS release)
        cmd += ['--model-name', config['model_name']]
    if config.get('vision_mmproj'):   # images: the model's own encoder on the GPU (tools/vision/strata-vision)
        cmd += ['--vision', config['vision_mmproj'], '--vision-tokens', str(config.get('vision_tokens', 2048))]
        if config.get('vision_helper'):
            cmd += ['--vision-helper', config['vision_helper']]
    cmd += ['--', '--ple-io', config['ple_io'], '--resident-experts',
            '--kv', kv, '--spec', str(config['spec'])]
    if resident:
        cmd += ['--kv-resident', str(resident)]
    extra = config.get('engine_args', [])
    if not isinstance(extra, list) or not all(isinstance(a, str) and a.startswith('--') or
                                                     (isinstance(a, str) and i > 0 and extra[i-1].startswith('--'))
                                                     for i, a in enumerate(extra)):
        raise ValueError('engine_args must be a list of --flag [value] strings')
    cmd += extra
    if config.get('amd_helper'):
        cmd += ['--amd-helper', config['amd_helper'], '--amd-device', config['amd_device'],
                '--amd-vram-mib', str(config.get('amd_vram_mib', 12288))]
        if config.get('amd_decode_only'):
            cmd += ['--amd-decode-only']
    return cmd


def environment(config=None):
    env = os.environ.copy()
    env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    env.setdefault('STRATA_RESIDENT_HEADROOM_GIB', '8')
    env.setdefault('OMP_NUM_THREADS', '1')
    for key, value in (config or {}).get('environment', {}).items():
        if not (key.startswith('STRATA_') or key in
                ('CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING', 'OMP_NUM_THREADS')):
            raise ValueError('unsupported optimization environment variable: ' + key)
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def optimization_environment(config=None):
    """Explicitly forward controls through systemd; never forward shell secrets."""
    return {k: v for k, v in environment(config).items()
            if k.startswith('STRATA_') or k in
            ('CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING', 'OMP_NUM_THREADS')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--print-command', action='store_true')
    args, extra = parser.parse_known_args()
    config = json.loads(args.config.read_text())
    cmd = command(config)
    if extra:
        parser.error('configure context/threads in ' + str(args.config))
    if args.print_command:
        import shlex
        print(shlex.join(cmd))
        return
    missing = [p for p in [cmd[0], config['model'], config['pack'] + '/expert-profile.bin',
                          config['mtp'] + '/experts.bin'] + [config[k] for k in ('vision_mmproj', 'vision_helper') if config.get(k)]
               if not Path(p).exists()]
    if missing:
        sys.exit('Missing local artifacts: ' + ', '.join(missing))
    os.execve(cmd[0], cmd, environment(config))


if __name__ == '__main__':
    main()
