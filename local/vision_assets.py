#!/usr/bin/python3
"""Render two 2560x1440 synthetic screenshots with known text, plus the questions and the exact answers to score.

Usage: vision_assets.py OUT_DIR      (needs Pillow: run it with a Python that has it, e.g. upstream Strata's .venv, -I)
Both pictures are larger than the 2048-token picture budget, so a reader always runs at its cap: the heaviest
picture a coding session sends. Nothing is captured from the screen; every string is fixed below.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 2560, 1440
MONO = '/usr/share/fonts/liberation/LiberationMono-Regular.ttf'
BOLD = '/usr/share/fonts/liberation/LiberationMono-Bold.ttf'

STATUS_TITLE = 'LILAC SWITCHBOARD - NODE STATUS'
STATUS_ROWS = [  # (node, address, state, model)
    ('bigrig-a', '10.20.30.41:4100', 'serving', 'flash-iq3_s'),
    ('bigrig-b', '10.20.30.42:4101', 'idle', 'coder-q4_k_m'),
    ('unraid-gpu', '10.20.30.57:4210', 'loading', 'embed-q8_0'),
    ('pi-logger', '10.20.30.88:9104', 'offline', '-'),
    ('mac-mini', '10.20.30.93:8740', 'serving', 'vision-q8_0'),
]
SELECTED = 2  # unraid-gpu is boxed and marked
FILES = ['amber-falcon-7B-Q4_K_M.gguf', 'mmproj-teal-heron-Q8_0.gguf', 'violet-otter-30B-A3B-IQ3_S.gguf',
         'mtp-ochre-lynx-Q2_0.gguf', 'indigo-wren-12B-Q6_K.gguf', 'cobalt-moth-4B-Q8_0.gguf']

CODE = [
    'import time',
    '',
    'MAX_RETRIES = 7',
    'BACKOFF_SECONDS = 2.5',
    '',
    'def fetch_with_retry(client, url):',
    '    for attempt in range(MAX_RETRIES):',
    '        try:',
    '            return client.get(url, timeout=30)',
    '        except TimeoutError:',
    '            time.sleep(BACKOFF_SECONDS * (attempt + 1))',
    '    raise RuntimeError("gave up after retries")',
]
TABLE = [('node', 'VRAM GB', 'decode tok/s'), ('atlas', '24.0', '61.4'), ('borealis', '31.6', '92.7'),
         ('cirrus', '12.0', '48.9'), ('dorado', '16.0', '55.2')]
ERROR = 'ERROR 0x5A3F: checkpoint shard 3 of 8 failed its checksum'


def render_status(path):
    img = Image.new('RGB', (W, H), (30, 30, 46))
    d = ImageDraw.Draw(img)
    title, body, small = ImageFont.truetype(BOLD, 44), ImageFont.truetype(MONO, 30), ImageFont.truetype(MONO, 26)
    d.rectangle([0, 0, W, 90], fill=(49, 50, 68))
    d.text((40, 20), STATUS_TITLE, font=title, fill=(203, 166, 247))
    d.text((40, 140), f'{"NODE":<14}{"ADDRESS":<22}{"STATE":<12}MODEL', font=body, fill=(166, 173, 200))
    for i, (node, addr, state, model) in enumerate(STATUS_ROWS):
        y = 200 + i * 70
        mark = '◆ ' if i == SELECTED else '  '
        if i == SELECTED:
            d.rectangle([30, y - 12, 1500, y + 46], outline=(249, 226, 175), width=3)
        d.text((40, y), f'{mark}{node:<12}{addr:<22}{state:<12}{model}', font=body, fill=(205, 214, 244))
    d.text((40, 620), 'Recent model files:', font=body, fill=(166, 173, 200))
    for i, f in enumerate(FILES):
        d.text((80, 680 + i * 52), f'{i + 1}. {f}', font=small, fill=(186, 194, 222))
    d.text((1700, 200), 'queue depth: 3', font=body, fill=(148, 226, 213))
    d.text((1700, 260), 'uptime: 4d 07h 12m', font=body, fill=(148, 226, 213))
    img.save(path)


def render_code(path):
    img = Image.new('RGB', (W, H), (24, 24, 37))
    d = ImageDraw.Draw(img)
    body, head = ImageFont.truetype(MONO, 30), ImageFont.truetype(BOLD, 34)
    d.text((40, 30), 'retry.py', font=head, fill=(137, 180, 250))
    for i, line in enumerate(CODE):
        d.text((40, 100 + i * 46), f'{i + 1:>3}  {line}', font=body, fill=(205, 214, 244))
    d.text((1400, 30), 'cluster.csv', font=head, fill=(137, 180, 250))
    for r, row in enumerate(TABLE):
        for c, cell in enumerate(row):
            d.text((1400 + c * 330, 110 + r * 60), cell, font=body, fill=(249, 226, 175) if r == 0 else (205, 214, 244))
    d.rectangle([30, 1300, W - 30, 1390], fill=(60, 30, 40))
    d.text((60, 1322), ERROR, font=body, fill=(243, 139, 168))
    img.save(path)


QUESTIONS = [  # (picture, question, substrings every correct answer contains)
    ('status.png', 'What is the title in the bar at the very top? Answer with the exact text only.', [STATUS_TITLE]),
    ('status.png', 'Which node is selected (the row inside the highlighted frame)? Answer with its node name and address.',
     ['unraid-gpu', '10.20.30.57:4210']),
    ('status.png', 'List every address with its port in the table, exactly as written, one per line.',
     [r[1] for r in STATUS_ROWS]),
    ('status.png', 'Transcribe the last four file names in the "Recent model files" list exactly.', FILES[-4:]),
    ('code.png', 'In retry.py, what are the values of MAX_RETRIES and BACKOFF_SECONDS? Answer as NAME=value.',
     ['MAX_RETRIES=7', 'BACKOFF_SECONDS=2.5']),
    ('code.png', 'In cluster.csv, which node has 31.6 GB of VRAM, and what is its decode tok/s?', ['borealis', '92.7']),
    ('code.png', 'Transcribe the red error message at the bottom exactly.', [ERROR]),
]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    render_status(out / 'status.png')
    render_code(out / 'code.png')
    (out / 'questions.json').write_text(json.dumps(
        [{'image': img, 'question': q, 'expect': e} for img, q, e in QUESTIONS], indent=1) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
