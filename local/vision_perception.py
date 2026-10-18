#!/usr/bin/python3
"""Perception check on a real screenshot: start a temporary vision service from a profile and ask three questions.

Usage: vision_perception.py PROFILE.json OUTPUT_DIR IMAGE [IMAGE...]
Prints each answer with its timing and the prompt token count, so settings (vision_tokens) can be compared.
"""
import base64, json, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from long_context import Server, BASE

QUESTIONS = [
    'What is the title at the very top of this screenshot? Answer with the exact text only.',
    'Which entry is currently selected (it has a filled diamond marker and a box around it)? Quote its first line exactly.',
    'List every IP address with its port number that appears anywhere in the screenshot, exactly as written.',
    'Transcribe the names of the last four files listed at the bottom (the ones ending in .gguf), exactly.',
]


def ask(image_bytes, mime, question, max_tokens=200):
    body = {'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,' + base64.b64encode(image_bytes).decode()}},
        {'type': 'text', 'text': question}]}], 'temperature': 0, 'reasoning_effort': 'none', 'max_tokens': max_tokens}
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d['choices'][0]['message']['content'].strip(), time.monotonic() - t0, d['usage']['prompt_tokens']


profile = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
results = []
with Server(profile, out) as s:
    print(f'startup {s.startup:.0f}s  vision_tokens={profile.get("vision_tokens")}', flush=True)
    for path in sys.argv[3:]:
        data = Path(path).read_bytes()
        mime = 'image/png' if data[:4] == b'\x89PNG' else 'image/jpeg'
        for q in QUESTIONS:
            a, dt, ptok = ask(data, mime, q)
            results.append({'image': path, 'question': q, 'answer': a, 'seconds': dt, 'prompt_tokens': ptok})
            print(f'[{dt:.1f}s, {ptok} prompt tok] Q: {q[:60]}\n   A: {a[:300]}', flush=True)
(out / 'answers.json').write_text(json.dumps(results, indent=1))
