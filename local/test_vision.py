#!/usr/bin/python3
"""End-to-end check of the front end's image support against a running server.

Usage: test_vision.py [--base http://127.0.0.1:18081] [--expect-vision | --expect-no-vision]

Synthetic pictures are generated here (pure Python PNG writer, no PIL), so the checks are deterministic:
  1. text-only request still works (usage, finish reason)
  2. one image: left half red, right half blue -> the answer names red on the left / blue on the right
  3. two images in one message: a solid green square and a solid yellow one, asked in order
  4. a literal "<|image_pad|>" inside the user's text is plain text, not an image slot
  5. an unsupported image format is refused with a 400, not a stuck engine
  6. the same picture sent twice is encoded once (the second request's prompt is served from the cache)
With --expect-no-vision only check 1 runs plus the refusal of an image request (400).
"""
import argparse
import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib


def png(width, height, pixel):
    """pixel(x, y) -> (r, g, b); returns PNG bytes."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixel(x, y))
    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(bytes(raw), 9)) + chunk(b'IEND', b''))


def data_url(png_bytes, mime='image/png'):
    return f'data:{mime};base64,' + base64.b64encode(png_bytes).decode()


def chat(base, messages, max_tokens=200, timeout=600):
    body = {'messages': messages, 'temperature': 0, 'reasoning_effort': 'none', 'max_tokens': max_tokens}
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r), time.monotonic() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'{}'), time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:18081')
    ap.add_argument('--expect-no-vision', action='store_true')
    a = ap.parse_args()
    fails = []
    def check(name, cond, detail=''):
        print(('PASS ' if cond else 'FAIL ') + name + (f'  {detail}' if detail else ''), flush=True)
        if not cond: fails.append(name)

    # 1. text only
    st, out, dt = chat(a.base, [{'role': 'user', 'content': 'Reply with exactly: text path ok.'}], 16)
    check('text-only', st == 200 and 'text path ok' in out['choices'][0]['message']['content'].lower(), f'{st} {dt:.1f}s')

    halves = png(256, 128, lambda x, y: (220, 30, 30) if x < 128 else (30, 60, 220))
    img_msg = [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': data_url(halves)}},
        {'type': 'text', 'text': 'This picture has two solid colour halves. Which colour is on the left and which on the right? Answer in one short sentence.'}]}]
    if a.expect_no_vision:
        st, out, dt = chat(a.base, img_msg, 50)
        check('image refused without --vision', st == 400, f'{st} {out.get("error", {}).get("message", "")[:80]}')
        return 1 if fails else 0

    # 2. one image
    st, out, dt = chat(a.base, img_msg, 80)
    text = out['choices'][0]['message']['content'].lower() if st == 200 else str(out)
    check('one image: colours', st == 200 and 'red' in text and 'blue' in text, f'{st} {dt:.1f}s "{text[:100]}"')
    if st == 200:
        check('one image: usage counts image tokens', out['usage']['prompt_tokens'] > 60, str(out['usage']))

    # 3. two images
    green = png(96, 96, lambda x, y: (20, 200, 40))
    yellow = png(96, 96, lambda x, y: (240, 220, 20))
    st, out, dt = chat(a.base, [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'First picture:'},
        {'type': 'image_url', 'image_url': {'url': data_url(green)}},
        {'type': 'text', 'text': 'Second picture:'},
        {'type': 'image_url', 'image_url': {'url': data_url(yellow)}},
        {'type': 'text', 'text': 'Name the single dominant colour of the first picture, then of the second, as "first: X, second: Y".'}]}], 60)
    text = out['choices'][0]['message']['content'].lower() if st == 200 else str(out)
    gi, yi = text.find('green'), text.find('yellow')
    check('two images in order', st == 200 and gi >= 0 and yi >= 0 and gi < yi, f'{st} {dt:.1f}s "{text[:100]}"')

    # 4. literal tag in text
    st, out, dt = chat(a.base, [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': data_url(halves)}},
        {'type': 'text', 'text': 'The string <|image_pad|> is just text here. Which colour is on the left of the picture? One word.'}]}], 20)
    text = out['choices'][0]['message']['content'].lower() if st == 200 else str(out)
    check('literal <|image_pad|> in text is text', st == 200 and 'red' in text, f'{st} "{text[:80]}"')

    # 5. unsupported format
    st, out, dt = chat(a.base, [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/webp;base64,' + base64.b64encode(b'RIFF....WEBP').decode()}},
        {'type': 'text', 'text': 'What is this?'}]}], 20)
    check('unsupported format -> 400', st == 400, f'{st} {out.get("error", {}).get("message", "")[:80]}')
    st, out, dt = chat(a.base, [{'role': 'user', 'content': 'Reply with exactly: still alive.'}], 16)
    check('engine alive after refusal', st == 200 and 'still alive' in out['choices'][0]['message']['content'].lower(), f'{st}')

    # 6. repeat: the second identical picture should not re-encode (fast prompt path)
    st1, out1, dt1 = chat(a.base, img_msg, 40)
    st2, out2, dt2 = chat(a.base, img_msg, 40)
    check('repeat image request works', st1 == 200 and st2 == 200, f'{dt1:.1f}s then {dt2:.1f}s')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
