"""Charts for the 2026-10-25 update note (power cap, upstream A/B, memory map, conversation cache).

Run: uv run --no-project --with matplotlib python local/v100/report/update_20261025.py
Numbers are the measured ones in docs/V100_TUNING.md (rounds 8-10); the conversation cache chart reads
local/benchmarks/v100-convcache/results.json and the memory map reads memory.json next to this script.
"""
import json
import pathlib
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path('/home/aim/Documents/Obsidian/_attachments/strata-v100-update-20261025')
OUT.mkdir(parents=True, exist_ok=True)
GREY, TEAL, AMBER, RED = '#8c9aa8', '#127c87', '#d99a2b', '#c0504d'
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})


def label_bars(ax, bars, fmt, color='black'):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', color=color)


def power():
    caps = ['135 W', '150 W', '165 W']
    prefill, peak = [1298, 1393, 1464], [231, 251, 282]
    colors = [GREY, TEAL, GREY]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    label_bars(a, a.bar(caps, prefill, color=colors), lambda v: f'{v:,.0f}')
    a.set_title('Reading an 80K-token prompt (tokens/s, higher is faster)')
    a.set_ylim(0, 1700)
    label_bars(b, b.bar(caps, peak, color=colors), lambda v: f'{v:.0f} W')
    b.axhline(250, color=RED, ls='--', lw=1.5)
    b.text(-0.42, 256, "card's 250 W rating", color=RED, ha='left', va='bottom')
    b.set_title('Highest power spike measured (W)')
    b.set_ylim(0, 320)
    fig.suptitle('Power cap: 150 W is the fastest setting whose spikes stay at the card rating', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'power-cap.png', dpi=110)


def upstream():
    rows = [('Reading an 80K prompt (tok/s)', 1395, 1526), ('Reading a 128K prompt (tok/s)', 1288, 1407),
            ('Writing, short prompt (tok/s)', 89.1, 91.3), ('Coding tasks passed (of 24)', 23, 24),
            ('Median coding task (s, lower is better)', 28.3, 32.2)]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    y = range(len(rows))
    for i, (name, prod, new) in enumerate(rows):
        top = max(prod, new)
        ax.barh(i - 0.19, prod / top, 0.36, color=TEAL)
        ax.barh(i + 0.19, new / top, 0.36, color=GREY)
        ax.text(prod / top + 0.01, i - 0.19, f'{prod:g}', va='center', color=TEAL)
        ax.text(new / top + 0.01, i + 0.19, f'{new:g}', va='center')
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlim(0, 1.15)
    ax.spines['bottom'].set_visible(False)
    ax.legend(['in use: engine 0.1.43.1 + Volta kernels', 'tested: upstream engine 0.1.50'], loc='upper center',
              bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.suptitle('Upstream engine: faster at reading long prompts, no better at real coding work\n'
                 '(each row on its own scale)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'upstream-ab.png', dpi=110)


def memory():
    m = json.loads((HERE / 'memory.json').read_text())
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4))
    for ax, key, title in ((axes[0], 'vram', 'Graphics card memory (V100, 32 GB)'),
                           (axes[1], 'ram', 'System memory (64 GB)')):
        left, handles = 0.0, []
        for part in m[key]:
            w = part['gib']
            handles.append(ax.barh(0, w, left=left, color=part['color'], edgecolor='white',
                                   label=f"{part['label']} {w:.1f} GB"))
            if w >= 8:   # only the wide segments carry their label inline; the legend names every part
                ax.text(left + w / 2, 0, f"{part['label']}\n{w:.1f} GB", ha='center', va='center', fontsize=10,
                        color='white' if part['color'] in (TEAL, '#c0504d', '#4a6378') else 'black')
            left += w
        ax.set_xlim(0, left)
        ax.set_yticks([])
        ax.set_title(title, loc='left', fontweight='bold')
        ax.spines['left'].set_visible(False)
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False,
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'memory-map.png', dpi=110)


def convcache():
    res = json.loads(pathlib.Path('/home/aim/opt/UnbiasedStrata/local/benchmarks/v100-convcache/results.json')
                     .read_text())
    def mean(cand, step):
        return statistics.mean(s['seconds'] for r in res if r['candidate'] == cand for s in r['steps']
                               if s['step'].startswith(step))
    steps = [('A1', 'First long message\n(80K tokens + screenshot)'),
             ('A2', 'Next message after two\nside requests'),
             ('A3', 'Next message with\nnothing in between')]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = range(len(steps))
    off = [mean('off', s) for s, _ in steps]
    on = [mean('cache4g', s) for s, _ in steps]
    label_bars(ax, ax.bar([i - 0.2 for i in x], off, 0.38, color=GREY, label='conversation cache off (before)'),
               lambda v: f'{v:.1f} s')
    label_bars(ax, ax.bar([i + 0.2 for i in x], on, 0.38, color=TEAL, label='conversation cache on (now)'),
               lambda v: f'{v:.1f} s', TEAL)
    ax.set_xticks(list(x), [s[1] for s in steps])
    ax.set_ylabel('seconds until the reply is done')
    ax.legend(frameon=False)
    ax.set_title('A long chat interrupted by side requests: the cache saves the re-read', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'conversation-cache.png', dpi=110)


def headline():
    panels = [('Long chat, next message after\nan interruption (seconds)', ['before', 'now'], [62.0, 1.0], '{:.0f} s'),
              ('Reading an 80K-token prompt\n(tokens per second)', ['135 W', '150 W'], [1298, 1393], '{:,.0f}'),
              ('Screenshot questions\nanswered correctly (of 7)', ['before', 'now'], [7, 7], '{:.0f}')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (title, names, vals, fmt) in zip(axes, panels):
        label_bars(ax, ax.bar(names, vals, color=[GREY, TEAL], width=0.6), fmt.format)
        ax.set_title(title, fontsize=12)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.suptitle('2026-10-25 at a glance: much faster long chats, a bit faster reading, vision unchanged',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'headline.png', dpi=110)


if __name__ == '__main__':
    headline()
    power()
    upstream()
    if (HERE / 'memory.json').exists():
        memory()
    if pathlib.Path('/home/aim/opt/UnbiasedStrata/local/benchmarks/v100-convcache/results.json').exists():
        convcache()
    print('wrote', sorted(p.name for p in OUT.iterdir()))
