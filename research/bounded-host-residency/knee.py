#!/usr/bin/env python3
"""RAM-knee table from bench logs TAG-first, TAG-wc (warmup + 12 turns), TAG-long per RAM size.

usage: knee.py LOGDIR WARMUP_TURNS BANK_GIB HOT_GIB FULL_TAG TAG:WARM_GIB...
Loss is elapsed time against FULL_TAG; the throughput it leaves is 1 / (1 + loss).
"""
import re, sys


def read(path):
    try:
        return open(path, errors='replace').read()
    except OSError:
        return ''


def num(pat, text, default=None):
    m = re.search(pat, text)
    return float(m.group(1)) if m else default


def stats(d, tag, warm):
    f, w, l = (read(f'{d}/{tag}-{k}.log') for k in ('first', 'wc', 'long'))
    turns = [(int(a), int(b), float(c)) for a, b, c in re.findall(r'TURN (\d+) at decode step \d+, prompt (\d+) tokens, prefill ([\d.]+) ms', w)]
    dec = {int(a): (int(b), float(c)) for a, b, c in re.findall(r'TURN (\d+) decode (\d+) tokens, mean ([\d.]+) ms', w)}
    after = [t for t in turns if t[0] >= warm]
    n = sum(dec[t][0] for t in dec if t >= warm)
    tokens = sum(t[1] for t in turns) + sum(v[0] for v in dec.values())
    m = re.search(r'TOKENS p50 ([\d.]+) p95 ([\d.]+) p99 ([\d.]+) worst ([\d.]+)', l)
    return dict(
        rss=num(r'MEM end: rss ([\d.]+) GiB', w), locked=num(r'([\d.]+) MiB locked for experts', w, 0) / 1024,
        ready=num(r'READY ([\d.]+) s', f), ttft=num(r'RESULT ttft ([\d.]+) ms', f), kl=num(r'KL mean ([\d.e+-]+)', f),
        top1=num(r'top-1 agree ([\d.]+)%', f), nan=num(r'non-finite logits (\d+)', f),
        short=sum(t[2] for t in after if t[1] <= 100), p516=sum(t[2] for t in after if 400 < t[1] < 700),
        p3381=sum(t[2] for t in after if t[1] > 3000), conv=sum(t[2] for t in after),
        dec=sum(dec[t][0] * dec[t][1] for t in dec if t >= warm) / max(1, n),
        first_touch=(num(r'first touch (\d+)', w, 0)) / max(1, tokens), capacity=(num(r'capacity miss (\d+)', w, 0)) / max(1, tokens),
        readback=(num(r'read-back (\d+)\n', w, 0)) / max(1, tokens), hot=num(r'hit rate ([\d.]+)%', w),
        long=num(r'decode steps \d+ \(ub 1\), ms/token mean ([\d.]+)', l), pct=[float(x) for x in m.groups()] if m else [0, 0, 0, 0])


def main():
    d, warm, bank, hot, full = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    b = stats(d, full, warm)
    rows = [('full', bank, b)] + [(t.split(':')[0], float(t.split(':')[1]), stats(d, t.split(':')[0], warm)) for t in sys.argv[6:]]
    loss = lambda x, y: (x / y - 1) * 100 if y else 0
    print('| RAM cache | RSS | RAM saved | coverage | short prefill | 516 | 3381 | 12-turn prefill | decode | 2048 decode | p50/p95/p99/worst | TTFT |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|')
    for tag, gib, s in rows:
        cov = min(1.0, (hot + gib) / bank) if tag != 'full' else 1.0
        print(f'| {tag} {gib:g} GiB | {s["rss"]} | {b["rss"] - s["rss"]:.1f} | {cov:.2f} | {s["short"]:.0f} ({loss(s["short"], b["short"]):+.0f}%) | '
              f'{s["p516"]:.0f} ({loss(s["p516"], b["p516"]):+.0f}%) | {s["p3381"]:.0f} ({loss(s["p3381"], b["p3381"]):+.0f}%) | '
              f'{s["conv"]:.0f} ({loss(s["conv"], b["conv"]):+.0f}%) | {s["dec"]:.2f} ({loss(s["dec"], b["dec"]):+.1f}%) | '
              f'{s["long"]:.2f} ({loss(s["long"], b["long"]):+.1f}%) | {"/".join(f"{x:.1f}" for x in s["pct"])} | {s["ttft"]:.0f} |')
    print()
    for tag, gib, s in rows:
        print(f'{tag}: locked {s["locked"]:.2f} GiB, ready {s["ready"]} s, HOT hit {s["hot"]}%, per token: first touch {s["first_touch"]:.2f}, '
              f'capacity {s["capacity"]:.3f}, read-back {s["readback"]:.2f} | KL {s["kl"]} top-1 {s["top1"]} non-finite {s["nan"]} | '
              f'short-prefill throughput {100 / (1 + loss(s["short"], b["short"]) / 100):.0f}% of full')


if __name__ == '__main__':
    main()
