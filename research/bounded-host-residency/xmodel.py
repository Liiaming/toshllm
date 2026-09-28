#!/usr/bin/env python3
"""Full host bank against the bounded cache per model, in real speeds, from bench logs.

usage: xmodel.py LOGDIR WARMUP_TURNS NAME:FULL_PREFIX:BOUNDED_PREFIX:HOT_GIB ...
Expects PREFIX-first, PREFIX-wc (optional), PREFIX-long and BOUNDED_PREFIX-np-first.
"""
import re, sys


def read(p):
    try:
        return open(p, errors='replace').read()
    except OSError:
        return ''


def num(pat, t):
    m = re.search(pat, t)
    return float(m.group(1)) if m else None


def run(d, prefix, warm):
    f, w, l = (read(f'{d}/{prefix}-{k}.log') for k in ('first', 'wc', 'long'))
    r = dict(rss=num(r'MEM end: rss ([\d.]+) GiB', f), locked=(num(r'([\d.]+) MiB locked for experts', f) or 0) / 1024,
             ready=num(r'READY ([\d.]+) s', f), ttft=num(r'RESULT ttft ([\d.]+) ms', f),
             pp_tok=num(r'prompt (\d+) tokens, [\d.]+ t/s', f), pp=num(r'prompt \d+ tokens, ([\d.]+) t/s', f),
             dec_first=num(r'decode steps \d+ \(ub 1\), ms/token mean [\d.]+ .*t/s ([\d.]+)', f),
             long_ms=num(r'decode steps \d+ \(ub 1\), ms/token mean ([\d.]+)', l), long_ts=num(r'decode steps \d+ \(ub 1\).*t/s ([\d.]+)', l),
             hot=num(r'hit rate ([\d.]+)%', l), warm=num(r'decode  requests \d+, warm hits \d+ \(([\d.]+)%', l))
    m = re.search(r'TOKENS p50 ([\d.]+) p95 ([\d.]+) p99 ([\d.]+) worst ([\d.]+)', l)
    r['pct'] = [float(x) for x in m.groups()] if m else None
    turns = [(int(a), int(b), float(c)) for a, b, c in re.findall(r'TURN (\d+) at decode step \d+, prompt (\d+) tokens, prefill ([\d.]+) ms', w)]
    dec = {int(a): (int(b), float(c)) for a, b, c in re.findall(r'TURN (\d+) decode (\d+) tokens, mean ([\d.]+) ms', w)}
    after = [t for t in turns if t[0] >= warm]
    if after:
        short = [t for t in after if t[1] <= 100]
        r['short_tok'], r['short_ms'] = sum(t[1] for t in short), sum(t[2] for t in short)
        r['conv_tok'], r['conv_ms'] = sum(t[1] for t in after), sum(t[2] for t in after)
        n = sum(dec[t][0] for t in dec if t >= warm)
        r['conv_dec_ms'] = sum(dec[t][0] * dec[t][1] for t in dec if t >= warm) / max(1, n)
    cold = num(r'file reads: first touch (\d+)', l)
    r['cold_long'] = num(r'cold reads \d+ \(([\d.]+)/token\)', l)
    return r


def delta(new, old, unit, higher_better=True):
    if new is None or old is None:
        return 'n/a'
    d = new - old
    pct = abs(d) / old * 100 if old else 0
    better = (d > 0) == higher_better
    word = 'FASTER' if better else 'SLOWER'
    if abs(d) < 1e-9:
        return f'0 {unit} (same)'
    return f'{d:+.1f} {unit} ({pct:.1f}% {word})'


def main():
    d, warm = sys.argv[1], int(sys.argv[2])
    for spec in sys.argv[3:]:
        name, full, bnd, hot = spec.split(':')
        a, b = run(d, full, warm), run(d, bnd, warm)
        np = run(d, bnd + '-np', warm)
        print(f'### {name}')
        print(f'memory: full RSS {a["rss"]} GiB | bounded RSS {b["rss"]} GiB, RAM cache {b["locked"]:.2f} GiB locked, VRAM experts {hot} GiB | saved {a["rss"] - b["rss"]:.2f} GiB')
        print(f'prefill {int(a["pp_tok"] or 0)} tokens: full {a["pp"]:.1f} t/s ({a["ttft"]:.0f} ms) | bounded {b["pp"]:.1f} t/s ({b["ttft"]:.0f} ms) | {delta(b["pp"], a["pp"], "t/s")}')
        if 'short_ms' in a and 'short_ms' in b:
            fa, fb = a['short_tok'] / a['short_ms'] * 1000, b['short_tok'] / b['short_ms'] * 1000
            ca, cb = a['conv_tok'] / a['conv_ms'] * 1000, b['conv_tok'] / b['conv_ms'] * 1000
            print(f'short prompts after warmup ({a["short_tok"]} tokens): full {fa:.1f} t/s | bounded {fb:.1f} t/s | {delta(fb, fa, "t/s")}')
            print(f'all 12-turn prefill ({a["conv_tok"]} tokens): full {ca:.1f} t/s | bounded {cb:.1f} t/s | {delta(cb, ca, "t/s")}')
            print(f'12-turn decode: full {1000 / a["conv_dec_ms"]:.1f} t/s ({a["conv_dec_ms"]:.2f} ms) | bounded {1000 / b["conv_dec_ms"]:.1f} t/s ({b["conv_dec_ms"]:.2f} ms) | '
                  f'{delta(1000 / b["conv_dec_ms"], 1000 / a["conv_dec_ms"], "t/s")}')
        print(f'decode after the long prompt: full {a["dec_first"]} t/s | bounded {b["dec_first"]} t/s | {delta(b["dec_first"], a["dec_first"], "t/s")}')
        print(f'2048-token decode: full {a["long_ts"]} t/s ({a["long_ms"]} ms) | bounded {b["long_ts"]} t/s ({b["long_ms"]} ms) | {delta(b["long_ts"], a["long_ts"], "t/s")}')
        if a['pct'] and b['pct']:
            print(f'latency p50/p95/p99/worst ms: full {"/".join(map(str, a["pct"]))} | bounded {"/".join(map(str, b["pct"]))} | p99 {delta(b["pct"][2], a["pct"][2], "ms", False)}')
        print(f'launch to first token: full {a["ready"] + a["ttft"] / 1000:.1f} s | bounded prewarm {b["ready"] + b["ttft"] / 1000:.1f} s (ready {b["ready"]}) | '
              f'bounded cold {np["ready"] + np["ttft"] / 1000:.1f} s (ready {np["ready"]})' if np['ready'] else '')
        print(f'2048 decode HOT hit {b["hot"]}%, RAM hit {b["warm"]}%, storage reads {b["cold_long"]}/token')
        print()


if __name__ == '__main__':
    main()
