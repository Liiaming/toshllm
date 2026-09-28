#!/usr/bin/env python3
"""Summary of bench logs from the prototype runs: TAG-first, TAG-wc, TAG-long per WARM size.

usage: summ.py LOGDIR WARMUP_TURNS TAG...
Steady prefill is the sum over the prompts of at most 100 tokens after the warmup turns.
"""
import re, sys


def grab(path):
    try:
        return open(path, errors='replace').read()
    except OSError:
        return ''


def num(pat, text, g=1, default=None):
    m = re.search(pat, text)
    return float(m.group(g)) if m else default


def main():
    d, warm = sys.argv[1], int(sys.argv[2])
    rows = []
    for tag in sys.argv[3:]:
        f, w, l = (grab(f'{d}/{tag}-{k}.log') for k in ('first', 'wc', 'long'))
        r = {'tag': tag}
        r['rss'] = num(r'MEM end: rss ([\d.]+) GiB', w)
        r['locked'] = num(r'([\d.]+) MiB locked for experts', w, default=0) / 1024
        r['ready'] = num(r'READY ([\d.]+) s', f)
        r['ttft'] = num(r'RESULT ttft ([\d.]+) ms', f)
        r['first_dec'] = num(r'decode steps \d+ \(ub 1\), ms/token mean ([\d.]+)', f)
        r['kl'] = num(r'KL mean ([\d.e+-]+)', f)
        r['top1'] = num(r'top-1 agree ([\d.]+)%', f)
        r['nan'] = num(r'non-finite logits (\d+)', f)
        turns = [(int(a), int(b), float(c), int(e)) for a, b, c, e in
                 re.findall(r'TURN (\d+) at decode step \d+, prompt (\d+) tokens, prefill ([\d.]+) ms .*?cold reads (\d+)', w)]
        dec = {int(a): (int(b), float(c)) for a, b, c in re.findall(r'TURN (\d+) decode (\d+) tokens, mean ([\d.]+) ms', w)}
        after = [t for t in turns if t[0] >= warm]
        r['steady_pf'] = sum(t[2] for t in after if t[1] <= 100)
        r['steady_n'] = sum(1 for t in after if t[1] <= 100)
        r['big_pf'] = ' '.join(f'{t[1]}:{t[2]:.0f}' for t in after if t[1] > 100)
        n = sum(dec[t][0] for t in dec if t >= warm)
        r['dec_after'] = sum(dec[t][0] * dec[t][1] for t in dec if t >= warm) / max(1, n)
        r['dec_all'] = num(r'multi-turn\), ms/token mean ([\d.]+)', w)
        r['cold_after_tok'] = sum(e for t in turns if t[0] >= warm for e in [t[3]]) / max(1, n)
        r['hot'] = num(r'hit rate ([\d.]+)%', w)
        r['warm_dec'] = num(r'decode  requests \d+, warm hits \d+ \(([\d.]+)%', w)
        r['evict'] = num(r'evictions (\d+), reloads', w, default=0)
        r['reload'] = num(r'reloads (\d+) \|', w, default=0)
        r['dem'] = num(r'demand    reads (\d+)', w, default=0)
        r['bg'] = num(r'read-back reads (\d+)', w, default=0)
        r['long'] = num(r'decode steps \d+ \(ub 1\), ms/token mean ([\d.]+)', l)
        m = re.search(r'TOKENS p50 ([\d.]+) p95 ([\d.]+) p99 ([\d.]+) worst ([\d.]+)', l)
        r['long_pct'] = '/'.join(m.groups()) if m else ''
        r['long_cold'] = num(r'cold reads \d+ \(([\d.]+)/token\)', l)
        m = re.search(r'COLDCURVE(.*)', l)
        r['curve'] = m.group(1).strip() if m else ''
        rows.append(r)
    keys = ['rss', 'locked', 'ready', 'ttft', 'first_dec', 'kl', 'top1', 'nan', 'steady_pf', 'steady_n', 'dec_after', 'dec_all',
            'cold_after_tok', 'hot', 'warm_dec', 'evict', 'reload', 'dem', 'bg', 'long', 'long_pct', 'long_cold']
    for k in keys:
        print(f'{k:>14s} ' + ' '.join(f'{(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k])):>18s}' for r in rows))
    for r in rows:
        print(f'{r["tag"]}: big prefills after warmup {r["big_pf"]} | curve {r["curve"]}')


if __name__ == '__main__':
    main()
