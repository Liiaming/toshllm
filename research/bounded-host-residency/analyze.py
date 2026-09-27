#!/usr/bin/env python3
"""Aggregate simulator output over workloads and turn storage reads into time.

usage: analyze.py GEOMETRY.json BASE_DECODE_MS BASE_PREFILL_TPS NONEXPERT_GIB RESULT.json...
Storage figures are this machine's NVMe (pread, F_NOCACHE), see storage.c.
"""
import collections, json, sys

PART_LAT_MS = 0.61      # one 576 KiB read
BW_GBS = 3.2            # large or parallel reads
STAGING_GIB = 0.25


def main():
    geo = json.load(open(sys.argv[1]))
    base_ms, base_pp, nonexpert = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    xb_mib = geo['per_expert_mib']
    parts = len(geo['parts'])
    rows = []
    for f in sys.argv[5:]:
        rows += json.load(open(f))
    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        k = (r['warm'], r['policy'], r.get('bypass', 0))
        a = agg[k]
        a['tok'] += r['decode_tokens']
        a['pre_tok'] += r['prefill_tokens']
        a['cap'] += r['decode_capacity_per_token'] * r['decode_tokens']
        a['layers'] += r['decode_capacity_layers_per_token'] * r['decode_tokens']
        a['acc'] += r['decode_host_access']
        a['hit'] += r['decode_warm_hit'] * r['decode_host_access']
        a['pre_cap'] += r['prefill_capacity']
        a['pre_cold'] += r['prefill_cold']
        a['hot'] += r['hot_hit'] * r['decode_tokens']
    order = lambda w: 1e9 if w == 'full' else float(w)
    print('warm_gib policy bypass | rss_gib | decode: host hit, capacity reads/token, MiB/token, +ms serial, +ms overlapped, slowdown | '
          'prefill: capacity MiB/ktok, +s/ktok, slowdown')
    for (w, pol, bp), a in sorted(agg.items(), key=lambda x: (order(x[0][0]), x[0][1], x[0][2])):
        tok = max(1, a['tok'])
        cap, layers = a['cap'] / tok, a['layers'] / tok
        serial = cap * parts * PART_LAT_MS
        overlap = layers * PART_LAT_MS + max(0.0, cap - layers) * xb_mib / 1024 / BW_GBS * 1000
        pre_mib_k = a['pre_cap'] * xb_mib / max(1, a['pre_tok']) * 1000
        pre_s_k = pre_mib_k / 1024 / BW_GBS
        warm_gib = geo['bank_gib'] if w == 'full' else float(w)
        rss = nonexpert + warm_gib + (0 if w == 'full' else STAGING_GIB)
        print(f"{w:>4} {pol:3s} {bp} | {rss:5.1f} | {a['hit'] / max(1, a['acc']):.3f} {cap:6.2f} {cap * xb_mib:6.2f} "
              f"{serial:6.2f} {overlap:6.2f} {overlap / base_ms * 100:5.1f}% | {pre_mib_k:8.0f} {pre_s_k:5.2f} "
              f"{pre_s_k / (1000 / base_pp) * 100:6.1f}%")


if __name__ == '__main__':
    main()
