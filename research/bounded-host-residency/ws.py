#!/usr/bin/env python3
"""Expert working set of a routing trace: unique expert bytes over rolling token windows.

usage: ws.py GEOMETRY.json TRACE.ids [TRACE.ids ...]
Tokens are taken in execution order (prefill rows, then decode steps); a turn starts where a
prefill follows decode steps.
"""
import collections, json, sys


def tokens(path):
    """-> list of (phase, frozenset of layer*256+expert) per token, and turn start indices."""
    chunks, last = [], None     # prefill steps repeat across turns, so keep them in order
    with open(path) as f:
        for line in f:
            v = line.split()
            if len(v) < 4:
                continue
            phase, step, layer = int(v[0]), int(v[1]), int(v[2])
            if (phase, step) != last:
                chunks.append((phase, collections.defaultdict(list)))
                last = (phase, step)
            chunks[-1][1][layer].append([layer * 256 + int(e) for e in v[3:]])
    out, turns, last_phase = [], [], None
    for phase, layers in chunks:
        if phase == 0 and last_phase != 0:
            turns.append(len(out))
        n = max(len(r) for r in layers.values())
        for i in range(n):
            ks = set()
            for rows in layers.values():
                if i < len(rows):
                    ks.update(rows[i])
            out.append((phase, frozenset(ks)))
        last_phase = phase
    return out, turns


def main():
    geo = json.load(open(sys.argv[1]))
    mib = geo['per_expert_mib']
    gib = lambda n: n * mib / 1024
    for path in sys.argv[2:]:
        toks, turns = tokens(path)
        name = path.rsplit('/', 1)[-1]
        pre = [t for t in toks if t[0] == 0]
        print(f'{name}: {len(toks)} tokens ({len(pre)} prefill), {len(turns)} turns, whole {gib(len(set().union(*[t[1] for t in toks]))):.2f} GiB')
        if turns:
            first_end = next((i for i, t in enumerate(toks) if t[0] != 0), len(toks))
            print(f'  first prefill ({first_end} tokens): {gib(len(set().union(*[t[1] for t in toks[:first_end]]))):.2f} GiB unique')
        for w in (256, 512, 1024, 2048):
            if w > len(toks):
                continue
            vals = []
            for s in range(0, len(toks) - w + 1, max(1, w // 4)):
                vals.append(gib(len(set().union(*[t[1] for t in toks[s:s + w]]))))
            vals.sort()
            print(f'  window {w:5d}: median {vals[len(vals) // 2]:.2f} p95 {vals[int(len(vals) * 0.95)]:.2f} max {vals[-1]:.2f} GiB')
        per_turn = []
        for i, s in enumerate(turns):
            e = turns[i + 1] if i + 1 < len(turns) else len(toks)
            per_turn.append(gib(len(set().union(*[t[1] for t in toks[s:e]]))))
        if per_turn:
            print('  per turn: ' + ' '.join(f'{v:.2f}' for v in per_turn) + ' GiB')
            seen, grow = set(), []
            for i, s in enumerate(turns):
                e = turns[i + 1] if i + 1 < len(turns) else len(toks)
                seen.update(*[t[1] for t in toks[s:e]])
                grow.append(gib(len(seen)))
            print('  cumulative: ' + ' '.join(f'{v:.2f}' for v in grow) + ' GiB')


if __name__ == '__main__':
    main()
