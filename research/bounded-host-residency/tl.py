#!/usr/bin/env python3
"""Prefill time attribution from a Dynamic MoE timeline (TOSH_DMOE_TIMELINE) and the bench log.

usage: tl.py RUN.tl [BASE.tl]
Kinds: 0 main waits for the chunk's routing ids, 3 main issues the bank uploads, inside it
33 pinning cached experts, 7 waiting for a ring chunk, 5 copying into the ring; 1 host executor
wide job, 31 its pin wait; 2 ring blit on the GPU; 8/9 demand read queued/reading, 10/11 the
same for read-backs; 40 one prefill (bench, layer field = turn).
The main thread runs 0, 3 and what is left of the prefill window serially, so those partition
it; the rest is graph building, splits that are not expert banks and the final GPU wait.
"""
import collections, sys


def load(path):
    ev = collections.defaultdict(list)
    for line in open(path):
        k, il, t0, t1 = line.split()
        ev[int(k)].append((float(t0), float(t1), int(il)))
    for v in ev.values():
        v.sort()
    return ev


def clip(iv, a, b):
    return sum(max(0.0, min(t1, b) - max(t0, a)) for t0, t1, _ in iv if t1 > a and t0 < b)


def union(iv, a, b):
    """length of the union of intervals clipped to [a, b]"""
    segs = sorted((max(t0, a), min(t1, b)) for t0, t1, _ in iv if t1 > a and t0 < b)
    tot, cur0, cur1 = 0.0, None, None
    for s0, s1 in segs:
        if cur1 is None or s0 > cur1:
            if cur1 is not None:
                tot += cur1 - cur0
            cur0, cur1 = s0, s1
        else:
            cur1 = max(cur1, s1)
    if cur1 is not None:
        tot += cur1 - cur0
    return tot


def turns(ev):
    rows = []
    for a, b, turn in ev[40]:
        w = b - a
        ids = clip(ev[0], a, b)
        issue = clip(ev[3], a, b)
        acq = clip(ev[33], a, b)
        ring = clip(ev[7], a, b)
        mem = clip(ev[5], a, b)
        rows.append(dict(turn=turn, wall=w, ids=ids, issue_acq=acq, issue_ring=ring, issue_copy=mem,
                         issue_other=issue - acq - ring - mem, rest=w - ids - issue,
                         exec_busy=union(ev[1], a, b), exec_pin=union(ev[31], a, b),
                         ids_while_exec_pins=sum(union(ev[31], max(t0, a), min(t1, b)) for t0, t1, _ in ev[0] if t1 > a and t0 < b),
                         dem_reads=sum(1 for t0, t1, _ in ev[9] if a <= t0 < b), bg_reads=sum(1 for t0, t1, _ in ev[11] if a <= t0 < b),
                         read_busy=union(ev[9] + ev[11], a, b),
                         dem_q=sorted(t1 - t0 for t0, t1, _ in ev[8] if a <= t0 < b)))
    return rows


COLS = ['wall', 'ids', 'issue_acq', 'issue_ring', 'issue_copy', 'issue_other', 'rest', 'exec_busy', 'exec_pin', 'ids_while_exec_pins', 'read_busy']


def show(rows, base=None):
    print('turn ' + ' '.join(f'{c:>10s}' for c in COLS) + '  demand/bg reads  demand queue p50/p95 ms')
    for i, r in enumerate(rows):
        line = f'{r["turn"]:4d} ' + ' '.join(f'{r[c]*1e3:10.1f}' for c in COLS)
        q = r['dem_q']
        line += f'  {r["dem_reads"]:5d}/{r["bg_reads"]:<5d}  {q[len(q)//2]*1e3 if q else 0:.2f}/{q[int(len(q)*0.95)]*1e3 if q else 0:.2f}'
        print(line)
        if base and i < len(base):
            b = base[i]
            print('  d  ' + ' '.join(f'{(r[c]-b[c])*1e3:10.1f}' for c in COLS))


def main():
    rows = turns(load(sys.argv[1]))
    base = turns(load(sys.argv[2])) if len(sys.argv) > 2 else None
    show(rows, base)


if __name__ == '__main__':
    main()
