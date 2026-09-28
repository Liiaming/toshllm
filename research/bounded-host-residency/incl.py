#!/usr/bin/env python3
"""HOT/WARM inclusivity economics on a routing trace, with the VRAM policy of sim.py.

usage: incl.py GEOMETRY.json ARENA_MIB WARM_GIB TRACE.ids [TRACE.ids ...]
Modes for an expert promoted to VRAM (its RAM copy counts against the RAM budget in all of them):
  excl     the RAM copy is freed; eviction from VRAM refills it (a read or a copy)
  lazy     the copy is kept, first in line for eviction (0117 TOSH_DMOE_HOST_DROP=0)
  incl     the copy is kept where it is in the LRU
  protect  the copy is kept and cannot be evicted while the expert is in VRAM
  sel:N    kept in place when the expert left VRAM in the last N tokens, else lazy
"""
import collections, json, sys
from sim import load_trace, simulate_hot


def replay(events, cap, mode):
    items = collections.OrderedDict()      # RAM cache, LRU order
    hot, seen, evicted_at = set(), set(), {}
    st = collections.Counter()
    dups, promo_at, lifetimes, reuse_gap, last_evict = [], {}, [], [], {}
    sel_n = int(mode.split(':')[1]) if mode.startswith('sel:') else 0

    def insert(k):
        while len(items) >= cap:
            victim = next((x for x in items if not (mode == 'protect' and x in hot)), None)
            if victim is None:
                return False
            del items[victim]
            st['warm_evict'] += 1
            if victim in hot:
                st['dup_dropped'] += 1
        items[k] = True
        return True

    for gi, clock, phase, kind, l, e, rows in events:
        k = (l, e)
        if kind == 'evict':
            hot.discard(k)
            lifetimes.append(clock - promo_at.pop(k, clock))
            last_evict[k] = clock
            evicted_at[k] = clock
            if k in items:
                items.move_to_end(k)       # needed again soon: no data moves
                st['evict_kept'] += 1
            else:
                st['refill'] += 1          # a read from the file or a copy from VRAM
                insert(k)
            continue
        if kind == 'promote':
            hot.add(k)
            promo_at[k] = clock
            if k not in items:
                st['promote_miss'] += 1    # carried by a prefill upload or read now
                seen.add(k)
                if mode != 'excl':
                    insert(k)
                    if mode == 'lazy' or (sel_n and not (k in evicted_at and clock - evicted_at[k] < sel_n)):
                        items.move_to_end(k, last=False)
            elif mode == 'excl':
                del items[k]
            elif mode == 'lazy' or (sel_n and not (k in evicted_at and clock - evicted_at[k] < sel_n)):
                items.move_to_end(k, last=False)
            dups.append(sum(1 for x in hot if x in items) if gi % 16 == 0 else dups[-1] if dups else 0)
            continue
        # a host access: decode miss or prefill upload
        st['access'] += 1
        if k in last_evict:
            reuse_gap.append(clock - last_evict.pop(k))
        if k in items:
            items.move_to_end(k)
            st['hit'] += 1
        else:
            st['first' if k not in seen else 'capacity'] += 1
            seen.add(k)
            insert(k)
    dups.sort(); lifetimes.sort(); reuse_gap.sort()
    q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else 0
    return st, dict(dup_mean=sum(dups) / max(1, len(dups)), dup_p95=q(dups, 0.95), dup_max=dups[-1] if dups else 0,
                    life_p50=q(lifetimes, 0.5), life_p90=q(lifetimes, 0.9), gap_p50=q(reuse_gap, 0.5), gap_p90=q(reuse_gap, 0.9),
                    n_gap=len(reuse_gap))


def main():
    geo = json.load(open(sys.argv[1]))
    arena, warm = float(sys.argv[2]), float(sys.argv[3])
    xb, L, E = geo['per_expert_mib'], geo['moe_layers'], geo['n_expert']
    slots = min(E, int(arena / (L * xb)))
    cap = int(warm * 1024 / xb)
    gib = lambda n: n * xb / 1024
    for tr in sys.argv[4:]:
        graphs = load_trace(tr)
        tokens = sum(1 if g[0] == 1 else max(1, sum(next(iter(g[2].values())).values()) // geo['n_used']) for g in graphs)
        n_dec = sum(1 for g in graphs if g[0] == 1)
        events, _, _ = simulate_hot(graphs, geo['n_used'], slots)
        print(f'{tr.rsplit("/", 1)[-1]}: {tokens} tokens ({n_dec} decode), HOT {slots} slots/layer = {gib(slots * L):.2f} GiB, RAM cache {warm} GiB')
        for mode in ('excl', 'lazy', 'incl', 'protect', 'sel:64', 'sel:256'):
            st, d = replay(events, cap, mode)
            print(f'  {mode:8s} refills/token {st["refill"] / tokens:5.2f} ({gib(st["refill"]):6.1f} GiB) evict_kept {st["evict_kept"]:6d} '
                  f'capacity/token {st["capacity"] / tokens:5.2f} first {st["first"]:5d} hit {st["hit"] / max(1, st["access"]):.3f} | '
                  f'duplicates mean {gib(d["dup_mean"]):.2f} p95 {gib(d["dup_p95"]):.2f} max {gib(d["dup_max"]):.2f} GiB, dropped while HOT {st["dup_dropped"]}')
        print(f'  HOT lifetime p50 {d["life_p50"]} p90 {d["life_p90"]} tokens; after eviction requested again: {d["n_gap"]}, gap p50 {d["gap_p50"]} p90 {d["gap_p90"]} tokens')


if __name__ == '__main__':
    main()
