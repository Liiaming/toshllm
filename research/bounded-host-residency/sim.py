#!/usr/bin/env python3
"""Offline HOT (VRAM) / WARM (RAM) / COLD (GGUF) expert residency simulator.

Stage 1 replays a routing trace through an approximation of the Dynamic MoE VRAM policy and
records every access the host has to serve: decode misses computed on the CPU, promotions read
for upload, and the routed non-resident experts of each prefill chunk. Stage 2 replays those
accesses through a bounded RAM cache under several policies.

Trace rows (from trace.cpp): phase step layer id0 id1 ...  (prefill steps are -(chunk + 1))
"""
import argparse, collections, json, os

DECAY = 0.98        # per graph, as the runtime policy
ADMIT = 1.8         # a candidate must beat the weakest resident by this factor
PROMO_BUDGET = 8    # uploads in flight per graph (the stage ring)


def load_trace(path):
    """-> list of graphs: (phase, key, {layer: [(expert, rows)]}) in execution order."""
    graphs, cur_key, cur = [], None, None
    with open(path) as f:
        for line in f:
            v = line.split()
            if len(v) < 4:
                continue
            phase, step, layer = int(v[0]), int(v[1]), int(v[2])
            key = (phase, step)
            if key != cur_key:
                cur = collections.defaultdict(collections.Counter)
                graphs.append((phase, step, cur))
                cur_key = key
            for e in v[3:]:
                cur[layer][int(e)] += 1
    return graphs


class Decayed:
    """Routing frequency with per-token decay, kept as (value, clock) and decayed on read."""

    def __init__(self):
        self.v = {}

    def get(self, k, now):
        x = self.v.get(k)
        return x[0] * DECAY ** (now - x[1]) if x else 0.0

    def add(self, k, w, now):
        self.v[k] = (self.get(k, now) + w, now)


def simulate_hot(graphs, n_used, slots):
    """VRAM residency. Returns host accesses and HOT membership changes, in order.
    event = (graph, clock, phase, kind, layer, expert, rows); kind in cpu, promote, prefill, evict
    """
    score, resident = Decayed(), collections.defaultdict(set)
    events, hot_hits, routed_decode, clock = [], 0, 0, 0
    for gi, (phase, step, layers) in enumerate(graphs):
        # a chunk of T tokens routes T * k selections in every layer
        tokens = 1 if phase == 1 else max(1, sum(next(iter(layers.values())).values()) // n_used)
        clock += tokens
        for l, cnt in layers.items():
            res = resident[l]
            for e, rows in cnt.items():
                score.add((l, e), rows, clock)
                if phase == 1:
                    routed_decode += 1
                    if e in res:
                        hot_hits += 1
                    else:
                        events.append((gi, clock, 1, 'cpu', l, e, 1))
                elif e not in res:
                    events.append((gi, clock, 0, 'prefill', l, e, rows))
        # after the graph: promote the strongest missed experts, within the per-graph budget
        budget = PROMO_BUDGET if phase == 1 else max(1, min(8, slots // 8)) * len(layers)
        cands = sorted(((score.get((l, e), clock), l, e) for l, cnt in layers.items() for e in cnt
                        if e not in resident[l]), reverse=True)
        for s, l, e in cands:
            res = resident[l]
            if len(res) < slots:
                # free slots are lent to the prefill uploads, which carry these experts anyway
                res.add(e)
                events.append((gi, clock, phase, 'promote', l, e, 0))
                budget -= phase == 1
                continue
            if budget <= 0:
                continue
            victim = min(res, key=lambda x: score.get((l, x), clock))
            if s > ADMIT * score.get((l, victim), clock) + 1e-9:
                res.discard(victim)
                res.add(e)
                events.append((gi, clock, phase, 'evict', l, victim, 0))
                events.append((gi, clock, phase, 'promote', l, e, 0))
                budget -= 1
    return events, hot_hits, routed_decode


class Warm:
    """Bounded RAM cache of whole experts. lru, or lfu (decayed routing frequency seen by the host).
    exclusive: an expert promoted to VRAM leaves RAM and returns on eviction from VRAM (a
    VRAM->RAM copy, not a storage read)."""

    def __init__(self, cap, policy, exclusive):
        self.cap, self.policy, self.exclusive = cap, policy, exclusive
        self.items = collections.OrderedDict()
        self.freq = Decayed()
        self.clock = 0
        self.loaded_at, self.uses = {}, collections.Counter()
        self.lifetimes, self.uses_per_load = [], []

    def touch(self, k):
        if self.policy == 'lru':
            self.items.move_to_end(k)
        self.uses[k] += 1

    def _drop(self, k):
        del self.items[k]
        self.lifetimes.append(self.clock - self.loaded_at.pop(k, self.clock))
        self.uses_per_load.append(self.uses.pop(k, 0))

    def insert(self, k):
        if self.cap <= 0:
            return
        while len(self.items) >= self.cap:
            if self.policy == 'lru':
                victim = next(iter(self.items))
            else:
                victim = min(self.items, key=lambda x: self.freq.get(x, self.clock))
            self._drop(victim)
        self.items[k] = self.clock
        self.loaded_at[k] = self.clock

    def discard(self, k):
        if k in self.items:
            self._drop(k)


def simulate_warm(events, cap, policy, exclusive, prefill_bypass):
    w = Warm(cap, policy, exclusive)
    loaded_ever, st = set(), collections.Counter()
    per_graph_cold = collections.defaultdict(list)   # decode graph -> layers with a storage read
    for gi, clock, phase, kind, l, e, rows in events:
        w.clock = clock
        k = (l, e)
        if kind == 'evict':
            if exclusive:
                # the runtime reads it back from the file: storage traffic off the critical path
                if k not in w.items:
                    st['refill'] += 1
                w.insert(k)
            continue
        if kind != 'promote':
            w.freq.add(k, 1.0 if phase == 1 else 0.25, clock)
        if kind == 'promote':
            if k in w.items:
                w.touch(k)
                if exclusive:
                    w.discard(k)
                continue
            if phase == 0 or not exclusive:
                continue             # a prefill upload or a decode miss already read it this graph
            st['cold_promote'] += 1
            continue
        st[f'access_{phase}'] += 1
        if k in w.items:
            w.touch(k)
            st[f'hit_{phase}'] += 1
            continue
        st[f'cold_{phase}'] += 1
        # a first touch is paid once after load, whatever the cache size; a reload is capacity
        if k in loaded_ever:
            st['reload'] += 1
            st[f'capacity_{phase}'] += 1
            if phase == 1:
                per_graph_cold[gi].append(l)
        loaded_ever.add(k)
        if phase == 0 and prefill_bypass and rows > 1:
            continue                 # file -> staging -> VRAM, not kept in RAM
        w.insert(k)
    st['unique_cold'] = len(loaded_ever)
    return st, per_graph_cold, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', nargs='+', required=True)
    ap.add_argument('--geometry', required=True, help='json from geometry.py')
    ap.add_argument('--arena-mib', type=float, required=True)
    ap.add_argument('--warm-gib', default='0.5,1,2,3,4,6,8,10,12,full')
    ap.add_argument('--policies', default='lru,lfu')
    ap.add_argument('--exclusive', type=int, default=1)
    ap.add_argument('--prefill-bypass', type=int, default=0)
    args = ap.parse_args()
    geo = json.load(open(args.geometry))
    E, L = geo['n_expert'], geo['moe_layers']
    xb = geo['per_expert_mib']
    slots = min(E, int(args.arena_mib / (L * xb)))
    bank_mib = geo['bank_gib'] * 1024
    out = []
    for tr in args.trace:
        graphs = load_trace(tr)
        n_dec = sum(1 for g in graphs if g[0] == 1)
        n_pre_tok = sum(sum(next(iter(g[2].values())).values()) // geo['n_used'] for g in graphs if g[0] == 0)
        events, hot_hits, routed = simulate_hot(graphs, geo['n_used'], slots)
        for wg in args.warm_gib.split(','):
            cap_mib = bank_mib if wg == 'full' else float(wg) * 1024
            cap = int(cap_mib / xb)
            for pol in args.policies.split(','):
                st, pgc, w = simulate_warm(events, cap, pol, bool(args.exclusive), bool(args.prefill_bypass))
                dec_cold = st['cold_1']
                layers_with_cold = [len(set(v)) for v in pgc.values()]
                out.append(dict(trace=os.path.basename(tr), warm=wg, policy=pol, slots=slots,
                                hot_hit=round(hot_hits / max(1, routed), 4), decode_tokens=n_dec, prefill_tokens=n_pre_tok,
                                decode_host_access=st['access_1'] + st['warm_hit_promote'] * 0,
                                decode_warm_hit=round(st['hit_1'] / max(1, st['access_1']), 4),
                                decode_cold_per_token=round(dec_cold / max(1, n_dec), 4),
                                decode_capacity_per_token=round(st['capacity_1'] / max(1, n_dec), 4),
                                decode_capacity_mib_per_token=round(st['capacity_1'] * xb / max(1, n_dec), 3),
                                decode_capacity_layers_per_token=round(sum(layers_with_cold) / max(1, n_dec), 4),
                                decode_cold_layers_hist=collections.Counter(min(x, 6) for x in layers_with_cold),
                                prefill_cold=st['cold_0'], prefill_capacity=st['capacity_0'],
                                prefill_capacity_mib_per_ktok=round(st['capacity_0'] * xb / max(1, n_pre_tok) * 1000, 1),
                                prefill_access=st['access_0'],
                                reload_rate=round(st['reload'] / max(1, st['cold_0'] + st['cold_1']), 4),
                                unique_cold=st['unique_cold'],
                                refill_reads=st['refill'],
                                uses_per_load=round(sum(w.uses_per_load) / max(1, len(w.uses_per_load)), 2),
                                lifetime_p50=sorted(w.lifetimes)[len(w.lifetimes) // 2] if w.lifetimes else None,
                                lifetime_p95=sorted(w.lifetimes)[int(len(w.lifetimes) * 0.95)] if w.lifetimes else None))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
