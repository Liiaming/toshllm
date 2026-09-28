#!/usr/bin/env python3
"""Expert list for the prewarm experiment: "layer expert" lines, most routed first.

usage: prewarm_list.py OUT TRACE.ids [TRACE.ids ...]
With the trace of the workload being measured this is an oracle (an upper bound); with traces
of other workloads it stands for a frequency profile kept from earlier sessions.
"""
import collections, sys


def main():
    freq = collections.Counter()
    for path in sys.argv[2:]:
        with open(path) as f:
            for line in f:
                v = line.split()
                if len(v) < 4:
                    continue
                layer = int(v[2])
                for e in v[3:]:
                    freq[(layer, int(e))] += 1
    with open(sys.argv[1], 'w') as out:
        for (layer, e), _ in freq.most_common():
            out.write(f'{layer} {e}\n')
    print(f'{sys.argv[1]}: {len(freq)} experts')


if __name__ == '__main__':
    main()
