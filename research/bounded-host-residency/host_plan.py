#!/usr/bin/env python3
"""Dry-run of a generic host-RAM plan for Dynamic MoE: full host bank when it fits safely, else the
largest safe bounded RAM cache. No model table: the inputs are the model's bank bytes, the
engine's non-expert host RSS and the VRAM arena the existing planner already chose.

usage: host_plan.py [--bank GIB] [--nonexpert GIB] [--hot GIB] [--ram GIB ...]
Defaults are Qwen3.6-35B-A3B Q4_K_S on a 12 GiB card, measured on this machine.
"""
import argparse

STAGING_GIB = 0.13          # read-back stage and GPU copy ring
WIRE_FRACTION = 0.79        # macOS vm.user_wire_limit / hw.memsize measured here; read the sysctl at runtime
WIRED_BY_OTHERS_GIB = 2.0   # kernel, GPU driver and other processes' wired pages, taken from vm_stat at runtime
MIN_COVERAGE = 1.0          # (HOT + RAM cache) / bank below which every token streams experts from storage


def reserve_gib(ram):
    """RAM left to macOS, the app and the file cache."""
    return max(6.0, 0.20 * ram)


def plan(ram, bank, nonexpert, hot):
    reserve = reserve_gib(ram)
    budget = ram - reserve
    wire = WIRE_FRACTION * ram - WIRED_BY_OTHERS_GIB
    full_rss = bank + nonexpert
    if full_rss <= budget and bank <= wire:
        return dict(mode='FULL HOST BANK', warm=bank, rss=full_rss, locked=bank, headroom=ram - full_rss, coverage=(hot + bank) / bank)
    warm = max(0.0, min(bank, budget - nonexpert - STAGING_GIB, wire))
    coverage = (hot + warm) / bank
    mode = 'BOUNDED (max safe)' if coverage >= MIN_COVERAGE else 'NOT RECOMMENDED (streams per token)'
    rss = warm + nonexpert + STAGING_GIB
    return dict(mode=mode, warm=warm, rss=rss, locked=warm, headroom=ram - rss, coverage=coverage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', type=float, default=17.07)
    ap.add_argument('--nonexpert', type=float, default=1.53)
    ap.add_argument('--hot', type=float, default=6.47)
    ap.add_argument('--ram', type=float, nargs='+', default=[16, 24, 32, 48, 64, 128, 256])
    a = ap.parse_args()
    print(f'bank {a.bank} GiB, non-expert RSS {a.nonexpert} GiB, HOT {a.hot} GiB; reserve max(6, 20% RAM), '
          f'lockable {WIRE_FRACTION:.2f} x RAM - {WIRED_BY_OTHERS_GIB} GiB')
    print(f'{"RAM":>5} {"reserve":>8} {"mode":36s} {"WARM":>6} {"RSS":>6} {"locked":>7} {"headroom":>9} {"HOT+WARM/bank":>14}')
    for ram in a.ram:
        p = plan(ram, a.bank, a.nonexpert, a.hot)
        print(f'{ram:5.0f} {reserve_gib(ram):8.1f} {p["mode"]:36s} {p["warm"]:6.1f} {p["rss"]:6.1f} {p["locked"]:7.1f} '
              f'{p["headroom"]:9.1f} {p["coverage"]:14.2f}')


if __name__ == '__main__':
    main()
