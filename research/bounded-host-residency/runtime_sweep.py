#!/usr/bin/env python3
"""Offline sweep of the runtime-aware host plan: physical RAM x current reclaimable share x headroom
formula, for the measured models. Every budget is in bytes Tosh may still add (incremental).

usage: runtime_sweep.py
"""
MODELS = {'Qwen': (17.07, 1.55, 6.96), 'GPT-OSS': (9.46, 1.43, 7.79), 'Gemma': (13.00, 1.80, 5.30), 'GLM': (11.71, 1.21, 7.77)}
STAGING, WIRE_FRACTION, WIRED_BASE, WIRE_MARGIN = 0.13, 0.79, 3.6, 1.0
HEADROOM = {
    'H1 max(4, 10%)': lambda r: max(4.0, 0.10 * r),
    'H2 max(4, 15%)': lambda r: max(4.0, 0.15 * r),
    'H3 max(6, 15%)': lambda r: max(6.0, 0.15 * r),
    'H4 max(6, 20%)': lambda r: max(6.0, 0.20 * r),
    'SELECTED clamp(15%, 4, 12)': lambda r: min(12.0, max(4.0, 0.15 * r)),
}


def plan(ram, reclaim, model, headroom):
    bank, nonexp, hot = MODELS[model]
    static_inc = ram - max(6.0, 0.20 * ram)                 # nothing of Tosh loaded yet
    dyn_inc = max(0.0, reclaim - headroom(ram))
    lockable = WIRE_FRACTION * ram - WIRED_BASE - WIRE_MARGIN
    eff = min(static_inc, dyn_inc)
    if bank + nonexp <= eff and bank <= lockable:
        return 'FULL', bank, reclaim - bank - nonexp
    warm = max(0.0, min(bank, eff - nonexp - STAGING, lockable))
    cov = (hot + warm) / bank
    tag = 'B' if cov >= 1.3 else 'b' if cov >= 1.0 else '-'
    return f'{tag}{warm:.1f}', warm, reclaim - warm - nonexp - STAGING


def main():
    rams, fracs = (16, 24, 32, 48, 64, 128), (0.25, 0.40, 0.55, 0.70, 0.85)
    print('cells: FULL host bank, B<warm> bounded with coverage >= 1.3, b<warm> 1.0-1.3, -<warm> rejected (< 1.0); '
          'then the RAM left free after the load')
    for hname, h in HEADROOM.items():
        print(f'\n## {hname}')
        for model in MODELS:
            print(f'{model:8s}' + ''.join(f'{f"{int(f*100)}%":>11s}' for f in fracs))
            for ram in rams:
                cells = []
                for f in fracs:
                    mode, warm, left = plan(ram, f * ram, model, h)
                    cells.append(f'{mode}/{left:.0f}')
                print(f'  {ram:4d} ' + ''.join(f'{c:>11s}' for c in cells))


if __name__ == '__main__':
    main()
