#!/usr/bin/env python3
# dry-run plans of two engine builds on the same simulated machines
# usage: plan_regression.py OLD_LLAMA_SERVER NEW_LLAMA_SERVER
import json, os, subprocess, sys
old, new = sys.argv[1], sys.argv[2]
models = ['Qwen3.6-35B-A3B-UD-Q4_K_S', 'gpt-oss-20b-Q4_K_M', 'gemma-4-26B-A4B-it-MXFP4_MOE', 'GLM-4.7-Flash-REAP-23B-A3B-Q4_K_M']
keys = ['state', 'mode', 'kv', 'ubatch', 'ncmoe', 'arena_mib']
def plan(b, m, env):
    e = dict(os.environ, TOSH_AUTO='1', TOSH_AUTO_DRY_RUN='exit', **env)
    o = subprocess.run([b, '-m', os.path.expanduser(f'~/models/{m}.gguf'), '-c', '8192'], env=e, capture_output=True, text=True).stdout
    d = json.loads(o)
    h = d.get('host', {})
    return tuple(d[k] for k in keys) + (round(h.get('warm_mib', 0)), h.get('coverage')), d
for m in models:
    for label, env in [('static', {}), ('rt16', {'TOSH_AUTO_HOST_POLICY': 'runtime', 'TOSH_AUTO_SIM_RAM_MIB': '16384'}),
                       ('rt24', {'TOSH_AUTO_HOST_POLICY': 'runtime', 'TOSH_AUTO_SIM_RAM_MIB': '24576'}),
                       ('rt32', {'TOSH_AUTO_HOST_POLICY': 'runtime', 'TOSH_AUTO_SIM_RAM_MIB': '32768'}),
                       ('rt64', {'TOSH_AUTO_HOST_POLICY': 'runtime', 'TOSH_AUTO_SIM_RAM_MIB': '65536'})]:
        a, _ = plan(old, m, env); b, dn = plan(new, m, env)
        same = 'SAME' if a == b else 'DIFF'
        print(f'{m[:22]:22s} {label:6s} {same} old={a} new={b[:7]} {dn.get("code")} host_w={dn.get("host_weights_mib")} floor={dn.get("warm_floor_mib")}')
