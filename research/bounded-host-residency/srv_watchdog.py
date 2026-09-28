#!/usr/bin/env python3
# llama-server with a watchdog: start -> ready or timeout (kill -9), sampling RSS and VRAM in use;
# optional first prompt. usage: srv_watchdog.py TAG TIMEOUT_S [server args...]; env passes through
import json, os, re, subprocess, sys, time, urllib.request
tag, tmo = sys.argv[1], float(sys.argv[2])
S = os.environ.get('SRV_BIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../vendor/llama.cpp/build-static/bin/llama-server'))
os.makedirs('pf/r4', exist_ok=True)
env = dict(os.environ, TOSH_FA_AMD='1', TOSH_DMOE_STATS='1', TOSH_AUTO_PLAN_FILE=f'pf/r4/{tag}.plan.json')
log = open(f'pf/r4/{tag}.log', 'w')
def vram():
    o = subprocess.run(['ioreg', '-r', '-d', '1', '-c', 'IOAccelerator'], capture_output=True, text=True).stdout
    m = re.search(r'"inUseVidMemoryBytes"=(\d+)', o)
    return int(m.group(1))/2**30 if m else None
def rss(pid):
    r = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
    return int(r)/1048576 if r else None
v0 = vram()
t0 = time.time()
p = subprocess.Popen([S, '--port', '18091', '--host', '127.0.0.1'] + sys.argv[3:], env=env, stdout=log, stderr=subprocess.STDOUT)
ready, peak_v, peak_r, last = None, 0, 0, 0
while time.time() - t0 < tmo and p.poll() is None:
    try:
        if json.loads(urllib.request.urlopen('http://127.0.0.1:18091/health', timeout=1).read()).get('status') == 'ok':
            ready = time.time() - t0; break
    except Exception: pass
    if time.time() - last > 1:
        last = time.time(); v = vram(); r = rss(p.pid)
        peak_v = max(peak_v, (v or 0) - (v0 or 0)); peak_r = max(peak_r, r or 0)
    time.sleep(0.2)
out = {'tag': tag, 'ready_s': round(ready, 2) if ready else None, 'exit': p.poll(),
       'hung': ready is None and p.poll() is None}
if ready:
    out['vram_gib'] = round(vram() - v0, 2); out['rss_gib'] = round(rss(p.pid), 2)
    if os.environ.get('GW_PROMPT'):
        body = {'prompt': open(os.environ['GW_PROMPT']).read(), 'n_predict': int(os.environ.get('GW_N', '128')),
                'temperature': 0, 'cache_prompt': False}
        r = urllib.request.Request('http://127.0.0.1:18091/completion', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            t = json.loads(urllib.request.urlopen(r, timeout=tmo).read())['timings']
            out['first'] = {'n': t['prompt_n'], 'pp': round(t['prompt_per_second'], 1), 'tg': round(t['predicted_per_second'], 2)}
        except Exception as e:
            out['first'] = f'error: {e}'
out['peak_vram_gib'] = round(peak_v, 2); out['peak_rss_gib'] = round(peak_r, 2)
if p.poll() is None:
    p.kill() if out['hung'] else p.terminate()
    try: p.wait(60)
    except Exception: p.kill(); p.wait(30)
log.close()
time.sleep(3)
out['vram_after_gib'] = round((vram() or 0) - (v0 or 0), 2)
txt = open(f'pf/r4/{tag}.log', errors='replace').read()
out['plan'] = (re.search(r'"state": "([A-Z_]+)"', open(f'pf/r4/{tag}.plan.json').read()).group(1)
               if os.path.exists(f'pf/r4/{tag}.plan.json') else None)
keys = r'(tosh_plan|common_fit|fit_params|llama_params_fit|offloaded|CPU_Mapped|CPU model buffer|Metal.*buffer size|MTL0 model|error|failed|abort|ASSERT|exiting|unsupported|UNSUPPORTED)'
out['log'] = [l[:200] for l in txt.splitlines() if re.search(keys, l)][-25:]
print(json.dumps(out, indent=1))
