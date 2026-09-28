#!/usr/bin/env python3
# llama-server start -> ready -> first long prompt -> decode, with the host cache's warm-up marks
# usage: srv3.py MODEL CTX TAG [server args...]; extra env passes through
import json, os, re, subprocess, sys, time, urllib.request
model, ctx, tag = os.path.expanduser(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
S = os.environ.get('SRV_BIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../vendor/llama.cpp/build-static/bin/llama-server'))
os.makedirs('pf/r3', exist_ok=True)
env = dict(os.environ, TOSH_FA_AMD='1', TOSH_DMOE_STATS='1', TOSH_AUTO_PLAN_FILE=f'pf/r3/{tag}.plan.json')
env.setdefault('TOSH_AUTO', '1')
log = open(f'pf/r3/{tag}.log', 'w')
t0 = time.time()
p = subprocess.Popen([S, '-m', model, '-c', str(ctx), '--port', '18091', '--host', '127.0.0.1'] + sys.argv[4:], env=env, stdout=log, stderr=subprocess.STDOUT)
def post(body):
    r = urllib.request.Request('http://127.0.0.1:18091/completion', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=3600).read())
ready = None
while time.time() - t0 < 900 and p.poll() is None:
    try:
        if json.loads(urllib.request.urlopen('http://127.0.0.1:18091/health', timeout=2).read()).get('status') == 'ok':
            ready = time.time() - t0; break
    except Exception: pass
    time.sleep(0.1)
out = {'tag': tag, 'ready_s': round(ready, 2) if ready else None}
if ready:
    t_req = time.time()
    r = post({'prompt': open(os.environ.get('SRV_PROMPT', 'wl/longctx.txt')).read(), 'n_predict': 256, 'temperature': 0, 'cache_prompt': False})
    t = r['timings']
    out['first'] = {'n': t['prompt_n'], 'pp': round(t['prompt_per_second'], 1), 'prompt_ms': round(t['prompt_ms']), 'tg': round(t['predicted_per_second'], 2),
                    'first_token_s': round(t_req - t0 + t['prompt_ms']/1000, 2)}
    t = post({'prompt': 'Write a long story about a lighthouse keeper who finds a map.', 'n_predict': 512, 'temperature': 0, 'cache_prompt': False, 'ignore_eos': True})['timings']
    out['decode'] = {'tg': round(t['predicted_per_second'], 2), 'ms': round(t['predicted_per_token_ms'], 2)}
    rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(p.pid)], capture_output=True, text=True).stdout.strip()
    out['rss_gib'] = round(int(rss)/1048576, 2) if rss else None
p.terminate()
try: p.wait(120)
except Exception: p.kill()
log.close()
txt = open(f'pf/r3/{tag}.log', errors='replace').read()
g = lambda pat: (re.search(pat, txt).group(1) if re.search(pat, txt) else None)
out['plan'] = g(r'"state": "([A-Z_]+)"')
out['warm_mib'] = g(r'"warm_mib": ([\d.]+)')
out['ready_line'] = g(r'tosh_hostcache: ready (.*)')
out['reuse'] = g(r'tosh_hostcache: loaded before ready: (.*)')
out['swap'] = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout.split()[5]
print(json.dumps(out))
