#!/usr/bin/env python3
# llama-server batch-size routes: prompts of N tokens (1-8, 9-31, 32+, 512, 1024, 2048), greedy output
# usage: srv_batches.py TAG [server args...]; env passes through
import json, os, subprocess, sys, time, urllib.request
tag = sys.argv[1]
S = os.environ.get('SRV_BIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../vendor/llama.cpp/build-static/bin/llama-server'))
os.makedirs('pf/r4', exist_ok=True)
env = dict(os.environ, TOSH_FA_AMD='1', TOSH_DMOE_STATS='1', TOSH_AUTO_PLAN_FILE=f'pf/r4/{tag}.plan.json')
log = open(f'pf/r4/{tag}.log', 'w')
p = subprocess.Popen([S, '--port', '18091', '--host', '127.0.0.1'] + sys.argv[2:], env=env, stdout=log, stderr=subprocess.STDOUT)
def post(path, body, tmo=600):
    r = urllib.request.Request('http://127.0.0.1:18091' + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=tmo).read())
t0, ready = time.time(), None
while time.time() - t0 < 300 and p.poll() is None:
    try:
        if json.loads(urllib.request.urlopen('http://127.0.0.1:18091/health', timeout=1).read()).get('status') == 'ok':
            ready = time.time() - t0; break
    except Exception: pass
    time.sleep(0.2)
out = {'tag': tag, 'ready_s': round(ready, 2) if ready else None, 'runs': []}
if ready:
    toks = post('/tokenize', {'content': open(os.environ.get('BT_TEXT', 'wl/gemma-longctx.txt')).read()})['tokens']
    for n in [3, 8, 20, 31, 64, 512, 1024, 2048]:
        try:
            r = post('/completion', {'prompt': toks[:n], 'n_predict': 24, 'temperature': 0, 'cache_prompt': False})
            t = r['timings']
            out['runs'].append({'n': n, 'pp': round(t['prompt_per_second'], 1), 'tg': round(t['predicted_per_second'], 1), 'text': r['content']})
        except Exception as e:
            out['runs'].append({'n': n, 'error': str(e)})
p.terminate()
try: p.wait(60)
except Exception: p.kill()
log.close()
txt = open(f'pf/r4/{tag}.log', errors='replace').read()
out['errors'] = [l[:160] for l in txt.splitlines() if any(k in l for k in ('ASSERT', 'abort', 'error', 'without routing', 'nan'))][:5]
print(json.dumps(out))
