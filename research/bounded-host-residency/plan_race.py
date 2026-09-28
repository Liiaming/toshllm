#!/usr/bin/env python3
# another process takes memory while Auto is planning: does the check before committing see it?
# usage: plan_race.py TAG GIB DELAY_S MODEL [server args...]; env passes through
import json, os, subprocess, sys, time, urllib.request
tag, gib, delay, model = sys.argv[1], sys.argv[2], float(sys.argv[3]), os.path.expanduser(sys.argv[4])
S = os.environ.get('SRV_BIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../vendor/llama.cpp/build-static/bin/llama-server'))
base = f'pf/lad/{tag}'
env = dict(os.environ, TOSH_FA_AMD='1', TOSH_AUTO='1', TOSH_AUTO_HOST_POLICY='runtime', TOSH_AUTO_PLAN_FILE=base + '.plan.json')
log = open(base + '.log', 'w')
t0 = time.time()
p = subprocess.Popen([S, '-m', model, '-c', '8192', '--port', '18095', '--host', '127.0.0.1'] + sys.argv[5:], env=env, stdout=log, stderr=subprocess.STDOUT)
time.sleep(delay)
h = subprocess.Popen([os.environ['MEMHOLD'], gib, base + '.hold'], stdout=subprocess.PIPE, text=True)
held = h.stdout.readline().strip()
t_held = time.time() - t0
ready = None
while time.time() - t0 < 300 and p.poll() is None:
    try:
        if json.loads(urllib.request.urlopen('http://127.0.0.1:18095/health', timeout=1).read()).get('status') == 'ok':
            ready = time.time() - t0; break
    except Exception: pass
    time.sleep(0.2)
rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(p.pid)], capture_output=True, text=True).stdout.strip()
p.terminate()
try: p.wait(90)
except Exception: p.kill()
log.close()
os.remove(base + '.hold'); h.wait(60)
plan = json.load(open(base + '.plan.json'))['product']
txt = open(base + '.log', errors='replace').read().splitlines()
print(json.dumps({'tag': tag, 'held': held, 'held_at_s': round(t_held, 2), 'ready_s': round(ready, 2) if ready else None,
                  'rss_gib': round(int(rss)/1048576, 2) if rss else None, 'mode': plan['mode'], 'reason': plan['reason'],
                  'warnings': plan['warnings'], 'warm_gib': round(plan['dmoe']['warm_bytes']/2**30, 2), 'coverage': plan['dmoe']['coverage'],
                  'recheck_ms': plan['recheck_ms'], 'log': [l[:200] for l in txt if 'Tosh Auto' in l or 'memory changed' in l]}))
