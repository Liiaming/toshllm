#!/usr/bin/env python3
# real memory pressure step: hold N GiB, preview the Auto plan, load with Auto, optionally run the
# workload (3.4K prompt, 12-turn chat, 2048-token decode), sampling RSS, VRAM, swap and compression.
# usage: pressure_step.py TAG HOLD_GIB MODEL [--bench] [server args...]; env passes through
import json, os, re, subprocess, sys, time, urllib.request
tag, hold, model = sys.argv[1], float(sys.argv[2]), os.path.expanduser(sys.argv[3])
bench = '--bench' in sys.argv[4:]
extra = [a for a in sys.argv[4:] if a != '--bench']
S = os.environ.get('SRV_BIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../vendor/llama.cpp/build-static/bin/llama-server'))
HOLD = os.environ.get('MEMHOLD', 'memhold')
os.makedirs('pf/lad', exist_ok=True)
base = f'pf/lad/{tag}'
env = dict(os.environ, TOSH_FA_AMD='1', TOSH_AUTO=os.environ.get('TOSH_AUTO', '1'), TOSH_AUTO_HOST_POLICY=os.environ.get('TOSH_AUTO_HOST_POLICY', 'runtime'),
           TOSH_DMOE_STATS='1', TOSH_AUTO_PLAN_FILE=base + '.plan.json')
GiB = 2**30

def sh(*a): return subprocess.run(a, capture_output=True, text=True).stdout
def mem():
    v = sh('vm_stat'); pg = 4096
    g = lambda k: int(re.search(k + r':\s+(\d+)', v).group(1))
    sw = sh('sysctl', '-n', 'vm.swapusage').split()
    return {'swap_gib': round(float(sw[5].rstrip('M'))/1024, 2), 'compressed_gib': round(g('Pages occupied by compressor')*pg/GiB, 2),
            'free_gib': round(g('Pages free')*pg/GiB, 2), 'pressure': int(sh('sysctl', '-n', 'kern.memorystatus_vm_pressure_level') or 0)}
def vram():
    m = re.search(r'"inUseVidMemoryBytes"=(\d+)', sh('ioreg', '-r', '-d', '1', '-c', 'IOAccelerator'))
    return int(m.group(1))/GiB if m else 0
def cmprs(pid):
    # compressed memory of one process, from top
    o = sh('top', '-l', '1', '-pid', str(pid), '-stats', 'pid,cmprs').strip().splitlines()
    v = o[-1].split()[-1] if o else '0'
    m = re.match(r'([\d.]+)([KMG]?)', v)
    return round(float(m.group(1))*{'K': 1/2**20, 'M': 1/1024, 'G': 1, '': 1/2**30}[m.group(2)], 2) if m else None
def rss(pid):
    r = sh('ps', '-o', 'rss=', '-p', str(pid)).strip()
    return round(int(r)/1048576, 2) if r else None
def post(path, body, tmo=3600):
    r = urllib.request.Request('http://127.0.0.1:18095' + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=tmo).read())

out = {'tag': tag, 'hold_gib': hold}
sentinel = base + '.hold'
holder = None
if hold > 0:
    holder = subprocess.Popen([HOLD, str(hold), sentinel] + os.environ.get('MEMHOLD_ARGS', '').split(), stdout=subprocess.PIPE, text=True)
    out['holder'] = holder.stdout.readline().strip()
    time.sleep(10)
out['mem_before'] = mem()
pv = subprocess.run([S, '-m', model, '-c', '8192'] + extra, env=dict(env, TOSH_AUTO_DRY_RUN='exit'), capture_output=True, text=True).stdout
out['preview'] = json.loads(pv.strip().splitlines()[-1])['product'] if pv.strip() else None
time.sleep(5)
v0 = vram()
log = open(base + '.log', 'w')
t0 = time.time()
p = subprocess.Popen([S, '-m', model, '-c', '8192', '--port', '18095', '--host', '127.0.0.1'] + extra, env=env, stdout=log, stderr=subprocess.STDOUT)
ready = None
while time.time() - t0 < 300 and p.poll() is None:
    try:
        if json.loads(urllib.request.urlopen('http://127.0.0.1:18095/health', timeout=1).read()).get('status') == 'ok':
            ready = time.time() - t0; break
    except Exception: pass
    time.sleep(0.2)
out['ready_s'] = round(ready, 2) if ready else None
out['exit'] = p.poll()
if ready:
    time.sleep(2)
    out['mem_ready'] = mem(); out['rss_ready'] = rss(p.pid); out['vram_ready'] = round(vram() - v0, 2)
    out['cmprs_server_ready'] = cmprs(p.pid)
    try: out['actual'] = json.load(open(base + '.plan.json.actual'))
    except Exception: pass
    if bench:
        t = post('/completion', {'prompt': open('wl/longctx.txt').read(), 'n_predict': 128, 'temperature': 0, 'cache_prompt': False})['timings']
        out['first'] = {'n': t['prompt_n'], 'pp': round(t['prompt_per_second'], 1), 'tg': round(t['predicted_per_second'], 2)}
        out['mem_first'] = mem()
        turns = [x.strip() for x in open('wl/conv12.txt').read().split('\n=====\n')]
        msgs, pn, pms, dn, dms = [], 0, 0.0, 0, 0.0
        for u in turns:
            msgs.append({'role': 'user', 'content': u})
            r = post('/v1/chat/completions', {'messages': msgs, 'max_tokens': 160, 'temperature': 0})
            tm = r['timings']
            pn += tm['prompt_n']; pms += tm['prompt_ms']; dn += tm['predicted_n']; dms += tm['predicted_ms']
            msgs.append({'role': 'assistant', 'content': r['choices'][0]['message'].get('content') or ''})
        out['chat'] = {'turns': len(turns), 'prefill_tokens': pn, 'pp': round(pn*1000/pms, 1), 'tg': round(dn*1000/dms, 2)}
        out['mem_chat'] = mem()
        # 2048-token decode, token arrival times from the stream
        body = {'prompt': 'Write a long story about a lighthouse keeper who finds a map.', 'n_predict': 2048, 'temperature': 0,
                'cache_prompt': False, 'ignore_eos': True, 'stream': True}
        r = urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18095/completion', data=json.dumps(body).encode(),
                                                          headers={'Content-Type': 'application/json'}), timeout=3600)
        ts, last = [], None
        for line in r:
            if not line.startswith(b'data: '): continue
            d = json.loads(line[6:])
            now = time.time()
            if last is not None and d.get('content'): ts.append((now - last)*1000)
            if d.get('content'): last = now
            if d.get('stop'): out['long'] = {'tg': round(d['timings']['predicted_per_second'], 2), 'ms': round(d['timings']['predicted_per_token_ms'], 2)}
        ts.sort()
        if ts: out['long'].update({f'p{q}': round(ts[min(len(ts) - 1, int(len(ts)*q/100))], 2) for q in (50, 95, 99)})
        out['mem_long'] = mem(); out['rss_end'] = rss(p.pid); out['vram_end'] = round(vram() - v0, 2)
        out['cmprs_server'] = cmprs(p.pid); out['cmprs_holder'] = cmprs(holder.pid) if holder else None
p.terminate()
try: p.wait(90)
except Exception: p.kill()
log.close()
if not ready:
    try: out['actual'] = json.load(open(base + '.plan.json.actual'))
    except Exception: pass
out['log_tail'] = [l[:220] for l in open(base + '.log', errors='replace').read().splitlines() if 'Tosh Auto' in l or 'E ' in l[:20] or 'memory changed' in l][:6]
if holder:
    try: os.remove(sentinel)
    except Exception: pass
    holder.wait(60)
time.sleep(5)
out['mem_after'] = mem()
print(json.dumps(out))
