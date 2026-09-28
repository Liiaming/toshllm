# Bounded host expert residency for Dynamic MoE (feasibility, 2026-09-27)

Question: can Dynamic MoE keep only part of the expert bank in RAM (WARM, mlocked) and read the
rest from the GGUF on demand (COLD, `pread`), instead of locking the whole bank?

Not productized. Production Dynamic MoE still locks the full bank.

Engine code in the series: `patches/llama/0106-experimental-bounded-host-expert-cache.patch`, the
selected design only (RAM copy of a promoted expert kept last in line, background read-back of an
evicted one, demand reads first, optional prewarm). The async VRAM copy and the inclusive modes
below were removed from the series after they measured NO-GO; their patches (then 0118 and 0119)
and tools are in this directory's git history.

## Tools

- `trace.cpp` (`build.sh`): router trace per token and 512-token prefill chunk. `TRACE_VERIFY=N`
  checks the first N rows against the top-k of the router's contiguous scores;
  `TRACE_NEGATIVE_CONTROL=1` reads the view as contiguous on purpose and must fail.
- `run_traces.sh`: workloads in `workloads/`.
- `geometry.py`: bank size, bytes per expert, file layout per GGUF.
- `sim.py`: HOT (VRAM, the Dynamic MoE policy approximated) then WARM (RAM: lru or lfu,
  exclusive with VRAM) over a trace. First touches are reported apart from capacity misses.
- `analyze.py`: capacity misses into ms/token and prefill s/1000 tokens with the measured storage.
- `storage.c`: read latency of expert-sized ranges: F_NOCACHE pread, page cache, mmap faults.

## Results, Qwen3.6-35B-A3B on RX 6700 XT, 32 GB, 7.1 GiB VRAM arena

Trace check: 160k rows over 8 workloads, 0 not the top-k of a router score; the negative
control failed 2835 of 3863 rows. Simulated VRAM hit 81-85% against 86% measured.

Storage (this NVMe, pread F_NOCACHE): 576 KiB 0.61 ms, 2 MiB 0.81 ms, 16 MiB 5.1 ms (3.2 GB/s);
mmap faults 2.6x slower (1.58 ms at 576 KiB); page cache 14 GB/s.

| RAM for experts | RSS est. | decode added | prefill added |
|---:|---:|---:|---:|
| full bank (today) | 18.5 GiB | 0 | 0 |
| 10 GiB | 11.7 GiB | 0.0% | +3% |
| 8 GiB | 9.7 GiB | +0.9% (up to +24% right after long prompts) | +23% |
| 6 GiB | 7.7 GiB | +4.9% | +222% |
| 4 GiB | 5.7 GiB | +13.6% | +264% |

Prefill decides the knee: each chunk uploads many distinct non-resident experts, and those not in
RAM come from storage at 3.2 GB/s against PCIe. Streaming prefill-only experts straight to VRAM
without keeping them in RAM does not help at 8-10 GiB (they are reused later).

Layout: each expert is 3 ranges (gate, up, down; 2 on Gemma 4), one per part tensor, 0.56 MiB
each on Qwen; parts of one layer sit 1-3 MiB apart. No repack is needed for per-expert reads.

Not yet done: GPT-OSS, Gemma 4 and GLM (traces for Qwen and GPT-OSS exist).

## Engine prototype, Qwen3.6-35B-A3B (first version, now in patch 0106)

`TOSH_DMOE_HOST_CACHE_MIB=10240` with `DMOE_LOAD=none`: the loader leaves the expert bank unread,
experts are read with `pread` into a locked RAM cache (one LRU pool per expert size), and the
untouched bank pages are PROT_NONE so any reader that bypasses the cache faults. Exclusive with the
VRAM arena: a promoted expert leaves RAM, an evicted one is read back asynchronously
(`TOSH_DMOE_HOST_DEMOTE=read`, default; `copy` reads the arena back and blocks the policy thread
6.3 ms per eviction, which halved promotions and dropped the VRAM hit from 80% to 67%). Demand
reads go ahead of read-backs. `TOSH_DMOE_HOST_IO` readers (4 selected), `TOSH_DMOE_HOST_NOCACHE=1`
reads past the page cache. Needs `GGML_OP_OFFLOAD_MIN_BATCH=9`.

RX 6700 XT, 32 GB, 7.1 GiB arena; full bank = the same engine with the bank locked in RAM.

| | full bank | 10 GiB cache | 8 GiB cache |
|---|---:|---:|---:|
| RSS | 18.59 GiB | 11.53 GiB | 9.51 GiB |
| 8K, first prompt after load (3381 tok): TTFT | 5.88 s | 10.82 s | |
| 8K decode 256 tok | 56.2 t/s | 55.4 t/s | |
| 12 turns, decode over all turns | 48.9 t/s | 39.5-40.1 t/s | 35.6 t/s |
| 12 turns, decode turns 5-11 | 21.2 ms | 22.6 ms | 25.5 ms |
| 12 turns, 20-39 token prompts, turns 5-11 | 1328 ms | 2290-2614 ms | |
| 2048-token decode (cold start) | 57.6 t/s | 47.3 t/s | |
| 16K prompt at 32K ctx: TTFT | 36.2 s | 49.3 s | |
| topic shift, 4 topics | 50.0 t/s | 31.3 t/s | |

Correctness: top-1 99.6-100% and mean KL 3-4e-4 against the full bank (same size as run-to-run
host/GPU reordering), 5175 promoted slots compared against the file with 0 mismatches, 0 hits on a
changed slot. Real reads: 1.7-2.4 ms p50 per 1.7 MiB expert with or without the page cache, since
loading the engine evicts most of the model file from it on this machine.

Against the simulation (conv12, 10 GiB): prefill cold reads 7937 real vs 6696 simulated (+19%);
decode cold reads 2.75/token vs 1.47 (1.9x) and 0.76/token in turns 5-11 against ~0 capacity
misses simulated. Most of the cost left is first touches, which the simulation priced apart:
a cold start pays about 2.4 reads per token for the first 2000 tokens.

## WARM size, prefill attribution and prewarm (now in patch 0106)

`ws.py` (working set per token window), `tl.py` (prefill attribution from `TOSH_DMOE_TIMELINE`),
`summ.py` (bench log summary), `prewarm_list.py` (expert lists by trace frequency). `sim.py`
now counts the read-backs that exclusivity costs (`refill_reads`): 3.7 per token on conv12,
against 3.8 measured with lazy drop and 4.9 with the drop of 0116.

Working set (traces): a 50-token prompt touches 7.2 GiB of experts, a 256-token window 11.4 GiB,
the 3381-token prompt 15.8 GiB. HOT holds 6.5 GiB (97 slots x 40 layers), so HOT + WARM covers the
whole 17.07 GiB bank from about 10.6 GiB of WARM: past that, capacity misses are not the cost.

Warmup conversation + 12 turns, 4 readers, NVMe direct (F_NOCACHE):

| WARM | RSS | steady prefill (9 short prompts) | decode after warmup | cold reads/token |
|---:|---:|---:|---:|---:|
| full bank | 18.6 | 2320 ms | 21.6 ms | 0 |
| 10 | 11.5 | 4101 (+77%) | 24.3 | 2.17 |
| 11 | 12.5 | 3932 (+69%) | 24.7 | 0.82 |
| 12 | 13.5 | 3520 (+52%) | 23.1 | 0.31 |
| 13 | 14.5 | 3431 (+48%) | 22.8 | 0.33 |
| 12, boost + lazy drop | 13.5 | 2964 (+28%) | 22.4 | 0.34 |
| 12, same + full prewarm | 13.5 | 2858 (+23%) | 22.3 | 0.16 |
| 14, boost + lazy drop | 15.5 | 2725 (+17%) | 21.8 | 0.31 |
| 17.1, lazy drop (no read-backs) | 18.6 | 2534 (+9%) | 22.6 | 0.25 |

Where short-prefill time goes (main thread partitioned, residual within 3% of wall): demand reads
are latency bound and serial per layer (routing is known only at each layer); demand waits behind
queued read-backs (fixed by moving a read-back a demand waits on to the front: +52% -> +29%); with
waits gone, about 19 points are the read-back traffic slowing the host executor and 9 points first
touches and the pinning path. Read-backs are reused 92-94% before eviction, so skipping them loses
(no read-backs: +58%, decode +27%); low-priority readers of their own starve them (+55%).
More readers than 4 only saturate the NVMe. Adjacent cold experts are rare (1.02 per run), so
merging demand reads saves nothing.

First 3381-token prompt, 12 GiB: cold reads are 16.5 GiB whatever the WARM size. Prewarm trades
start time for TTFT almost one to one:

| prewarm | ready | TTFT | launch to first token |
|---|---:|---:|---:|
| full bank (no prewarm) | 11.9 s | 5.86 s | 17.8 s |
| none | 4.4 s | 10.87 s | 15.3 s |
| layer order, 12 GiB, file order | 8.5 s | 7.02 s | 15.5 s |
| trace oracle 2 / 4 / 6 GiB | 4.8 / 5.3 / 6.0 s | 10.25 / 9.58 / 8.85 s | 15.1 / 14.9 / 14.8 s |
| trace oracle, 12 GiB | 7.6 s | 6.97 s | 14.6 s |
| other workloads' profile, 12 GiB | 7.6 s | 7.25 s | 14.9 s |

File-order reads of the same set: 6.2k merged reads instead of 21.6k, 3.64 against 3.58 GB/s.

## Async VRAM -> RAM copies instead of rereads: NO-GO, removed

`TOSH_DMOE_HOST_DEMOTE=gpu TOSH_DMOE_DOWN_STAGE=1 TOSH_DMOE_DOWN_QUEUE=own`: an expert leaving VRAM
is blitted into a 64 x 2 MiB device-allocated ring on a queue of its own and copied into its RAM
slot on a dispatch queue; the slot stays loading until then (requests wait for the copy, never for
the file) and the VRAM slot is not reused before the copy finished. A failed or ring-full copy
falls back to a background file read. The microbenchmark went with the code.

Microbenchmark (RX 6700 XT): 1.69-1.95 MiB per copy, 130-165 us of GPU time, 12.5 GB/s, 5-7 us
to enqueue, not slowed by and not slowing a compute queue. Wrapping the mlocked RAM cache itself
with newBufferWithBytesNoCopy works in isolation, but in the engine the driver pages those wraps
in and out: 1 GiB pieces gave copy p99 752 ms and 165 ms decode steps, 16 MiB pieces p99 35 ms;
the staging ring gives p50 1 ms, p99 10-24 ms, at 0.2 ms of CPU copy per expert.

Result (12 GiB, lazy drop, warmup + 12 turns): file rereads caused by VRAM evictions 30.5k -> 0
(plus 2.5-3k ring-full fallbacks), file bytes 58.6 -> 12-22 GiB, yet short prefill 2858 ms (0117)
-> 3156 ms and decode 22.3 -> 23.0 ms: no gain. Isolation at 17 GiB (capacity to spare): keeping
the copy on promotion +12% short prefill, dropping it and rereading from the file +31%, dropping it
and copying from VRAM +39% (full bank 2259 ms). The cost follows moving an expert back into RAM on
every eviction, whatever the path; skipping prefill-time moves and running the copies off Metal's
completion thread did not change it. Cause not found yet.

`host_plan.py`: dry run of a generic host-RAM plan (full bank when it fits a reserve of
max(6 GiB, 20% RAM) and the wire limit, else the largest safe RAM cache; below HOT + RAM cache =
bank it streams every token and is not recommended).

## Inclusive VRAM/RAM residency: NO-GO, removed

`TOSH_DMOE_HOST_DROP`: what happens to the RAM copy of an expert promoted to VRAM. 1 frees it (0116),
0 keeps it first in line for eviction (0117), 2 keeps it in place (inclusive), 3 keeps it in place
only while its decayed use is under `TOSH_DMOE_HOST_KEEP_BELOW`. Copies held in both places count
against the RAM budget. Keep-last is now the only behaviour and the switch is gone.

Trace (conv12, 12 GiB): exclusive 3.71 refills/token and no capacity misses; keep-last 2.58
refills and none; inclusive 1.08 refills but 0.91 capacity misses/token; protecting copies of
experts in VRAM 0 refills and 6.6 capacity misses/token. Duplicates average 2.6-6.0 GiB.

Engine (warmup + 12 turns, short prefill against 2259 ms full bank):

| RAM copy on promotion | 12 GiB | decode | 14 GiB |
|---|---:|---:|---:|
| keep last (0117) | 2933 (+30%) | 22.5 | 2725 (+21%) |
| keep in place (inclusive) | 3650 (+62%) | 25.6 | 2826 (+25%) |
| keep in place under score 4 | 3340 (+48%) | 24.0 | |
| keep in place under score 16 | 3915 (+73%) | 28.2 | |

At a fixed RAM budget a duplicate takes the place of an expert found nowhere else, so rereads in
the background turn into misses on the critical path. The saving of the bounded cache is exactly
not holding VRAM residents twice; keeping them all (the 17 GiB control, +12%) needs the full bank.
Repeated bounded runs vary by up to 10% (full bank 3%).

## RAM knee with the selected design (patch 0106)

Same runtime at every size, prewarm of the whole RAM cache in file order; `knee.py` builds the
table. Loss is elapsed time against the full bank (short prefill: 9 prompts of 20-70 tokens after
a warmup conversation; the throughput left is 1/(1+loss)).

| RAM cache | RSS | saved | short prefill | 516 | 3381 | 12-turn prefill | decode | 2048 decode | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full bank 17.07 | 18.6 | 0 | 2290 ms | 1202 | 5466 | 8958 | 21.75 ms | 16.97 ms | 27.5 |
| 12 GiB | 13.5 | 5.1 | +29% | +10% | +5% | +12% | +5.2% | +8.1% | 35.8 |
| 14 GiB | 15.5 | 3.1 | +19% | +5% | +2% | +7% | +4.7% | +3.9% | 29.2 |
| 15 GiB | 16.5 | 2.1 | +16% | +7% | +2% | +6% | +4.6% | +3.4% | 29.8 |
| 16 GiB | 17.5 | 1.1 | +7% | +1% | 0% | +2% | -0.4% | +2.4% | 28.4 |
| 16 GiB, no prewarm | 17.5 | 1.1 | +6% | +1% | +2% | +3% | -3.0% | +19.6% | 57.4 |

Read-backs per token fall from 4.1 (12 GiB) to 0.85 (16 GiB); capacity misses are under 0.12 per
token everywhere. The curve has no sharp knee: each GiB kept in RAM buys a few points of short
prefill. Saving 3 GiB costs about 19% of short prefill (7% over the whole conversation); near-full
speed needs 16 GiB, which saves 1.1 GiB. Without prewarm the first 2048 tokens pay +20% and p99 57 ms.

Machines (with `host_plan.py`): 32 GiB and up keep the full bank. 24 GiB cannot hold it safely
(RSS 18.6 against an 18 GiB budget) and gets about 16 GiB of RAM cache, the measured near-full
point. 16 GiB gets about 8 GiB, where the earlier 8 GiB runs showed +20% steady decode and +211%
on a long prompt: not recommended.

## Other MoE models (patch 0107)

0107 drops the Qwen-only gate: parts keep fixed places (0 gate or fused gate/up, 1 up, 2 down;
a fused layout has no part 1 and its up rows follow gate inside part 0), biases and scales stay
loaded, every size pool holds at least one layer's full expert set (Gemma's second size class had
79 slots for 128 experts), and the generic expert copy in the scheduler and the routing-less bank
upload take their bytes from the cache instead of the protected bank.

At 8K on this 12 GiB card Auto picks Dynamic MoE for all four; a 24 GiB machine holds the whole
bank of GPT-OSS, Gemma and GLM, so their bounded runs use a forced 16 GiB budget (RAM cache from
`host_plan.py`: 8.4, 8.1 and 8.7 GiB), Qwen a forced 24 GiB one (16.3 GiB). `xmodel.py` prints the
comparison. Real speeds, full bank first:

| model | RSS full / bounded | 3.4K prompt t/s | 12-turn prefill t/s | 12-turn decode t/s | 2048 decode t/s | p99 ms |
|---|---|---|---|---|---|---|
| Qwen, 16.3 GiB | 18.6 / 17.5 | 563 / 508 | 471 / 451 | 46.0 / 44.1 | 58.9 / 57.0 | 27.5 / 30.3 |
| GPT-OSS, 8.4 GiB | 10.5 / 9.5 | 735 / 718 | 586 / 586 | 70.4 / 65.0 | 89.3 / 89.2 | 16.7 / 16.8 |
| Gemma 4, 8.1 GiB | 14.5 / 9.2 | 630 / 523 | n/a | n/a | 54.2 / 52.4 | 26.1 / 35.8 |
| GLM, 8.7 GiB | 12.7 / 9.6 | 367 / 352 | 212 / 207 | 27.4 / 26.4 | 43.5 / 42.3 | 33.5 / 34.2 |

Gemma has no multi-turn run: the bench cannot apply its chat template. Correctness: slot checks
with 0 mismatches and 0 stale hits on all four, 0 non-finite logits; KL at run-to-run level for
Qwen, GPT-OSS and GLM. Gemma bounded shows KL 1.1e-3 (top-1 98%): the full bank reproduces it when
more rows go to the host (KL 1.7e-3 at a fixed threshold of 48), and bounded with room for the whole
bank gives 2.1e-5, so it is the host path's arithmetic under a different host share, not the cache.

Coverage (VRAM experts + RAM cache)/bank separates the results: 1.34-1.71 (Qwen 16.3, GLM,
GPT-OSS) stay within 10% on representative prefill and 5% on long decode; 1.00-1.20 (Gemma, Qwen at
12-14 GiB) lose 17-29% on prompts and more tail; under 1.0 (Qwen at 8 GiB) is not usable. Prewarm
did not pay off end to end on any model (Qwen: 18.8 s to the first token with it, 17.4 s cold).

## Runtime RAM-aware host plan (patch 0108, experimental)

`TOSH_AUTO_HOST_POLICY=runtime` makes Auto plan host memory from the machine's state at load, not
from installed RAM alone. Every cap is memory this load may still add:

- reclaimable = free + speculative + purgeable + min(inactive, file-backed) pages
  (`host_statistics64`; inactive anonymous pages need the compressor or swap, so they do not count)
- static cap = RAM - max(6 GiB, 20% RAM) - this process's RSS
- dynamic cap = reclaimable - clamp(15% RAM, 4 GiB, 12 GiB)
- lockable = `vm.user_wire_limit` - wired - 1 GiB, for the locked bank or RAM cache only
- effective = min(static, dynamic); pressure from `kern.memorystatus_vm_pressure_level` plus swap
  written during a 0.4 s sample (swap left from before does not count)

The whole bank is taken only with NORMAL pressure when engine + bank fit the effective cap and the
bank fits the lockable one; otherwise the RAM cache is min(bank, effective - engine - staging,
lockable), halved under ELEVATED pressure, none under CRITICAL, and a DMoE candidate needs
(arena + cache)/bank >= 1.0. Reason codes: FULL_HOST_SAFE, FULL_HOST_STATIC_LIMIT,
FULL_HOST_CURRENT_RAM_LIMIT, FULL_HOST_WIRE_LIMIT, HOST_MEMORY_PRESSURE_HIGH, BOUNDED_HOST_SELECTED
(_BORDERLINE under 1.3), BOUNDED_HOST_COVERAGE_TOO_LOW, BOUNDED_HOST_CURRENT_RAM_TOO_LOW,
HOST_MEMORY_LOCK_LIMIT. Without the switch the plan is unchanged. `runtime_sweep.py` compares
headroom formulas offline; `memhold.c` holds touched memory to put the machine in a known state.

This 32 GiB machine, Qwen at 8K (other load held by memhold):

| state | reclaimable | plan | RAM cache | coverage | 12-turn prefill t/s | 12-turn decode t/s |
|---|---:|---|---:|---:|---:|---:|
| light | 26.6 GiB | full host bank | 17.1 | 1.41 | 470.6 | 46.0 |
| 6 GiB held | 20.6 | bounded | 14.1 | 1.23 | 420.6 | 43.8 |
| 10 GiB held | 16.8 | bounded | 10.4 | 1.01 | 337.0 | 42.9 |
| released | 26.6 | full host bank | 17.1 | 1.41 | | |

No swap was written and pressure stayed normal during these runs. GPT-OSS stays on the full bank in
all four states, Gemma and GLM go bounded only with 10 GiB held. Bounded plans start without
prewarm: the first 3.4K prompt runs at 265-275 t/s against 563 and the first 2048 tokens at 46-47
t/s against 59. Coverage near 1.0 is slow in steady state too (prefill 28% slower): 1.0 lets it run,
it does not make it good.

## Startup priming and the coverage gate (patch 0109, experimental)

llama-server's warm-up is one decode of BOS and EOS (2 tokens) before the server is ready. With a
bounded cache it reads about 600 Qwen experts (1.0 GiB) and 99% of them are used afterwards, but it
lasts about 0.6 s: a fill limited to it loaded 0.1-0.2 GiB. The real window is the lock of the RAM
cache: mlock zero-fills every page, 3-4 s for 15 GiB. 0109 starts the cache as soon as the loader
knows the bank, locks it on its own thread, and meanwhile reads experts in file order into free
slots (16 readers, published only when read, last in line for eviction, never evicting); the
server waits for the lock and the fill stops at ready. Writing the pages with experts costs less
than zeroing them, so ready does not move.

Qwen, 14.8 GiB RAM cache (`srv_startup.py`, forced plan):

| fill | ready | loaded before ready | first 3.4K prompt | first token |
|---|---:|---:|---:|---:|
| none | 9.08 s | 1.0 GiB (warm-up) | 289.6 t/s | 20.73 s |
| 4 readers | 8.48 s | 6.1 GiB | 350.1 t/s | 18.12 s |
| 8 readers | 8.68 s | 8.8 GiB | 382.8 t/s | 17.49 s |
| 16 readers | 8.85 s | 10.4 GiB | 413.8 t/s | 17.01 s |

Full host bank on the same machine: ready 14.7 s, 578 t/s, first token 20.5 s. With the moderate
14.1 GiB plan in the bench the fill reached 12.1 GiB: first prompt 451.1 t/s against 275.3 cold,
2048-token decode 54.65 t/s (p99 43.2 ms) against 47.24 (65.8 ms). GPT-OSS, Gemma and GLM bounded
plans fill 1.9, 8.2 and 3.6 GiB the same way, 85-99% used afterwards.

Coverage gate: GOOD >= 1.25, CONSTRAINED 1.0-1.25, rejected under 1.0. A gate at 1.10 was tried
and dropped: Auto's fallback for Qwen with 10 GiB held (23 layers of experts on the host) ran the
12-turn prefill at 234.2 t/s and chat decode at 28.9 t/s, against 337.0 and 42.9 for the 1.01
bounded plan it would replace.

Also fixed here: a plan taken by Auto crashed llama-server on 9-31 token prompts (the offload
threshold and the cache switch had been read by the planner's probes before the plan set them),
and 0108 did not build without Dynamic MoE.

## Product hardening: fail-closed Auto, split models, Gemma (patch 0110)

**Gemma.** With TOSH_AUTO under a tight RAM budget (headroom forced to 19.5 GiB, 7.9 GiB allowed)
every candidate was rejected: full GPU by VRAM, the bounded cache at coverage 0.83-0.96, and the
classic expert offload (15-17 layers on the host need 8.2-9.1 GiB). The plan was UNSUPPORTED, but
`common_init_result` ignored the result and loaded with the user's arguments. Upstream `fit` then
put 15 layers of experts in a CPU_REPACK buffer over mmap, and the first long prompt wedged the GPU:
the main thread waited in `commandBufferWithUnretainedReferences` with no command buffer ever
completing. The same layout with `GGML_CPU_NO_REPACK=1` ran (262.5 t/s), and so did an explicit
ncmoe with no repack (386.6 t/s): the expert maths were never the problem. 0110 refuses the load
(exit 1 in 5 s, no VRAM touched) and, when the budget allows it, picks the classic offload
explicitly (PLAN_CLASSIC_NCMOE, no repack, no mmap). Same budget, bounded against classic:
262.5 / 47.42 t/s prefill / decode against 416.4 / 25.32.

Gemma, first 3.4K prompt and 2048-token decode (`dmoe_bench`, ub 1024 / 512 for ncmoe):

| mode | prefill | 12-turn prefill | chat decode | 2048 decode | p50/p95/p99 ms | RSS | KL vs full |
|---|---:|---:|---:|---:|---|---:|---:|
| full host DMoE | 633.2 | 397.2 | 44.85 | 54.34 | 18.3/20.5/24.4 | 14.5 GiB | ref |
| bounded 8.9 GiB (cov 1.105) | 573.9 | 408.8 | 37.86 | 53.94 | 18.3/21.2/26.1 | 10.0 GiB | 0 / 5.3e-4 |
| classic ncmoe 17 | 394.1 | 183.9 | 22.30 | 23.01 | 43.6/45.7/47.3 | 8.8 GiB | 2.4e-3 |

The bounded run with the same CPU/GPU split as the full bank (5974 host experts in both) matched it
bit for bit; the classic offload's KL is host arithmetic on 17 layers. The multi-turn bench now
renders turns with the model's Jinja template (`common_chat_templates_apply`), as llama-server does.

**Planner accounting.** The planner now sees every shard of a split GGUF, counts every weight the
probe leaves in host memory (Flash Next keeps 27.5 GiB of per-layer token embeddings there, which the
old estimate missed: projected host 73.4 GiB against a real RSS of 116 GiB, now 100.2), and rejects a
RAM cache smaller than one layer per expert size (1028 MiB for Gemma, which has two sizes). Reasons
carry codes: PLAN_FULL_GPU, PLAN_FULL_HOST_DMOE, PLAN_BOUNDED_DMOE, PLAN_CLASSIC_NCMOE,
PLAN_UNSUPPORTED, with limits VRAM_LIMIT, STATIC_RAM_LIMIT, CURRENT_RAM_LIMIT, WIRE_LIMIT,
COVERAGE_TOO_LOW, STRUCTURAL_WARM_TOO_SMALL, SPLIT_GGUF_METADATA_ERROR, NO_SAFE_FALLBACK. Plans for
Qwen, GPT-OSS, Gemma and GLM are identical to 0109 on 20 simulated machines (`plan_regression.py`).

**Split models and the cold tier.** The RAM cache keeps one descriptor per shard and reads each
expert from its own file. On a 64-lane card the loader fuses a file's gate and up experts into one
tensor; the cache now reads both halves, and the fusion is skipped where it has no place (a mapped
CPU buffer, "tensor buffer not set", or a repacking one, the repack.cpp:5153 assert, both on the
default path with `fit`).

Qwen3.8 Flash Next UD-Q4_K_XL, one Radeon Pro Vega II die, Auto without DMoE arguments:

| | pp512 | tg128 | 3.4K prompt | 512 decode | RSS | HOT hit |
|---|---:|---:|---:|---:|---:|---:|
| Auto, full host (ub 1024, arena 20.2 GiB) | 301.2 | 23.12 | 202.4 | 21.22 | 116.6 GiB | 82.1% |
| manual reference, same session | 308.7 | 23.00 | 179.4 | 21.22 | 116.3 GiB | 82.3% |
| Auto, bounded 62.8 GiB (cov 1.18) | | | 127.1 | 20.54 | 65.5 GiB | 82.7% |

Expert bytes read from three shards matched the loaded bank (576 checked, 0 differ), and 7866 slots
promoted from the bounded cache verified against the file.

## Real memory pressure, plan reporting and server memory (patch 0111)

`memhold.c` now grows in 256 MiB steps and stops growing on critical pressure, swap growth or low
free memory (`--mlock` wires what it holds); `pressure_step.py` holds N GiB, previews the Auto plan,
loads with Auto and runs the workload (3.4K prompt, 12-turn chat, 2048-token decode) sampling RSS,
VRAM, swap and compression; `plan_race.py` takes memory while Auto is planning.

Qwen3.6-35B-A3B under real held memory (32 GiB machine, runtime policy, final 0111):

| held | reclaimable | plan | RAM cache | HOT | coverage | projected RSS | RSS at end | footprint | 3.4K prompt | 12-turn prefill | chat decode | 2048 decode, p99 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 26.6 | full host | | 6.3 | 1.37 | 21.89 | 22.21 | 17.77 | 547.8 | 354.1 | 42.73 | 49.02, 34.3 ms |
| 4 | 21.6 | bounded | 15.2 | 6.3 | 1.26 | 16.88 | 16.73 | 15.68 | 365.5 | 334.5 | 38.78 | 39.32, 44.3 ms |
| 8 | 19.3 | bounded | 12.9 | 6.3 | 1.12 | 14.59 | 13.01 | 13.66 | 365.8 | 289.0 | 31.28 | 31.17, 83.7 ms |
| 11 | 16.9 | bounded (q8, ub 512) | 10.4 | 6.7 | 1.00 | 12.10 | 10.58 | 11.26 | 275.5 | 233.8 | 28.57 | 33.61, 67.3 ms |
| 12-13 | 14-15 | unsupported, exit 1 | | | | | | | | | | |
| released | 27.3 | full host | | 6.3 | 1.37 | 22.55 | 18.52 at ready | 17.77 | | | | |

GiB and t/s. No swap was written in any step (1.53 GiB before and after), VRAM was 9.25 GiB against
9.93 projected. The classic expert offload never won: with this card's free VRAM it needs as much
host memory as a bounded cache at coverage 1.0, so Auto goes from bounded to unsupported. Gemma 4
26B-A4B: full host at 0 and 6 GiB held, bounded at 10-12 (coverage 1.08-1.00), unsupported from 13.
GPT-OSS and GLM stay on the full bank up to 8 GiB held and go bounded at 14.

What the plan did not count, found by comparing projected and measured memory:

- llama-server's RAM prompt cache (`--cache-ram`, 8 GiB by default) filled after ready: Qwen full
  host ended at 22.2 GiB against 18.65 projected.
- context checkpoints, up to 32 per slot, each holding the sliding-window cache of a sequence:
  Gemma grew from 14.2 to 20.6 GiB with the prompt cache off, and wrote 2.1 GiB to swap during a
  long decode with it on.

The plan now gives both what its budget leaves, model first and checkpoints before the prompt
cache (their size comes from the file's sliding-window metadata), lowers them when they do not
fit, and counts them in the projection. Gemma full host then ended at 17.2 GiB against 21.2
projected with no swap. RSS also counts clean file pages the system can drop: Flash Next reads
115.5 GiB resident at ready but its footprint is 72.9 GiB against 108.2 projected, so the load
report carries both.

Compression in constrained bounded plans (1.4-2.4 GiB during a workload with 8-11 GiB held) is the
server's own cold pages: runtime state written once after ready, which macOS compresses first when
free memory is short. The holder alone at 8 GiB compresses nothing in 240 s. Pressure stayed
normal and no swap was written.

Planning takes 3.7 s of probes, so the plan checks memory again before committing: a 4 GiB
allocation 1 s into planning moved reclaimable from 26.1 to 21.8 GiB, and 0111 planned again to a
15.3 GiB bounded cache, where 0110 loaded the full bank from the old snapshot. A failed lock of the
RAM cache now fails the load cleanly (ALLOCATION_FAILED, exit 1); it used to abort at exit with the
lock thread still attached. A failed lock of the full bank keeps running with mixed execution off
and says so (`runtime_code`).

The plan JSON carries `plan_schema_version` 1 and a `product` object in bytes (mode, reason,
warnings, limits, limiting resource, memory, dmoe, runtime, model, unsupported requirements);
`<plan>.actual` has the load result (READY, PLAN_REFUSED, MODEL_LOAD_FAILED, ALLOCATION_FAILED,
CONTEXT_FAILED) with projected and measured RSS, footprint and VRAM.

Flash Next, one Vega II die, final 0111: pp512 302.7 t/s, tg128 23.10, 3.4K prompt 201.7, decode
21.19, footprint 72.9 GiB, VRAM 26.6 GiB against 30.0 projected, no swap.

## Generic MoE prefill with Dynamic MoE (after 0112, no engine patch kept)

Full host bank, same session, RX 6700 XT, `prefill/ubatch_sweep.sh` (llama-bench, t/s):

| model | ubatch | pp512 | pp1024 | pp2048 | pp4096 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 512 / 1024 / 2048 / 4096 / 8192 | 889 / 889 / 889 / 900 / 884 | 899 / 1045 / 1040 / 1050 / 1050 | 875 / 1011 / 1058 / 1054 / 1049 | 845 / 989 / 1045 / 854 / 858 |
| GPT-OSS 20B | same | 1216 / 1188 / 1200 / 1205 / 1214 | 1215 / 1368 / 1375 / 1379 / 1372 | 1175 / 1353 / 1346 / 1342 / 1340 | 1131 / 1291 / 1278 / 1164 / 1157 |
| Gemma 4 26B-A4B | same | 827 / 868 / 847 / 842 / 781 | 813 / 937 / 956 / 958 / 924 | 811 / 920 / 958 / 958 / 951 | 785 / 896 / 915 / 736 / 729 |
| GLM-4.7-Flash-REAP | same | 792 / 806 / 813 / 809 / 776 | 741 / 789 / 783 / 786 / 797 | 603 / 648 / 603 / 600 / 602 | 440 / 465 / 447 / 374 / 375 |

2048 against Auto's 1024 at pp4096: Qwen 989.4 -> 1045.1 (5.6% faster), Gemma 896.3 -> 914.8 (2.1%),
GPT-OSS 1290.8 -> 1278.2 (1.0% slower), GLM 465.2 -> 447.3 (3.8% slower). 4096 and up lose 10-20% on
all four: compute memory doubles with each doubling of the batch (Qwen 440 / 735 / 1326 MiB at
512 / 1024 / 2048, GPT-OSS 1.2 / 2.3 / 5.5 GiB, where attention without flash attention dominates)
and comes out of the expert arena, so more experts come from the host (Qwen 43.9 -> 68.4 GiB
uploaded from 2048 to 4096). Auto's 1024 is its rule: a larger batch only when it costs under 10% of
the bank in arena, 2048 only for high expert dispersion. ggml's allocator already reuses memory by
lifetime inside the graph, so the peak is one operation's live set, not attention plus MoE.

Where the prefill time goes (`prefill/prefill_timelines.sh`, ubatch 1024, 3.4K prompt): the main
thread waits for the router's ids while the GPU computes 56-70% of the prefill, and copies the
routed host experts into the upload ring 24-35% (Qwen 2067 of 5847 ms, GPT-OSS 1145 of 3948, Gemma
1738 of 5503, GLM 2141 of 8868). The copy already runs on every core (`dispatch_apply`), at about
9.4 GB/s for Qwen's 19.4 GiB, and the GPU blits at about 6.4 GB/s: moving experts is the limit, not
thread count.

Blitting the routed experts straight from the wrapped host bank instead of the ring
(`prefill/direct_blit_ab.sh`, two alternating rounds): Qwen 581.0 -> 444.9 t/s (23.4% slower),
GPT-OSS 815.5 -> 553.9 (32.1% slower), Gemma 630.4 -> 477.5 (24.3% slower), GLM 369.6 -> 115-314:
the GPU reading host pages is slower than a write-combined ring. Dropped.

Next layer's experts are not known before its router runs, and its attention needs this layer's
output, so the overlap left is inside a layer: resident experts (about 46% of the routed ones) could
compute while the rest upload on a separate blit queue, or one micro-batch's layer L+1 could run
while the next micro-batch is at layer L. Both change the graph; not attempted.

Flash Next on one Vega II die: 2048 against 1024 gives pp4096 327.5 -> 350.6 t/s (7.0% faster),
pp2048 346.2 -> 359.9; there the GPU compute (router wait 54 s) outweighs uploads.
