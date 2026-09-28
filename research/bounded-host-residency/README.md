# Bounded host expert residency for Dynamic MoE (feasibility, 2026-09-27)

Question: can Dynamic MoE keep only part of the expert bank in RAM (WARM, mlocked) and read the
rest from the GGUF on demand (COLD, `pread`), instead of locking the whole bank?

Not productized. Production Dynamic MoE still locks the full bank.

## Tools

- `trace.cpp` (`build.sh`): router trace per token and 512-token prefill chunk. `TRACE_VERIFY=N`
  checks the first N rows against the top-k of the router's contiguous scores;
  `TRACE_NEGATIVE_CONTROL=1` reads the view as contiguous on purpose and must fail.
- `run_traces.sh`, `run_traces_subset.sh`: workloads in `workloads/`.
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

## Engine prototype, Qwen3.6-35B-A3B (patch 0116, experimental)

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

## WARM size, prefill attribution and prewarm (patch 0117, experimental)

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

## Async VRAM -> RAM copies instead of rereads (patch 0118, experimental)

`TOSH_DMOE_HOST_DEMOTE=gpu TOSH_DMOE_DOWN_STAGE=1 TOSH_DMOE_DOWN_QUEUE=own`: an expert leaving VRAM
is blitted into a 64 x 2 MiB device-allocated ring on a queue of its own and copied into its RAM
slot on a dispatch queue; the slot stays loading until then (requests wait for the copy, never for
the file) and the VRAM slot is not reused before the copy finished. A failed or ring-full copy
falls back to a background file read. `demote_bench.m` is the microbenchmark.

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
