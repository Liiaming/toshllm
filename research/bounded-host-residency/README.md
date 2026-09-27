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

Not yet done: GPT-OSS, Gemma 4 and GLM (traces for Qwen and GPT-OSS exist), a prototype that
measures a real cold read against the model.
