# Handoff: ToshLLM 0.87.11 and Dynamic MoE

State after the local release preparation of 0.87.11. Nothing in this document has been pushed,
tagged or published; those steps wait for the owner's approval.

## 1. Repository state

- Branch `main`. The release-preparation commit (`release: prepare ToshLLM 0.87.11`) sits on top of
  `246e101`; `git log v0.87.10..HEAD` lists the whole release.
- Engine patch series `patches/llama` 0001-0112, 107 files, applied on llama.cpp `9575389609d6`
  (`LLAMA_COMMIT` in `scripts/build-engines.sh`). The series reproduces `vendor/llama.cpp` exactly
  (fresh-index tree hash `aa5303d7`).
- Not pushed, no tag, no GitHub release.

## 2. Release state

- Version 0.87.11 (`VERSION`, `Sources/App/AboutTab.swift`, the bundle's Info.plist), proposed tag
  `v0.87.11`. Previous release v0.87.10.
- `./scripts/release.sh` stamps `## [Unreleased]` as `## [0.87.11] - <date>`, commits, creates the
  annotated tag and pushes; CI then builds, signs, notarizes and publishes the DMG with that
  CHANGELOG section as the release body. Run it only after approval.
- Local artifact (ad-hoc signed, not uploaded; CI signs and notarizes the published one):
  `dist/ToshLLM-v0.87.11.dmg`, 74162125 bytes, SHA-256
  `e2b917221043396f67e34fa0f5b2b42e3e955a84294dd442033fcf41463f1531`.
- Tests: 301 executed, 1 skipped, 0 failures (the slot prototype's tests went with it); translation
  files in sync; engines built clean with and without Dynamic MoE.

## 3. Dynamic MoE architecture

Three tiers of expert memory:

- **HOT**: a VRAM arena of expert slots per layer, filled by dynamic residency (decayed use counts).
- **WARM**: host RAM. Either the whole expert bank, locked (full host), or a bounded cache of
  expert-sized slots, locked, one pool per expert size (bounded host).
- **COLD**: the GGUF itself, read with `pread` per expert part, one descriptor per shard, both halves
  of a gate/up pair the loader fuses on 64-lane cards.

Prefill uploads only the experts a chunk routes to (compact upload): residents copy device to
device, the rest go through a write-combined ring and a blit. Rare experts (few rows) run on the
CPU executor. Decode misses fall back to the CPU executor while promotions fill the arena.

Plans, from `common/tosh-plan.cpp`:

| mode | meaning |
|---|---|
| `PLAN_FULL_GPU` | model and requested context fit in VRAM with the reserve |
| `PLAN_FULL_HOST_DMOE` | whole bank locked in RAM, HOT arena in VRAM |
| `PLAN_BOUNDED_DMOE` | bounded WARM cache + HOT arena, the rest COLD |
| `PLAN_CLASSIC_NCMOE` | explicit `--n-cpu-moe` layout, no repack, no mmap |
| `PLAN_UNSUPPORTED` | fail closed: exit 1, nothing loaded |

## 4. Product behavior

- `--dynamic-moe on|off`, default off. App: Settings → Performance & Memory → "Dynamic MoE
  (experimental)", key `dynamicMoeEnabled`, default off for new and existing users.
- Off: the standard expert offload, exactly as before; the engine logs `Tosh MoE: standard`.
- On: the planner owns memory for MoE models (fit off, KV type, batch, offload, cache sizes).
- Precedence: `TOSH_DMOE_*` set by hand > `TOSH_AUTO` (developer override; the app uses it only for
  the forced `dmoe` mode and the retry `nodmoe`) > `--dynamic-moe` > standard.
- The slot-based prototype (`TOSH_MOE_*`, "Optimize dMoE") is gone from the app. Its engine code is
  still compiled and inert.

## 5. Planner

- Host signals (`host_statistics64`): reclaimable = free + speculative + purgeable +
  min(inactive, file-backed); compressed, swap used, swap written in a 0.4 s sample,
  `kern.memorystatus_vm_pressure_level`, this process's RSS.
- static cap = RAM - max(6 GiB, 20% RAM) - RSS; dynamic cap = reclaimable - clamp(15% RAM, 4, 12 GiB);
  lockable = `vm.user_wire_limit` - wired - 1 GiB; effective = min(static, dynamic).
- Full host needs bank + host weights (measured by the probe, e.g. Flash Next's 27.5 GiB of
  per-layer embeddings) within effective and lockable, with normal pressure.
- Bounded WARM = min(bank, effective - engine - staging, lockable), halved under elevated pressure,
  never below one layer's experts per expert size (`STRUCTURAL_WARM_TOO_SMALL`).
- The server's prompt cache and context checkpoints get what the plan leaves, checkpoints first
  (`CONTEXT_CHECKPOINTS_REDUCED`, `PROMPT_CACHE_REDUCED`); checkpoint size comes from the file's
  sliding-window metadata.
- Stale snapshot: planning spends ~3.7 s in probes; memory is read again before committing and the
  plan is made once more if reclaimable dropped over 1 GiB or pressure rose
  (`REPLANNED_AFTER_MEMORY_CHANGE`).
- Output: plan JSON with `plan_schema_version` 1 and a `product` object in bytes; `<plan>.actual`
  with the load result (READY, PLAN_REFUSED, MODEL_LOAD_FAILED, ALLOCATION_FAILED, CONTEXT_FAILED),
  projected against measured RSS, `phys_footprint` and private VRAM. Compare the projection with the
  footprint: RSS also counts clean file pages.

## 6. Coverage

(HOT arena + WARM) / bank: GOOD >= 1.25, CONSTRAINED 1.0-1.25, REJECT < 1.0.

## 7. Startup preload

The bounded cache starts in the loader, locks on its own thread (mlock zero-fills pages for 3-4 s
on 15 GiB) and meanwhile 16 readers fill free slots in file order; the server waits for the lock
before it is ready. No separate blocking warm-up. Fill before ready varies with disk state
(8-12 GiB on Qwen in the same window on different days).

## 8. Validated models

RX 6700 XT 12 GB, 32 GB RAM: Qwen3.6-35B-A3B UD-Q4_K_S, GPT-OSS 20B Q4_K_M, Gemma 4 26B-A4B
MXFP4, GLM-4.7-Flash-REAP-23B Q4_K_M. One Radeon Pro Vega II die, 512 GB RAM: Qwen3.8 Flash Next
UD-Q4_K_XL, 4 shards, 103.7 GiB, bank 71.7 GiB.

## 9. Performance references

3.4K-token prompt, RX 6700 XT, t/s (prompt / generation):

| model | standard offload | Dynamic MoE, full host |
|---|---|---|
| Qwen3.6-35B-A3B | 310.6 / 29.0 | 580.6 / 43.8 |
| GPT-OSS 20B | 496.7 / 40.7 | 901.9 / 68.1 |
| Gemma 4 26B-A4B | 384.6 / 22.5 | 595.3 / 38.8 |
| GLM-4.7-Flash-REAP | 411.6 / 28.5 | 401.8 / 30.9 |

Qwen under real held memory (first prompt, 12-turn prefill / decode, 2048 decode):

| held | plan | WARM | coverage | first | 12-turn | 2048 |
|---:|---|---:|---:|---:|---|---:|
| 0 GiB | full host | | 1.37 | 547.8 | 354.1 / 42.7 | 49.0 |
| 4 GiB | bounded | 15.2 | 1.26 | 365.5 | 334.5 / 38.8 | 39.3 |
| 8 GiB | bounded | 12.9 | 1.12 | 365.8 | 289.0 / 31.3 | 31.2 |
| 11 GiB | bounded, q8 KV | 10.4 | 1.00 | 275.5 | 233.8 / 28.6 | 33.6 |
| 12+ GiB | unsupported | | | | | |

Gemma bounded (8.6 GiB cache, coverage 1.1): prompt 574 t/s, decode 45 t/s, KL 5e-4 against the
full bank. Gemma classic offload under 12 GiB held (forced): 306.3 / 21.5.
Flash Next, one die: pp512 302.7, tg128 23.1; server 3.4K prompt 201.7, decode 21.2; footprint
72.9 GiB; ubatch 2048 gives pp4096 350.6 against 327.5.

## 10. Bugs fixed on the way

- Auto's UNSUPPORTED was ignored and upstream `fit` loaded a CPU_REPACK layout that wedged the GPU.
- The offload threshold and the cache switch were read by the planner's probes before the plan set
  them (crash on 9-31 token prompts in bounded mode).
- Split GGUFs: the planner read only the first shard; the cache used one file descriptor.
- The loader's fused gate/up had no COLD descriptor, and could not live in a mapped or repacking CPU
  buffer ("tensor buffer not set", `repack.cpp:5153`).
- The server's prompt cache and context checkpoints were not in the plan (Gemma swapped 2 GiB).
- A failed lock of the RAM cache aborted at exit; now ALLOCATION_FAILED, exit 1.
- A memory change during planning went unseen; now one replan.

## 11. Known risks

- Legacy `TOSH_MOE_*` engine code still compiled (inert).
- Constrained bounded plans: macOS compresses 1.4-2.4 GiB of the server's cold pages under load;
  no swap, normal pressure.
- Prompt speed drops with coverage (about half at 1.0).
- Context checkpoints of recurrent models are not estimated (~0.7 GiB residual on Qwen3.6).
- Preload before ready depends on disk state.
- The classic offload rarely wins under real pressure: it needs about as much host memory as a
  bounded cache at coverage 1.0.

## 12. Prefill research result

- Larger chunks: NO-GO generically (1024 -> 2048: Qwen +5.6%, Gemma +2.1%, GPT-OSS -1.0%, GLM -3.8%;
  4096+ loses 10-20% because compute memory comes out of the HOT arena).
- Scratch sharing: NOT NEEDED (ggml's allocator already reuses by lifetime).
- Helper threads: NOT NEEDED (the ring copy already runs on every core, ~9.4 GB/s).
- Direct blit from the wrapped host bank: NO-GO (23-32% slower).
- Resident/upload overlap: CONTINUE DIAGNOSIS.
Details: `research/bounded-host-residency/README.md`, section "Generic MoE prefill".

## 13. Next research target

Overlap execution of experts already HOT with the upload of non-resident ones. Not another chunk
sweep, not PLE as a generic feature, not predicted expert ids, not MTP, not split-K, unless new
evidence appears.

Questions: are HOT experts blocked until every upload of the layer lands; what share of routed
expert work is HOT per layer (about 46% of routed experts on Qwen at 1024); how long is the GPU
idle during ring copy and blits (the copy is 24-35% of prefill); can a second Metal queue run the
blits while the resident part of MUL_MAT_ID computes (on Vega a blit queue overlaps compute); which
dependencies (the join, the scheduler's split boundaries, the ids read-back) prevent it. Primary
models Qwen, GPT-OSS, Gemma, GLM; Flash Next only as a stress test.

## 14. Legacy cleanup

Remove the inert `TOSH_MOE_*` implementation (Metal context and ops, graph, loader, llama-bench,
`tosh-moe/tosh-moe-cache.cpp`) after the release, as a behavior-preserving cleanup with the same
smoke matrix before and after. Some pieces are shared (gate/up fusion, staging); separate them first.

## 15. Starting instruction for the next session

Start from the release-preparation commit (or from the `v0.87.11` tag once published). Do not modify
the production planner or cache architecture. First verify the repository and release state: clean
tree, 107 patches applying on the pinned commit, vendor reproduced, and whether 0.87.11 was pushed and
tagged. The next research target is generic MoE expert execution/upload overlap: measure how much
routed work is already HOT per layer and how long the GPU idles during uploads before writing code.
Use Qwen3.6-35B-A3B, GPT-OSS 20B, Gemma 4 26B-A4B and GLM-4.7-Flash-REAP as primary models and
Flash Next only as a stress test. Follow MEASURE -> EXPLAIN -> RECONCILE -> GATE -> IMPLEMENT.
