#!/bin/zsh
# prefill timelines, full host bank, ubatch 1024, 3.4K prompt
cd "$(dirname "$0")/.."
export TOSH_FA_AMD=1 DMOE_NO_SAMPLER=1 GGML_SCHED_PREFETCH_EXPERTS=1 GGML_CPU_NO_REPACK=1 TOSH_DMOE_STATS=1
D="TOSH_DMOE_CACHE_MIB=auto TOSH_DMOE_RESERVE_MIB=969 TOSH_DMOE_RARE_ROWS=auto TOSH_DMOE_RARE_TAIL=0 DMOE_EXPW=1 DMOE_LOAD=mlock DMOE_NCMOE=99"
for m in Qwen3.6-35B-A3B-UD-Q4_K_S gpt-oss-20b-Q4_K_M gemma-4-26B-A4B-it-MXFP4_MOE GLM-4.7-Flash-REAP-23B-A3B-Q4_K_M; do
  w=wl/longctx.txt; [[ $m == gemma* ]] && w=wl/gemma-longctx.txt
  env ${=D} DMOE_CTX=8192 DMOE_NUB=${TL_NUB:-1024} TOSH_DMOE_TIMELINE=pf/pp/tl-$m.tl ./dmoe_bench ~/models/$m.gguf $w pf/pp/tl-$m 32 > pf/pp/tl-$m.log 2>&1
  echo "$m rc=$?"; sleep 25
done
echo TL4-DONE
