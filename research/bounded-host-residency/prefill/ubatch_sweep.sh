#!/bin/zsh
# prefill against ubatch with a full host bank (DMoE), same session: pp512..pp4096 at each ubatch
cd "$(dirname "$0")/.."
B=../../vendor/llama.cpp/build-static/bin/llama-bench
export TOSH_FA_AMD=1 GGML_SCHED_PREFETCH_EXPERTS=1 GGML_CPU_NO_REPACK=1 TOSH_DMOE_CACHE_MIB=auto TOSH_DMOE_RESERVE_MIB=969 TOSH_DMOE_RARE_ROWS=auto TOSH_DMOE_RARE_TAIL=0 TOSH_DMOE_STATS=1
OT='\.ffn_(up|down|gate|gate_up)_exps\.weight=CPU'
for m in ${=PP_MODELS:-Qwen3.6-35B-A3B-UD-Q4_K_S gpt-oss-20b-Q4_K_M gemma-4-26B-A4B-it-MXFP4_MOE GLM-4.7-Flash-REAP-23B-A3B-Q4_K_M}; do
  for ub in ${=PP_UBS:-512 1024 2048 4096 6144 8192}; do
    b=$(( ub > 4096 ? ub : 4096 ))
    $B -m ~/models/$m.gguf -ngl 99 -fa 1 --load-mode mlock -ot "$OT" -p 512,1024,2048,4096 -n 0 -ub $ub -b $b -r 2 -o csv > pf/pp/$m-ub$ub.csv 2> pf/pp/$m-ub$ub.err
    echo "$m ub$ub rc=$?"; sleep 25
  done
done
echo UBSWEEP-DONE
