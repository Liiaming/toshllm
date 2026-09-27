#!/bin/zsh
# routing traces for the residency simulator; outputs go to $1 (not tracked)
cd "$(dirname "$0")"
OUT=${1:?out dir}; BIN=${2:-./trace}
mkdir -p $OUT
export TOSH_FA_AMD=1 GGML_CPU_NO_REPACK=1 TRACE_VERIFY=20000
for spec in Qwen3.6-35B-A3B-UD-Q4_K_S:q35:26 gpt-oss-20b-Q4_K_M:oss:6 gemma-4-26B-A4B-it-MXFP4_MOE:gem:20 GLM-4.7-Flash-REAP-23B-A3B-Q4_K_M:glm:20; do
  m=${spec%%:*}; rest=${spec#*:}; tag=${rest%%:*}; nc=${rest#*:}
  mkdir -p $OUT/$tag
  for w in prose:2048 code:512 reasoning:512 multilingual:512 chat:256 shift4:256 conv12:200 longctx:256 n4-16k:64; do
    name=${w%%:*}; ngen=${w#*:}
    $BIN ~/models/$m.gguf workloads/$name.txt $OUT/$tag/$name $ngen $nc > $OUT/$tag/$name.log 2>&1
    echo "$tag $name rc=$? $(grep -o 'trace verify.*' $OUT/$tag/$name.log) $(grep -c '' $OUT/$tag/$name.ids) rows"
    sleep 10
  done
done
echo TRACES-DONE
