#!/bin/zsh
# shorter set for the slow models: long decode, topic shift, many turns, long prompt
cd "$(dirname "$0")"
OUT=${1:?out dir}; BIN=${2:-./trace}
export TOSH_FA_AMD=1 GGML_CPU_NO_REPACK=1 TRACE_VERIFY=8000
for spec in gemma-4-26B-A4B-it-MXFP4_MOE:gem:20 GLM-4.7-Flash-REAP-23B-A3B-Q4_K_M:glm:20; do
  m=${spec%%:*}; rest=${spec#*:}; tag=${rest%%:*}; nc=${rest#*:}
  mkdir -p $OUT/$tag
  for w in prose:1024 shift4:160 conv12:100 n4-16k:32; do
    name=${w%%:*}; ngen=${w#*:}
    [ -s $OUT/$tag/$name.ids ] && grep -q "trace verify" $OUT/$tag/$name.log && { echo "$tag $name kept"; continue; }
    $BIN ~/models/$m.gguf workloads/$name.txt $OUT/$tag/$name $ngen $nc > $OUT/$tag/$name.log 2>&1
    echo "$tag $name rc=$? $(grep -o 'trace verify.*' $OUT/$tag/$name.log) $(grep -c '' $OUT/$tag/$name.ids) rows"
    sleep 5
  done
done
echo TRACES-DONE
