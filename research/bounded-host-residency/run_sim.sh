#!/bin/zsh
# simulator over every trace of a model; results as one json per line in $3
cd "$(dirname "$0")"
TR=${1:?trace dir}; GEO=${2:?geometry dir}; OUT=${3:?out file}; shift 3
for spec in $@; do   # tag:arena_mib
  m=${spec%%:*}; a=${spec#*:}
  for f in $TR/$m/*.ids; do
    for bypass in 0 1; do
      python3 sim.py --trace $f --geometry $GEO/$m.json --arena-mib $a --policies lru,lfu --prefill-bypass $bypass \
        | python3 -c "import json,sys; [print(json.dumps(dict(r, model='$m', bypass=$bypass))) for r in json.load(sys.stdin)]" >> $OUT
    done
  done
done
echo SIM-DONE >> $OUT
