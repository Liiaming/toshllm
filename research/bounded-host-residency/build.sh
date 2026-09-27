#!/bin/zsh
# builds trace against the engine in vendor/llama.cpp/build-static
cd "$(dirname "$0")"
V=../../vendor/llama.cpp; B=$V/build-static; O=${1:-./trace}
clang++ -std=c++17 -O2 trace.cpp -I$V/include -I$V/ggml/include $B/src/libllama.a $B/ggml/src/libggml.a $B/ggml/src/ggml-metal/libggml-metal.a \
  $B/ggml/src/ggml-blas/libggml-blas.a $B/ggml/src/libggml-cpu.a $B/ggml/src/libggml-base.a \
  -framework Metal -framework Foundation -framework Accelerate -framework MetalKit -framework IOKit -framework CoreFoundation -o $O
