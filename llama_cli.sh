#!/bin/bash

# --- Environment Setup ---
# Your specific ROCm 7.12 build for gfx1151
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-~/llama.cpp/build}
LLAMA_CPP_DIR=$(readlink -f "$LLAMA_CPP_DIR")
export ROCM_PATH=${ROCM_PATH:-~/therock-dist-linux-gfx1151-latest}
export LD_LIBRARY_PATH=$ROCM_PATH/lib:${LLAMA_CPP_DIR}/bin

# CRITICAL: Fix for ROCm 7.2+ hipBLASLt workspace faults on Strix Halo
#export ROCBLAS_USE_HIPBLASLT=0

# Enable Unified Memory for that massive 128GB pool
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export HSA_OVERRIDE_GFX_VERSION=11.5.1

# --- Performance Tweaks ---
# Using numactl to bind to the first CCD (0-7) and its local memory. 
# This reduces Infinity Fabric noise while the GPU is slamming the memory controller.
BIND_CMD=""
if command -v numactl &> /dev/null; then
    BIND_CMD="numactl --cpunodebind=0 --membind=0"
fi

exec $BIND_CMD ${LLAMA_CPP_DIR}/bin/llama-cli \
    -lv 1 \
    --context-shift \
    --jinja \
    -ctk f16 \
    -ctv f16 \
    --temp 0 \
    --top-p 0 \
    --min-p 0 \
    --color on \
    --no-mmap \
    --no-warmup \
    --mlock \
    -ngl 999 \
    --flash-attn on \
    -b 2048 \
    -ub 256 \
    -t 16 \
    -tb 16 \
    -ctxcp 32 \
    --cache-ram 65536 \
    -sm none \
    --spec-draft-ngl all \
    --spec-draft-type-k f16 \
    --spec-draft-type-v f16 \
    --spec-draft-n-max 5 \
    --spec-draft-n-min 0 \
    --spec-draft-p-min 0.0 \
    --spec-draft-p-split 0.10 \
    --reasoning on \
    "$@"
