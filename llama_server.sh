#!/bin/bash

# This script sets up the environment and runs the llama.cpp server with
# optimal settings for a ROCm 7.12 build on a Strix Halo (gfx1151) GPU.
LLAMA_CPP_DIR=${LLAMA_CPP_DIR?"Please set LLAMA_CPP_DIR to the build directory of llama.cpp (e.g., ~/llama.cpp/build)"}
LLAMA_CPP_DIR=$(readlink -f "$LLAMA_CPP_DIR")

# Your specific ROCm 7.12 build for gfx1151
export ROCM_PATH=${ROCM_PATH?"Please set ROCM_PATH to your ROCm 7.12 build directory (e.g., ~/therock-dist-linux-gfx1151-7.12.0a20260218)"}
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LLAMA_CPP_DIR/"
echo "Using ROCm from: $ROCM_PATH"

# Fix for ROCm 7.2+ hipBLASLt workspace faults on Strix Halo
export ROCBLAS_USE_HIPBLASLT=1
unset ROCBLAS_USE_HIPBLASLT
export HIPBLASLT_LOG_MASK=32
export HIP_FORCE_DEV_KERNARG=1
export GGML_CUDA_FORCE_MMQ=1

# Enable Unified Memory for that massive 128GB pool
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export GGML_ROCM_FORCE_TILING_ALLOCATOR=1
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HIP_VISIBLE_DEVICES=0

# Ensure the recurrent states (DeltaNet) are offloaded to the GPU
# This is critical for Qwen3-Next performance
export GGML_HIP_FORCE_RS_GPU=1
export GGML_HIP_FORCE_KV_GPU=1
export GGML_HIP_GRAPHS=0
export GGML_DEBUG=1

export HSA_FORCE_FINE_GRAIN_PCIE=1

# Force a larger workspace
export GGML_HIP_ALLOC_GRAPH_RESERVE=2048
export ROCM_METADATA_WAIT_TIMEOUT=100
export ROCM_ALLOW_INT8_MIXED_PRECISION=1

export LLAMA_LOG_COLORS=1
export LLAMA_LOG_TIMESTAMPS=1
export LLAMA_LOG_PREFIX=1

export HSA_ENABLE_SDMA=0
export AMD_DEBUG=high

# Using numactl to bind to the first CCD (0-7) and its local memory. 
# This reduces Infinity Fabric noise while the GPU is slamming the memory controller.
BIND_CMD=""
if command -v numactl &> /dev/null; then
    BIND_CMD="numactl --cpunodebind=0 --membind=0"
fi
# disable for now
BIND_CMD=""

HERE="$BASH_SOURCE"
HERE="${HERE%/*}"
MODELS_DIR=${MODELS_DIR?"Please set MODELS_DIR to the directory where your llama.cpp models are stored (e.g., /models)"}
LLAMA_PRESETS="${LLAMA_PRESETS:-$HERE/llama_server.ini}"
cd $MODELS_DIR || exit $?
exec $BIND_CMD $LLAMA_CPP_DIR/bin/llama-server \
    --models-preset "${LLAMA_PRESETS}" \
    --models-dir "$MODELS_DIR" \
    --no-webui \
    --verbose \
    --split-mode none \
    --log-verbosity 3 \
    --host :: \
    --port 8000 \
    "$@"
