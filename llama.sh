#!/bin/bash

export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export LD_LIBRARY_PATH=${ROCM_PATH}/lib
export PATH=${ROCM_PATH}/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export ROCBLAS_USE_HIPBLASLT=0
export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0
export GGML_HIP_FORCE_RS_GPU=1
export GGML_HIP_FORCE_KV_GPU=1
export GGML_HIP_GRAPHS=0
export GGML_HIP_ALLOC_GRAPH_RESERVE=2048
export ROCM_METADATA_WAIT_TIMEOUT=100
export ROCM_ALLOW_INT8_MIXED_PRECISION=1
export HIPBLASLT_LOG_MASK=32
export HIP_FORCE_DEV_KERNARG=1
export GGML_CUDA_FORCE_MMQ=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/llama/bin
export XDG_CACHE_HOME=/hf

if [ "$1" = "bash" ] || [ "$1" = "/bin/bash" ]; then
    exec "$@"
fi
set -- /llama/bin/llama-server \
    --models-preset /llamacpp_presets.ini \
    --models-max 4 \
    --models-dir /models/ \
    --models-autoload \
    --metrics \
    --log-timestamps \
    --log-prefix \
    --log-colors on \
    -fit off \
    --embeddings \
    --verbose \
    --mlock \
    --split-mode none \
    --log-verbosity 2 \
    --no-webui \
    --host :: \
    --port 8000 \
    "$@"

exec "$@"
