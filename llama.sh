#!/bin/bash

# Unified llama.cpp wrapper for Strix Halo (gfx1151) ROCm builds.
# Usage:
#   llama.sh [subcommand] [args...]
#   llama.sh                          # server (default)
#   llama.sh cli -m /models/model.gguf
#   llama.sh rocm-server model.gguf   # ROCm::Inference only, at /rocm/v1/
#   llama.sh quantize in.gguf out.gguf [quant-type]
#   llama.sh bench -m /models/model.gguf

# --- Environment Setup ---
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export LD_LIBRARY_PATH=${ROCM_PATH}/lib
export PATH=${ROCM_PATH}/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export HIP_VISIBLE_DEVICES=0
# HIP_LAUNCH_BLOCKING=1 / AMD_SERIALIZE_KERNEL=1 serialize every kernel launch (debugging, or a workaround for
# GPU hangs): off unless set (run.sh passes them through)
export GGML_HIP_FORCE_RS_GPU=1
export GGML_HIP_FORCE_KV_GPU=1
export GGML_HIP_ALLOC_GRAPH_RESERVE=2048
export ROCM_METADATA_WAIT_TIMEOUT=1000
export ROCM_ALLOW_INT8_MIXED_PRECISION=1
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_CONCURRENT_EXEC=1
# export HSA_FORCE_FINE_GRAIN_PCIE=1  # Can cause GPU lockups on some firmware
export AMD_LOG_LEVEL=0
export HSA_LOG_LEVEL=0
export HSA_DISABLE_CRASH_DUMP=1
export HSA_ENABLE_SDMA=0
export XDG_CACHE_HOME=/hf

export LLAMA_ATTN_ROT_DISABLE=1

# --- Mode-specific overrides ---
export ROCBLAS_USE_HIPBLASLT=${ROCBLAS_USE_HIPBLASLT:-0}
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0}
export GGML_HIP_GRAPHS=${GGML_HIP_GRAPHS:-0}
export GGML_CUDA_FORCE_MMQ=${GGML_CUDA_FORCE_MMQ:-1}

# --- Resolve binary paths ---
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-/llama}
LLAMA_SERVER=${LLAMA_CPP_DIR}/bin/llama-server
LLAMA_CLI=${LLAMA_CPP_DIR}/bin/llama-cli
LLAMA_COMPLETION=${LLAMA_CPP_DIR}/bin/llama-completion
LLAMA_QUANTIZE=${LLAMA_CPP_DIR}/bin/llama-quantize
LLAMA_BENCH=${LLAMA_CPP_DIR}/bin/llama-bench
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LLAMA_CPP_DIR}/bin

# --- ROCm::Inference (rocm-hip): rocm-inference-server, one model, OpenAI-compatible, on
# /tmp/rocm-inference.sock, reached through nginx at /rocm/v1/... ---
ROCM_HIP=${ROCM_HIP:-/opt/rocm-hip}
start_rocm_inference() {
    local model=${ROCM_MODEL:?ROCM_MODEL=<model.gguf> (a path, or a file in /models) is required}
    [ -f "$model" ] || model=${M_DIR:-/models}/$model
    # never with the kernel-serializing debug switches, even when set for llama-server
    env -u HIP_LAUNCH_BLOCKING -u AMD_SERIALIZE_KERNEL \
        PERL5LIB="$ROCM_HIP/perl5/lib/perl5${PERL5LIB:+:$PERL5LIB}" \
        perl "$ROCM_HIP/perl5/bin/rocm-inference-server" \
            --model "$model" \
            --listen /tmp/rocm-inference.sock \
            --ctx "${ROCM_CTX:-32768}" \
            ${ROCM_DRAFT:+--draft "$ROCM_DRAFT"} \
            ${ROCM_STORE:+--store "$ROCM_STORE"} \
            ${ROCM_ARGS:-} &
}

# --- Subcommand dispatch ---
SUBCMD="${1:-server}"

case "$SUBCMD" in
    # ---- server (default) ----
    bash|sh)
        exec "$@"
        ;;
    server)
        shift
        if [ "$1" = "bash" ] || [ "$1" = "/bin/bash" ]; then
            exec "$@"
        fi
        "$LLAMA_SERVER" \
            --models-preset "${L_INI:-${PRESET_FILE:-/llamacpp_presets.ini}}" \
            --models-max 4 \
            --models-dir "${M_DIR:-/models}" \
            --models-autoload \
            --metrics \
            --slot-save-path "${S_DIR:-/llama.cpp/slots}" \
            --slots \
            --cache-idle-slots \
            --slot-prompt-similarity 0.0 \
            --log-timestamps \
            --log-prefix \
            --log-colors on \
            -fit off \
            --embeddings \
            --split-mode none \
            --no-webui \
            --host /tmp/llama.sock \
            --timeout 3600 \
            "$@" &
          # also ROCm::Inference next to it, at /rocm/v1/..., when a model is given
          [ -n "${ROCM_MODEL:-}" ] && start_rocm_inference
          set -x
          exec /usr/sbin/nginx -c /nginx.conf -g "daemon off;"
        ;;

    # ---- rocm-server: ROCm::Inference only (no llama-server), at /rocm/v1/... ----
    rocm-server)
        shift
        ROCM_MODEL=${1:-${ROCM_MODEL:-}}
        start_rocm_inference
        set -x
        exec /usr/sbin/nginx -c /nginx.conf -g "daemon off;"
        ;;

    # ---- cli: interactive chat ----
    cli)
        shift
        exec "$LLAMA_CLI" \
            -lv 1 \
            --context-shift \
            --jinja \
            -ctk q8_0 \
            -ctv q8_0 \
            --temp 0 \
            --top-p 0 \
            --min-p 0 \
            --color on \
            --no-warmup \
            -ngl 999 \
            --flash-attn on \
            -b 512 \
            -ub 64 \
            -t 16 \
            -tb 16 \
            -ctxcp 32 \
            --cache-ram 65536 \
            -sm none \
            --reasoning on \
            "$@"
        ;;

    # ---- completion: interactive chat ----
    completion)
        shift
        exec "$LLAMA_COMPLETION" \
            -lv 1 \
            --context-shift \
            --jinja \
            -ctk f16 \
            -ctv f16 \
            --temp 0 \
            --top-p 0 \
            --min-p 0 \
            --color on \
            --no-warmup \
            -ngl 999 \
            --flash-attn on \
            -b 2048 \
            -ub 256 \
            -t 16 \
            -tb 16 \
            -sm none \
            --simple-io \
            -no-cnv \
            --reasoning on \
            "$@"
        ;;

    # ---- quantize: BF16/FP16 -> ROCmFP4 ----
    quantize)
        shift
        SOURCE="${1:?Usage: llama.sh quantize source.gguf output.gguf [quant-type]}"
        shift
        OUTPUT="${1:?Usage: llama.sh quantize source.gguf output.gguf [quant-type]}"
        shift
        QUANT_TYPE="${1:-Q4_0_ROCMFP4_STRIX_LEAN}"
        shift
        exec "$LLAMA_QUANTIZE" "$SOURCE" "$OUTPUT" "$QUANT_TYPE" "$@"
        ;;

    # ---- bench: benchmark performance ----
    bench)
        shift
        # Set bench-optimized env overrides
        export ROCBLAS_USE_HIPBLASLT=1
        export HSA_ENABLE_SDMA=1
        export GGML_HIP_GRAPHS=1
        export GGML_CUDA_FORCE_MMQ=0
        export AMD_DEBUG=high

        # If user provides arguments, pass them directly to llama-bench
        if [ $# -gt 0 ]; then
            exec "$LLAMA_BENCH" \
                -mmp 0 \
                -b 4096 \
                -ub 1024 \
                -t 8 \
                -fa 1 \
                "$@"
        fi

        # Otherwise run comprehensive test suite
        echo "======================================================================"
        echo "Running comprehensive benchmark suite for Strix Halo"
        echo "======================================================================"
        echo ""

        echo ">>> Test 1: Prompt Processing (pp) at various sizes"
        "$LLAMA_BENCH" \
            -mmp 0 -b 4096 -ub 1024 -t 8 -fa 1 \
            -pg 512,0 -pg 1024,0 -pg 2048,0 -pg 4096,0 -pg 8192,0

        echo ""
        echo ">>> Test 2: Text Generation (tg) at various batch sizes"
        "$LLAMA_BENCH" \
            -mmp 0 -b 4096 -ub 1024 -t 8 -fa 1 \
            -pg 0,32 -pg 0,64 -pg 0,128 -pg 0,256 -pg 0,512

        echo ""
        echo ">>> Test 3: Mixed workloads (realistic usage patterns)"
        "$LLAMA_BENCH" \
            -mmp 0 -b 4096 -ub 1024 -t 8 -fa 1 \
            -pg 512,128 -pg 1024,128 -pg 2048,256 -pg 4096,512 -pg 8192,512

        echo ""
        echo ">>> Test 4: Long context processing (stress test)"
        "$LLAMA_BENCH" \
            -mmp 0 -b 4096 -ub 1024 -t 8 -fa 1 \
            -pg 16384,0 -pg 32768,0 -pg 65536,0

        echo ""
        echo "======================================================================"
        echo "Benchmark suite complete!"
        echo "======================================================================"
        ;;

    # ---- help ----
    --help|-h|help)
        echo "Usage: llama.sh [subcommand] [args...]"
        echo ""
        echo "Subcommands:"
        echo "  (none) / server   Start llama.cpp server (default)"
        echo "  rocm-server       ROCm::Inference only, at /rocm/v1/ (e.g. llama.sh rocm-server Qwen3.5-4B-ROCMFP4.gguf);"
        echo "                    with server, ROCM_MODEL=<model.gguf> runs it next to llama-server"
        echo "  cli               Interactive CLI chat (e.g. llama.sh cli -m /models/model.gguf)"
        echo "  quantize          Quantize GGUF to ROCmFP4 (e.g. llama.sh quantize in.gguf out.gguf Q4_0_ROCMFP4_STRIX_LEAN)"
        echo "  bench             Run benchmark suite (e.g. llama.sh bench -m /models/model.gguf)"
        echo "  help              Show this help"
        echo ""
        echo "Default quantize type: Q4_0_ROCMFP4_STRIX_LEAN (4.38 BPW, memory-efficient)"
        ;;

    *)
        echo "Unknown subcommand: $SUBCMD" >&2
        echo "Run 'llama.sh help' for usage." >&2
        exit 1
        ;;
esac
