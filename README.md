# llama.cpp on AMD Strix Halo (gfx1151) with ROCmFP4

[GitHub]: https://github.com/ggerganov/llama.cpp
[Lemonade SDK]: https://github.com/lemonade-sdk/llamacpp-rocm

Optimized container infrastructure for running llama.cpp models on AMD Strix Halo (gfx1151) GPU with ROCmFP4 quantization support. Built using the [lemonade-sdk fork](https://github.com/lemonade-sdk/llamacpp-rocm) to produce MoE-capable container images.

## Hardware

| Specification | Value |
|--------------|-------|
| GPU | AMD gfx1151 (Strix Halo) |
| Memory | 128GB unified CPU/GPU (UMA) |
| ROCm | ROCm 7.13 nightly (gfx1151) |
| Fork | aardbeiplantje/rocmfp4-llama (mtp-rocmfp4-strix branch) |

## Quick Start

### Running Containers

```bash
./run.sh                              # Start server (REST API)
./run.sh cli -m /models/model.gguf    # Interactive chat
./run.sh quantize in.gguf out.gguf    # Quantize model
./run.sh bench -m /models/model.gguf  # Benchmark
./run.sh help                         # Show subcommands
```

### Build & Push to Docker Hub/GHCR

```bash
# Build for local testing
docker buildx bake --target _local

# Build and push to GHCR (requires Docker Hub auth)
docker buildx bake --target containers
```

## Features

- **ROCmFP4 Quantization**: Dual-scale 4.50 BPW optimized for Strix Halo
- **MoE Support**: MTP (Multi-Token Predict) for Qwen3.6, Nemotron models
- **Unified Memory**: 128GB UMA with proper allocation graphs
- **Flash Attention**: rocWMMA kernels on RDNA3.5
- **KV Cache Optimization**: GPU offload, f16/q8_0 support
- **Benchmark Suite**: 4 workloads (prompt processing, generation, mixed, long-context)
- **Request Tracking**: Built-in request ID generation for session management
- **Slot Assignment**: Per-request slot targeting via `id_slot` parameter

## Model Recommendations

| Model | Type | ROCmFP4 Size | KV Cache | Total | Notes |
|-------|------|-------------|----------|-------|-------|
| Gemma 4 26B A4B MoE | MoE | ~16 GB | ~16 GB | ~32 GB | ~4B active speed, excellent coding |
| Qwen3-Coder 30B A3B MoE | MoE | ~18 GB | ~19 GB | ~37 GB | Purpose-built for coding |
| Gemma 4 12B Unified | Dense | ~7 GB | ~10 GB | ~17 GB | Fast, good for multi-model |
| Gemma 4 31B Dense | Dense | ~23 GB | ~16 GB | ~39 GB | High quality, moderate speed |
| Qwen3.6-35B-A3B MoE | MoE | ~17 GB | ~17 GB | ~34 GB | MTP support with draft heads |

## Docker Environment Variables

| Variable | Default | Description |
|----------|--------|-------------|
| `MODELS_DIR` | (required) | Host path, mounted as `/models` inside container |
| `LLAMA_DOCKER_IMAGE` | local/ai/llama.cpp-gfx1151:latest | Docker image tag |
| `LLAMA_PRESETS` | ./llamacpp_presets.ini | Path to presets file |
| `HF_HOME` | /hf | HuggingFace cache directory |

## Quantization Workflow

### HF Model → BF16 GGUF

```bash
cd ~/rocmfp4-llama
PYTHONPATH=~/rocmfp4-llama/gguf-py/:$PYTHONPATH \
LD_LIBRARY_PATH=~/rocmfp4-llama/build-strix-rocmfp4/bin:$LD_LIBRARY_PATH \
python3 convert_hf_to_gguf.py \
  /path/to/hf-model-dir \
  --outtype bf16 \
  -o /path/to/model-bf16.gguf
```

### BF16 GGUF → ROCmFP4 Quantization

```bash
# Via Docker
./run.sh quantize /models/source-bf16.gguf /models/output-rocmfp4.gguf Q4_0_ROCMFP4_STRIX_LEAN

# Or inside container
llama-quantize source-bf16.gguf output-rocmfp4.gguf Q4_0_ROCMFP4_STRIX_LEAN
```

### ROCmFP4 Quantization Types

| Type | Format | BPW | Use Case |
|------|--------|-----|---------|
| `Q4_0_ROCMFP4` | dual-scale 4.50 BPW (pure) | 4.50 | Dense storage |
| `Q4_0_ROCMFP4_FAST` | single-scale 4.25 BPW (pure) | 4.25 | Pure speed |
| `Q4_0_ROCMFP4_STRIX` | FAST dense + Q6_K embeddings | 4.49 | Balanced |
| `Q4_0_ROCMFP4_STRIX_LEAN` | FAST dense + Q5_K embeddings | **4.38** | **Recommended (64-128GB)** |

**⚠️ Important**: Source must be F16/BF16 GGUF. Do NOT requantize from an already-quantized GGUF.

## Benchmarking

```bash
./run.sh bench                              # Full benchmark suite
./run.sh bench -m /models/model.gguf        # Single model benchmark
```

### Benchmark Workloads

1. **Prompt Processing**: Various prompt sizes (512 → 8192 tokens)
2. **Text Generation**: Various batch sizes (32 → 512)
3. **Mixed Workloads**: Realistic usage patterns
4. **Long Context**: 16K → 65K context stress test

### Bench-optimized Environment Variables

```bash
ROCBLAS_USE_HIPBLASLT=1
HSA_ENABLE_SDMA=1
GGML_HIP_GRAPHS=1
GGML_CUDA_FORCE_MMQ=0
AMD_DEBUG=high
```

## Build System

### docker-bake.hcl

Defines Docker Compose bake targets:
- `_local` → Local testing image
- `containers` → GHCR push with attestation + SBOM

### Dockerfile

Multi-stage build:
1. **base**: Debian trixie-slim + dependencies
2. **builder**: ROCm + llama.cpp fork + build
3. **runtime**: ROCm libs + binaries + user setup

### build_llama.cpp.sh

Build script with explicit HIP/ROCm configuration:

| Setting | Value | Purpose |
|---------|-------|--------|
| `CMAKE_HIP_ARCHITECTURES` | gfx1151 | Explicit GPU target |
| `GGML_HIP_UMA=ON` | - | Unified memory pool |
| `GGML_HIP_FORCE_RS_GPU=ON` | - | Recurrent state on GPU |
| `GGML_HIP_FORCE_KV_GPU=ON` | - | KV cache on GPU |
| `GGML_HIP_GRAPHS=ON` | - | Graph optimization |
| `GGML_HIP_ROCWMMA=ON` | - | Flash attention kernels |
| `JOBS=32` | - | Multi-core parallelism |

## Configuration

### llamacpp_presets.ini

Per-model preset configuration:

- **Default**: threads=16, f16 KV cache, SWA, flash-attn
- **Nemotron models**: reasoning=true, auto reasoning format
- **Qwen3.6/A3B MoE**: optimized KV cache types
- **Qwen3-Coder**: code-focused settings

### Environment Variables (llama.sh)

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
GGML_HIP_FORCE_RS_GPU=1
GGML_HIP_FORCE_KV_GPU=1
GGML_HIP_ALLOC_GRAPH_RESERVE=2048
HSA_ENABLE_SDMA=0
flash-attn = true
```

### llamacpp_presets.ini Settings

Key configuration for session management:

| Parameter | Value | Description |
|----------|-------|-------------|
| `parallel` | `1` (or `8` for Strix Halo) | Number of concurrent slots for processing |
| `cache-prompt` | `true` | Enable prompt caching for session tracking |
| `no-slots` | `false` | Enable slot-based request management |
| `cache-idle-slots` | `true` | Reuse idle slots for faster resume |

**For 128GB UMA + Strix Halo:**
- Set `parallel = 8` to utilize all 8 batch threads
- Combined with `threads = 16` and `threads-batch = 8`, this provides:
  - 8 concurrent request slots
  - Each slot uses up to 16 threads
  - Batch processing within requests uses 8 threads
- This matches the 128GB UMA capacity (8 slots × 16 threads × ~2GB/slot ≈ 4KB/slot cache, well within limits)

## API Request Tracking

llama-server includes native support for request/session tracking that allows pi.dev and similar platforms to manage concurrent sessions without continuous prompt processing:

### Request ID Generation

- **Response ID**: Each completion includes a unique `id` field (e.g., `"chatcmpl-xxxxx"`)
- **Session Tracking**: Store response IDs to correlate requests with session state
- **Caching**: Responses with identical IDs can be cached and reused

### Slot Assignment (Per-Request)

Use the `id_slot` parameter to assign completion tasks to specific processing slots:

```json
{
  "model": "your-model",
  "messages": [{"role": "user", "content": "Hello"}],
  "id_slot": 0  // Assign to slot 0 (first available)
}
```

**Benefits:**
- `id_slot = -1`: Auto-assign to idle slot (default)
- `id_slot >= 0`: Force use of specific slot for priority/queue management
- Enables deterministic task scheduling for session isolation

### Response Structure Example

```json
{
  "choices": [
    {
      "message": {
        "id": "chatcmpl-ecQULm0WqPrftUqjPZO1CFYeDjGZNbDu",
        "content": "Hello! How can I help you?",
        "role": "assistant"
      },
      "finish_reason": "stop"
    }
  ],
  "created": 1757141666,
  "model": "your-model",
  "id": "chatcmpl-ecQULm0WqPrftUqjPZO1CFYeDjGZNbDu"
}
```

### Session Management for pi.dev

To track sessions without continuous prompting:

1. **Create session**: Send request with `id_slot` assignment
2. **Store response ID**: Extract and cache the `id` field
3. **Resume session**: If using streaming, continue from stored ID
4. **Cancel session**: Use `/v1/chat/completions/{id}` to abort in-progress completions
5. **Cache prompts**: Enable `cache_prompt: true` for common prefixes

**Recommended settings for session tracking:**
- `--cache-prompt` (default: enabled)
- `--slot-save-path` directory for prompt persistence
- Monitor `/v1/metrics` for `llamacpp:requests_processing` count

See also:
- Slot management: `POST /slots/{id_slot}` endpoints
- Request metrics: `GET /v1/metrics`
- Completion control: `POST /v1/chat/completions/{id}/cancel` (experimental)

## Git Workflow

- Branch from `main`, open PRs against `main`
- Commit messages: imperative and short (~72 chars)
- Temporary files go to `.tmp/` (do not commit)

## License

[AGENTS.md for full guidelines](./AGENTS.md)

---

*Built with [Lemonade SDK](https://github.com/lemonade-sdk/llamacpp-rocm) and ROCmFP4 fork*
