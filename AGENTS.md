# AGENTS.md - Guidelines for AI Coding Assistants

## Project Overview
This repository contains build and deployment infrastructure for [llama.cpp](https://github.com/ggerganov/llama.cpp) on AMD Strix Halo (gfx1151) via ROCm. Built using the [lemonade-sdk fork](https://github.com/lemonade-sdk/llamacpp-rocm) to produce container images serving MoE models. Hardware: AMD Ryzen AI 9 HX 395+ with 128GB unified memory.

## Build System
- **`docker-bake.hcl`**: Docker Compose bake file defining build targets via `docker buildx bake`. Pushes to GHCR (`ghcr.io/ai/llama.cpp-gfx1151:latest`) and internal registry by default. Version variable controls the lemonade-sdk release tag (default b9586).
- **`Dockerfile`**: Multi-stage build using Debian trixie-slim. Builder stage clones the llama.cpp fork and builds. Runtime stage packages ROCm libs + the built binaries. Entrypoint is `llama.sh` which auto-loads models from `/models/`.
- **`llamacpp_presets.ini`**: Per-model preset configuration. `[ * ]` section holds defaults (threads=16, f16 KV cache, SWA, flash-attn, cache-ram=-1). Individual model sections override as needed. MoE models need their own `threads` override (default 16 is fine; scale up if needed).
- **`llama.sh`**: Persistent server wrapper with host networking, webui disabled, presets loaded from `/llamacpp_presets.ini`.

## ROCmFP4 Fork
The build uses the custom fork at [`aardbeiplantje/rocmfp4-llama`](https://github.com/aardbeiplantje/rocmfp4-llama) with ROCmFP4 quantization support (Codebook10 4-bit format optimized for Strix Halo gfx1151).

- **`mtp-rocmfp4-strix`** branch: Main ROCmFP4 + MTP (Multi-Token Predict) support for Qwen3.6 MoE and similar models.
- **`nemotron-mtp-rocmfp4-strix`** branch: Extends `mtp-rocmfp4-strix` with additional fixes specific to NVIDIA Nemotron MoE models with ROCmFP4 quantization. Used in the Dockerfile for building.

## Quantization Workflow
ROCmFP4 quantization is done via the forked llama.cpp tools (built to `build-strix-rocmfp4/bin/` or inside Docker to `/llama/`).

### Converting HuggingFace → BF16 GGUF
Use `convert_hf_to_gguf.py` from the rocmfp4-llama fork (copied from ggml-org/llama.cpp):

```bash
cd ~/rocmfp4-llama
PYTHONPATH=~/rocmfp4-llama/gguf-py/:$PYTHONPATH \
LD_LIBRARY_PATH=~/rocmfp4-llama/build-strix-rocmfp4/bin:$LD_LIBRARY_PATH \
python3 convert_hf_to_gguf.py \
  /path/to/hf-model-dir \
  --outtype bf16 \
  -o /path/to/model-bf16.gguf
```

This converts any supported HF model (Qwen3.6, Gemma 3/4, etc.) to BF16 GGUF format — the source for ROCmFP4 quantization.

### Quantizing BF16 → ROCmFP4
```bash
llama-quantize source-bf16.gguf output-rocmfp4.gguf Q4_0_ROCMFP4_STRIX_LEAN
```

Available quantize types:
- `Q4_0_ROCMFP4` — dual-scale 4.50 BPW (pure)
- `Q4_0_ROCMFP4_FAST` — single-scale 4.25 BPW (pure)
- `Q4_0_ROCMFP4_STRIX` — FAST dense + Q6_K embeddings + dual-scale K/V (4.49 BPW)
- `Q4_0_ROCMFP4_STRIX_LEAN` — FAST dense + Q5_K embeddings + dual-scale K/V (4.38 BPW, recommended for 64–128GB)

**Critical**: Source must be F16/BF16 GGUF. Do NOT requantize from an already-quantized GGUF — quality loss is significant.

## Key Commands
- Build and push to GHCR: `docker buildx bake release`
- Build locally only: `docker buildx bake` (or `docker buildx bake local`)
- Run persistent server: `./llama.sh` (then `./llama.sh` again for another instance)
- Quantize models: See Quantization Workflow section above

## Code Style
When modifying llama.cpp source (not just this infra repo):
- Follow C99 style for C code, modern C++ idioms for `.cpp` files
- Use clang-format with the existing formatting configuration if present
- Keep commits focused and messages descriptive but concise

## Docker / CI Notes
- Target `release` pushes to GHCR; target `local` loads into local Docker
- The Dockerfile clones `mtp-rocmfp4-strix` (or `nemotron-mtp-rocmfp4-strix` for Nemotron-specific builds) and builds with `JOBS=32`
- `.tmp/` is ignored — local temp files go here, do not commit them

## AMD ROCm / Strix Halo Specifics
- **GPU**: AMD gfx1151 (Strix Halo) — `HSA_OVERRIDE_GFX_VERSION=11.5.1` required for ROCm 7.x compatibility
- **Unified memory**: 128GB shared CPU/GPU RAM — mandatory `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (ROCm has no CPU offload path)
- **KV cache**: f16 is default; q8_0 for memory-constrained scenarios. At 128GB, f16 KV cache is generally fine.
- **Thread scaling**: ROCmFP4 dense models run well at 16 threads. MoE models can scale higher if needed.
- **ROCBLAS**: `ROCBLAS_USE_HIPBLASLT` can be toggled 0/1; benchmarks suggest `=1` works well on ROCm 7.13+
- **Performance env vars**: Set in `llama.sh` — `GGML_HIP_FORCE_RS_GPU=1`, `GGML_HIP_FORCE_KV_GPU=1`, `GGML_HIP_ALLOC_GRAPH_RESERVE=2048`, `HSA_FORCE_FINE_GRAIN_PCIE=1`
- **SDMA**: Disabled by default (`HSA_ENABLE_SDMA=0`) to avoid transfer faults; enable if large prompt processing is a bottleneck
- **Flash attention**: Enabled (`flash-attn = true`) — uses rocWMMA kernels on Strix Halo

## Model Recommendations for 128GB UMA

| Model | Type | ROCmFP4 Size | KV Cache (256K, f16) | Total | Notes |
|-------|------|-------------|---------------------|-------|-------|
| Gemma 4 26B A4B MoE | MoE (25.2B/3.8B active) | ~16 GB | ~16 GB | ~32 GB | ~4B active speed, excellent coding |
| Qwen3-Coder 30B A3B MoE | MoE (31B/3B active) | ~18 GB | ~19 GB | ~37 GB | Purpose-built for coding |
| Gemma 4 12B Unified | Dense | ~7 GB | ~10 GB | ~17 GB | Fast, good for multi-model |
| Gemma 4 31B Dense | Dense | ~23 GB | ~16 GB | ~39 GB | High quality, moderate speed |
| Qwen3.6-35B-A3B MoE | MoE (35B/3B active) | ~17 GB | ~17 GB | ~34 GB | MTP support with draft heads |

For coding workloads, prioritize Gemma 4 26B A4B MoE or Qwen3-Coder 30B A3B MoE — both fit comfortably with 256K context and room for concurrent sessions.

## Git Workflow
- Branch from `main`, open PRs against `main`.
- Keep commit messages imperative and short (~72 chars).
