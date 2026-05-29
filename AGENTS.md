# AGENTS.md - Guidelines for AI Coding Assistants

## Project Overview
This repository contains build and deployment infrastructure for [llama.cpp](https://github.com/ggerganov/llama.cpp) on AMD Strix Halo (gfx1151) via ROCm. Built using the [lemonade-sdk fork](https://github.com/lemonade-sdk/llamacpp-rocm) to produce container images serving MoE models like Qwen3.6-35B-A3B.

## Build System
- **`docker-bake.hcl`**: Docker Compose bake file defining build targets via `docker buildx bake`. Pushes to GHCR (`ghcr.io/ai/llama.cpp-gfx1151:latest`) and internal registry by default. Version variable controls the lemonade-sdk release tag (default b1282).
- **`Dockerfile`**: Base Debian trixie-slim image with ROCm-enabled llama.cpp server, entrypoint configured for auto-loading models from `/models/`.
- **`llamacpp_presets.ini`**: Per-model preset configuration — `[ * ]` section holds defaults (threads=4, q8_0 KV cache, SWA), individual model sections override as needed. Each MoE model that should scale beyond 4 threads needs its own `threads` override.
- **`llama.sh`**: Persistent server wrapper (`--detach`, host networking, binds presets from repo). Sets `MODELS_DIR` env var to the models directory.

## Key Tools & Commands
- Build: `docker buildx bake release` (pushes to registry) or use local-only output
- Run locally: `./llama.sh` then run again for persistent server
- `llama.sh` — persistent server wrapper (sets up host networking, webui, binds preset file from repo dir)

## Code Style
When modifying llama.cpp source (not just this infra repo):
- Follow C99 style for C code, modern C++ idioms for `.cpp` files
- Use clang-format with the existing formatting configuration if present
- Keep commits focused and messages descriptive but concise

## Docker / CI Notes
- Targets in `docker-bake.hcl` often define GPU-specific builds (cuda, rocm, metal). Adjust targets accordingly.
- Tests may run inside the built images; check each target's command section.
- `.tmp/` is ignored — local temp files go here, do not commit them.

## AMD ROCm / Strix Halo Specifics
- Image tagged as `gfx1151` — HSA_OVERRIDE_GFX_VERSION=11.5.1 required for older ROCm versions on gfx1151 hardware.
- Unified memory mode is mandatory (`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`) since ROCm has no CPU offload path.
- KV cache quantization (q6_k or q8_0) matters — at 64GB shared RAM, every MB counts for concurrent contexts.
- `ROCBLAS_USE_HIPBLASLT` can be toggled between 0/1 to test GEMM throughput on your ROCm version.

## Git Workflow
- Branch from `main`, open PRs against `main`.
- Keep commit messages imperative and short (~72 chars).
