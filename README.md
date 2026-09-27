# llama.cpp Docker Build for AMD Strix Halo (gfx1151)

Docker-based deployment of llama.cpp with ROCm FP4 support for AMD Strix Halo (gfx1151).

## Project Structure

```
.
├── docker-bake.hcl      # Docker Buildx bake configuration (_local + release targets)
├── Dockerfile           # Builds local/ai/llama.cpp-gfx1151:latest from the ROCmFPX fork
├── build_llama.cpp.sh   # Build script for llama.cpp inside container
├── llama.sh             # In-container entrypoint: server/cli/completion/quantize/bench dispatch
├── run.sh               # Host runner: creates/removes container, passes GPU + env through
├── litellm.sh           # Optional LiteLLM proxy on :4000 in front of llama-server :8000
├── nginx.conf           # nginx reverse proxy [::]:8000 -> unix:/tmp/llama.sock (+ Lua/Perl fixup)
├── llamacpp_presets.ini # Per-model presets driving --models-preset
├── lib/
│   └── perl/llama.pm    # Request fixup: slot id, conv_id, system-prompt boilerplate cleanup
└── AGENTS.md            # Agent instructions
```

The image source is **not** upstream llama.cpp: it clones the fork https://github.com/aardbeiplantje/ROCmFPX.git (branch `nemotron-mtp-rocmfp4-strix`) at build time. See AGENTS.md for details.

## Build

```bash
# Build Docker image
docker buildx bake -f docker-bake.hcl _local

# Release: push ghcr.io/ai/llama.cpp-gfx1151:latest with provenance + SBOM attestations
docker buildx bake -f docker-bake.hcl release
```

## Run

```bash
# Start server container (detached by default; pass args to run attached)
bash run.sh server

# Interactive CLI chat inside container
MODELS_DIR=/path/to/models bash run.sh cli

# Quantize a GGUF to ROCm FP4 (default type Q4_0_ROCMFP4_STRIX_LEAN)
bash run.sh quantize in.gguf out.gguf [quant-type]

# Benchmarks (no args runs the full suite)
bash run.sh bench

# Tail container logs without starting it
bash run.sh tail
```

Key env vars for `run.sh`: `MODELS_DIR` (mount at `/models`), `LLAMA_PRESETS` (override presets ini), `HF_HOME` / `HF_TOKEN` (model download cache).

```bash
# Optional LiteLLM proxy on :4000 in front of llama-server :8000
bash litellm.sh
```
