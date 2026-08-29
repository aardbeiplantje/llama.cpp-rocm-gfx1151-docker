# llama.cpp Docker Build for AMD Strix Halo (gfx1151)

Docker-based deployment of llama.cpp with ROCm FP4 support for AMD Strix Halo (gfx1151).

## Project Structure

```
.
├── docker-bake.hcl      # Docker Buildx bake configuration
├── Dockerfile           # Builds local/ai/llama.cpp-gfx1151:latest
├── build_llama.cpp.sh   # Build script for llama.cpp inside container
├── llama.sh             # Wrapper for running llama.cpp
├── minijinja/           # Minijinja submodule (for Jinja2 template support)
└── AGENTS.md            # Agent instructions
```

## Build

```bash
# Build Docker image
docker buildx bake -f docker-bake.hcl _local
```

## Run

```bash
# Run completion
LLAMA_CPP_DIR=llama.cpp/build bash llama.sh completion \
  --model Qwen3.5-4B-ROCMFP4.gguf \
  --prompt "2+2=?" \
  --perf -n 100 \
  --reasoning off \
  --temp 0.7 \
  --top-p 0.9 \
  --jinja
```

## Submodules

- `minijinja` — Upstream minijinja for Jinja2 template support