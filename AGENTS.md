# AGENTS.md

## Project Structure

This is a Docker-based llama.cpp deployment for AMD Strix Halo (gfx1151) with ROCm FP4 support.

## Key Ideas

1. **Docker** — Builds `local/ai/llama.cpp-gfx1151:latest` via `docker buildx bake -f docker-bake.hcl _local`

## Key Commands

```bash
# Build Docker image
docker buildx bake -f docker-bake.hcl _local
```