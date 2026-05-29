# AGENTS.md - Guidelines for AI Coding Assistants

## Project Overview
This repository contains build and test infrastructure for [llama.cpp](https://github.com/ggerganov/llama.cpp), a C++ implementation of Large Language Models.

## Build System
- **`docker-bake.hcl`**: Docker Compose bake file defining build targets via `docker buildx bake`. Used to build llama.cpp container images with various GPU/CPU backends (CUDA, ROCm, Metal, etc.).
- **`llamacpp_presets.ini`**: Preset configuration for llama.cpp builds.
- **`Dockerfile`**: Base Docker image definition.

## Key Tools & Commands
- Build: `docker buildx bake <target>`
- Run tests inside containers as defined in `docker-bake.hcl`
- Shell scripts (`run_llama.sh`) may be used for quick invocation

## Code Style
When modifying llama.cpp source (not just this infra repo):
- Follow C99 style for C code, modern C++ idioms for `.cpp` files
- Use clang-format with the existing formatting configuration if present
- Keep commits focused and messages descriptive but concise

## Docker / CI Notes
- Targets in `docker-bake.hcl` often define GPU-specific builds (cuda, rocm, metal). Adjust targets accordingly.
- Tests may run inside the built images; check each target's command section.
- `.tmp/` is ignored — local temp files go here, do not commit them.

## Git Workflow
- Branch from `main`, open PRs against `main`.
- Keep commit messages imperative and short (~72 chars).
