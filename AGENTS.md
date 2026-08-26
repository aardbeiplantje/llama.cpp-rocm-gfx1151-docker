# AGENTS.md

## Repo structure

This is a Docker-based llama.cpp deployment for AMD Strix Halo (gfx1151) with ROCm FP4 support.
Three submodules: `llama.cpp` (fork on `rocmfp4-strix` branch), `minijinja` (upstream), `rocmfp4-llama` (source branch with ROCmFP4 patches).

## Building the Docker image

```bash
docker buildx bake -f docker-bake.hcl _local
```

Produces `local/ai/llama.cpp-gfx1151:latest`. The ROCm tarball is fetched from an S3 bucket and cached under `/var/cache/downloads` during build.

## Running

```bash
./run.sh server          # detached (default)
./run.sh cli -m /models/model.gguf   # interactive TTY
./run.sh bench           # benchmark suite
./run.sh quantize in.gguf out.gguf   # BF16/FP16 -> ROCmFP4
./run.sh tail            # follow logs
./run.sh bash            # interactive shell in container
```

Mount models via `MODELS_DIR=/path/to/models ./run.sh`. HF cache via `HF_HOME`.

## llama.sh subcommands (inside container)

- `server` (default) — starts llama.cpp server + nginx on unix socket `/tmp/llama.sock`
- `cli` — interactive CLI chat
- `quantize` — quantize GGUF (default output type: `Q4_0_ROCMFP4_STRIX_LEAN`)
- `bench` — comprehensive benchmark suite

## Presets

`/llamacpp_presets.ini` — `[<ModelName>]` sections override defaults. Model presets are matched by filename prefix. Add new model presets by copying the `[Qwen3.6-35B-A3B-ROCMFP4]` block and adjusting `ctx-size`/other params.

## Key env vars (host or container)

`HSA_OVERRIDE_GFX_VERSION=11.5.1`, `GGML_HIP_FORCE_KV_GPU=1`, `GGML_HIP_ALLOC_GRAPH_RESERVE=2048`, `HSA_ENABLE_SDMA=0`.

## Perl XS module (nginx integration)

`Llama/` — Perl XS bindings to llama.cpp, built into the Docker image. Tests run during build (`make test`). The module is loaded by nginx via `perl_require llama.pm` in `nginx.conf`.

## ROCmFP4 development

The ROCmFP4 patches live in the `rocmfp4-llama` submodule (branch `nemotron-mtp-rocmfp4-strix`). Merging to `llama.cpp` submodule branch `rocmfp4-strix`. See `TODO.md` for migration status. See `PLAN.md` for the 13-week MFMA/KV-cache optimization roadmap.

## Build script flags

`build_llama.cpp.sh` — key CMake flags: `-DGGML_USE_ROCMFP4=ON`, `-DAMDGPU_TARGETS=gfx1151`, `-DGGML_HIP_UMA=ON`, `-DGGML_HIP_FORCE_KV_GPU=ON`.
