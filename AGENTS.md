# AGENTS.md

## Project Structure

Docker-based llama.cpp deployment for AMD Strix Halo (gfx1151) with ROCm FP4 support.

**Not upstream llama.cpp.** The image builds a fork: https://github.com/aardbeiplantje/ROCmFPX.git, branch `nemotron-mtp-rocmfp4-strix`, cloned at build time inside the Dockerfile and renamed to `llama.cpp`. The fork is what adds ROCm FP4 quantization support; bumping the source means changing that clone line (the `LEMONADE_LLAMACPP_VERSION` bake var is currently unused).

The ROCm toolchain comes from the AMD "Therock" nightly tarball for gfx1151, downloaded via curl into a build cache mount (repo.amd.com copy is cached badly — see Dockerfile comments).

## Architecture

- **Build stages**: `base` (debian trixie + build tools + the ROCm tarball) → `builder` (cmake/ninja build of the fork, JOBS=32) and `rocm-hip-build`; `rocm-runtime` extracts only the ROCm runtime libraries; `runtime` is a fresh `debian:trixie-slim` with those, `/llama`, `/opt/rocm-hip` and runtime packages (no compilers, no tarball layer). A build step fails the image when `ldd` finds a missing library for llama-server or ROCm::Inference: add the package to the runtime `apt-get install` then. See `build_llama.cpp.sh` for all cmake flags (rocWMMA flash attention, FP4, graph opt, RS/KV GPU offload).
- **ROCm::Inference** (`rocm-hip-build` stage): builds rocm-hip from the `rocm-hip-src` build context (`ROCM_HIP_CONTEXT`, default `../rocm-hip`; private, hence not cloned) into `/opt/rocm-hip`: `make install` (libinference, kernels embedded), Minijinja (perl-minijinja-xs + the Rust minijinja C library, Debian's rustc/cargo), ROCm::Inference, Web::Route, `bin/rocm-inference-server`. The context and the stage must have different names: a named context replaces a stage of the same name.
- **Server runtime**: nginx listens on `[::]:8000` and proxies to `unix:/tmp/llama.sock`; `/rocm/` goes to `unix:/tmp/rocm-inference.sock` (rocm-inference-server, prefix stripped, `proxy_buffering off`). A Lua+Perl request fixup (`lib/lua/fixup.lua` -> `lib/perl/llama.pm`, both locations) rewrites chat requests before proxying: injects `id_slot`/`conv_id`, converts repeated system messages to user role, strips boilerplate (model ID, env block, MCP/skills/AGENTS instructions) into "Understood." pairs. POST bodies are logged to `/tmp/request-logs/`.
- **Entrypoint**: `/llama.sh` dispatches subcommands; default is `server`, which starts llama-server then execs nginx foreground.
- **Model presets**: `/llamacpp_presets.ini` drives `--models-preset`: global `[*]` defaults plus per-model `.gguf` sections. Server autoloads up to 4 models from `/models`, parallel slots.
- **Container layout**: `/models` (mounted from `$MODELS_DIR`), `/hf` (HF cache), named volume `llama.cpp-data` at `/llama.cpp` for slot save paths and lookup caches.

## Key Commands

```bash
# Build local image (local/ai/llama.cpp-gfx1151:latest)
docker buildx bake -f docker-bake.hcl _local

# Release: push ghcr.io/ai/llama.cpp-gfx1151:latest with provenance + SBOM attestations
docker buildx bake -f docker-bake.hcl release

# Run container — arg selects mode/name; server runs detached by default, add args for -it
bash run.sh [server|cli|bench|quantize|convert|bash|tail] [args...]

# LiteLLM proxy on :4000 in front of llama-server :8000 (model list inside)
bash litellm.sh
```

### run.sh notable env vars

- `MODELS_DIR` — host dir mounted at `/models`; `LLAMA_PRESETS` — host file mounted over `/llamacpp_presets.ini` (the file llama-server reads; `PRESET_FILE` / `L_INI` select another)
- `HF_HOME`, `HF_TOKEN` — HF model download cache (`/hf`)
- `HSA_OVERRIDE_GFX_VERSION=11.5.1` and ROCm tuning vars (`GGML_HIP_FORCE_RS_GPU`, `GGML_HIP_FORCE_KV_GPU`, `GGML_HIP_ALLOC_GRAPH_RESERVE=2048`, `HSA_ENABLE_SDMA=0`) are exported by default
- GPU access: `--device /dev/kfd --device /dev/dri`, groups 109/986/992, memlock unlimited

### llama.sh subcommands (in-container entrypoint)

- `server` (default) — llama-server + nginx; preset-driven autoload, slots with idle-cache; with `ROCM_MODEL` set also rocm-inference-server (at `/rocm/v1/`)
- `rocm-server [model.gguf]` — rocm-inference-server + nginx only (`ROCM_MODEL`, `ROCM_CTX`, `ROCM_DRAFT`, `ROCM_STORE`, `ROCM_ARGS`); started without `HIP_LAUNCH_BLOCKING` / `AMD_SERIALIZE_KERNEL`
- `cli` / `completion` — interactive chat (jinja, flash-attn, q8_0/turbo4 KV caches)
- `quantize in.gguf out.gguf [type]` — default quant type `Q4_0_ROCMFP4_STRIX_LEAN` (~4.38 BPW)
- `bench [args]` — no args runs the full suite (pp sizes, tg batches, mixed workloads, long-context stress); bench mode forces `ROCBLAS_USE_HIPBLASLT=1 HSA_ENABLE_SDMA=1 GGML_HIP_GRAPHS=1`

## Gotchas

- `HSA_FORCE_FINE_GRAIN_PCIE=1` can cause GPU lockups on some firmware — off unless set (run.sh passes it only when set)
- `HIP_LAUNCH_BLOCKING=1` / `AMD_SERIALIZE_KERNEL=1` serialize every kernel launch: off unless set (were exported by llama.sh since 7610baa); set them again if GPU hangs come back. rocm-inference-server never gets them
- ROCm tarball URL is pinned to a specific nightly date in the Dockerfile; bump deliberately and test the build
- The Perl fixup only acts when request header `X-LLamaCPP-Id-slot` is numeric; other requests pass through unmodified. It moves the boilerplate out of a first system (or developer) message only; a conversation that starts with a user message keeps it
- Both proxied locations use `proxy_http_version 1.1` and `proxy_buffering off`, so streamed replies (SSE) reach clients as they come
