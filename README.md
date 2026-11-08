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
│   ├── lua/fixup.lua    # access_by_lua_file: POST body -> /fixup (llama.pm) -> proxied request
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

The image's `/version` names the commits it was built from: this project's and rocm-hip's (with its
submodule), and the build time: `docker exec llama cat /version`. The bake file passes this project's `.git` as
the named context `docker-git`; a plain `docker build` needs `--build-context docker-git=.git`.

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

## ROCm::Inference next to llama.cpp

The image also carries [rocm-hip](https://github.com/CowboyTim/rocm-hip)'s engine: `libinference` with the Perl
modules ROCm::Inference, Web::Route and Minijinja (via the `perl-minijinja-xs` XS extension), and `rocm-inference-server`, an OpenAI-compatible server for
one model (streamed `reasoning_content` / `content`, tool calls). It listens on `/tmp/rocm-inference.sock`;
nginx serves it at **`/rocm/v1/...`** with the same request fixups as llama-server's `/v1/...`.

The `perl-minijinja-xs` Perl XS module provides low-level bindings to the Rust Minijinja C library (built via `cargo`), enabling fast template rendering from within Perl â it's the XS component that links Perl into the ROCm::Inference pipeline.

```bash
# next to llama-server: llama.cpp at http://host:8000/v1, ROCm::Inference at http://host:8000/rocm/v1 with every
# model in /models (GET /rocm/v1/models), each loaded when a request's "model" names it
MODELS_DIR=/path/to/models bash run.sh server

# one model, loaded at start
ROCM_MODEL=Qwen3.5-4B-ROCMFP4.gguf MODELS_DIR=/path/to/models bash run.sh server

# ROCm::Inference only (a model, or none: all of /models)
MODELS_DIR=/path/to/models bash run.sh rocm-server Qwen3.8-Flash-Next-...-00001-of-00003.gguf
```

Without `ROCM_MODEL` the server takes the models of `ROCM_MODELS_DIR` (default `/models`): nothing is loaded until a
request names one, and it unloads the least recently used idle ones when memory runs short (what llama-server holds
counts); `POST /rocm/models/load` / `/rocm/models/unload` with `{"model": id}` as llama.cpp's router.
`ROCM_INFERENCE=0` leaves it out of `run.sh server`. `ROCM_MODEL` (a path or a file in `/models`), `ROCM_CTX` (default: the model's trained context from the GGUF, within the KV budget), `ROCM_DRAFT` (an MTP draft head),
`ROCM_STORE_DIR` (a host directory for the KV block store: mounted at `/kvstore` and passed as `ROCM_STORE=/kvstore`;
it survives restarts, unlike `/dev/shm`), `ROCM_STORE` (a store directory inside the container, instead) and `ROCM_ARGS`
(more server options) are passed through by `run.sh`. The build clones rocm-hip's `main` (with its perl-minijinja-xs submodule); bump
`CACHEBUST` to pick up new commits.

Key env vars for `run.sh`: `MODELS_DIR` (mount at `/models`), `LLAMA_PRESETS` (override presets ini), `HF_HOME` / `HF_TOKEN` (model download cache); `HSA_FORCE_FINE_GRAIN_PCIE`, `HIP_LAUNCH_BLOCKING` and `AMD_SERIALIZE_KERNEL` are off unless set.

```bash
# Optional LiteLLM proxy on :4000 in front of llama-server :8000
bash litellm.sh
```
