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

## ROCmFP4 fixes in llama.cpp submodule (branch `rocmfp4-strix-upstream-merged`)

The following fixes were applied to make ROCmFP4 work like in `rocmfp4-llama` submodule:

### 1. HIP dequantization kernels (`ggml/src/ggml-hip/dequantize_hip.cuh`)
- **Problem**: ROCmFP4 dequant functions (`dequantize_rocmfp4`, `dequantize_rocmfp4_fast`) were stubs returning zeros
- **Fix**: Added proper implementation using `rocmfp4_ue4m3_to_fp32_half_finite()` and `rocmfp4_decode_i8()` from `rocmfp4_hip_scale.cuh`
- **Added**: `#include "../../rocmfp4/rocmfp4_hip_scale.cuh"`

### 2. MTP tensor loading for qwen35 (`src/models/qwen35.cpp`)
- **Problem**: MTP (NextN) tensors skipped when `load_mtp=false` (default), causing "unused tensor" warnings
- **Fix**: Changed `mtp_flags` from `!ml.load_mtp ? TENSOR_SKIP : 0` to always `0` (required for qwen35 models)

### 3. Build verification
```bash
cd llama.cpp && ROCM_PATH=/opt/rocm bash ../build_llama.cpp.sh
LLAMA_CPP_DIR=llama.cpp/build bash llama.sh completion --model Qwen3.5-4B-ROCMFP4.gguf --prompt "2+2=?" --perf -n 100 --reasoning off --temp 0.7 --top-p 0.9 --jinja
```

### Known limitation
Qwen3.5-4B-ROCMFP4.gguf (qwen35 arch with recurrent/MTP layers) still produces garbled output in llama.cpp due to API divergence from rocmfp4-llama's working qwen35.cpp. The `rocmfp4-llama` submodule (branch `nemotron-mtp-rocmfp4-strix`) has the working implementation.

## HIP Dequantization Implementation Plan (Port from SYCL)

The HIP backend is missing dequantize implementations for several quantization types that exist in the SYCL backend (`ggml/src/ggml-sycl/dequantize.hpp`). This causes build failures or runtime issues when using IQ/MXFP4/NVFP4 quantized models.

### Missing Quantization Types (SYCL → HIP)

| Type | SYCL Function | HIP Status |
|------|---------------|------------|
| IQ2_XXS | `dequantize_iq2_xxs` | Missing |
| IQ2_XS | `dequantize_iq2_xs` | Missing |
| IQ2_S | `dequantize_iq2_s` | Missing |
| IQ3_XXS | `dequantize_iq3_xxs` | Missing |
| IQ3_S | `dequantize_iq3_s` | Missing |
| IQ1_S | `dequantize_iq1_s` | Missing |
| IQ1_M | `dequantize_iq1_m` | Missing |
| IQ4_NL | `dequantize_iq4_nl` | Missing |
| IQ4_XS | `dequantize_iq4_xs` | Missing |
| MXFP4 | `dequantize_mxfp4` | Missing |
| NVFP4 | `dequantize_nvfp4` | Missing |

### Phased Implementation

**Phase 1: IQ2 Family** (IQ2_XXS, IQ2_XS, IQ2_S)
- Port from SYCL `dequantize_iq2_xxs`, `dequantize_iq2_xs`, `dequantize_iq2_s`
- Add constant lookup tables: `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `ksigns_iq2xs`, `kmask_iq2xs` as `__constant__` memory

**Phase 2: IQ3 Family** (IQ3_XXS, IQ3_S)
- Port from SYCL `dequantize_iq3_xxs`, `dequantize_iq3_s`
- Add lookup tables: `iq3xxs_grid`, `iq3s_grid`, reuse `ksigns_iq2xs`, `kmask_iq2xs`

**Phase 3: IQ1 Family** (IQ1_S, IQ1_M)
- Port from SYCL `dequantize_iq1_s`, `dequantize_iq1_m`
- Add lookup table: `iq1s_grid_gpu`

**Phase 4: IQ4 Family** (IQ4_NL, IQ4_XS)
- Port from SYCL `dequantize_iq4_nl`, `dequantize_iq4_xs`
- Add lookup table: `kvalues_iq4nl`

**Phase 5: MXFP4/NVFP4**
- Port from SYCL `dequantize_mxfp4`, `dequantize_nvfp4`
- Add lookup tables: `kvalues_mxfp4`, scale conversion functions

**Phase 6: Block Dequantize Kernels**
- Add `__global__` kernels for each new type (e.g., `dequantize_block_iq2_xxs`, etc.)
- Required by `get_rows_cuda_kq` and `convert.cu`

### Key Files
- `/workdir/llama.cpp.git/llama.cpp/ggml/src/ggml-hip/dequantize_hip.cuh` — Main dequantize functions
- Source: `/workdir/llama.cpp.git/llama.cpp/ggml/src/ggml-sycl/dequantize.hpp` (SYCL reference)

### Implementation Notes
- Adapt SYCL `dfloat2` → HIP `float2`, `__dpct_inline__` → `__device__ __forceinline__`
- SYCL `sycl::fma` → `fmaf`, `sycl::vec` → manual operations
- Add `__constant__` lookup tables at file scope
- Include necessary headers: `ggml.h`, `rocmfp4_hip_scale.cuh`
