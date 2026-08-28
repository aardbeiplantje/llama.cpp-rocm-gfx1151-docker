# AGENTS.md

## Project Structure

This is a Docker-based llama.cpp deployment for AMD Strix Halo (gfx1151) with ROCm FP4 support.

**Three submodules:**
- `llama.cpp` — fork on `rocmfp4-strix-upstream-merged` branch
- `minijinja` — upstream
- `rocmfp4-llama` — source branch `nemotron-mtp-rocmfp4-strix` with ROCmFP4 patches

## Key Ideas

1. **Docker** — Builds `local/ai/llama.cpp-gfx1151:latest` via `docker buildx bake -f docker-bake.hcl _local`
2. **llama.cpp** — Fork with ROCm FP4 support for AMD Strix Halo (gfx1151)
3. **rocmfp4-llama** — Source of ROCm FP4 patches; merge target for llama.cpp submodule

## Key Commands

```bash
# Build Docker image
docker buildx bake -f docker-bake.hcl _local

# Build llama.cpp inside container
cd llama.cpp && ROCM_PATH=/opt/rocm bash ../build_llama.cpp.sh

# Run completion
LLAMA_CPP_DIR=llama.cpp/build bash llama.sh completion --model Qwen3.5-4B-ROCMFP4.gguf --prompt "2+2=?" --perf -n 100 --reasoning off --temp 0.7 --top-p 0.9 --jinja
```

## Current Status

- ✅ Qwen3-0.6B-Q8_0.gguf works
- ❌ Qwen3.5-4B-ROCMFP4.gguf produces garbled output (qwen35 arch with recurrent/MTP layers)
- See `LLAMA_ROCMFP4.md` for detailed status and investigation history