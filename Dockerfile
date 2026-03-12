FROM debian:trixie-slim AS runtime
WORKDIR /tmp
RUN apt update && apt install -y --no-install-recommends \
    ca-certificates \
    unzip \
    libcurl4 \
    libssl3 \
    libgomp1 \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

ARG CACHEBUST=1
RUN apt update && apt upgrade -y \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -N -M -d /llama-server/ -u 1000 llama-runtime
RUN mkdir -p /models      && chown -R llama-runtime:users /models
RUN mkdir -p /hf          && chown -R llama-runtime:users /hf

WORKDIR /llama
USER root
ARG LEMONADE_LLAMACPP_VERSION=b1212
ADD https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/${LEMONADE_LLAMACPP_VERSION}/llama-${LEMONADE_LLAMACPP_VERSION}-ubuntu-rocm-gfx1151-x64.zip llama-rocm.zip
RUN    \
       unzip llama-rocm.zip \
    && chmod +x llama* \
    && chown -R llama-runtime:users . \
    && rm -f llama-rocm.zip

COPY llamacpp_presets.ini llamacpp_presets.ini

RUN mkdir -p /llama.cpp/slots && chown -R llama-runtime:users /llama.cpp/
USER llama-runtime
WORKDIR /models
ENV TMPDIR=/dev/shm
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_CACHE=/hf/hub
ENV HF_HOME=/hf
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1
ENV GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
ENV ROCBLAS_USE_HIPBLASLT=0
ENV HIP_VISIBLE_DEVICES=0
ENV HSA_ENABLE_SDMA=0
ENV GGML_HIP_FORCE_RS_GPU=1
ENV GGML_HIP_FORCE_KV_GPU=1
ENV GGML_HIP_GRAPHS=0
ENV GGML_HIP_ALLOC_GRAPH_RESERVE=2048
ENV ROCM_METADATA_WAIT_TIMEOUT=100
ENV ROCM_ALLOW_INT8_MIXED_PRECISION=1
ENV HIPBLASLT_LOG_MASK=32
ENV HIP_FORCE_DEV_KERNARG=1
ENV GGML_CUDA_FORCE_MMQ=1
ENV LLAMA_LOG_COLORS=1
ENV LLAMA_LOG_TIMESTAMPS=1
ENV LLAMA_LOG_PREFIX=1
ENV HSA_FORCE_FINE_GRAIN_PCIE=1
ENTRYPOINT ["/llama/llama-server", "--models-preset", "/llama/llamacpp_presets.ini", "--models-max", "4", "--models-dir", "/models/", "--models-autoload", "--no-webui", "--host", "::", "--port", "8000"]
