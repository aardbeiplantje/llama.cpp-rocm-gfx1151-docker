FROM debian:trixie-slim AS runtime
WORKDIR /tmp
RUN apt update && apt install -y --no-install-recommends \
    ca-certificates \
    unzip \
    libcurl4 \
    libssl3 \
    libgomp1 \
    libatomic1 \
    make \
    gcc \
    g++ \
    git \
    cmake \
    ninja-build \
    build-essential \
    clang \
    pkg-config \
    glslc \
    vulkan-tools \
    libvulkan-dev \
    spirv-headers \
    ccache \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -N -M -d /llama-server/ -u 1000 llama-runtime
RUN mkdir -p /models      && chown -R llama-runtime:users /models
RUN mkdir -p /hf          && chown -R llama-runtime:users /hf

WORKDIR /llama
RUN git clone --depth=1 https://github.com/aardbeiplantje/rocmfp4-llama.git
ADD https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx1151-7.13.0.tar.gz /tmp/rocm.tar.gz
RUN mkdir -p /opt/rocm \
     && cd /opt/rocm \
     && cat /tmp/rocm.tar.gz|tar xzf -
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PATH=${ROCM_PATH}/bin:$PATH
RUN \
    cd rocmfp4-llama && \
    git checkout mtp-rocmfp4-strix && \
    env JOBS=32 scripts/build-strix-rocmfp4-mtp.sh && \
    mv build-strix-rocmfp4 / && \
    rm -rf /llama && mv /build-strix-rocmfp4 /llama && \
    rm -rf rocmfp4-llama


COPY llamacpp_presets.ini llamacpp_presets.ini

RUN mkdir -p /llama.cpp/slots && chown -R llama-runtime:users /llama.cpp/
USER llama-runtime
WORKDIR /models
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
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/llama/bin
ENTRYPOINT ["/llama/bin/llama-server", "--models-preset", "/llama/llamacpp_presets.ini", "--models-max", "4", "--models-dir", "/models/", "--models-autoload", "--no-webui", "--host", "::", "--port", "8000"]
