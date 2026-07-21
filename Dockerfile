FROM debian:trixie-slim AS base
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
    glslang-tools \
    vulkan-tools \
    libvulkan-dev \
    spirv-headers \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ADD https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx1151-7.13.0.tar.gz /tmp/rocm.tar.gz
# this is on AWS S3 and cached badly
# ADD https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx1151-7.13.0a20260515.tar.gz /tmp/rocm.tar.gz
# hence: curl
RUN --mount=type=cache,target=/var/cache/downloads \
    curl -qsSk -L https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx1151-7.13.0a20260515.tar.gz \
        -z file -o /var/cache/downloads/rocm.tar.gz
RUN --mount=type=cache,target=/var/cache/downloads \
    cp /var/cache/downloads/rocm.tar.gz /rocm.tar.gz

FROM base AS builder

WORKDIR /llama-build
COPY --from=base /rocm.tar.gz /
RUN mkdir -p /opt/rocm \
    && tar -xzf /rocm.tar.gz -C /opt/rocm \
    && rm -f /rocm.tar.gz
RUN git clone --depth=1 --single-branch -b nemotron-mtp-rocmfp4-strix https://github.com/aardbeiplantje/rocmfp4-llama.git
RUN git clone --depth=1 --single-branch -b master https://github.com/ggml-org/llama.cpp.git
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PATH=${ROCM_PATH}/bin:$PATH
ARG W=rocmfp4-llama
COPY build_llama.cpp.sh /llama-build
RUN \
    cd $W && \
    env JOBS=32 bash /llama-build/build_llama.cpp.sh && \
    mv build /llama && \
    rm -rf $W

FROM base AS runtime
WORKDIR /llama
COPY --from=base /rocm.tar.gz /
RUN mkdir -p /opt/rocm \
    && tar -xzf /rocm.tar.gz -C /opt/rocm \
        --wildcards \
        "*/lib/*.so*" \
        "*/lib/rocblas/*" \
        "*/lib/hipblaslt/*" \
    && rm -f /rocm.tar.gz
COPY --from=builder /llama /llama

RUN useradd -N -M -d /llama-server/ -u 1000 llama-runtime
RUN mkdir -p /models      && chown -R llama-runtime:users /models
RUN mkdir -p /hf          && chown -R llama-runtime:users /hf

COPY llamacpp_presets.ini /
COPY llama.sh /

RUN mkdir -p /llama.cpp/slots && chown -R llama-runtime:users /llama.cpp/
USER llama-runtime
WORKDIR /models
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PATH=${ROCM_PATH}/bin:$PATH
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_CACHE=/hf/hub
ENV HF_HOME=/hf
ENTRYPOINT ["/llama.sh"]
