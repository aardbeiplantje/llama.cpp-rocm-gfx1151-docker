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
    strace \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -N -M -d /llama-server/ -u 1000 llama-runtime
RUN mkdir -p /models      && chown -R llama-runtime:users /models
RUN mkdir -p /hf          && chown -R llama-runtime:users /hf

WORKDIR /llama
RUN git clone --depth=1 --single-branch -b nemotron-mtp-rocmfp4-strix https://github.com/aardbeiplantje/rocmfp4-llama.git
ADD https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx1151-7.13.0.tar.gz /tmp/rocm.tar.gz
RUN mkdir -p /opt/rocm \
     && cd /opt/rocm \
     && cat /tmp/rocm.tar.gz|tar xzf -
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PATH=${ROCM_PATH}/bin:$PATH
COPY build_llama.cpp.sh /llama
RUN \
    cd rocmfp4-llama && \
    env JOBS=32 bash /llama/build_llama.cpp.sh && \
    mv build / && \
    rm -rf /llama && mv /build /llama && \
    rm -rf rocmfp4-llama


COPY llamacpp_presets.ini llamacpp_presets.ini
COPY llama.sh /

RUN mkdir -p /llama.cpp/slots && chown -R llama-runtime:users /llama.cpp/
USER llama-runtime
WORKDIR /models
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_CACHE=/hf/hub
ENV HF_HOME=/hf
ENTRYPOINT ["/llama.sh"]
