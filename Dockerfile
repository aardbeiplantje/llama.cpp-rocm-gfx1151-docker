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
    perl \
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
ARG CACHEBUST=1
RUN git clone --depth=1 --single-branch -b nemotron-mtp-rocmfp4-strix https://github.com/aardbeiplantje/rocmfp4-llama.git rocmfp4-llama.cpp.git
RUN git clone --depth=1 --single-branch -b master https://github.com/ggml-org/llama.cpp.git llama.cpp.git
RUN git clone --depth=1 --single-branch -b main https://github.com/aardbeiplantje/ROCmFPX.git ROCmFPX.git
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PATH=${ROCM_PATH}/bin:$PATH
ARG W=ROCmFPX.git
COPY build_llama.cpp.sh /llama-build
RUN \
    mv $W llama.cpp && \
    cd llama.cpp && \
    env JOBS=32 bash /llama-build/build_llama.cpp.sh && \
    mv build /llama && \
    rm -rf $W

FROM base AS perl-builder
WORKDIR /llama-perl
COPY --from=builder /llama-build/llama.cpp /llama.cpp
COPY --from=builder /llama /llama
COPY Llama /llama-perl/Llama
ARG CACHEBUST=1
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
RUN \
    cd /llama-perl/Llama && \
    ROCM_PATH=/opt/rocm \
    LLAMA_SRC=/llama.cpp \
    LLAMA_BUILD=/llama \
    perl Makefile.PL && \
    make test && \
    make install DESTDIR=/app/lib/perl

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

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    libnginx-mod-http-lua \
    libnginx-mod-http-perl \
    libnginx-mod-http-auth-pam \
    libjson-xs-perl \
    libsys-mmap-perl \
    ca-certificates \
    strace \
    lsof \
    lua-cjson \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log
#COPY --from=perl-builder /app/lib/perl/usr/local/lib /usr/local/lib 
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:/llama/bin
#RUN perl -MLlama -we 'print "OK\n"'
COPY nginx.conf /nginx.conf

RUN useradd -N -M -d /llama-server/ -u 1000 llama-runtime
RUN mkdir -p /models      && chown -R llama-runtime:users /models
RUN mkdir -p /hf          && chown -R llama-runtime:users /hf
RUN mkdir -p /var/lib/nginx/body  && chown llama-runtime /var/lib/nginx/body
RUN mkdir -p /var/lib/nginx/proxy && chown llama-runtime /var/lib/nginx/proxy
RUN mkdir -p /var/lib/nginx/fastcgi && chown llama-runtime /var/lib/nginx/fastcgi
RUN mkdir -p /var/lib/nginx/uwsgi && chown llama-runtime /var/lib/nginx/uwsgi
RUN mkdir -p /var/lib/nginx/scgi && chown llama-runtime /var/lib/nginx/scgi

COPY llamacpp_presets.ini /
COPY llama.sh /
WORKDIR /app
COPY lib/ /app/lib/

RUN mkdir -p /llama.cpp/slots && chown -R llama-runtime:users /llama.cpp/
USER llama-runtime
RUN nginx -t -c /nginx.conf
WORKDIR /models
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PATH=${ROCM_PATH}/bin:$PATH
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_CACHE=/hf/hub
ENV HF_HOME=/hf
ENTRYPOINT ["/llama.sh"]
