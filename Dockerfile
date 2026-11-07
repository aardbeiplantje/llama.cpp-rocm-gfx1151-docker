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

# ADD https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx1151-10.1.0.tar.gz /tmp/rocm.tar.gz
# this is on AWS S3 and cached badly
# ADD https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx1151-7.14.0a20260612.tar.gz /tmp/rocm.tar.gz
# hence: curl
RUN --mount=type=cache,target=/var/cache/downloads \
    curl -qsSk -L https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-10.1.0.tar.gz \
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
#RUN git clone --depth=1 --single-branch -b nemotron-mtp-rocmfp4-strix https://github.com/aardbeiplantje/rocmfp4-llama.git rocmfp4-llama.cpp.git
#RUN git clone --depth=1 --single-branch -b master https://github.com/ggml-org/llama.cpp.git llama.cpp.git
# llama.cpp (ROCmFPX) is not built for now: BUILD_LLAMA=1 builds it (the image then has llama-server)
ARG BUILD_LLAMA=0
RUN if [ "$BUILD_LLAMA" = 1 ]; then git clone --depth=1 --single-branch -b nemotron-mtp-rocmfp4-strix https://github.com/aardbeiplantje/ROCmFPX.git ROCmFPX.git; fi
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
ENV PATH=${ROCM_PATH}/bin:$PATH
ARG W=ROCmFPX.git
COPY build_llama.cpp.sh /llama-build
RUN \
    if [ "$BUILD_LLAMA" = 1 ]; then \
        mv $W llama.cpp && \
        cd llama.cpp && \
        env JOBS=32 bash /llama-build/build_llama.cpp.sh && \
        mv build /llama; \
    else \
        echo "llama.cpp: not built (BUILD_LLAMA=0)" && mkdir -p /llama/bin; \
    fi

# ROCm::Inference (https://github.com/aardbeiplantje/purhie):
# libinference (kernels embedded), the Perl modules ROCm::Inference, Web::Route and Minijinja (with the Rust
# minijinja C library), and the OpenAI-compatible rocm-inference-server, all under /opt/rocm-hip
FROM base AS rocm-hip-build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libperl-dev \
    rustc \
    cargo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/rocm \
    && tar -xzf /rocm.tar.gz -C /opt/rocm
ENV ROCM_PATH=/opt/rocm
# with the perl-minijinja-xs submodule (Minijinja's source); CACHEBUST makes a rebuild take new commits
ARG CACHEBUST=1
RUN git clone --depth=1 --single-branch -b main --recurse-submodules --shallow-submodules \
        https://github.com/aardbeiplantje/purhie.git /src/rocm-hip
WORKDIR /src/rocm-hip
# Minijinja first: its C library goes into /opt/rocm-hip/lib, which the install below creates as well
RUN mkdir -p /opt/rocm-hip/lib
RUN cd perl-minijinja-xs \
    && bash scripts/setup-minijinja.sh \
    && cp minijinja/target/release/libminijinja_cabi.so /opt/rocm-hip/lib/ \
    && MINIJINJA_RPATH=/opt/rocm-hip/lib perl Makefile.PL INSTALL_BASE=/opt/rocm-hip/perl5 \
    && make && make install
RUN make -j32 install PREFIX=/opt/rocm-hip
RUN cd perl/ROCm-Inference \
    && ROCM_INFERENCE_PREFIX=/opt/rocm-hip perl Makefile.PL INSTALL_BASE=/opt/rocm-hip/perl5 \
    && make && make install
RUN cd perl/Web-Route \
    && perl Makefile.PL INSTALL_BASE=/opt/rocm-hip/perl5 \
    && make && make install

# The ROCm runtime libraries only: the runtime image gets these, not the tarball (a lower layer keeps a
# file even when a later one deletes it) or the build tools of base
FROM base AS rocm-runtime
RUN mkdir -p /opt/rocm \
    && tar -xzf /rocm.tar.gz -C /opt/rocm \
        --wildcards \
        "*/lib/*.so*" \
        "*/lib/rocblas/*" \
        "*/lib/hipblaslt/*"

FROM debian:trixie-slim AS runtime
WORKDIR /llama
# nginx and the runtime's packages: their own layer, before the copies from the build stages (a rocm-hip or llama
# change does not reinstall them); --build-arg APT_CACHEBUST=<new value> to refresh them
ARG APT_CACHEBUST=1
# (the echo makes the value part of this RUN's cache key: a new value runs apt again)
RUN echo "apt: $APT_CACHEBUST" && apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 \
    libssl3 \
    libgomp1 \
    libatomic1 \
    perl \
    curl \
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
COPY --from=rocm-runtime /opt/rocm /opt/rocm
COPY --from=builder /llama /llama
COPY --from=rocm-hip-build /opt/rocm-hip /opt/rocm-hip

# every shared library llama.cpp and ROCm::Inference need is here (the image no longer inherits base's packages)
RUN ! LD_LIBRARY_PATH=/opt/rocm/lib:/llama/bin ldd /llama/bin/llama-server /llama/bin/*.so* \
        /opt/rocm-hip/lib/*.so* $(find /opt/rocm-hip/perl5 -name '*.so') 2>/dev/null | grep 'not found'
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:/llama/bin
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
ENV PERL5LIB=/opt/rocm-hip/perl5/lib/perl5
ENTRYPOINT ["/llama.sh"]
