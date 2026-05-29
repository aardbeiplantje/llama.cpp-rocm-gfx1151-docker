#!/bin/bash
HF_HOME=${HF_HOME:-$(pwd)/huggingface}
MODELS_DIR=${MODELS_DIR:-$(pwd)/models}
DOCKER_IMAGE=${DOCKER_IMAGE:-local/ai/llama.cpp-gfx1151:latest}
CONTAINER_NAME=${CONTAINER_NAME:-llama.cpp}
CONTAINER_ARGS=${CONTAINER_ARGS:-}
extra_args=""
if [ ! -z "$MODELS_PRESETS" ]; then
    extra_args="$extra_args -v $MODELS_PRESETS:/llama/llamacpp_presets.ini:ro"
fi
LLAMA_DATA=${LLAMA_DATA:-llama.cpp-data-$LOGNAME}
exec docker run --rm \
    --pull=always \
    --name ${CONTAINER_NAME} \
    -p 8000:8000 \
    -v $HF_HOME:/hf:ro \
    -v $MODELS_DIR:/models:ro \
    -v $LLAMA_DATA:/llama.cpp:rw \
    $extra_args \
    $CONTAINER_ARGS \
    --memory=128g \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add=video \
    --ipc=host \
    --shm-size=4g \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --group-add=109 \
    --group-add 992 \
        $DOCKER_IMAGE \
        $@

