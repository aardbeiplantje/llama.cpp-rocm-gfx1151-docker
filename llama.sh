#!/bin/bash
docker stop llama || true
LLAMA_DOCKER_IMAGE=${LLAMA_DOCKER_IMAGE:-"registry.aardbeiplantje.link/ai/llama.cpp-gfx1151:latest"}
HERE="$BASH_SOURCE"
HERE="${HERE%/*}"
MODELS_DIR=${MODELS_DIR?"Please set MODELS_DIR to the directory where your llama.cpp models are stored (e.g., /models)"}
LLAMA_PRESETS="${LLAMA_PRESETS:-$HERE/llama_server.ini}"
docker rm llama || true
exec docker run \
    --pull=always \
    --name llama \
    --detach \
    --network=host \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    --group-add=video \
    --ipc=host \
    --shm-size=128GB \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --group-add=109 \
    --group-add 992 \
    --device /dev/kfd \
    --device /dev/dri \
    -v $MODELS_DIR:/models:ro \
    -v $LLAMA_PRESETS:/llama/llamacpp_presets.ini \
    -e HF_HOME \
    -e HF_TOKEN \
    -e HF_HUB_CACHE \
    $LLAMA_DOCKER_IMAGE \
        --models-preset /llama/llamacpp_presets.ini \
        --models-dir /models \
        --verbose \
        --mlock \
        --split-mode none \
        --log-verbosity 3 \
        --webui \
        $*
