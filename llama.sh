#!/bin/bash
MODELS_DIR=${MODELS_DIR?"Please set MODELS_DIR to the directory where your llama.cpp models are stored (e.g., /models)"}
LLAMA_DOCKER_IMAGE=${LLAMA_DOCKER_IMAGE:-local/ai/llama.cpp-gfx1151:latest}
HERE="$BASH_SOURCE"
HERE="${HERE%/*}"
LLAMA_PRESETS="${LLAMA_PRESETS:-$HERE/llamacpp_presets.ini}"
docker stop llama >/dev/null 2>&1 || true
while [ "$(docker ps -a -q -f name=^llama$)" ]; do
    docker rm llama >/dev/null 2>&1 || true
    sleep 1
done
exec docker run \
    --name llama \
    --rm \
    --detach \
    --network=host \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --cap-add=SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --group-add=109 \
    --group-add=986 \
    --group-add=992 \
    --device /dev/kfd \
    --device /dev/dri \
    --tmpfs /tmp:rw,suid,exec,size=1G \
    --tmpfs /var/tmp:rw,suid,exec,size=1G \
    -v $MODELS_DIR:/models:ro \
    -v $LLAMA_PRESETS:/llama/llamacpp_presets.ini \
    -v llama.cpp-data:/llama.cpp:rw \
    -v $HF_HOME:/hf:rw \
    -e HF_HOME=/hf \
    -e HF_TOKEN \
    -e HF_HUB_CACHE=/hf/hub \
    $LLAMA_DOCKER_IMAGE \
        --verbose \
        --mlock \
        --split-mode none \
        --log-verbosity 3 \
        --webui \
        "$@"
