#!/bin/bash
LLAMA_DOCKER_IMAGE=${LLAMA_DOCKER_IMAGE:-local/ai/llama.cpp-gfx1151:latest}
HERE="$BASH_SOURCE"
HERE="${HERE%/*}"

# Switch case?
DOCKER_OPTS=${DOCKER_OPTS:-""}
case "${1:-}" in
    quantize|cli|bench)
        D=$LOGNAME-${1}
        DOCKER_OPTS="$DOCKER_OPTS -it"
        ;;
    convert)
        D=$LOGNAME-${1}
        DOCKER_OPTS="$DOCKER_OPTS -it"
        ;;
    bash)
        D=$LOGNAME-${1}
        DOCKER_OPTS="$DOCKER_OPTS -it"
        ;;
    server)
        D=llama
        DOCKER_OPTS="$DOCKER_OPTS -d"
        ;;
    tail)
        # Just tail the logs, don't start the container
        docker logs -f llama
        exit $?
        ;;
    *)
        D=llama
        DOCKER_OPTS="$DOCKER_OPTS -d"
        ;;
esac

# Clean up any existing container
docker stop $D >/dev/null 2>&1 || true
while [ "$(docker ps -a -q -f name=^$D$)" ]; do
    docker rm $D >/dev/null 2>&1 || true
    sleep 1
done

# Now run
exec docker run \
    --name $D \
    --rm \
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
    $([ -n "$MODELS_DIR"    ] && echo "-v $MODELS_DIR:/models:rw") \
    $([ -n "$LLAMA_PRESETS" ] && echo "-v $LLAMA_PRESETS:/llama/llamacpp_presets.ini") \
    -v llama.cpp-data:/llama.cpp:rw \
    $([ -n "$HF_HOME" ] && echo "-v $HF_HOME:/hf:rw") \
    -e XDG_CACHE_HOME=/dev/shm \
    -e HF_HOME=/hf \
    -e HF_TOKEN \
    -e HF_HUB_CACHE=/hf/hub \
    $DOCKER_OPTS \
        $LLAMA_DOCKER_IMAGE \
            "$@"
