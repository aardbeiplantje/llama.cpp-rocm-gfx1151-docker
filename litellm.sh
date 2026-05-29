#!/bin/bash
docker stop litellm || true
ip=$(ip -6 addr show scope global|grep /128|awk '{print $2}'|cut -d/ -f1|head -n1)
echo "Using IP: $ip"
cat >/tmp/litellm-$LOGNAME.yaml <<EOcfg
model_list:
  - model_name: gemma-3n-E4B-it-Q8_0
    litellm_params:
      model: openai/gemma-3n-E4B-it-Q8_0
      api_base: http://[$ip]:8000
      api_key: "not-needed"
      temperature: 0.6
      max_tokens: 8192
      drop_params: true
      extra_body:
        num_ctx: 98304
        options:
          include_special: true
  - model_name: Nemotron-3-Nano-30B-A3B-Q8_0
    litellm_params:
      model: openai/Nemotron-3-Nano-30B-A3B-Q8_0
      api_base: http://[$ip]:8000
      api_key: "not-needed"
      temperature: 0.6
      max_tokens: 8192
      drop_params: true
      extra_body:
        num_ctx: 98304
        options:
          include_special: true
        chat_template: "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"

litellm_settings:
  cache: true # full-response cache — bypasses llama.cpp for exact-dup requests
  cache_params:
    type: disk # persists across LiteLLM restarts; uses ./.litellm_cache by default
    ttl: 86400 # TTL: 24 hours before expired entries are evicted
  set_verbose: false

router_settings:
  routing_strategy: simple-shuffle
EOcfg
export OPENAI_API_KEY="not-needed"
exec docker run \
    --pull=always \
    --name litellm \
    --detach \
    --rm \
    -e OPENAI_API_KEY \
    -v /tmp/litellm-$LOGNAME.yaml:/app/config.yaml:ro \
    --network=host \
    docker.litellm.ai/berriai/litellm:main-latest \
        --port 4000 \
        --host '::' \
        --config /app/config.yaml \
        --detailed_debug
