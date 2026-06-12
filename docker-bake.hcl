group "default" {
  targets = ["local"]
}
group "release" {
  targets = ["containers"]
}
group "local" {
  targets = ["_local"]
}
variable "DOCKER_REGISTRY" {
  default = "ghcr.io"
}
variable "DOCKER_REPOSITORY" {
  default = "ai"
}
variable "DOCKER_IMAGE_NAME" {
  default = "llama.cpp"
}
variable "LEMONADE_LLAMACPP_VERSION" {
  default = "b1293"
}
variable "DOCKER_TAG" {
  default = "latest"
}
variable "GFX_VERSION" {
  default = "-gfx1151"
}
target "_common" {
  context = "."
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
  networks = ["host"]
  buildkit = true
  target = "runtime"
  args = {
    CACHEBUST = "1"
    LEMONADE_LLAMACPP_VERSION = "${LEMONADE_LLAMACPP_VERSION}"
  }
  progress = ["plain", "tty"]
}

target "_local" {
  inherits = ["_common"]
  tags = [
    "local/${DOCKER_REPOSITORY}/${DOCKER_IMAGE_NAME}${GFX_VERSION}:${DOCKER_TAG}",
  ]
  output = [
    "type=docker,name=local/${DOCKER_REPOSITORY}/${DOCKER_IMAGE_NAME}${GFX_VERSION}:${DOCKER_TAG}"
  ]
}

target "containers" {
  inherits = ["_common"]
  pull = true
  name = "containers-${env}"
  matrix = {
    env = ["release"]
  }
  tags = [
    "${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}/${DOCKER_IMAGE_NAME}${GFX_VERSION}:${DOCKER_TAG}",
  ]
  output = [
    "type=image,name=${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}/${DOCKER_IMAGE_NAME}${GFX_VERSION}:${DOCKER_TAG},push=true"
  ]
  cache-to = [
    "type=registry,ref=${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}/${DOCKER_IMAGE_NAME}${GFX_VERSION}:cache,mode=max"
  ]
  cache-from = [
    "type=registry,ref=${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}/${DOCKER_IMAGE_NAME}${GFX_VERSION}:cache",
    "type=registry,ref=${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}/${DOCKER_IMAGE_NAME}${GFX_VERSION}:${DOCKER_TAG}"
  ]
  attest = [
    "type=provenance,mode=max",
    "type=sbom",
  ]
}
