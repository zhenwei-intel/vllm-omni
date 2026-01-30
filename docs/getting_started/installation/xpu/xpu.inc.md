# --8<-- [start:requirements]

- GPU: Intel Data Center GPU Max Series (formerly Ponte Vecchio) or Intel Data Center GPU Flex Series
- Driver: Intel GPU drivers that support Intel Extension for PyTorch (IPEX)

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

vLLM-Omni currently recommends the steps in under setup through Docker Images.

# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]

# --8<-- [start:build-wheel-from-source]

# --8<-- [end:build-wheel-from-source]

# --8<-- [start:build-docker]

#### Build docker image

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.xpu -t vllm-omni-xpu .
```

#### Launch the docker image

##### Launch with OpenAI API Server

```
docker run --rm \
--device /dev/dri \
--ipc=host \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=$HF_TOKEN" \
-p 8091:8091 \
vllm-omni-xpu \
--model Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

##### Launch with interactive session for development

```
docker run --rm -it \
--network=host \
--device /dev/dri \
--ipc=host \
-v <path/to/model>:/app/model \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--entrypoint bash \
vllm-omni-xpu
```

# --8<-- [end:build-docker]

# --8<-- [start:pre-built-images]

vLLM-Omni offers an official docker image for deployment. These images are built on top of Intel Extension for PyTorch (IPEX) docker images and available on Docker Hub as [vllm/vllm-omni-xpu](https://hub.docker.com/r/vllm/vllm-omni-xpu/tags). The version of vLLM-Omni indicates which release of vLLM it is based on.

#### Launch vLLM-Omni Server
Here's an example deployment command that has been verified on Intel Data Center GPU Max Series:
```bash
docker run --rm \
  --device /dev/dri \
  --ipc=host \
  -v <path/to/model>:/app/model \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8091:8091 \
  vllm/vllm-omni-xpu:v0.14.0rc1 \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

#### Launch an interactive terminal with prebuilt docker image.
If you want to run in dev environment you can launch the docker image as follows:
```bash
docker run --rm -it \
  --network=host \
  --device /dev/dri \
  --ipc=host \
  -v <path/to/model>:/app/model \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  --entrypoint bash \
  vllm/vllm-omni-xpu:v0.14.0rc1
```

# --8<-- [end:pre-built-images]
