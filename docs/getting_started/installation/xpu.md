# XPU

vLLM-Omni supports Intel Data Center GPUs (XPU) through Intel Extension for PyTorch (IPEX). This is a community maintained hardware plugin for running vLLM on XPU.

## Requirements

- OS: Linux
- Python: 3.12

!!! note
    vLLM-Omni is currently not natively supported on Windows.

=== "XPU"

    --8<-- "docs/getting_started/installation/xpu/xpu.inc.md:requirements"

## Set up using Docker

### Build your own docker image

=== "XPU"

    --8<-- "docs/getting_started/installation/xpu/xpu.inc.md:build-docker"

### Pre-built images

=== "XPU"

    --8<-- "docs/getting_started/installation/xpu/xpu.inc.md:pre-built-images"
