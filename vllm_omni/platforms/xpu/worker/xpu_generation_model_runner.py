# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""XPU Generation Model Runner for vLLM-Omni.

Inherits from GPU Generation Model Runner with XPU-specific initialization.
"""

from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner


class XPUGenerationModelRunner(GPUGenerationModelRunner):
    """XPU generation model runner for vLLM-Omni.

    Inherits all functionality from GPUGenerationModelRunner as XPU follows
    the same execution model as GPU/CUDA for non-autoregressive generation.
    """

    pass
