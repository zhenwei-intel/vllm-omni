# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generation XPU Model Runner for vLLM-Omni.

Handles generation model execution on Intel XPU devices.
"""

from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner


class XPUGenerationModelRunner(GPUGenerationModelRunner):
    """Generation model runner for Intel XPU devices.
    
    Inherits functionality from GPUGenerationModelRunner. XPU-specific 
    optimizations can be added here as needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
