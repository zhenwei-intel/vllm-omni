# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AR XPU Model Runner for vLLM-Omni.

Exposes per-request hidden representations via ModelRunnerOutput.pooler_output
and also outputs sampled tokens on Intel XPU devices.
"""

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


class XPUARModelRunner(GPUARModelRunner):
    """AR model runner for Intel XPU devices.
    
    Inherits functionality from GPUARModelRunner. XPU-specific optimizations
    can be added here as needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
