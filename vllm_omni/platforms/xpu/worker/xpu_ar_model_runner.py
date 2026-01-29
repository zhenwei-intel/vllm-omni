# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""XPU AR Model Runner for vLLM-Omni.

Inherits from GPU AR Model Runner with XPU-specific initialization.
"""

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


class XPUARModelRunner(GPUARModelRunner):
    """XPU autoregressive model runner for vLLM-Omni.
    
    Inherits all functionality from GPUARModelRunner as XPU follows
    the same execution model as GPU/CUDA.
    """
    pass
