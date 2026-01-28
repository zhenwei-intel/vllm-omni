# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base XPU Model Runner for vLLM-Omni.

Extends GPU model runner with XPU-specific device handling.
"""

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner


class XPUModelRunner(OmniGPUModelRunner):
    """Base model runner for Intel XPU devices.

    Inherits most functionality from OmniGPUModelRunner since PyTorch XPU API
    is largely compatible with CUDA API. Device-specific operations are handled
    through PyTorch's device abstraction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
