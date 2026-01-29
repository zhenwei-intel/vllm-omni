# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.platforms.xpu.worker.xpu_ar_worker import XPUARWorker
from vllm_omni.platforms.xpu.worker.xpu_generation_worker import XPUGenerationWorker

__all__ = [
    "XPUARWorker",
    "XPUGenerationWorker",
]
