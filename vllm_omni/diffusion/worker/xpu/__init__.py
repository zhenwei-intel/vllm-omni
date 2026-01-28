# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU worker classes for diffusion models."""

from vllm_omni.diffusion.worker.xpu.xpu_worker import XPUWorker, XPUWorkerProc

__all__ = ["XPUWorker", "XPUWorkerProc"]
