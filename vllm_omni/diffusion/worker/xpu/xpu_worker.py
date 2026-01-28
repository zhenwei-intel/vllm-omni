# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU Worker for Diffusion Models.

Handles Intel XPU device initialization and delegates model operations
to GPUDiffusionModelRunner (which is device-agnostic through PyTorch).
"""

import multiprocessing as mp
import os
from contextlib import AbstractContextManager, nullcontext

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import GiB_bytes

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.profiler import CurrentProfiler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.gpu_diffusion_model_runner import GPUDiffusionModelRunner
from vllm_omni.diffusion.worker.gpu_diffusion_worker import WorkerProc
from vllm_omni.lora.request import LoRARequest

logger = init_logger(__name__)


class XPUWorker:
    """
    A worker that executes the model on a single Intel XPU device.
    Similar to GPUWorker but with XPU-specific device initialization.
    """

    def __init__(
        self,
        local_rank: int,
        rank: int,
        od_config: OmniDiffusionConfig,
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.od_config = od_config
        self.device: torch.device | None = None
        self.vllm_config: VllmConfig | None = None
        self.model_runner: GPUDiffusionModelRunner | None = None
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self.lora_manager: DiffusionLoRAManager | None = None
        self.init_device_and_model()

    def init_device_and_model(self) -> None:
        """Initialize the XPU device and load the model."""
        world_size = self.od_config.num_gpus
        rank = self.rank
        # Set environment variables for distributed initialization
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.od_config.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Setup XPU device
        self.device = torch.device(f"xpu:{rank}")
        torch.xpu.set_device(self.device)

        # Create vllm_config for parallel configuration
        vllm_config = VllmConfig()
        vllm_config.parallel_config.tensor_parallel_size = self.od_config.parallel_config.tensor_parallel_size
        vllm_config.parallel_config.data_parallel_size = self.od_config.parallel_config.data_parallel_size
        self.vllm_config = vllm_config

        # Initialize distributed environment
        with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=self.od_config):
            init_distributed_environment(world_size=world_size, rank=rank)
            logger.info(f"Worker {self.rank}: Initialized device and distributed environment.")

            parallel_config = self.od_config.parallel_config
            initialize_model_parallel(
                data_parallel_size=parallel_config.data_parallel_size,
                cfg_parallel_size=parallel_config.cfg_parallel_size,
                sequence_parallel_size=parallel_config.sequence_parallel_size,
                ulysses_degree=parallel_config.ulysses_degree,
                ring_degree=parallel_config.ring_degree,
                tensor_parallel_size=parallel_config.tensor_parallel_size,
                pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            )

        # Create model runner and load model
        self.model_runner = GPUDiffusionModelRunner(
            vllm_config=self.vllm_config,
            od_config=self.od_config,
            device=self.device,
        )
        self.model_runner.load_model(
            memory_pool_context_fn=self._maybe_get_memory_pool_context,
        )
        assert self.model_runner.pipeline is not None
        self.lora_manager = DiffusionLoRAManager(
            pipeline=self.model_runner.pipeline,
            device=self.device,
            dtype=self.od_config.dtype,
            max_cached_adapters=self.od_config.max_cpu_loras,
            lora_path=self.od_config.lora_path,
            lora_scale=self.od_config.lora_scale,
        )
        logger.info(f"Worker {self.rank}: Initialization complete.")

    def generate(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """Generate output for the given requests."""
        return self.execute_model(requests, self.od_config)

    @classmethod
    def start_profile(cls, trace_path_template: str) -> str:
        """Start profiling for this XPU worker."""
        return CurrentProfiler.start(trace_path_template)

    @classmethod
    def stop_profile(cls) -> dict | None:
        """Stop profiling and return the result dictionary."""
        return CurrentProfiler.stop()

    def execute_model(self, reqs: list[OmniDiffusionRequest], od_config: OmniDiffusionConfig) -> DiffusionOutput:
        """Execute a forward pass by delegating to the model runner."""
        assert self.model_runner is not None, "Model runner not initialized"
        if self.lora_manager is not None and reqs:
            req = reqs[0]

            if len(reqs) > 1:
                # This worker (and the current diffusion model runner) applies
                # a single LoRA to the whole batch. Reject inconsistent LoRA
                # settings to avoid silently applying the wrong adapter.
                for r in reqs:
                    if r.lora_request != req.lora_request:
                        raise ValueError("All requests in a batch must have the same lora_request")

            self._apply_lora(req.lora_request)

        return self.model_runner.execute_model(reqs, od_config)

    def _apply_lora(self, lora_request: LoRARequest | None) -> None:
        """Apply LoRA to the model."""
        assert self.lora_manager is not None
        if lora_request is None:
            self.lora_manager.clear_active_adapter()
        else:
            self.lora_manager.set_active_adapter(lora_request)

    def load_weights(self, weights):
        """Load weights into the model."""
        assert self.model_runner is not None
        return self.model_runner.load_weights(weights)

    def sleep(self, level: int = 1) -> bool:
        """
        Put the worker to sleep. The worker should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level. Level 1 sleep will offload the model
                weights and discard the kv cache.
                Currently only support level 1.
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.xpu.mem_get_info(self.device)[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            assert self.model_runner is not None
            model = self.model_runner.pipeline
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.xpu.mem_get_info(self.device)
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
            freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes,
        )
        return True

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """
        Wake up the worker from sleep mode. See the sleep function
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the worker memory
                for specific memory allocations. Values must be in
                `("weights")`. If None, all memory is reallocated.
                wake_up should be called with all tags (or None) before the
                worker is used again.
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            assert self.model_runner is not None
            model = self.model_runner.pipeline
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
        return True

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if self.od_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            return allocator.use_memory_pool(tag=tag)
        else:
            return nullcontext()

    def shutdown(self) -> None:
        destroy_distributed_env()


class XPUWorkerProc(WorkerProc):
    """Wrapper that runs one XPUWorker in a separate process."""

    def _create_worker(self, gpu_id: int, od_config: OmniDiffusionConfig) -> XPUWorker:
        """Create an XPUWorker instead of GPUWorker."""
        return XPUWorker(
            local_rank=gpu_id,
            rank=gpu_id,
            od_config=od_config,
        )

    @staticmethod
    def worker_main(
        rank: int,
        od_config: OmniDiffusionConfig,
        pipe_writer: mp.connection.Connection,
        broadcast_handle,
    ) -> None:
        """Worker initialization and execution loops."""

        worker_proc = XPUWorkerProc(
            od_config,
            gpu_id=rank,
            broadcast_handle=broadcast_handle,
        )
        logger.info(f"Worker {rank}: Scheduler loop started.")
        pipe_writer.send(
            {
                "status": "ready",
                "result_handle": worker_proc.result_mq_handle if rank == 0 else None,
            }
        )
        worker_proc.worker_busy_loop()
        logger.info(f"Worker {rank}: Shutdown complete.")
