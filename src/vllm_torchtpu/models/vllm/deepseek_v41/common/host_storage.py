# SPDX-License-Identifier: Apache-2.0
"""Node-local Engram storage without CUDA registration or accelerator views."""

import os
import tempfile
from contextlib import ExitStack

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.shm_broadcast import (
    SHM_PATH,
    check_shm_free_space,
)
from vllm.models.deepseek_v41.common.engram import DPSharedEngramStorage


class HostSharedEngramStorage(DPSharedEngramStorage):
    """Reuse upstream packed FP8/scale views with a CPU-only shared mapping."""

    def _allocate(self, num_bytes: int) -> torch.Tensor:
        # Adapted from vLLM common/engram.py:DPSharedEngramStorage._allocate.
        # Torch owns the mmap; no CUDA registration or resource tracker is needed.
        group = self.group
        with ExitStack() as stack:
            path = error = None
            if group.rank_in_group == 0:
                try:
                    check_shm_free_space(num_bytes)
                    backing = stack.enter_context(
                        tempfile.NamedTemporaryFile(prefix="vllm_engram_", dir=SHM_PATH)
                    )
                    backing.truncate(num_bytes)
                    path = backing.name
                except (OSError, RuntimeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            path, error = group.broadcast_object((path, error))
            if error is not None:
                raise RuntimeError("Engram shared-memory creation failed: " + error)
            tensor = None
            try:
                # Reserve pages before loading to report ENOSPC instead of SIGBUS.
                # Each rank faults its portion on its own host NUMA node.
                start = num_bytes * group.rank_in_group // group.world_size
                end = num_bytes * (group.rank_in_group + 1) // group.world_size
                with open(path, "r+b") as backing:
                    os.posix_fallocate(backing.fileno(), start, end - start)
                tensor = torch.from_file(
                    path, shared=True, size=num_bytes, dtype=torch.uint8, device="cpu"
                )
            except (OSError, RuntimeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            errors: list[str | None] = [None] * group.world_size
            dist.all_gather_object(errors, error, group=group.cpu_group)
            failures = "; ".join(
                f"EDP rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            )
            if failures:
                raise RuntimeError("Engram shared-memory mapping failed: " + failures)
            assert tensor is not None
            # All ranks mapped the file before the leader unlinks its name.
            return tensor

    def get_views(
        self, weight: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (weight.data_ptr(), scales.data_ptr()) != (
            self.weight.data_ptr(),
            self.weight_scale_inv.data_ptr(),
        ):
            raise RuntimeError("Shared Engram parameter storage must not be replaced")
        return weight.data, scales.data
