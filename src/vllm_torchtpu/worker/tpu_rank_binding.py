# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Mapping

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TpuWorkerBinding:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    init_rank: int
    init_world_size: int
    init_local_rank: int
    native_local_rank: int
    dp_rank: int = 0
    dp_size: int = 1
    local_rank_offset: int = 0
    pcp_local_rank_remap: tuple[int, ...] | None = None
    pcp_remap_source: str | None = None

    def as_env(self) -> dict[str, str]:
        return {
            "RANK": str(self.rank),
            "LOCAL_RANK": str(self.local_rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
        }


def _get_int_env(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, "") or default)
    except ValueError:
        return default


def get_tpu_worker_binding(
    parallel_config,
    rank: int,
    local_rank: int,
    *,
    env: Mapping[str, str],
    use_spawned_pcp_local_rank: bool = False,
) -> TpuWorkerBinding:
    world_size = parallel_config.world_size
    dp_size = (_get_int_env(env, "TORCH_TPU_DP_SIZE", 0)
               or parallel_config.data_parallel_size)
    local_rank_offset = _get_int_env(env, "TPU_LOCAL_RANK_OFFSET", 0)

    rank = int(rank)
    local_rank = int(local_rank)
    if dp_size > 1:
        dp_rank = parallel_config.data_parallel_index
        assert dp_rank is not None, (
            "ParallelConfig.data_parallel_index must be resolved when "
            "data_parallel_size > 1")

        native_local_rank = dp_rank * world_size + local_rank
        global_rank = dp_rank * world_size + rank
        global_world = world_size * dp_size
        tpu_local_rank = native_local_rank
        return TpuWorkerBinding(
            rank=global_rank,
            local_rank=local_rank_offset + tpu_local_rank,
            world_size=global_world,
            local_world_size=local_rank_offset + global_world,
            init_rank=rank,
            init_world_size=world_size,
            init_local_rank=tpu_local_rank,
            native_local_rank=native_local_rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            local_rank_offset=local_rank_offset,
        )

    native_local_rank = local_rank
    tpu_local_rank = native_local_rank
    return TpuWorkerBinding(
        rank=rank,
        local_rank=local_rank_offset + tpu_local_rank,
        world_size=world_size,
        local_world_size=local_rank_offset + world_size,
        init_rank=rank,
        init_world_size=world_size,
        init_local_rank=tpu_local_rank,
        native_local_rank=native_local_rank,
        dp_size=1,
        local_rank_offset=local_rank_offset,
    )
