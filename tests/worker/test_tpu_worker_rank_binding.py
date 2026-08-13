# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

from vllm_torchtpu.worker import tpu_rank_binding as binding


def _parallel_config(**kwargs):
    values = {
        "world_size": 8,
        "data_parallel_size": 1,
        "data_parallel_rank": 0,
        "data_parallel_index": None,
        "prefill_context_parallel_size": 8,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_single_host_worker_binding():
    pc = _parallel_config(world_size=8, prefill_context_parallel_size=8)
    b = binding.get_tpu_worker_binding(pc, rank=2, local_rank=2, env={})

    assert b.as_env() == {
        "RANK": "2",
        "LOCAL_RANK": "2",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
    }
    assert b.rank == 2
    assert b.local_rank == 2
    assert b.init_rank == 2
    assert b.init_world_size == 8
    assert b.init_local_rank == 2
    assert b.native_local_rank == 2


def test_single_host_worker_binding_with_offset():
    pc = _parallel_config(world_size=8, prefill_context_parallel_size=8)
    env = {"TPU_LOCAL_RANK_OFFSET": "4"}
    b = binding.get_tpu_worker_binding(pc, rank=2, local_rank=2, env=env)

    assert b.as_env() == {
        "RANK": "2",
        "LOCAL_RANK": "6",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "12",
    }
    assert b.local_rank == 6
    assert b.local_world_size == 12


def test_dp_binding():
    pc = _parallel_config(
        world_size=4,
        data_parallel_size=1,
        data_parallel_rank=0,
        data_parallel_index=1,
        prefill_context_parallel_size=4,
    )
    env = {
        "TORCH_TPU_DP_SIZE": "2",
        "TPU_LOCAL_RANK_OFFSET": "1",
    }
    b = binding.get_tpu_worker_binding(pc, rank=2, local_rank=2, env=env)

    assert b.dp_rank == 1
    assert b.as_env() == {
        "RANK": "6",
        "LOCAL_RANK": "7",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "9",
    }
    assert b.rank == 6
    assert b.local_rank == 7
    assert b.world_size == 8
    assert b.init_rank == 2
    assert b.init_world_size == 4
    assert b.init_local_rank == 6


def test_dp_binding_rejects_unresolved_data_parallel_index():
    pc = _parallel_config(data_parallel_size=2, data_parallel_index=None)

    with pytest.raises(AssertionError, match="data_parallel_index"):
        binding.get_tpu_worker_binding(pc, rank=0, local_rank=0, env={})
