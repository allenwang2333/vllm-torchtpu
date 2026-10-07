# Copyright 2025 Google LLC
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
"""Host-side caches on the set_forward_context path (CPU-runnable).

With sequence-parallel MoE on (even at DP == 1), upstream set_forward_context
builds a DPMetadata on every call; TpuPlatform.set_additional_forward_context
then copied its token-count tensor to the TPU each time, and the patched
ParallelConfig.use_sequence_parallel_moe re-read os.environ each time.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm_torchtpu
import vllm_torchtpu.platforms.tpu_platform as tpu_platform
from vllm_torchtpu import envs


@dataclass
class _DPMeta:
    num_tokens_across_dp_cpu: torch.Tensor


@pytest.fixture
def h2d_counter(monkeypatch):
    copies = []

    def fake_copy(t):
        out = t.clone()  # a distinct object standing in for the device copy
        copies.append(out)
        return out

    monkeypatch.setattr(tpu_platform, "_copy_dp_token_counts_to_tpu", fake_copy)
    monkeypatch.setattr(tpu_platform, "_dp_token_counts_on_tpu_cache", {})
    return copies


def _enter(n, dtype=torch.int32):
    meta = _DPMeta(torch.tensor(n, dtype=dtype))
    out = tpu_platform.TpuPlatform.set_additional_forward_context(dp_metadata=meta)
    assert out == {}
    return meta.num_tokens_across_dp_cpu


def test_same_counts_reuse_one_device_copy(h2d_counter):
    a = _enter([16])
    b = _enter([16])
    assert a is b
    assert len(h2d_counter) == 1
    assert a.tolist() == [16]


def test_different_counts_get_new_device_copy(h2d_counter):
    a = _enter([16])
    b = _enter([32])
    c = _enter([16, 16])
    d = _enter([16], dtype=torch.int64)
    assert len({id(x) for x in (a, b, c, d)}) == 4
    assert len(h2d_counter) == 4
    assert b.tolist() == [32] and c.tolist() == [16, 16]
    assert d.dtype == torch.int64
    assert _enter([32]) is b
    assert len(h2d_counter) == 4


def test_cache_is_bounded(h2d_counter, monkeypatch):
    monkeypatch.setattr(tpu_platform, "_DP_TOKEN_COUNTS_CACHE_MAX", 4)
    for n in range(10):
        _enter([n])
    assert len(tpu_platform._dp_token_counts_on_tpu_cache) <= 4
    assert _enter([9]).tolist() == [9]


def test_no_dp_metadata_is_a_no_op(h2d_counter):
    out = tpu_platform.TpuPlatform.set_additional_forward_context(dp_metadata=None)
    assert out == {}
    assert h2d_counter == []


def test_cached_tensor_survives_downstream_reads(h2d_counter):
    # DPMetadata consumers derive new tensors; the cached copy stays intact.
    from vllm.forward_context import DPMetadata

    t = _enter([13])
    meta = DPMetadata(t)
    with meta.sp_local_sizes(4) as sizes:
        assert sizes == [4, 4, 4, 4]
    meta.cu_tokens_across_sp(4)
    assert _enter([13]) is t and t.tolist() == [13]


# --- ParallelConfig.use_sequence_parallel_moe memo -------------------------


@pytest.fixture
def sp_gate(monkeypatch):
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed import parallel_state

    if not getattr(ParallelConfig, "_tpu_no_sp_moe_patch", False):
        vllm_torchtpu._patch_disable_sequence_parallel_moe()
    monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False)
    monkeypatch.setattr(envs, "VLLM_TPU_MOE_SEQUENCE_PARALLEL", "auto")

    def set_dp_group(world_size):
        monkeypatch.setattr(
            parallel_state,
            "_DP",
            None if world_size is None else SimpleNamespace(world_size=world_size),
        )

    config = SimpleNamespace(
        enable_expert_parallel=True,
        tensor_parallel_size=4,
        prefill_context_parallel_size=1,
        data_parallel_size=1,
    )
    return ParallelConfig.use_sequence_parallel_moe.fget, config, set_dp_group


def test_sp_gate_memoized_once_dp_group_exists(sp_gate, monkeypatch):
    gate, config, set_dp_group = sp_gate
    set_dp_group(1)
    assert gate(config) is False
    # os.environ is not re-read on a memo hit.
    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "4")
    assert gate(config) is False
    # A fresh config object is evaluated afresh.
    fresh = SimpleNamespace(**vars(config))
    assert gate(fresh) is True


def test_sp_gate_memo_invalidates_on_inputs(sp_gate, monkeypatch):
    gate, config, set_dp_group = sp_gate
    set_dp_group(1)
    assert gate(config) is False
    config.data_parallel_size = 2
    assert gate(config) is True
    config.data_parallel_size = 1
    monkeypatch.setattr(envs, "VLLM_TPU_MOE_SEQUENCE_PARALLEL", "force")
    assert gate(config) is True
    monkeypatch.setattr(envs, "VLLM_TPU_MOE_SEQUENCE_PARALLEL", "auto")
    assert gate(config) is False
    set_dp_group(4)  # new live DP group
    assert gate(config) is True


def test_sp_gate_not_memoized_before_dp_init(sp_gate, monkeypatch):
    gate, config, set_dp_group = sp_gate
    set_dp_group(None)
    assert gate(config) is False
    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "4")
    assert gate(config) is True
