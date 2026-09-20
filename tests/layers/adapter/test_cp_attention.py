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
"""CPU tests for DCP layout propagation and kernel registry isolation.

Real TPU pass correctness is covered by test_attention_dcp.py. Here the
kernel builder and collectives are replaced so both adapter entry points and
the registry can be checked together, including reuse across layers/layouts.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import (DeviceConfig, VllmConfig, get_current_vllm_config,
                         set_current_vllm_config)
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx.configs import (
    AttentionScope, KVLayout)
from vllm_torchtpu.layers.adapter import attention, cp_attention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def cpu_vllm_config_context():
    # CPU CI does not register torch's TPU device type. These tests only
    # exercise layout propagation with CPU tensors and mocked kernels.
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _impl(cls=attention.PallasBatchedRPAAttentionBackendImpl, head_size=256):
    return cls(num_heads=8,
               head_size=head_size,
               scale=head_size**-0.5,
               num_kv_heads=1,
               alibi_slopes=None,
               sliding_window=None,
               kv_cache_dtype="fp8")


def _config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=8,
                                        prefill_context_parallel_size=1,
                                        decode_context_parallel_size=2))


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("compiled", [False, True])
def test_dcp_layout_survives_prebuild_forward_and_registry_reuse(
        monkeypatch, rank, compiled, cpu_vllm_config_context):
    built, calls = [], []

    def build_op(_name, fn):
        built.append(fn)

        def kernel(_cache, query, *args):
            scope = fn.keywords["attention_scope"]
            calls.append((fn.keywords["kv_layout"], scope, query.shape))
            value = 1 if scope == AttentionScope.CACHE_ONLY else 3
            return torch.full_like(query, value), torch.zeros(query.shape[:2])

        return kernel

    group = SimpleNamespace(
        world_size=2,
        rank_in_group=rank,
        all_gather=lambda tensor, dim: torch.cat([tensor, tensor], dim=dim))
    monkeypatch.setattr(attention, "_get_dcp_group", lambda: group)
    monkeypatch.setattr(cp_attention, "get_dcp_group", lambda: group)
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op", build_op)
    monkeypatch.setattr(cp_attention, "synchronize_tensors",
                        lambda *a, **k: None)
    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    implementations = []
    # Every layer resolves NHD: it is the only layout any backend allocates
    # now, and `initialize_kernel` refuses HND outright (see below). Three
    # layers, so a later one has to reuse the registry rather than rebuild.
    get_current_vllm_config().cache_config.kv_cache_layout = "LBNHC"
    for _ in range(3):
        impl = _impl()
        with set_vllm_model_wrapper_context(mesh=None, vllm_config=_config()):
            impl.initialize_kernel(layer)
        implementations.append(impl)

    assert len(built) == 2  # One layout, two scopes; shared by every layer.
    for other in implementations[1:]:
        assert implementations[0].rpa_dcp_cache_kernel is (
            other.rpa_dcp_cache_kernel)
        assert implementations[0].rpa_dcp_new_kernel is (
            other.rpa_dcp_new_kernel)

    metadata = SimpleNamespace(seq_lens=torch.tensor([2, 3, 4],
                                                     dtype=torch.int32),
                               block_tables=torch.zeros(3,
                                                        2,
                                                        dtype=torch.int32),
                               query_start_loc=torch.arange(4,
                                                            dtype=torch.int32),
                               request_distribution=torch.tensor(
                                   [3, 3, 3], dtype=torch.int32))
    query = torch.ones(3, 8, 256)
    key = torch.ones(3, 1, 256)
    for impl in implementations:
        # The layout baked in at prebuild must survive a config that changes
        # underneath the layer -- forward must not re-resolve it.
        get_current_vllm_config().cache_config.kv_cache_layout = "LBHNC"
        forward = impl._run_dcp_forward
        if compiled:
            forward = torch.compile(forward, backend="eager", fullgraph=True)
        output = forward(layer, query, key, key, torch.empty(1), metadata)
        assert calls[-2:] == [
            (KVLayout.HEAD_ALONG_SUBLANE, AttentionScope.CACHE_ONLY,
             (3, 16, 256)),
            (KVLayout.HEAD_ALONG_SUBLANE, AttentionScope.NEW_TOKENS_ONLY,
             (3, 8, 256)),
        ]
        # Two cached partials (value=1, weight=1 each) and one new partial
        # (value=3, weight=1), including both production LSE merges.
        torch.testing.assert_close(output, torch.full_like(query, 5 / 3))
    assert len(built) == 2  # Forward must reuse the prebuilt ops.
    assert all(type(fn.keywords["kv_layout"]) is KVLayout for fn in built)


def test_dcp_kernel_registry_is_keyed_by_layout(monkeypatch):
    """The adapter only ever asks for NHD now, but the registry key still has
    to carry the layout: `build_dcp_kernels` is public and the DCP kernels
    reinterpret the pool from this argument alone. Also pins the default --
    omitting `kv_layout` must land on the same entry as passing NHD."""
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    monkeypatch.setattr(cp_attention, "_build_dcp_kernel_op",
                        lambda name, fn: fn)
    kwargs = dict(sliding_window=None,
                  sm_scale=0.0625,
                  logits_soft_cap=None,
                  q_scale=None,
                  k_scale=None,
                  v_scale=None,
                  cp_group_size=2,
                  cp_rank=0)
    implicit = cp_attention.build_dcp_kernels(**kwargs)
    explicit = cp_attention.build_dcp_kernels(
        **kwargs, kv_layout=KVLayout.HEAD_ALONG_SUBLANE)
    assert all(a is b for a, b in zip(implicit, explicit))

    hnd = cp_attention.build_dcp_kernels(**kwargs,
                                         kv_layout=KVLayout.SEQ_ALONG_LANE)
    assert all(a is not b for a, b in zip(implicit, hnd))


@pytest.mark.parametrize("cls,head_size", [
    (attention.PallasAttentionBackendImpl, 256),
    (attention.PallasBatchedRPAAttentionBackendImpl, 256),
    (attention.PallasBatchedRPAAttentionBackendImpl, 64),
])
def test_dcp_rejects_hnd_on_every_backend(cls, head_size,
                                          cpu_vllm_config_context):
    """Every backend allocates the NHD pool now -- the CUSTOM backend's HND
    shape override went away with the in-tree batched kernel -- so the guard
    is no longer conditional on the class or the head width."""
    get_current_vllm_config().cache_config.kv_cache_layout = "LBHNC"
    impl = _impl(cls, head_size)
    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    with set_vllm_model_wrapper_context(mesh=None, vllm_config=_config()):
        with pytest.raises(NotImplementedError,
                           match="VLLM_KV_CACHE_LAYOUT=HND"):
            impl.initialize_kernel(layer)
