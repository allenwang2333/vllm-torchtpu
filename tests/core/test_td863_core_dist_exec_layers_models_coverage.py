# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""100% statement and branch coverage tests for TD8.63.

Covers:
  1. src/vllm_torchtpu/core/tpu_scheduler.py
  2. src/vllm_torchtpu/distributed/dcp.py
  3. src/vllm_torchtpu/distributed/kv_transfer/tpu_multi_connector.py
  4. src/vllm_torchtpu/executors/tpu_multiproc_executor.py
  5. src/vllm_torchtpu/kernels/mla/masked_dense/cost_model.py
  6. src/vllm_torchtpu/layers/adapter/custom_ops/deepseek_v41/deepseek_v41_backend.py
  7. src/vllm_torchtpu/layers/adapter/linear.py
  8. src/vllm_torchtpu/models/vllm/deepseek_v41/v2/async_fence.py
  9. src/vllm_torchtpu/models/vllm/deepseek_v41/v2/engram.py
  10. src/vllm_torchtpu/omni/attention.py
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

# Stub vllm_omni modules if not installed. The stubs are scoped to this
# module's import: they (and the omni.attention module bound to them) are
# removed again below so later test modules import against their own mocks.
if "vllm_omni" not in sys.modules:
    vllm_omni_mod = ModuleType("vllm_omni")
    diff_mod = ModuleType("vllm_omni.diffusion")
    attn_pkg = ModuleType("vllm_omni.diffusion.attention")
    backends_mod = ModuleType("vllm_omni.diffusion.attention.backends")
    sdpa_mod = ModuleType("vllm_omni.diffusion.attention.backends.sdpa")

    class _StubSDPABackend:
        pass

    class _StubSDPAImpl:
        def _forward_impl(self, query, key, value, attn_metadata, mask_mode):
            return SimpleNamespace(
                query=query,
                key=key,
                value=value,
                attn_metadata=attn_metadata,
                mask_mode=mask_mode,
            )

    sdpa_mod.SDPABackend = _StubSDPABackend  # type: ignore[attr-defined]
    sdpa_mod.SDPAImpl = _StubSDPAImpl  # type: ignore[attr-defined]

    sys.modules["vllm_omni"] = vllm_omni_mod
    sys.modules["vllm_omni.diffusion"] = diff_mod
    sys.modules["vllm_omni.diffusion.attention"] = attn_pkg
    sys.modules["vllm_omni.diffusion.attention.backends"] = backends_mod
    sys.modules["vllm_omni.diffusion.attention.backends.sdpa"] = sdpa_mod

import pytest  # type: ignore[missing-import]
import torch
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu.core.tpu_scheduler import TpuDpScheduler
from vllm_torchtpu.distributed import dcp
from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import (
    TPUMultiConnector,
)
from vllm_torchtpu.executors.tpu_multiproc_executor import TpuMultiprocExecutor
from vllm_torchtpu.kernels.mla.masked_dense.cost_model import (
    GLM52_TPU7X_PREFILL_COST_MODEL,
    KV_BLOCK_SIZE,
    QUERY_BLOCK_SIZE,
    MaskedDensePrefillCostModel,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_backend import (
    AttentionBackend,
)
from vllm_torchtpu.layers.adapter.linear import TpuRowParallelLinear
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import async_fence, engram
from vllm_torchtpu.omni.attention import TpuSDPABackend, TpuSDPAImpl

_omni_attention = sys.modules["vllm_torchtpu.omni.attention"]
if _omni_attention.SDPABackend.__name__ == "_StubSDPABackend":
    for _name in (
        "vllm_omni",
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.attention",
        "vllm_omni.diffusion.attention.backends",
        "vllm_omni.diffusion.attention.backends.sdpa",
        "vllm_torchtpu.omni.attention",
    ):
        sys.modules.pop(_name, None)
    _omni_pkg = sys.modules.get("vllm_torchtpu.omni")
    if _omni_pkg is not None and hasattr(_omni_pkg, "attention"):
        delattr(_omni_pkg, "attention")

pytestmark = pytest.mark.cpu_test


# ===========================================================================
# 1. TpuDpScheduler Coverage
# ===========================================================================


def test_tpu_dp_scheduler_coverage(monkeypatch) -> None:
    schedule_calls: list[bool] = []

    def mock_super_schedule(self, allow_async: bool = False):
        schedule_calls.append(allow_async)
        return MagicMock(spec=SchedulerOutput)

    monkeypatch.setattr(
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler.schedule",
        mock_super_schedule,
    )
    monkeypatch.setattr(
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler.__init__",
        lambda self, *args, **kwargs: None,
    )

    scheduler = TpuDpScheduler()
    assert scheduler._drain_prefill is False
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.waiting = []
    scheduler.skipped_waiting = []

    # 1. schedule with throttle_prefills=False -> opens drain window
    # Case 1a: drain_prefill is True, waiting is non-empty -> stays draining
    scheduler.waiting = [MagicMock()]
    out = scheduler.schedule(throttle_prefills=False)
    assert scheduler.prefill_capacity_bound is True
    assert scheduler._drain_prefill is True
    assert out is not None
    assert schedule_calls[-1] is False

    # Case 1b: drain_prefill is True and skipped_waiting is non-empty -> stays draining
    scheduler.waiting = []
    scheduler.skipped_waiting = [MagicMock()]
    out = scheduler.schedule(throttle_prefills=False)
    assert scheduler._drain_prefill is True

    # Case 1c: waiting & skipped_waiting empty -> resets _drain_prefill = False
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    out = scheduler.schedule(throttle_prefills=False)
    assert scheduler._drain_prefill is False

    # 2. schedule throttle_prefills=True, not _drain_prefill -> paused_new()
    assert scheduler._drain_prefill is False
    out = scheduler.schedule(throttle_prefills=True)
    assert scheduler._pause_state == PauseState.UNPAUSED

    # 2b. schedule throttle_prefills=True when _drain_prefill is True (covers 51->54)
    scheduler._drain_prefill = True
    scheduler.waiting = [MagicMock()]
    out = scheduler.schedule(throttle_prefills=True)
    assert scheduler._drain_prefill is True
    assert schedule_calls[-1] is False

    # 3. _schedule_paused_new: prev_pause_state == PauseState.UNPAUSED
    scheduler._pause_state = PauseState.UNPAUSED
    out = scheduler._schedule_paused_new()
    assert scheduler._pause_state == PauseState.UNPAUSED

    # 4. _schedule_paused_new: prev_pause_state != PauseState.UNPAUSED (fast path)
    scheduler._pause_state = PauseState.PAUSED_NEW
    out = scheduler._schedule_paused_new()
    assert scheduler._pause_state == PauseState.PAUSED_NEW


# ===========================================================================
# 2. DCP Accessors Coverage
# ===========================================================================


def test_dcp_coverage(monkeypatch) -> None:
    # 0. Initial import resolution succeeds (covers line 49)
    monkeypatch.setattr(dcp, "_vllm_get_dcp_group_resolved", False)
    monkeypatch.setattr(dcp, "_vllm_get_dcp_group", None)
    mock_init_group = SimpleNamespace(rank_in_group=1, world_size=2)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dcp_group",
        lambda: mock_init_group,
    )
    assert dcp.get_dcp_group() is mock_init_group
    assert dcp._vllm_get_dcp_group_resolved is True

    # 1. Import resolution fails
    monkeypatch.setattr(dcp, "_vllm_get_dcp_group_resolved", False)
    monkeypatch.setattr(dcp, "_vllm_get_dcp_group", None)
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "vllm.distributed.parallel_state", None)
        assert dcp.get_dcp_group() is None
        assert dcp._vllm_get_dcp_group_resolved is True

    # 2. Cached None resolution
    assert dcp.get_dcp_group() is None

    # 3. get_dcp_group when resolved group function raises AssertionError
    def mock_group_raises():
        raise AssertionError("Uninitialized")

    monkeypatch.setattr(dcp, "_vllm_get_dcp_group", mock_group_raises)
    monkeypatch.setattr(dcp, "_vllm_get_dcp_group_resolved", True)
    assert dcp.get_dcp_group() is None

    # 4. get_dcp_group when resolved group function returns a valid group
    mock_group = SimpleNamespace(rank_in_group=2, world_size=4)
    monkeypatch.setattr(dcp, "_vllm_get_dcp_group", lambda: mock_group)
    assert dcp.get_dcp_group() is mock_group

    # 5. get_dcp_rank and get_dcp_world_size
    # With active group
    assert dcp.get_dcp_rank() == 2
    assert dcp.get_dcp_world_size() == 4

    # Without active group
    monkeypatch.setattr(dcp, "_vllm_get_dcp_group", lambda: None)
    assert dcp.get_dcp_rank() == 0
    assert dcp.get_dcp_world_size() == 1

    # 6. get_dcp_group_layout
    monkeypatch.setattr(
        dcp,
        "get_cp_group_layout",
        lambda grp: SimpleNamespace(group=grp, ranks=[0, 1]),
    )
    layout = dcp.get_dcp_group_layout()
    assert layout.ranks == [0, 1]

    # 7. get_or_create_dcp_mesh
    captured_mesh_args: dict[str, Any] = {}

    def mock_get_or_create_cp_mesh(axis_name, group):
        captured_mesh_args["axis_name"] = axis_name
        captured_mesh_args["group"] = group
        return "mesh_obj"

    monkeypatch.setattr(dcp, "get_or_create_cp_mesh", mock_get_or_create_cp_mesh)

    # Default axis name (None -> DCP_AXIS_NAME)
    res = dcp.get_or_create_dcp_mesh(None)
    assert res == "mesh_obj"
    assert captured_mesh_args["axis_name"] == dcp.DCP_AXIS_NAME

    # Explicit custom axis name
    res = dcp.get_or_create_dcp_mesh("custom_dcp")
    assert captured_mesh_args["axis_name"] == "custom_dcp"


# ===========================================================================
# 3. TPUMultiConnector Coverage
# ===========================================================================


def test_tpu_multi_connector_coverage() -> None:
    # 1. Empty connectors fast path
    empty_conn = object.__new__(TPUMultiConnector)
    empty_conn._connectors = []
    empty_conn.register_runner("dummy_runner")
    assert empty_conn.get_block_ids_with_load_errors_group_index() is None

    # 2. register_runner with connectors
    runner_calls: list[Any] = []
    c1 = SimpleNamespace(register_runner=lambda r: runner_calls.append(r))
    c2 = SimpleNamespace()  # no register_runner attribute
    conn = object.__new__(TPUMultiConnector)
    conn._connectors = [c1, c2]
    conn.register_runner("runner_1")
    assert runner_calls == ["runner_1"]

    # 3. get_block_ids_with_load_errors_group_index: all combinations
    c_no_fn = SimpleNamespace()
    c_none = SimpleNamespace(get_block_ids_with_load_errors_group_index=lambda: None)
    c_idx0 = SimpleNamespace(get_block_ids_with_load_errors_group_index=lambda: 0)
    c_idx0_dup = SimpleNamespace(get_block_ids_with_load_errors_group_index=lambda: 0)
    c_idx1 = SimpleNamespace(get_block_ids_with_load_errors_group_index=lambda: 1)

    # Consistent index resolution
    conn_ok = object.__new__(TPUMultiConnector)
    conn_ok._connectors = [c_no_fn, c_none, c_idx0, c_idx0_dup]
    assert conn_ok.get_block_ids_with_load_errors_group_index() == 0

    # Conflicting index error
    conn_conflict = object.__new__(TPUMultiConnector)
    conn_conflict._connectors = [c_idx0, c_idx1]
    with pytest.raises(RuntimeError, match="Conflicting KV-load-error"):
        conn_conflict.get_block_ids_with_load_errors_group_index()


# ===========================================================================
# 4. TpuMultiprocExecutor Coverage
# ===========================================================================


def test_tpu_multiproc_executor_coverage(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.v1.executor.multiproc_executor.MultiprocExecutor._init_executor",
        lambda self: None,
    )
    monkeypatch.setattr(
        "vllm.v1.executor.multiproc_executor.MultiprocExecutor.get_kv_cache_specs",
        lambda self: [{"kv": "spec"}],
    )
    monkeypatch.setattr(
        "vllm.v1.executor.multiproc_executor.MultiprocExecutor.initialize_from_config",
        lambda self, configs: None,
    )

    executor = object.__new__(TpuMultiprocExecutor)
    executor.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        parallel_config=SimpleNamespace(world_size=4, pipeline_parallel_size=2),
    )

    # 1. _init_executor patches environment
    patch_called = False

    def mock_patch():
        nonlocal patch_called
        patch_called = True

    monkeypatch.setattr(
        "vllm_torchtpu._patch_multiproc_worker_global_rank_env", mock_patch
    )
    executor._init_executor()
    assert patch_called is True

    # 2. get_kv_cache_specs with override already present
    executor.vllm_config.cache_config.num_gpu_blocks_override = 1234
    specs = executor.get_kv_cache_specs()
    assert specs == [{"kv": "spec"}]
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 1234

    # 3. get_kv_cache_specs with override None -> reconciles override
    executor.vllm_config.cache_config.num_gpu_blocks_override = None
    rpc_calls: list[tuple[str, Any]] = []

    def mock_collective_rpc(method, args=None):
        rpc_calls.append((method, args))
        if method == "get_num_gpu_blocks_override":
            return [2048, 2048]
        return None

    executor.collective_rpc = mock_collective_rpc  # type: ignore[method-assign]
    specs = executor.get_kv_cache_specs()
    assert specs == [{"kv": "spec"}]
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 2048

    # 4. get_kv_cache_specs when reconcile returns None
    executor.vllm_config.cache_config.num_gpu_blocks_override = None
    monkeypatch.setattr(
        "vllm_torchtpu.executors.tpu_multiproc_executor.reconcile_num_gpu_blocks_override",
        lambda *args, **kwargs: None,
    )
    executor.get_kv_cache_specs()
    assert executor.vllm_config.cache_config.num_gpu_blocks_override is None

    # 5. initialize_from_config
    # 5a. empty shapes
    rpc_prewarms: list[tuple[str, Any]] = []

    def mock_rpc_shapes_empty(method, args=None):
        rpc_prewarms.append((method, args))
        if method == "get_kv_prewarm_shapes":
            return []
        return None

    executor.collective_rpc = mock_rpc_shapes_empty  # type: ignore[method-assign]
    executor.initialize_from_config([])
    assert len(rpc_prewarms) == 1

    # 5b. shapes[0] is empty
    rpc_prewarms.clear()

    def mock_rpc_shapes_first_empty(method, args=None):
        rpc_prewarms.append((method, args))
        if method == "get_kv_prewarm_shapes":
            return [[]]
        return None

    executor.collective_rpc = mock_rpc_shapes_first_empty  # type: ignore[method-assign]
    executor.initialize_from_config([])
    assert len(rpc_prewarms) == 1

    # 5c. shapes[0] populated with shapes
    rpc_prewarms.clear()

    def mock_rpc_shapes_populated(method, args=None):
        rpc_prewarms.append((method, args))
        if method == "get_kv_prewarm_shapes":
            return [["shape1", "shape2"]]
        return None

    executor.collective_rpc = mock_rpc_shapes_populated  # type: ignore[method-assign]
    executor.initialize_from_config([])
    assert ("prewarm_kv_offload_shape", ("shape1",)) in rpc_prewarms
    assert ("prewarm_kv_offload_shape", ("shape2",)) in rpc_prewarms


# ===========================================================================
# 5. MaskedDensePrefillCostModel Coverage
# ===========================================================================


def test_masked_dense_cost_model_coverage() -> None:
    cm = MaskedDensePrefillCostModel(
        analytic_fixed_ns=100,
        analytic_token_ns=10,
        bitmap_fixed_ns=200,
        bitmap_token_ns=20,
        q_kv_block_ns=5,
        sparse_fixed_ns=300,
        sparse_token_ns=30,
    )
    assert cm.estimate_analytic_cost_ns(num_tokens=10, num_kv_blocks=4) == (
        100 + 10 * 10 + 4 * 5
    )
    assert cm.estimate_bitmap_cost_ns(num_tokens=10, num_kv_blocks=4) == (
        200 + 10 * 20 + 4 * 5
    )
    assert cm.estimate_sparse_cost_ns(num_tokens=10) == (300 + 10 * 30)

    # Constant definitions
    assert KV_BLOCK_SIZE == 1024
    assert QUERY_BLOCK_SIZE == 32
    assert GLM52_TPU7X_PREFILL_COST_MODEL.analytic_fixed_ns == 215_000


# ===========================================================================
# 6. AttentionBackend Coverage
# ===========================================================================


def test_attention_backend_coverage() -> None:
    assert AttentionBackend.get_name() == "TORCH_TPU_DEEPSEEK_V41"
    shape = AttentionBackend.get_kv_cache_shape(
        num_blocks=16,
        block_size=64,
        num_kv_heads=8,
        head_size=128,
        extra_arg="unused",
    )
    assert shape == (16, 8, 64, 128)


# ===========================================================================
# 7. TpuRowParallelLinear Coverage
# ===========================================================================


def test_tpu_row_parallel_linear_coverage(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.RowParallelLinear.forward",
        lambda self, x: x + 999.0,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.linear.tensor_model_parallel_all_reduce",
        lambda x: x,
    )

    linear = object.__new__(TpuRowParallelLinear)
    torch.nn.Module.__init__(linear)
    linear.reduce_results = True
    linear.tp_size = 2
    linear.tp_rank = 0
    linear.skip_bias_add = False
    linear.bias = torch.nn.Parameter(torch.zeros(4))
    linear.input_is_parallel = True
    linear.return_bias = False
    linear.quant_method = SimpleNamespace(apply=lambda lin, x, bias: x * 2.0)

    x = torch.ones((2, 4))

    # 1. defer_bias is False branches:
    # 1a. reduce_results is False
    linear.reduce_results = False
    assert linear.forward(x)[0, 0] == 1000.0

    # 1b. tp_size <= 1
    linear.reduce_results = True
    linear.tp_size = 1
    assert linear.forward(x)[0, 0] == 1000.0

    # 1c. skip_bias_add is True
    linear.tp_size = 2
    linear.skip_bias_add = True
    assert linear.forward(x)[0, 0] == 1000.0

    # 1d. bias is None
    linear.skip_bias_add = False
    linear.bias = None  # type: ignore[assignment]
    assert linear.forward(x)[0, 0] == 1000.0

    # 2. defer_bias is True:
    linear.bias = torch.nn.Parameter(torch.ones(4))
    # 2a. input_is_parallel is True, return_bias is False
    linear.input_is_parallel = True
    linear.return_bias = False
    res = linear.forward(x)
    assert isinstance(res, torch.Tensor)
    # output = (1.0 * 2.0) + 1.0 = 3.0
    assert torch.allclose(res, torch.full((2, 4), 3.0))

    # 2b. input_is_parallel is False, return_bias is True
    linear.input_is_parallel = False
    linear.return_bias = True
    x_wide = torch.ones((2, 8))
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.linear.split_tensor_along_last_dim",
        lambda t, num_partitions: [t[:, :4], t[:, 4:]],
    )
    res_tuple = linear.forward(x_wide)
    assert isinstance(res_tuple, tuple)
    assert len(res_tuple) == 2
    assert res_tuple[1] is None
    assert torch.allclose(res_tuple[0], torch.full((2, 4), 3.0))


# ===========================================================================
# 8. AsyncFence Coverage
# ===========================================================================


class _MockTPUModelRunner:
    @staticmethod
    def _prepare_lookback_token_ids(self, num_reqs: int) -> torch.Tensor:
        return torch.ones((num_reqs,), dtype=torch.int32)


def test_async_fence_coverage(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.TPUModelRunner", _MockTPUModelRunner
    )

    event_calls = {"sync": 0, "record": 0}

    class MockEvent:
        def synchronize(self) -> None:
            event_calls["sync"] += 1

        def record(self) -> None:
            event_calls["record"] += 1

    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=MockEvent), raising=False)

    # Install fence
    async_fence.install_lookback_fence()
    runner_fn = _MockTPUModelRunner._prepare_lookback_token_ids
    assert getattr(runner_fn, "_deepseek_v41_fenced", False) is True

    # Call install a second time -> early return
    async_fence.install_lookback_fence()

    # Call prepare: first time (fence is None) -> records event
    runner_instance = SimpleNamespace()
    ids1 = runner_fn(runner_instance, 5)
    assert ids1.shape == (5,)
    assert event_calls["sync"] == 0
    assert event_calls["record"] == 1
    assert hasattr(runner_instance, "_deepseek_v41_lookback_fence")

    # Call prepare: second time (fence is not None) -> syncs existing and records
    ids2 = runner_fn(runner_instance, 3)
    assert ids2.shape == (3,)
    assert event_calls["sync"] == 1
    assert event_calls["record"] == 2


# ===========================================================================
# 9. Engram Coverage
# ===========================================================================


def test_engram_coverage(monkeypatch) -> None:
    cache_called: list[Any] = []

    def mock_write_cache(cache, slots, values, valid):
        cache_called.append((cache, slots, values, valid))

    monkeypatch.setattr(
        "vllm_torchtpu.models.vllm.deepseek_v41.v2.engram.write_cache",
        mock_write_cache,
    )

    state = object.__new__(engram.NgramHashState)
    state.write_cache("cache_obj", [1], [2], [True])
    assert cache_called == [("cache_obj", [1], [2], [True])]

    # Verify all exports
    for name in engram.__all__:
        assert hasattr(engram, name)

    # Verify all exports
    for name in engram.__all__:
        assert hasattr(engram, name)


# ===========================================================================
# 10. Omni Attention SDPA Coverage
# ===========================================================================


class _MockSDPAImplBase:
    def _forward_impl(self, query, key, value, attn_metadata, mask_mode):
        return SimpleNamespace(
            query=query,
            key=key,
            value=value,
            attn_metadata=attn_metadata,
            mask_mode=mask_mode,
        )


class _MockSDPABackendBase:
    pass


def test_omni_attention_coverage(monkeypatch) -> None:
    monkeypatch.setattr(
        TpuSDPAImpl,
        "_forward_impl",
        lambda self, q, k, v, attn_metadata, mask_mode: SimpleNamespace(
            query=q, key=k, value=v, attn_metadata=attn_metadata, mask_mode=mask_mode
        ),
    )

    impl = TpuSDPAImpl.__new__(TpuSDPAImpl)
    q = torch.ones((1, 2, 4))
    k = torch.ones((1, 2, 4))
    v = torch.ones((1, 2, 4))
    res = impl.forward(q, k, v, attn_metadata="meta")
    assert res.mask_mode == "broadcast_k"
    assert res.attn_metadata == "meta"

    assert TpuSDPABackend.get_name() == "TPU_SDPA"
    assert TpuSDPABackend.get_impl_cls() is TpuSDPAImpl
