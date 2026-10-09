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
"""100% CPU statement & branch coverage suite for TD8.74 targets:
- src/vllm_torchtpu/executors/ray_distributed_executor.py
- src/vllm_torchtpu/executors/ray_distributed_executor_v2.py
- src/vllm_torchtpu/models/vllm/deepseek_v41/v1/engram.py
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/engram.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
import torch
from vllm import platforms
from vllm.config import CacheConfig

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.executors import ray_distributed_executor as ray_executor_v1
from vllm_torchtpu.executors import ray_distributed_executor_v2 as ray_executor_v2
from vllm_torchtpu.executors.ray_distributed_executor_v2 import (
    RayDistributedExecutorV2,
    get_tpu_bundles_for_indices,
)
from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v1 import engram as v1_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import engram as v2_engram


def _make_v1_executor(monkeypatch: pytest.MonkeyPatch):
    instance = object.__new__(ray_executor_v1.RayDistributedExecutor)
    instance.parallel_config = SimpleNamespace(
        world_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        placement_group=None,
        ray_workers_use_nsight=False,
    )
    instance.vllm_config = SimpleNamespace(
        parallel_config=instance.parallel_config,
        model_config=SimpleNamespace(
            model="/cache/model",
            model_weights="gs://bucket/model",
            runner_type="generate",
        ),
        kv_transfer_config=None,
        ec_transfer_config=None,
    )
    instance.scheduler_config = SimpleNamespace(async_scheduling=True)
    instance.collective_rpc = Mock()
    instance.use_ray_spmd_worker = True
    instance.kv_output_aggregator = None
    instance.forward_dag = None
    instance.workers = []
    for key in (
        "TORCH_TPU_BASE_PORT",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_XPROF_SESSION_ID",
        "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE",
        "RAY_USAGE_STATS_ENABLED",
    ):
        tpu_envs.__dict__.pop(key, None)
        monkeypatch.delenv(key, raising=False)
    return instance


def test_ray_v1_remaining_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = _make_v1_executor(monkeypatch)
    backend = MagicMock()
    backend.is_initialized.return_value = True
    backend.get_runtime_context.return_value.get_node_id.return_value = "node0"
    backend.nodes.return_value = [
        {"NodeID": "node0", "Resources": {"TPU": 2}},
        {"NodeID": "node1", "Resources": {"TPU": 2}},
    ]
    platform = SimpleNamespace(
        ray_device_key="TPU",
        device_name="tpu",
        additional_env_vars=[],
        device_control_id_to_physical_device_id=int,
    )
    monkeypatch.setattr(ray_executor_v1, "ray", backend)
    monkeypatch.setattr(ray_executor_v1, "current_platform", platform)
    monkeypatch.setattr(platforms, "current_platform", platform)
    monkeypatch.setattr(ray_executor_v1, "get_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(ray_executor_v1, "get_open_port", lambda: 9000)
    monkeypatch.setattr(
        ray_executor_v1, "get_tpu_multihost_topology", Mock(return_value="1,2,2,1")
    )
    monkeypatch.setattr(ray_executor_v1, "_wait_until_pg_ready", Mock())
    monkeypatch.setenv("VLLM_RAY_BUNDLE_INDICES", "")

    # 1. RAY_USAGE_STATS_ENABLED == "1" branch in _init_executor
    monkeypatch.setenv("RAY_USAGE_STATS_ENABLED", "1")
    tpu_envs.__dict__.pop("RAY_USAGE_STATS_ENABLED", None)
    executor._initialize_ray_cluster = Mock()
    executor._init_workers_ray = Mock()
    executor._init_executor()
    assert os.environ["RAY_USAGE_STATS_ENABLED"] == "1"

    # 2. get_kv_connector_handshake_metadata returns None
    assert executor.get_kv_connector_handshake_metadata() is None

    # 3. _init_workers_ray with:
    #    - ray_workers_use_nsight=True
    #    - n_nodes != n_ips warning (2 nodes but 1 unique IP)
    #    - TORCH_TPU_XPROF_SESSION_ID not in os.environ (auto-generated)
    #    - env_vars_to_copy containing both present and missing env vars
    #    - use_ray_spmd_worker=False
    executor2 = _make_v1_executor(monkeypatch)
    executor2.parallel_config.ray_workers_use_nsight = True
    executor2.use_ray_spmd_worker = False
    executor2._configure_ray_workers_use_nsight = Mock(
        side_effect=lambda kwargs: {**kwargs, "runtime_env": {"nsight": "on"}}
    )
    pg = SimpleNamespace(bundle_specs=[{"TPU": 1}] * 4)
    actors = [MagicMock(name=f"worker{i}") for i in range(4)]
    backend.remote.return_value.return_value.remote.side_effect = actors
    # All workers report same IP "10.0.0.1" (matching driver_ip) -> n_ips == 1,
    # while node_ids are "node0" and "node1" -> n_nodes == 2 (triggers n_nodes != n_ips)
    backend.get.side_effect = [
        ["10.0.0.1", "10.0.0.1", "10.0.0.1", "10.0.0.1"],
        ("node0", [0]),
        ("node0", [1]),
        ("node1", [0]),
        ("node1", [1]),
    ]
    monkeypatch.setenv("PRESENT_ENV_VAR", "val1")
    monkeypatch.delenv("MISSING_ENV_VAR", raising=False)
    monkeypatch.setattr(
        ray_executor_v1,
        "get_env_vars_to_copy",
        Mock(return_value={"PRESENT_ENV_VAR", "MISSING_ENV_VAR"}),
    )

    executor2._init_workers_ray(pg)
    executor2._configure_ray_workers_use_nsight.assert_called_once()
    assert "TORCH_TPU_XPROF_SESSION_ID" in os.environ
    assert executor2.pp_tp_workers == []
    for env in executor2._env_vars_for_all_workers:
        assert env["PRESENT_ENV_VAR"] == "val1"
        assert "MISSING_ENV_VAR" not in env


def _make_v2_executor() -> RayDistributedExecutorV2:
    executor = object.__new__(RayDistributedExecutorV2)
    executor.shutdown = lambda: None
    pc = SimpleNamespace(
        world_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        ray_workers_use_nsight=False,
        placement_group=None,
        max_parallel_loading_workers=None,
        nnodes_within_dp=1,
        local_world_size=2,
        ray_runtime_env={},
        data_parallel_rank_local=None,
        data_parallel_master_port=0,
        data_parallel_master_ip="127.0.0.1",
        data_parallel_size=1,
        data_parallel_rank=0,
        data_parallel_index=None,
    )
    cache_cfg = MagicMock(spec=CacheConfig)
    cache_cfg.num_gpu_blocks_override = None
    vllm_cfg = SimpleNamespace(
        parallel_config=pc,
        model_config=MagicMock(),
        cache_config=cache_cfg,
        kv_transfer_config=None,
        instance_id="vllm-td874",
    )
    executor.vllm_config = vllm_cfg
    executor.parallel_config = pc
    executor.world_size = 2
    return executor


def test_ray_v2_dp_geometry_and_slice_host_layout_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _make_v2_executor()
    executor.parallel_config.data_parallel_size = 2

    # 1. data_parallel_index is None, falls back to data_parallel_rank
    executor.parallel_config.data_parallel_index = None
    executor.parallel_config.data_parallel_rank = 1
    assert executor._get_dp_geometry() == (2, 1, 4)

    # 2. Both data_parallel_index and data_parallel_rank are None -> ValueError
    executor.parallel_config.data_parallel_rank = None
    with pytest.raises(
        ValueError, match="ParallelConfig.data_parallel_index must be resolved"
    ):
        executor._get_dp_geometry()

    # 3. _slice_host_layout: node without NodeManagerAddress or node:<ip> resource
    with pytest.raises(RuntimeError, match="no address to reach it by"):
        executor._slice_host_layout(
            "TPU",
            ray_nodes=[{"NodeID": "n0", "Resources": {"TPU": 4.0}}],
        )

    # 4. _slice_host_layout: node with node:<ip> resource fallback & invalid
    #    worker-id label on single host (covers except block and len(hosts) == 1)
    host_order, chips = executor._slice_host_layout(
        "TPU",
        ray_nodes=[
            {
                "NodeID": "n0",
                "Resources": {"TPU": 4.0, "node:10.0.0.5": 1.0},
                "Labels": {"ray.io/tpu-worker-id": "invalid_int"},
            }
        ],
    )
    assert host_order == ["10.0.0.5"]
    assert chips == {"10.0.0.5": 4}

    # 5. _slice_host_layout: empty hosts list (no nodes with TPU)
    empty_order, empty_chips = executor._slice_host_layout(
        "TPU",
        ray_nodes=[{"NodeID": "cpu_only", "Resources": {"CPU": 8.0}}],
    )
    assert empty_order == []
    assert empty_chips == {}

    # 6. get_tpu_bundles_for_indices with bundle_to_node_id=None
    monkeypatch.setattr(
        ray_executor_v2,
        "get_bundles_sorted_by_node",
        lambda pg: [(0, "node_0", "10.0.0.1"), (1, "node_1", "10.0.0.2")],
    )
    assert get_tpu_bundles_for_indices("dummy_pg", [1]) == [(1, "node_1", "10.0.0.2")]


def test_ray_v2_cluster_and_init_executor_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _make_v2_executor()
    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True
    mock_ray.nodes.return_value = [{"NodeID": "other", "Resources": {"TPU": 4.0}}]
    mock_ray.get_runtime_context.return_value.get_node_id.return_value = "missing_node"
    platform = SimpleNamespace(ray_device_key="TPU", device_name="tpu")
    monkeypatch.setattr(ray_executor_v2, "ray", mock_ray)
    monkeypatch.setattr(ray_executor_v2, "current_platform", platform)
    monkeypatch.setattr(ray_executor_v2, "get_ip", lambda: "10.0.0.1")

    # 1. _initialize_ray_cluster: current node has no TPU available
    with pytest.raises(ValueError, match="Current node has no TPU available"):
        executor._initialize_ray_cluster()

    # 2. _initialize_ray_cluster: platform does not support ray
    platform.ray_device_key = None
    with pytest.raises(ValueError, match="does not support ray"):
        executor._initialize_ray_cluster()
    platform.ray_device_key = "TPU"

    # 3. _init_executor: ray is None -> ImportError
    monkeypatch.setattr(ray_executor_v2, "ray", None)
    with pytest.raises(ImportError, match="requires installation of ray"):
        executor._init_executor()
    monkeypatch.setattr(ray_executor_v2, "ray", mock_ray)

    # 4. _init_executor: dp_size > 1 and pcp_size > 1 -> NotImplementedError
    executor.parallel_config.data_parallel_size = 2
    executor.parallel_config.data_parallel_index = 0
    executor.parallel_config.prefill_context_parallel_size = 2
    executor.world_size = 4
    executor._get_parallel_sizes = Mock(return_value=(2, 1, 2))
    with pytest.raises(
        NotImplementedError, match="Prefill context parallelism is not supported"
    ):
        executor._init_executor()


def _run_v2_init_executor_scenario(
    monkeypatch: pytest.MonkeyPatch,
    *,
    worker_node_and_tpus: list[tuple[str, list[int]]],
    dp_size: int = 1,
    cluster_chips: int = 2,
    bundle_indices_env: str = "",
    init_status: str = "READY",
    has_connector: bool = False,
) -> RayDistributedExecutorV2:
    executor = _make_v2_executor()
    executor.parallel_config.data_parallel_size = dp_size
    executor.parallel_config.data_parallel_index = 0
    executor.parallel_config.placement_group = SimpleNamespace(
        bundle_specs=[{"TPU": 1.0}, {"TPU": 1.0}]
    )
    if has_connector:
        executor.vllm_config.kv_transfer_config = object()
    executor._get_parallel_sizes = Mock(return_value=(2, 1, 1))
    executor._is_driver_worker = Mock(side_effect=lambda r: r == 0)
    executor._build_runtime_env = Mock(return_value={})
    executor._post_init_executor = Mock()
    executor.start_worker_monitor = Mock()
    executor._get_output_rank = Mock(return_value=0)
    executor.collective_rpc = Mock(return_value=["10.0.0.1:8100"])

    monkeypatch.setenv("VLLM_RAY_BUNDLE_INDICES", bundle_indices_env)
    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True
    mock_ray.get_runtime_context.return_value.get_node_id.return_value = "node_1"
    mock_ray.nodes.return_value = [
        {
            "NodeID": "node_1",
            "NodeManagerAddress": "10.0.0.1",
            "Resources": {"TPU": float(cluster_chips)},
        }
    ]
    actors = [MagicMock(), MagicMock()]
    mock_remote_class = MagicMock()
    mock_remote_class.options.return_value.remote.side_effect = actors
    mock_ray.remote.return_value = mock_remote_class
    mock_ray.get.side_effect = [
        worker_node_and_tpus,
        "tcp://10.0.0.1:9000",
        [None, None],
        [
            {"status": init_status, "handle": "h0"},
            {"status": init_status, "handle": "h1"},
        ],
    ]

    platform = SimpleNamespace(ray_device_key="TPU", device_name="tpu")
    monkeypatch.setattr(ray_executor_v2, "ray", mock_ray)
    monkeypatch.setattr(ray_executor_v2, "current_platform", platform)
    monkeypatch.setattr(ray_executor_v2, "get_driver_env_vars", lambda **kw: {})
    monkeypatch.setattr(ray_executor_v2, "get_open_port", lambda: 9000)
    monkeypatch.setattr(ray_executor_v2, "get_tpu_multihost_topology", lambda n: "2x1")
    monkeypatch.setattr(
        ray_executor_v2,
        "get_bundles_sorted_by_node",
        lambda pg: [(0, "node_1", "10.0.0.1"), (1, "node_1", "10.0.0.1")],
    )
    mock_mq = MagicMock()
    monkeypatch.setattr(ray_executor_v2, "MessageQueue", mock_mq)
    try:
        executor._init_executor()
    finally:
        executor.ray_worker_handles = []
    return executor


def test_ray_v2_init_executor_validation_and_connector_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. dp_size > 1 and sum(chips_per_host.values()) != slice_world_size
    with pytest.raises(ValueError, match="Cluster reports 2 chips across 1 hosts"):
        _run_v2_init_executor_scenario(
            monkeypatch,
            worker_node_and_tpus=[("node_1", [0]), ("node_1", [1])],
            dp_size=2,
            cluster_chips=2,
        )

    # 2. node_id != assignment["node_id"]
    with pytest.raises(RuntimeError, match="started on Ray node wrong_node"):
        _run_v2_init_executor_scenario(
            monkeypatch,
            worker_node_and_tpus=[("wrong_node", [0]), ("node_1", [1])],
        )

    # 3. len(physical_ids) != 1
    with pytest.raises(RuntimeError, match="requires exactly one chip per worker"):
        _run_v2_init_executor_scenario(
            monkeypatch,
            worker_node_and_tpus=[("node_1", [0, 1]), ("node_1", [1])],
        )

    # 4. slice_local_rank out of bounds (not 0 <= slice_local_rank < local_world_size)
    with pytest.raises(RuntimeError, match="got chip 5 on 10.0.0.1"):
        _run_v2_init_executor_scenario(
            monkeypatch,
            worker_node_and_tpus=[("node_1", [5]), ("node_1", [1])],
        )

    # 5. Worker failed to initialize (status != READY_STR)
    with pytest.raises(RuntimeError, match="Worker 0 failed to initialize"):
        _run_v2_init_executor_scenario(
            monkeypatch,
            worker_node_and_tpus=[("node_1", [0]), ("node_1", [1])],
            init_status="FAILED",
        )

    # 6. VLLM_RAY_BUNDLE_INDICES fallback + has_connector=True
    registered: list[str] = []
    monkeypatch.setattr(
        ray_executor_v2, "set_node_kv_ip_port", lambda item: registered.append(item)
    )
    executor = _run_v2_init_executor_scenario(
        monkeypatch,
        worker_node_and_tpus=[("node_1", [0]), ("node_1", [1])],
        bundle_indices_env="0",
        has_connector=True,
    )
    assert executor.has_connector is True
    assert registered == ["10.0.0.1:8100"]
    assert executor.vllm_config.parallel_config.assigned_physical_gpu_ids == [0, 1]


def test_ray_v2_kv_cache_specs_and_initialize_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _make_v2_executor()
    sentinel_specs = [{"layer0": MagicMock()}]
    monkeypatch.setattr(
        ray_executor_v2.RayExecutorV2,
        "get_kv_cache_specs",
        lambda self: sentinel_specs,
    )
    monkeypatch.setattr(
        ray_executor_v2.RayExecutorV2,
        "initialize_from_config",
        lambda self, cfg: None,
    )

    # 1. num_gpu_blocks_override is None and reconcile returns None
    executor.collective_rpc = Mock(return_value=[None, None])
    monkeypatch.setattr(
        ray_executor_v2, "reconcile_num_gpu_blocks_override", lambda vals, **kw: None
    )
    assert executor.get_kv_cache_specs() is sentinel_specs
    assert executor.vllm_config.cache_config.num_gpu_blocks_override is None

    # 2. num_gpu_blocks_override is None and reconcile returns agreed value
    monkeypatch.setattr(
        ray_executor_v2, "reconcile_num_gpu_blocks_override", lambda vals, **kw: 256
    )
    assert executor.get_kv_cache_specs() is sentinel_specs
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 256

    # 3. num_gpu_blocks_override is already set -> skips reconcile
    executor.collective_rpc.reset_mock()
    assert executor.get_kv_cache_specs() is sentinel_specs
    executor.collective_rpc.assert_not_called()

    # 4. initialize_from_config: empty shapes -> early return
    executor.collective_rpc = Mock(return_value=[[]])
    executor.initialize_from_config([])
    executor.collective_rpc.assert_called_once_with("get_kv_prewarm_shapes")

    # 5. initialize_from_config: non-empty shapes -> prewarm each shape
    executor.collective_rpc = Mock(return_value=[[(4, 128), (8, 128)]])
    executor.initialize_from_config([])
    assert executor.collective_rpc.call_args_list == [
        call("get_kv_prewarm_shapes"),
        call("prewarm_kv_offload_shape", args=((4, 128),)),
        call("prewarm_kv_offload_shape", args=((8, 128),)),
    ]


def test_v1_and_v2_engram_full_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    # V1 write_cache
    v1_writes: list[tuple] = []
    monkeypatch.setattr(
        v1_engram, "write_cache", lambda c, s, v, val: v1_writes.append((c, s, v, val))
    )
    state_v1 = object.__new__(v1_engram.NgramHashState)
    c_t = torch.zeros(4, dtype=torch.int32)
    s_t = torch.zeros(2, dtype=torch.int32)
    v_t = torch.ones(2, dtype=torch.int32)
    val_t = torch.tensor([True, False])
    state_v1.write_cache(c_t, s_t, v_t, val_t)
    assert len(v1_writes) == 1

    # V1 ensure_cache when is_compiling() is True
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        v1_engram.NgramHashState.__bases__[0], "ensure_cache", lambda self: True
    )
    assert state_v1.ensure_cache() is True

    # V1 ensure_cache when is_compiling() is False
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    state_v1._cache = torch.zeros(4, dtype=torch.int32)
    state_v1._kv_cache_ref = None
    state_v1.use_slot_cache = True
    state_v1.block_size = 8
    state_v1.swa_cache_module = SimpleNamespace(
        bound_cache=torch.empty(0, dtype=torch.int32),
        block_size=8,
    )
    assert state_v1.ensure_cache() is False
    assert state_v1._cache is None
    assert state_v1._kv_cache_ref is None

    bound = torch.zeros((4, 8), dtype=torch.int32)
    state_v1.swa_cache_module = SimpleNamespace(bound_cache=bound, block_size=8)
    state_v1.use_slot_cache = False
    assert state_v1.ensure_cache() is True

    state_v1.use_slot_cache = True
    assert state_v1.ensure_cache() is True
    assert state_v1._cache is not None and state_v1._cache.shape == (32,)
    cached_ref = state_v1._cache
    assert state_v1.ensure_cache() is True
    assert state_v1._cache is cached_ref

    state_v1.swa_cache_module.block_size = 16
    assert state_v1.ensure_cache() is True
    assert state_v1.block_size == 16
    assert state_v1._cache is not None and state_v1._cache.shape == (64,)

    # V2 write_cache and exports
    v2_writes: list[tuple] = []
    monkeypatch.setattr(
        v2_engram, "write_cache", lambda c, s, v, val: v2_writes.append((c, s, v, val))
    )
    state_v2 = object.__new__(v2_engram.NgramHashState)
    state_v2.write_cache(c_t, s_t, v_t, val_t)
    assert len(v2_writes) == 1
    assert v2_engram.Engram is common_engram.Engram
