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
"""Full-module unit tests for TD7.92 distributed modules.

Covers 100% of statements and branches in:
- `src/vllm_torchtpu/distributed/utils.py`
- `src/vllm_torchtpu/distributed/tpu_mp_multihost.py`
- `src/vllm_torchtpu/distributed/pp_push.py`
- `src/vllm_torchtpu/distributed/mesh_utils.py`
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import Future
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import jax
import pytest
import torch

from vllm_torchtpu.distributed import (
    mesh_utils,
    pp_push,
    utils,
)
from vllm_torchtpu.distributed import (
    tpu_mp_multihost as mp,
)
from vllm_torchtpu.platforms import tpu_platform

pytestmark = pytest.mark.cpu_test


def test_distributed_utils_full_coverage(monkeypatch):
    """Exercise every statement and branch in distributed/utils.py."""
    monkeypatch.setattr(utils, "_NODES_KV_IP_PORT", {})
    monkeypatch.setattr(utils, "_NAMESPACE_CACHE", ("", ""))

    # Multi-host KV IP/port registry and getters (Ray vs single-host)
    utils.set_node_kv_ip_port((1, "10.0.0.2", 9101))
    utils.set_node_kv_ip_port((0, "10.0.0.1", 9100))

    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    assert utils.get_kv_ips() == ["10.0.0.1", "10.0.0.2"]
    assert utils.get_kv_ports() == [9100, 9101]

    monkeypatch.delenv("TPU_MULTIHOST_BACKEND", raising=False)
    monkeypatch.setattr(utils, "get_ip", lambda: "192.168.1.10")
    monkeypatch.setenv("TPU_KV_TRANSFER_PORT", "9200")
    monkeypatch.setenv("TPU_SIDE_CHANNEL_PORT", "9300")
    monkeypatch.setenv("TPU_NODE_ID", "3")
    assert utils.get_host_ip() == "192.168.1.10"
    assert utils.get_kv_ips() == "192.168.1.10"
    assert utils.get_kv_transfer_port() == "9200"
    assert utils.get_kv_ports() == "9200"
    assert utils.get_side_channel_port() == "9300"
    assert utils.get_node_id() == 3

    # Executor & channel & timeout getters
    monkeypatch.setenv("TPU_KV_TRANSFER_CHANNEL_NUMBER", "4")
    monkeypatch.setenv("TPU_KV_COORD_EXECUTOR_MAX_WORKERS", "8")
    monkeypatch.setenv("TPU_KV_CHANNEL_EXECUTOR_MAX_WORKERS", "16")
    monkeypatch.setenv("TPU_P2P_WAIT_PULL_TIMEOUT", "100")
    assert utils.get_transfer_channel_number() == 4
    assert utils.get_kv_coord_executor_max_workers() == 8
    assert utils.get_kv_channel_executor_max_workers() == 16
    assert utils.get_p2p_wait_pull_timeout() == 100
    assert utils.get_raiden_registry_ttl_s() == 160.0

    monkeypatch.setenv("TPU_P2P_WAIT_PULL_TIMEOUT", "1000")
    assert utils.get_raiden_registry_ttl_s() == 1100.0

    # Stage 3 & KV staging getters
    monkeypatch.setenv("TPU_RAIDEN_STAGE3_STATUS_PROBE_S", "-5.0")
    assert utils.get_stage3_status_probe_s() == 0.0
    monkeypatch.setenv("TPU_RAIDEN_STAGE3_STATUS_PROBE_S", "2.5")
    assert utils.get_stage3_status_probe_s() == 2.5

    monkeypatch.setenv("TPU_RAIDEN_INLINE_LOAD", "1")
    assert utils.get_raiden_inline_load() is True
    assert utils.get_stage3_deferred_submit_enabled() is False

    monkeypatch.setenv("TPU_RAIDEN_INLINE_LOAD", "0")
    monkeypatch.setenv("TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT", "1")
    assert utils.get_stage3_deferred_submit_enabled() is True

    monkeypatch.setenv("TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S", "0")
    monkeypatch.setenv("TPU_P2P_WAIT_PULL_TIMEOUT", "120")
    assert utils.get_stage3_registration_wait_s() == 30.0
    monkeypatch.setenv("TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S", "45.0")
    assert utils.get_stage3_registration_wait_s() == 45.0

    monkeypatch.setenv("TPU_RAIDEN_TEST_REGISTRATION_DELAY_S", "-1.0")
    assert utils.get_stage3_test_registration_delay_s() == 0.0
    monkeypatch.setenv("TPU_RAIDEN_TEST_REGISTRATION_DELAY_S", "1.5")
    assert utils.get_stage3_test_registration_delay_s() == 1.5

    monkeypatch.setenv("TPU_KV_STAGE_WAIT_TIMEOUT_SECS", "0")
    assert utils.get_kv_stage_wait_timeout_secs() == 30.0
    monkeypatch.setenv("TPU_KV_STAGE_WAIT_TIMEOUT_SECS", "12.0")
    assert utils.get_kv_stage_wait_timeout_secs() == 12.0

    monkeypatch.setenv("TPU_KV_STAGE_WAITER_POOL_SIZE", "6")
    monkeypatch.setenv("TPU_KV_SHM_POOL_GB", "8.5")
    assert utils.get_kv_stage_waiter_pool_size() == 6
    assert utils.get_kv_shm_pool_gb() == 8.5

    # Namespace sanitization & 1-entry cache hit/miss
    monkeypatch.delenv("TPU_KV_TRANSFER_NAMESPACE", raising=False)
    assert utils.get_ipc_socket_path(0) == "ipc:///tmp/tpu_conn_node0_0.sock"
    assert utils.get_shm_name(0) == "tpu_conn_kv_node0_0"

    monkeypatch.setenv("TPU_KV_TRANSFER_NAMESPACE", "decode:9400-a_1")
    assert (
        utils.get_ipc_socket_path(2, 1)
        == "ipc:///tmp/tpu_conn_decode_9400-a_1_node2_1.sock"
    )
    # Second call hits _NAMESPACE_CACHE
    assert utils.get_shm_name(2, 1) == "tpu_conn_kv_decode_9400-a_1_node2_1"

    # Remaining boolean/int/float config getters
    monkeypatch.setenv("TPU_KV_WARMUP_ENABLED", "1")
    monkeypatch.setenv("TPU_KV_LATENCY_LOG_INTERVAL", "5.0")
    monkeypatch.setenv("TPU_KV_PIN_SHM", "1")
    monkeypatch.setenv("TPU_USE_RAIDEN_CONNECTOR", "1")
    monkeypatch.setenv("TPU_RAIDEN_TRANSFER_NUM_SLOTS", "12")
    monkeypatch.setenv("TPU_RAIDEN_POOL_Staging_LEASES", "2")
    monkeypatch.setenv("TPU_RAIDEN_POOL_STAGING_LEASES", "2")
    assert utils.get_kv_warmup_enabled() is True
    assert utils.get_kv_latency_log_interval() == 5.0
    assert utils.get_kv_pin_shm() is True
    assert utils.get_use_raiden_connector() is True
    assert utils.get_raiden_transfer_num_slots() == 12
    assert utils.get_raiden_pool_staging_leases() == 2

    # Raiden telemetry module import & configuration branches
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_MODULE", None)
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_CONFIGURED", False)
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_IMPORT_ATTEMPTED", False)

    # 1) Import failure path
    monkeypatch.setitem(sys.modules, "tpu_sync", None)
    monkeypatch.setitem(sys.modules, "tpu_sync.api", None)
    monkeypatch.setitem(sys.modules, "tpu_sync.api.torch", None)
    assert utils.get_raiden_telemetry_module() is None
    # Subsequent call when _RAIDEN_TELEMETRY_IMPORT_ATTEMPTED is True
    assert utils.get_raiden_telemetry_module() is None
    # configure_raiden_telemetry when telemetry is None
    utils.configure_raiden_telemetry([" sponge"])
    assert utils._RAIDEN_TELEMETRY_CONFIGURED is False

    # 2) Import success path + configure_telemetry success & failure
    fake_telemetry = SimpleNamespace()
    configured_args: list = []
    fake_telemetry.configure_telemetry = lambda b: configured_args.append(b)

    fake_kcm = ModuleType("tpu_sync.api.torch.kv_cache_manager")
    fake_kcm._torch_impl = lambda: fake_telemetry  # type: ignore[attr-defined]
    fake_torch_pkg = ModuleType("tpu_sync.api.torch")
    fake_torch_pkg.kv_cache_manager = fake_kcm  # type: ignore[attr-defined]
    fake_api_pkg = ModuleType("tpu_sync.api")
    fake_api_pkg.torch = fake_torch_pkg  # type: ignore[attr-defined]
    fake_root_pkg = ModuleType("tpu_sync")
    fake_root_pkg.api = fake_api_pkg  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "tpu_sync", fake_root_pkg)
    monkeypatch.setitem(sys.modules, "tpu_sync.api", fake_api_pkg)
    monkeypatch.setitem(sys.modules, "tpu_sync.api.torch", fake_torch_pkg)
    monkeypatch.setitem(sys.modules, "tpu_sync.api.torch.kv_cache_manager", fake_kcm)
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_MODULE", None)
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_IMPORT_ATTEMPTED", False)

    assert utils.get_raiden_telemetry_module() is fake_telemetry
    utils.configure_raiden_telemetry(["console"])
    assert configured_args == [["console"]]
    assert utils._RAIDEN_TELEMETRY_CONFIGURED is True

    # 3) configure_raiden_telemetry without configure_telemetry attribute
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_MODULE", SimpleNamespace())
    utils.configure_raiden_telemetry(["none"])

    # 4) configure_raiden_telemetry when exception is raised
    bad_telemetry = SimpleNamespace(
        configure_telemetry=Mock(side_effect=RuntimeError("telemetry boom"))
    )
    monkeypatch.setattr(utils, "_RAIDEN_TELEMETRY_MODULE", bad_telemetry)
    utils.configure_raiden_telemetry(["err"])


def test_tpu_mp_multihost_full_coverage(monkeypatch):
    """Exercise every statement and branch in distributed/tpu_mp_multihost.py."""
    for key in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCH_TPU_MP_RENDEZVOUS_PORT",
        "TORCH_TPU_BASE_PORT",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
        "TORCH_TPU_XPROF_SESSION_ID",
        "TPU_NUM_HOSTS",
        "NODE_RANK",
        mp._DONE_SENTINEL,
        mp._DP_DONE_SENTINEL,
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(mp, "_keepalive_stores", [])

    # _rendezvous_port default and override
    assert mp._rendezvous_port(9000) == 9001
    monkeypatch.setenv("TORCH_TPU_MP_RENDEZVOUS_PORT", "9123")
    assert mp._rendezvous_port(9000) == 9123
    monkeypatch.delenv("TORCH_TPU_MP_RENDEZVOUS_PORT")

    # _rendezvous_host_ips for master (rank 0) and worker (rank 1)
    values = {
        "pfx_ip_0": b"10.0.0.1",
        "pfx_ip_1": b"10.0.0.2",
        "pfx_xprof_session_id": b"99999",
    }
    store = Mock()
    store.get.side_effect = values.__getitem__
    create_store = Mock(return_value=store)
    monkeypatch.setattr(mp.dist, "TCPStore", create_store)
    monkeypatch.setattr(mp.time, "time_ns", lambda: 99999)

    for rank in (0, 1):
        monkeypatch.setattr(mp, "get_ip", lambda r=rank: f"10.0.0.{r + 1}")
        ips, sess = mp._rendezvous_host_ips(
            num_peers=2,
            peer_index=rank,
            master_addr="10.0.0.1",
            port=9001,
            key_prefix="pfx",
        )
        assert ips == ["10.0.0.1", "10.0.0.2"]
        assert sess == "99999"
    assert mp._keepalive_stores == [store]

    # prepare_mp_multihost_env: nnodes <= 1, normal bootstrap, and idempotent re-entry
    rendezvous = Mock(return_value=(["10.0.0.1", "10.0.0.2"], "sess_id"))
    topology = Mock(return_value="2,2,2")
    monkeypatch.setattr(mp, "_rendezvous_host_ips", rendezvous)
    monkeypatch.setattr(tpu_platform, "get_tpu_multihost_topology", topology)

    mp.prepare_mp_multihost_env(SimpleNamespace(nnodes=1))
    rendezvous.assert_not_called()

    cfg = SimpleNamespace(
        nnodes=2,
        node_rank=0,
        local_world_size=2,
        master_addr="10.0.0.1",
        master_port=9000,
    )
    mp.prepare_mp_multihost_env(cfg)
    assert os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == (
        "10.0.0.1:8070,10.0.0.1:8071,10.0.0.2:8070,10.0.0.2:8071"
    )
    assert os.environ[mp._DONE_SENTINEL] == "2"
    # Re-entry is a no-op
    mp.prepare_mp_multihost_env(cfg)
    assert rendezvous.call_count == 1

    # prepare_mp_multihost_dp_env: single-host fallback, invalid partition, & re-entry
    assert not mp.prepare_mp_multihost_dp_env(
        SimpleNamespace(data_parallel_size=4, data_parallel_size_local=0), 8
    )
    assert not mp.prepare_mp_multihost_dp_env(
        SimpleNamespace(data_parallel_size=4, data_parallel_size_local=4), 8
    )
    with pytest.raises(ValueError, match="must be a multiple"):
        mp.prepare_mp_multihost_dp_env(
            SimpleNamespace(data_parallel_size=5, data_parallel_size_local=2), 10
        )

    dp_cfg = SimpleNamespace(
        data_parallel_size=4,
        data_parallel_size_local=2,
        data_parallel_rank=2,
        data_parallel_rpc_port=9500,
        data_parallel_master_ip="10.0.0.1",
        world_size=2,
    )
    assert mp.prepare_mp_multihost_dp_env(dp_cfg, 8) is True
    assert dp_cfg._data_parallel_master_port_list == [9600, 9601, 9602, 9603, 9604]
    assert os.environ[mp._DP_DONE_SENTINEL] == "4"
    assert (
        call(
            num_peers=2,
            peer_index=1,
            master_addr="10.0.0.1",
            port=9501,
            key_prefix="tpu_mp_dp",
        )
        in rendezvous.call_args_list
    )
    # Re-entry updates _data_parallel_master_port_list without re-rendezvousing
    dp_cfg._data_parallel_master_port_list = []
    assert mp.prepare_mp_multihost_dp_env(dp_cfg, 8) is True
    assert dp_cfg._data_parallel_master_port_list == [9600, 9601, 9602, 9603, 9604]
    assert rendezvous.call_count == 2


def test_pp_push_and_mesh_utils_full_coverage(monkeypatch):
    """Exercise every statement and branch in pp_push.py and mesh_utils.py."""
    from torch_tpu._internal.distributed import tpu_distributed
    from vllm.distributed.device_communicators import shm_broadcast
    from vllm.v1.executor import multiproc_executor

    # --- pp_push.py ---
    monkeypatch.setattr(pp_push, "_ring_chunks_needed", 0)

    class _Queue:
        def __init__(self, n_reader, n_local_reader, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(shm_broadcast, "MessageQueue", _Queue)

    pp_push.widen_message_rings(None)
    cfg1 = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1))
    cfg4 = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=4))
    pp_push.widen_message_rings(cfg1)
    pp_push.widen_message_rings(cfg4)
    pp_push.widen_message_rings(cfg4)
    assert _Queue(1, 1).kwargs["max_chunks"] == 24

    exec_stub = SimpleNamespace(
        output_rank=3,
        calls=[],
        collective_rpc=lambda method, **kw: exec_stub.calls.append((method, kw)),
    )
    wave = pp_push._WaveState(stages=4)
    wave.before_waiting(exec_stub, 1, 1)

    wave.settles = 5
    s1 = wave.dispatched()
    wave.before_waiting(exec_stub, s1, wave.real_steps)
    assert wave.settles == 6

    wave.pushes = 5
    s2 = wave.dispatched()
    s3 = wave.dispatched()
    wave.before_waiting(exec_stub, s2, s2)
    assert wave.pushes == 6
    wave.before_waiting(exec_stub, s2, s2)

    wave.settles = 0
    wave.before_waiting(exec_stub, s3, wave.real_steps)
    assert wave.settles == 1

    wave.pushes = 0
    s4 = wave.dispatched()
    r4 = wave.real_steps
    _ = wave.dispatched()
    wave.before_waiting(exec_stub, s4, r4)
    assert wave.pushes == 1

    f_ok = Future()
    pf = pp_push._PushFuture(f_ok, exec_stub, pp_push._WaveState(stages=2), 1, 1)
    assert isinstance(pf, Future)
    assert not hasattr(wave, "__dict__")
    seen = []
    pf.add_done_callback(seen.append)
    assert not pf.done() and not pf.running()
    f_ok.set_result("val")
    assert pf.done() and seen == [pf]
    assert pf.result() == "val"
    assert pf.exception() is None

    f_err = Future()
    f_err.set_exception(ValueError("err"))
    pf_err = pp_push._PushFuture(f_err, exec_stub, pp_push._WaveState(stages=2), 1, 1)
    assert isinstance(pf_err.exception(), ValueError)

    f_cancel = Future()
    pf_cancel = pp_push._PushFuture(
        f_cancel, exec_stub, pp_push._WaveState(stages=2), 1, 1
    )
    assert pf_cancel.cancel() and pf_cancel.cancelled()

    pp_push.patch_executor_for_pp_wave(None)
    pp_push.patch_executor_for_pp_wave(cfg1)

    class _FakeMultiprocExecutor:
        def __init__(self):
            self.vllm_config = cfg4
            self.output_rank = 3
            self.calls = []

        def execute_model(self, scheduler_output, non_block=False):
            fut = Future()
            fut.set_result(("exec", scheduler_output.total_num_scheduled_tokens))
            return fut if non_block else fut.result()

        def sample_tokens(self, grammar_output, non_block=False):
            fut = Future()
            fut.set_result(("sample", grammar_output))
            return fut if non_block else fut.result()

        def collective_rpc(self, method, **kwargs):
            self.calls.append((method, kwargs))

    monkeypatch.setattr(multiproc_executor, "MultiprocExecutor", _FakeMultiprocExecutor)
    pp_push.patch_executor_for_pp_wave(cfg4)
    pp_push.patch_executor_for_pp_wave(cfg4)

    ex = _FakeMultiprocExecutor()
    assert ex.execute_model(SimpleNamespace(total_num_scheduled_tokens=0)) == (
        "exec",
        0,
    )
    assert ex.execute_model(SimpleNamespace(total_num_scheduled_tokens=8)) == (
        "exec",
        8,
    )
    fut_exec = ex.execute_model(
        SimpleNamespace(total_num_scheduled_tokens=8), non_block=True
    )
    assert isinstance(fut_exec, pp_push._PushFuture)
    fut_sample = ex.sample_tokens("g", non_block=True)
    assert isinstance(fut_sample, pp_push._PushFuture)
    assert ex.sample_tokens("g2", non_block=False) == ("sample", "g2")

    # --- mesh_utils.py ---
    monkeypatch.setattr(mesh_utils, "_LAYOUT_CACHE", {})
    monkeypatch.setattr(mesh_utils, "_MESH_CACHE", {})

    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    assert mesh_utils._get_current_global_rank() == 0
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    assert mesh_utils._get_current_global_rank() == 3

    monkeypatch.setattr(tpu_distributed, "global_device_id", lambda: 5)
    assert mesh_utils._get_tpu_global_device_id() == 5
    monkeypatch.setattr(
        tpu_distributed, "global_device_id", Mock(side_effect=RuntimeError)
    )
    monkeypatch.setattr(jax, "local_devices", lambda: [SimpleNamespace(id=11)])
    assert mesh_utils._get_tpu_global_device_id() == 11
    monkeypatch.setattr(jax, "local_devices", lambda: [])
    assert mesh_utils._get_tpu_global_device_id() == 0
    monkeypatch.setattr(jax, "local_devices", Mock(side_effect=RuntimeError))
    assert mesh_utils._get_tpu_global_device_id() == 0

    grp = SimpleNamespace(
        world_size=2, cpu_group=object(), ranks=(0, 1), rank_in_group=1
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, val, group: out.__setitem__(slice(None), [(0, 10), None]),
    )
    with pytest.raises(RuntimeError, match="empty entry"):
        mesh_utils._collect_rank_to_device_id(grp, 1, 20)

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, val, group: out.__setitem__(slice(None), [(0, 10)]),
    )
    with pytest.raises(RuntimeError, match="missing="):
        mesh_utils.get_cp_group_layout(grp)

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, val, group: out.__setitem__(slice(None), [(0, 10), (1, 20)]),
    )
    # Single-rank layout creation + cache hit
    single_layout = mesh_utils.get_cp_group_layout(None)
    assert single_layout.world_size == 1
    assert mesh_utils.get_cp_group_layout(None) is single_layout

    # Multi-rank layout creation + cache hit (both tuple and list ranks)
    layout = mesh_utils.get_cp_group_layout(grp)
    grp_list = SimpleNamespace(
        world_size=2, cpu_group=object(), ranks=[0, 1], rank_in_group=1
    )
    assert mesh_utils.get_cp_group_layout(grp_list) is layout
    assert (layout.device_id, layout.prev_rank_in_group, layout.next_rank_in_group) == (
        20,
        0,
        0,
    )

    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(id=10)])
    with pytest.raises(RuntimeError, match="Cannot build the 'dcp' mesh"):
        mesh_utils.get_or_create_cp_mesh("dcp", grp)

    monkeypatch.setattr(
        jax,
        "devices",
        lambda: [
            SimpleNamespace(id=10, coords=(0, 0), core_on_chip=0),
            SimpleNamespace(id=20, coords=(0, 0), core_on_chip=1),
        ],
    )
    monkeypatch.setattr(
        jax.sharding,
        "Mesh",
        lambda devs, axis_names: SimpleNamespace(devices=devs, axis_names=axis_names),
    )
    mesh = mesh_utils.get_or_create_cp_mesh("dcp", grp)
    assert mesh.axis_names == ("dcp",)
    assert mesh_utils.get_or_create_cp_mesh("dcp", grp) is mesh
