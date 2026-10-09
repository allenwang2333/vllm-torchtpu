# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TD7.62: offload block-major layout, Raiden connector, and Omni."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.v1.kv_cache_interface import KVCacheTensor
from vllm.v1.kv_offload.base import GPULoadStoreSpec, OffloadKey
from vllm.v1.outputs import KVConnectorOutput

from vllm_torchtpu.offload import block_major_layout as bml
from vllm_torchtpu.offload import raiden_connector as rc
from vllm_torchtpu.offload.raiden_store import (
    AdmissionOp,
    FinishedJob,
    RaidenLoadStoreSpec,
    RaidenOffloadingManager,
)

pytestmark = [pytest.mark.cpu_test]


# ---------------------------------------------------------------------------
# 1. offload/block_major_layout.py Tests
# ---------------------------------------------------------------------------


def test_block_major_layer_indices_error_branches():
    # 1. Remainder or row_bytes % fragment_row_bytes != 0
    cfg_bad_tile = SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * 500,
                layers=["layers.0"],
                layer_stride=500,
                block_stride=500,
                offset=0,
            )
        ],
    )
    with pytest.raises(ValueError, match="does not tile whole scheduler blocks"):
        bml.block_major_layer_indices(cfg_bad_tile, fragment_row_bytes=512)

    # 2. tensor.block_stride != row_bytes
    cfg_bad_stride = SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * 512,
                layers=["layers.0"],
                layer_stride=512,
                block_stride=1024,
                offset=0,
            )
        ],
    )
    with pytest.raises(ValueError, match="placement is not block-major"):
        bml.block_major_layer_indices(cfg_bad_stride, fragment_row_bytes=512)

    # 3. offset % fragment_row_bytes != 0 or offset out of bounds
    cfg_bad_offset = SimpleNamespace(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * 512,
                layers=["layers.0"],
                layer_stride=512,
                block_stride=512,
                offset=16,
            )
        ],
    )
    with pytest.raises(ValueError, match="fragment page offset does not match"):
        bml.block_major_layer_indices(cfg_bad_offset, fragment_row_bytes=512)


def test_resolve_block_major_contract_unified_and_multi_shapes(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_BLOCK_MAJOR_KV", "1")
    vllm_cfg = MagicMock()
    kv_cfg = MagicMock()

    # 1. Multi-shapes geometry raises ValueError
    with (
        patch(
            "vllm_torchtpu.block_major_pool.unified_block_major_enabled",
            return_value=False,
        ),
        patch(
            "vllm_torchtpu.offload.raiden_store.resolve_kernel_geometry",
            return_value=(16, {(16, 2, 4), (16, 2, 8)}, torch.bfloat16, 16),
        ),
        pytest.raises(ValueError, match="multi-shapes KV cache"),
    ):
        bml.resolve_block_major_contract(vllm_cfg, kv_cfg)

    # 2. Unified block-major pool contract resolution
    fake_spec = SimpleNamespace(dtype=torch.bfloat16, block_size=32)
    fake_layout = SimpleNamespace(
        split=2,
        num_pools=2,
        pool_index_by_layer={"layers.0": 0, "layers.1": 1, "layers.2": 0},
    )
    with (
        patch(
            "vllm_torchtpu.block_major_pool.unified_block_major_enabled",
            return_value=True,
        ),
        patch(
            "vllm_torchtpu.offload.raiden_store.resolve_attention_geometry",
            return_value=(fake_spec, 16, (16, 2, 4)),
        ),
        patch(
            "vllm_torchtpu.kv_cache_materializer.resolve_block_major_pool_layout",
            return_value=fake_layout,
        ),
    ):
        contract = bml.resolve_block_major_contract(vllm_cfg, kv_cfg)
        assert contract is not None
        assert contract.fragment_count == 2
        # split=2 * prod(16, 2, 4)=128 * bf16(2) = 512
        assert contract.fragment_row_bytes == 512
        assert contract.bundle_row_bytes == 1024


# ---------------------------------------------------------------------------
# 2. offload/raiden_connector.py Tests
# ---------------------------------------------------------------------------


def test_raiden_connector_event_helpers_and_worker_metadata(monkeypatch):
    # _enqueue_pool_writes
    sync_calls = []
    fake_sync = SimpleNamespace(
        synchronize=lambda tensors, wait=False: sync_calls.append((tensors, wait))
    )
    with patch.dict("sys.modules", {"torch_tpu._internal.sync": fake_sync}):
        import torch_tpu._internal as internal_pkg

        monkeypatch.setattr(internal_pkg, "sync", fake_sync, raising=False)
        t = torch.zeros(2)
        rc._enqueue_pool_writes([t])
        assert sync_calls == [([t], False)]

    # _record_fence_event and _event_is_retired
    class _FakeEvent:
        def __init__(self, ready: bool = False):
            self.recorded = False
            self.synced = False
            self.ready = ready

        def record(self):
            self.recorded = True

        def query(self) -> bool:
            return self.ready

        def synchronize(self):
            self.synced = True

    monkeypatch.setattr(
        torch,
        "tpu",
        SimpleNamespace(Event=lambda: _FakeEvent(ready=True)),
        raising=False,
    )
    ev = rc._record_fence_event()
    assert ev.recorded is True
    assert rc._event_is_retired(ev) is True
    assert ev.synced is True

    ev_not_ready = _FakeEvent(ready=False)
    assert rc._event_is_retired(ev_not_ready) is False
    assert ev_not_ready.synced is False

    # RaidenOffloadingWorkerMetadata.aggregate
    m1 = rc.RaidenOffloadingWorkerMetadata(
        completed_jobs={1: 1},
        fenced_jobs={10: 1},
    )
    m2 = rc.RaidenOffloadingWorkerMetadata(
        completed_jobs={1: 1, 2: 1},
        fenced_jobs={10: 1, 20: 1},
    )
    merged = m1.aggregate(m2)
    assert merged.completed_jobs == {1: 2, 2: 1}
    assert merged.fenced_jobs == {10: 2, 20: 1}


def test_tpu_raiden_offloading_scheduler_full_lifecycle(monkeypatch):
    fake_mgr = object.__new__(RaidenOffloadingManager)
    submitted_loads = []
    submitted_stores = []
    cancelled_keys = []
    drained_calls = []

    fake_mgr.submit_load_job = lambda job_id, keys, bids, req_id, pinned=False: (
        submitted_loads.append((job_id, keys, bids, req_id, pinned))
    )
    fake_mgr.submit_store_job = lambda job_id, keys, bids: submitted_stores.append(
        (job_id, keys, bids)
    )
    fake_mgr.on_store_job_cancelled = lambda keys: cancelled_keys.append(keys)
    fake_mgr.is_job_launched = lambda jid: jid == 200
    fake_mgr.drain_jobs = lambda jids: (
        drained_calls.append(set(jids))
        or [FinishedJob(job_id=200, op=AdmissionOp.STORE, success=True)]
    )
    fake_mgr.poll_finished_jobs = lambda: [
        FinishedJob(
            job_id=300,
            op=AdmissionOp.LOAD,
            success=False,
            req_id="req-300",
            failed_device_block_ids=[7, 8],
        )
    ]

    def _fake_super_init(self, spec, vllm_config, kv_cache_config):
        self.manager = fake_mgr
        self.config = SimpleNamespace(num_workers=2)

    monkeypatch.setattr(rc.OffloadingConnectorScheduler, "__init__", _fake_super_init)
    sched = rc.TPURaidenOffloadingScheduler(MagicMock(), MagicMock(), MagicMock())
    assert sched._raiden_manager is fake_mgr

    def _gpu_spec(ids):
        spec = object.__new__(GPULoadStoreSpec)
        spec.block_ids = np.array(ids, dtype=np.int64)
        return spec

    k1 = OffloadKey((0, b"h1"))
    store_job_keep = TransferJob(
        src_spec=_gpu_spec([10]),
        dst_spec=RaidenLoadStoreSpec([k1]),
        req_id="req-store-1",
    )
    store_job_cancel = TransferJob(
        src_spec=_gpu_spec([11]),
        dst_spec=RaidenLoadStoreSpec([k1]),
        req_id="req-store-2",
    )
    load_job = TransferJob(
        src_spec=RaidenLoadStoreSpec([k1], pinned=True),
        dst_spec=_gpu_spec([12]),
        req_id="req-load",
    )

    base_meta_1 = OffloadingConnectorMetadata(
        load_jobs={20: load_job},
        store_jobs={10: store_job_keep, 11: store_job_cancel},
        jobs_to_flush={11, 200, 999},
    )
    monkeypatch.setattr(
        rc.OffloadingConnectorScheduler,
        "build_connector_meta",
        lambda self, _out: base_meta_1,
    )
    meta_1 = sched.build_connector_meta(MagicMock())
    assert isinstance(meta_1, rc.RaidenOffloadingConnectorMetadata)
    assert meta_1.fence_job_ids == {10}
    assert set(meta_1.finished_store_job_ids) == {11, 200}
    assert meta_1.finished_load_jobs == {300: "req-300"}
    assert meta_1.failed_load_device_block_ids == [7, 8]
    assert len(submitted_loads) == 1
    assert len(cancelled_keys) == 1
    assert drained_calls == [{200}]

    # Step 2: build_connector_meta with no jobs_to_flush (empty drained fast path)
    base_meta_2 = OffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={},
        jobs_to_flush=set(),
    )
    fake_mgr.poll_finished_jobs = lambda: [
        FinishedJob(job_id=400, op=AdmissionOp.STORE, success=True)
    ]
    monkeypatch.setattr(
        rc.OffloadingConnectorScheduler,
        "build_connector_meta",
        lambda self, _out: base_meta_2,
    )
    meta_2 = sched.build_connector_meta(MagicMock())
    assert meta_2.finished_store_job_ids == [400]

    # update_connector_output branches:
    # 1. None / empty fenced_jobs
    super_updates = []
    monkeypatch.setattr(
        rc.OffloadingConnectorScheduler,
        "update_connector_output",
        lambda self, out: super_updates.append(out),
    )
    sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=None))

    # 2. Partial ack (1 of 2 workers) + unknown/cancelled job_id 999
    w_meta_partial = rc.RaidenOffloadingWorkerMetadata(
        completed_jobs={},
        fenced_jobs={10: 1, 999: 1},
    )
    sched.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=w_meta_partial)
    )
    assert submitted_stores == []

    # 3. Final ack (2 of 2 workers) -> submits store job 10
    w_meta_final = rc.RaidenOffloadingWorkerMetadata(
        completed_jobs={},
        fenced_jobs={10: 1},
    )
    sched.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=w_meta_final)
    )
    assert submitted_stores == [(10, [k1], [10])]
    assert len(super_updates) == 3


def test_tpu_raiden_offloading_connector_roles_and_worker_hooks(monkeypatch):
    vllm_cfg = MagicMock()
    kv_cfg = SimpleNamespace(num_blocks=4, kv_cache_groups=[object()])

    assert rc.TPURaidenOffloadingConnector.get_required_kvcache_layout(vllm_cfg) is None

    monkeypatch.setenv("VLLM_TPU_BLOCK_MAJOR_KV", "1")
    monkeypatch.setenv("RAIDEN_DISABLE_SINGLETON_WORKER", "0")
    with pytest.raises(ValueError, match="RAIDEN_DISABLE_SINGLETON_WORKER"):
        rc.TPURaidenOffloadingConnector(vllm_cfg, KVConnectorRole.SCHEDULER, kv_cfg)

    monkeypatch.setenv("RAIDEN_DISABLE_SINGLETON_WORKER", "1")
    monkeypatch.setattr(
        rc.offloading_config_module,
        "build_offloading_config",
        lambda _vc, _kc: MagicMock(),
    )
    monkeypatch.setattr(rc, "TPURaidenStoreOffloadingSpec", lambda *_a: MagicMock())

    fake_sched = SimpleNamespace(take_events=lambda: ["ev1"])
    monkeypatch.setattr(rc, "TPURaidenOffloadingScheduler", lambda *_a: fake_sched)

    # Scheduler role
    conn_sched = rc.TPURaidenOffloadingConnector(
        vllm_cfg, KVConnectorRole.SCHEDULER, kv_cfg
    )
    assert conn_sched.prefer_cross_layer_blocks is True
    assert conn_sched.reset_cache() is False
    assert list(conn_sched.take_events()) == ["ev1"]

    # Non-scheduler / non-worker role branch (374->exit)
    conn_none = rc.TPURaidenOffloadingConnector(vllm_cfg, MagicMock(), kv_cfg)
    assert list(conn_none.take_events()) == []

    # Worker role
    fake_worker = SimpleNamespace(
        kv_cache_config=kv_cfg,
        _init_worker=lambda _c: None,
    )
    monkeypatch.setattr(rc, "OffloadingConnectorWorker", lambda *_a: fake_worker)
    conn_worker = rc.TPURaidenOffloadingConnector(
        vllm_cfg, KVConnectorRole.WORKER, kv_cfg
    )

    pool = torch.zeros(4, 16, dtype=torch.int8)
    conn_worker.register_kv_caches({"layer.0": pool})
    assert len(conn_worker._pool_sync_tensors) == 1

    # No pending work -> build_connector_worker_meta returns None, load errors empty
    assert conn_worker.build_connector_worker_meta() is None
    assert conn_worker.get_block_ids_with_load_errors() == set()

    # Step with fence_job_ids, one retired event and one unretired event, plus echoes
    events_iter = iter(["ev_unretired", "ev_retired"])
    monkeypatch.setattr(rc, "_enqueue_pool_writes", lambda _t: None)
    monkeypatch.setattr(rc, "_record_fence_event", lambda: next(events_iter))
    monkeypatch.setattr(rc, "_event_is_retired", lambda ev: ev == "ev_retired")

    step1_meta = rc.RaidenOffloadingConnectorMetadata(
        load_jobs={},  # pyrefly: ignore[unexpected-keyword]
        store_jobs={},  # pyrefly: ignore[unexpected-keyword]
        jobs_to_flush=set(),  # pyrefly: ignore[unexpected-keyword]
        fence_job_ids={101},
        finished_store_job_ids=[],
        finished_load_jobs={},
        failed_load_device_block_ids=[],
    )
    conn_worker._connector_metadata = step1_meta
    assert conn_worker.get_finished(set()) == (set(), set())
    assert len(conn_worker._pending_fences) == 1

    step2_meta = rc.RaidenOffloadingConnectorMetadata(
        load_jobs={},  # pyrefly: ignore[unexpected-keyword]
        store_jobs={},  # pyrefly: ignore[unexpected-keyword]
        jobs_to_flush=set(),  # pyrefly: ignore[unexpected-keyword]
        fence_job_ids={102},
        finished_store_job_ids=[50],
        finished_load_jobs={60: "req-60"},
        failed_load_device_block_ids=[9],
    )
    conn_worker._connector_metadata = step2_meta
    sending, recving = conn_worker.get_finished(set())
    assert sending == set()
    assert recving == {"req-60"}

    # Load errors populated and cleared on read
    assert conn_worker.get_block_ids_with_load_errors() == {9}
    assert conn_worker.get_block_ids_with_load_errors() == set()

    w_meta = conn_worker.build_connector_worker_meta()
    assert isinstance(w_meta, rc.RaidenOffloadingWorkerMetadata)
    assert w_meta.fenced_jobs == {102: 1}
    assert w_meta.completed_jobs == {50: 1, 60: 1}
    assert conn_worker.build_connector_worker_meta() is None

    # Step 3: no fence_job_ids and _pending_fences emptied
    conn_worker._pending_fences.clear()
    step3_meta = rc.RaidenOffloadingConnectorMetadata(
        load_jobs={},  # pyrefly: ignore[unexpected-keyword]
        store_jobs={},  # pyrefly: ignore[unexpected-keyword]
        jobs_to_flush=set(),  # pyrefly: ignore[unexpected-keyword]
    )
    conn_worker._connector_metadata = step3_meta
    assert conn_worker.get_finished(set()) == (set(), set())
