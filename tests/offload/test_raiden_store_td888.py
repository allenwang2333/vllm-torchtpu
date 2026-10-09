# SPDX-License-Identifier: Apache-2.0
"""TD8.88 coverage completion for `offload/raiden_store.py` and the
prefix-aware scheduler `schedule()` override.

Every test runs on CPU: the Raiden native wheel (`tpu_sync`) is replaced by a
fake module in `sys.modules`, and the store itself by `FakeKVCacheStore`.
"""

import socket
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from tests.offload.test_raiden_store_offload import (
    FAKE_STORE_TYPES,
    FakeKVCacheStore,
    _ctx,
    _key,
    make_manager,
)
from vllm_torchtpu.offload import raiden_store as rs

_MOD = "vllm_torchtpu.offload.raiden_store"


def _mla(num_kv_heads=1, head_size=64, block_size=16, dtype=torch.uint8):
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_iter_leaf_kv_cache_specs_recurses_into_uniform_type_specs():
    a, b = _mla(head_size=64), _mla(head_size=128)
    uniform = UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs={"layer.0": a, "layer.1": b}
    )
    assert list(rs.iter_leaf_kv_cache_specs(uniform)) == [a, b]
    assert list(rs.iter_leaf_kv_cache_specs(a)) == [a]


def test_load_store_spec_repr():
    keys = [_key(1), _key(2)]
    assert repr(rs.RaidenLoadStoreSpec(keys)) == "RaidenLoadStoreSpec(2 keys)"
    assert (
        repr(rs.RaidenLoadStoreSpec(keys, pinned=False))
        == "RaidenLoadStoreSpec(2 keys, unpinned)"
    )


def test_resolve_multi_shapes_kv_geometry_collects_mla_shapes_and_dtypes():
    swa = SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        sliding_window=128,
    )
    a = _mla(head_size=64, dtype=torch.uint8)
    b = _mla(head_size=128, dtype=torch.bfloat16)
    groups = [
        SimpleNamespace(
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=16, kv_cache_specs={"a": a, "b": b}
            )
        ),
        SimpleNamespace(kv_cache_spec=swa),
    ]
    cfg = SimpleNamespace(kv_cache_groups=groups)
    block_size, shapes, dtypes, device_block = rs._resolve_multi_shapes_kv_geometry(cfg)
    assert block_size == device_block == 16
    assert shapes == {
        (a.num_states, a.num_kv_heads, a.head_size),
        (b.num_states, b.num_kv_heads, b.head_size),
    }
    assert dtypes == {torch.uint8, torch.bfloat16}


def test_resolve_multi_shapes_kv_geometry_rejects_non_mla_leaf():
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    cfg = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=full)])
    with pytest.raises(AssertionError, match="only MLA-family layers"):
        rs._resolve_multi_shapes_kv_geometry(cfg)


def test_resolve_kernel_geometry_dispatches_multi_shapes():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architecture="DeepseekV4ForCausalLM")
    )
    sentinel = (16, {(1, 1, 64)}, {torch.uint8}, 16)
    with patch(f"{_MOD}._resolve_multi_shapes_kv_geometry", return_value=sentinel) as m:
        assert rs.resolve_kernel_geometry(vllm_config, "kvc") == sentinel
    m.assert_called_once_with("kvc")


# ---------------------------------------------------------------------------
# RaidenOffloadingManager
# ---------------------------------------------------------------------------


def test_global_registry_without_store_server_ip_raises():
    with pytest.raises(ValueError, match="without store_server_ip"):
        make_manager(FakeKVCacheStore(), global_registry_address="registry:1")


def _fake_tpu_sync(store_cls):
    """Install a fake `tpu_sync.api.torch.kv_cache_manager` module tree."""
    impl = SimpleNamespace(
        KVCacheStore=store_cls,
        RaidenId=lambda *a: ("rid",) + a,
        BlockStatus=FAKE_STORE_TYPES.BlockStatus,
        RaidenBlockId=FAKE_STORE_TYPES.RaidenBlockId,
    )
    kcm = types.ModuleType("tpu_sync.api.torch.kv_cache_manager")
    kcm._torch_impl = lambda: impl
    mods = {
        "tpu_sync": types.ModuleType("tpu_sync"),
        "tpu_sync.api": types.ModuleType("tpu_sync.api"),
        "tpu_sync.api.torch": types.ModuleType("tpu_sync.api.torch"),
        "tpu_sync.api.torch.kv_cache_manager": kcm,
    }
    return kcm, mods


@pytest.mark.parametrize("enable_global", [False, True])
def test_manager_constructs_real_store_when_none_injected(enable_global):
    built = {}

    class _Store(FakeKVCacheStore):
        def __init__(self, **kwargs):
            super().__init__()
            built.update(kwargs)
            self.raiden_controller_address = "127.0.0.1:1"
            self.store_server_address = ""

    _, mods = _fake_tpu_sync(_Store)
    kwargs = {}
    if enable_global:
        kwargs = dict(global_registry_address="registry:1", store_server_ip="10.0.0.9")
    with (
        patch.dict(sys.modules, mods),
        patch.dict("os.environ", {"PYTHONHASHSEED": "0"}),
    ):
        mgr = rs.RaidenOffloadingManager(
            kernel_physical_blocks_capacity=8,
            offload_logical_blocks_capacity=4,
            device_block_size_factor=2,
            world_size=3,
            raiden_controller_port=7,
            raiden_job_name="job",
            **kwargs,
        )
    assert built["capacity"] == 8
    assert built["raiden_id"] == ("rid", "job", "0", "kv_cache", 0)
    assert built["expected_worker_count"] == 3
    assert built["store_server_ip"] == ("10.0.0.9" if enable_global else "127.0.0.1")
    assert mgr._block_status is FAKE_STORE_TYPES.BlockStatus
    assert mgr._enable_global is enable_global


def test_newly_inserted_returns_none_when_sub_hash_vanished():
    store = FakeKVCacheStore()
    mgr = make_manager(store, device_block_size_factor=1)
    # Nothing resident: the post-admission probe finds fewer than asked.
    assert mgr._newly_inserted_sub_hashes([b"x" * 9], [0]) is None


def test_on_new_request_and_short_sub_hash():
    mgr = make_manager(FakeKVCacheStore())
    assert type(mgr.on_new_request(_ctx())).__name__ == "RequestOffloadingContext"
    assert mgr._sub_hash_to_offload_key(b"abc") is None


def test_prepare_store_over_capacity_returns_none():
    mgr = make_manager(FakeKVCacheStore(), num_blocks=2)
    keys = [_key(i) for i in range(3)]
    assert mgr.prepare_store(keys, _ctx()) is None
    assert not mgr._inflight_store_keys


def test_shutdown_drains_live_admissions():
    mgr = make_manager(FakeKVCacheStore())
    mgr._admissions[5] = MagicMock()
    with patch.object(mgr, "drain_jobs") as drain:
        mgr.shutdown()
    drain.assert_called_once_with({5})


def _store_admission(job_id, sub_hash, **kw):
    return rs.Admission(
        job_id=job_id,
        op=rs.AdmissionOp.STORE,
        keys=[_key(job_id)],
        sub_hashes=[sub_hash],
        pending={sub_hash},
        **kw,
    )


def test_poll_ignores_hashes_without_admission():
    store = FakeKVCacheStore()
    mgr = make_manager(store)
    adm = _store_admission(1, b"live")
    mgr._admissions[1] = adm
    mgr._hash_to_admission[b"live"] = adm
    store._save_failed = [b"unknown-failed"]
    store._save_done = [b"unknown-done"]
    mgr._poll_once()
    assert adm.pending == {b"live"}
    assert not adm.failed
    assert mgr._finished == []


def test_save_retry_launch_failure_fails_job():
    store = FakeKVCacheStore()
    store.fail_save = True
    mgr = make_manager(store)
    adm = _store_admission(
        2, b"h", failed=True, failed_sub_hashes=[b"h"], retries_left=1
    )
    assert mgr._maybe_retry_save(adm) is False
    assert adm.retries_left == 0
    assert store.calls_named("save")[0].args[0] == [b"h"]


def test_drain_jobs_skips_unknown_job_ids():
    mgr = make_manager(FakeKVCacheStore())
    assert mgr.drain_jobs({999}) == []


# ---------------------------------------------------------------------------
# RaidenStoreOffloadingWorker
# ---------------------------------------------------------------------------


def test_worker_multi_shapes_registration_walks_canonical_tensors():
    t0 = torch.zeros(4, 8, dtype=torch.int8)
    t1 = torch.zeros(4, 16, dtype=torch.int8)
    kv_caches = SimpleNamespace(
        tensors=[
            SimpleNamespace(tensor=t0, page_size_bytes=8),
            SimpleNamespace(tensor=t1, page_size_bytes=16),
        ]
    )
    with (
        patch(
            f"{_MOD}.resolve_kernel_geometry",
            return_value=(16, {(1, 1, 8), (1, 1, 16)}, {torch.uint8}, 16),
        ),
        patch.object(rs.RaidenStoreOffloadingWorker, "_init_registration") as init,
    ):
        rs.RaidenStoreOffloadingWorker(
            kv_caches,
            vllm_config=None,
            kv_cache_config=None,
            host_blocks_to_allocate=11,
            controller_address="h:1",
            rank=2,
        )
    (tensors, host_blocks, addr, rank), _ = init.call_args
    assert tensors[0] is t0 and tensors[1] is t1
    assert (host_blocks, addr, rank) == (11, "h:1", 2)


def _bare_worker(address="127.0.0.1:1"):
    worker = object.__new__(rs.RaidenStoreOffloadingWorker)
    worker._controller_address = address
    worker._device_tensors = [["t"]]
    worker._host_blocks_to_allocate = 3
    worker._rank = 1
    worker._mgr = None
    return worker


def test_controller_reachable_true_and_false():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert _bare_worker(f"[127.0.0.1]:{port}")._controller_reachable()
    finally:
        srv.close()
    with patch(f"{_MOD}.socket.create_connection", side_effect=OSError):
        assert not _bare_worker(f"127.0.0.1:{port}")._controller_reachable()


def test_register_loop_notes_unreachable_retries_failure_then_registers():
    worker = _bare_worker()
    stop = MagicMock()
    stop.is_set.return_value = False
    worker._stop_registration = stop
    created = []

    class _Mgr:
        calls = 0

        def __init__(self, **kwargs):
            _Mgr.calls += 1
            if _Mgr.calls == 1:
                raise RuntimeError("controller not ready")
            created.append(kwargs)

    kcm, mods = _fake_tpu_sync(None)
    kcm.KVCacheManager = _Mgr
    mods["tpu_sync.api.torch"].kv_cache_manager = kcm
    # monotonic: next_note base, then >= next_note on the unreachable pass,
    # then the re-armed note.
    with (
        patch.dict(sys.modules, mods),
        patch.object(worker, "_controller_reachable", side_effect=[False, True, True]),
        patch(f"{_MOD}.time.monotonic", side_effect=[0.0, 100.0, 100.0]),
    ):
        worker._register_loop()
    assert worker._mgr is not None
    assert created[0]["worker_id"] == "shard_1"
    assert created[0]["kv_caches"] == [["t"]]
    assert [c.args for c in stop.wait.call_args_list] == [(0.5,), (1.0,)]


def test_register_loop_unreachable_before_note_and_stop():
    worker = _bare_worker()
    stop = MagicMock()
    stop.is_set.side_effect = [False, True]
    worker._stop_registration = stop
    with patch.object(worker, "_controller_reachable", return_value=False):
        worker._register_loop()
    assert worker._mgr is None
    stop.wait.assert_called_once_with(0.5)


# ---------------------------------------------------------------------------
# TPURaidenStoreOffloadingSpec
# ---------------------------------------------------------------------------


def _spec_inputs(
    *,
    cpu_bytes=1 << 20,
    pcp=1,
    dp_rank=0,
    dp_rank_local=0,
    groups=(),
    extra=None,
):
    vllm_config = MagicMock()
    vllm_config.model_config.is_hybrid = False
    vllm_config.parallel_config.data_parallel_rank = dp_rank
    vllm_config.parallel_config.data_parallel_rank_local = dp_rank_local
    vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [g for g, _ in groups]
    kv_cache_config.kv_cache_tensors = []
    config = MagicMock()
    config.cache.blocks_per_chunk = 1
    config.parallel.pcp_size = pcp
    config.parallel.dcp_size = 1
    config.parallel.world_size = 1
    config.extra_config = {"cpu_bytes_to_use": cpu_bytes, **(extra or {})}
    config.worker_kv_bytes_per_block = 1024
    config.groups = [gc for _, gc in groups]
    config.engine_id = "eng"
    return config, vllm_config, kv_cache_config


def _build_spec(config, vllm_config, kv_cache_config, ns_mock=None):
    with (
        patch(f"{_MOD}.resolve_kernel_geometry", return_value=(16, (1, 1), None, 16)),
        patch(f"{_MOD}.resolve_block_major_contract", return_value=None),
        patch(
            f"{_MOD}.derive_offload_namespace",
            new=ns_mock or MagicMock(return_value=b"ns"),
        ),
    ):
        return rs.TPURaidenStoreOffloadingSpec(config, vllm_config, kv_cache_config)


def test_spec_requires_cpu_bytes_to_use():
    with pytest.raises(ValueError, match="cpu_bytes_to_use"):
        _build_spec(*_spec_inputs(cpu_bytes=0))


def test_spec_cp_geometry_attention_groups_and_dp_suffix():
    attn = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    groups = [
        (SimpleNamespace(kv_cache_spec=attn), SimpleNamespace(tokens_per_block=32)),
        # Non-attention group: the CP scaling check skips it.
        (SimpleNamespace(kv_cache_spec=object()), SimpleNamespace(tokens_per_block=1)),
    ]
    ns = MagicMock(return_value=b"ns")
    spec = _build_spec(
        *_spec_inputs(pcp=2, dp_rank=3, dp_rank_local=None, groups=groups),
        ns_mock=ns,
    )
    assert ns.call_args.kwargs["cp_geometry"] == (2, 1, 1)
    # dp_rank_local None falls back to the global DP rank.
    assert spec.raiden_controller_port == rs._DEFAULT_CONTROLLER_PORT + 3
    assert spec.raiden_job_name == "vllm-offload-eng-dp3"


def test_spec_cp_scale_mismatch_raises():
    attn = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    groups = [
        (SimpleNamespace(kv_cache_spec=attn), SimpleNamespace(tokens_per_block=16))
    ]
    with pytest.raises(AssertionError, match="CP-aware"):
        _build_spec(*_spec_inputs(pcp=2, groups=groups))


def test_spec_rejects_raiden_shm_key(monkeypatch):
    monkeypatch.setitem(vars(rs.envs), "RAIDEN_SHM_KEY", "1")
    with pytest.raises(ValueError, match="RAIDEN_SHM_KEY"):
        _build_spec(*_spec_inputs())


def test_spec_get_manager_and_worker_are_memoized():
    spec = _build_spec(
        *_spec_inputs(extra={"store_server_ip": "10.1.1.1", "kv_pool_group": "g"})
    )
    with patch(f"{_MOD}.RaidenOffloadingManager") as mgr_cls:
        m1 = spec.get_manager()
        assert spec.get_manager() is m1
    mgr_cls.assert_called_once()
    kw = mgr_cls.call_args.kwargs
    assert kw["store_server_ip"] == "10.1.1.1"
    assert kw["kv_pool_group"] == "g"
    assert kw["key_namespace"] == b"ns"

    group = SimpleNamespace(rank_in_group=4)
    with (
        patch("vllm.distributed.parallel_state.get_world_group", return_value=group),
        patch(f"{_MOD}.RaidenStoreOffloadingWorker") as worker_cls,
    ):
        w1 = spec.get_worker("kv")
        assert spec.get_worker("kv") is w1
    worker_cls.assert_called_once()
    assert worker_cls.call_args.kwargs["controller_address"] == (
        f"10.1.1.1:{spec.raiden_controller_port}"
    )
    assert worker_cls.call_args.kwargs["rank"] == 4


# ---------------------------------------------------------------------------
# PrefixAwareScheduler.schedule
# ---------------------------------------------------------------------------


def test_prefix_aware_schedule_selects_queue_then_delegates():
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

    sched = object.__new__(PrefixAwareScheduler)
    order = []
    with (
        patch.object(
            PrefixAwareScheduler,
            "_select_waiting_queue_for_scheduling",
            side_effect=lambda: order.append("select"),
        ),
        patch.object(
            AsyncScheduler,
            "schedule",
            side_effect=lambda *a, **k: order.append(("super", a, k)) or "out",
        ),
    ):
        assert sched.schedule(1, flag=True) == "out"
    assert order == ["select", ("super", (1,), {"flag": True})]
