# SPDX-License-Identifier: Apache-2.0
"""TD8.109 CPU coverage tests for KV transfer and Raiden modules."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_torchtpu.distributed.kv_transfer.connector_metadata as connector_metadata
import vllm_torchtpu.distributed.kv_transfer.host_kv_shm as host_kv_shm
import vllm_torchtpu.distributed.kv_transfer.kv_scatter as kv_scatter
import vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans as byte_spans
import vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint as layout_fp
import vllm_torchtpu.distributed.kv_transfer.raiden.seq_on_lane as seq_on_lane
import vllm_torchtpu.distributed.kv_transfer.raiden.tags as tags
import vllm_torchtpu.distributed.kv_transfer.raiden.telemetry_utils as telemetry_utils
import vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats as stats_mod
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (
    PoolEntry,
    PoolManifest,
    RegionSpec,
)


def _mk_pool(
    tag: str,
    layer_name: str,
    storage_index: int,
    block_stride_bytes: int,
    num_blocks: int,
    regions: tuple[RegionSpec, ...],
) -> PoolEntry:
    return PoolEntry(
        tag=tag,
        layer_name=layer_name,
        storage_index=storage_index,
        base_offset_bytes=0,
        block_stride_bytes=block_stride_bytes,
        num_blocks=num_blocks,
        regions=regions,
        dtype_tag="fp8",
    )


def _mk_manifest(
    pools: tuple[PoolEntry, ...], storages: tuple[object, ...]
) -> PoolManifest:
    return PoolManifest(
        binding="private_typed",
        storages=list(storages),
        pools=list(pools),
    )


def test_connector_metadata_dataclasses() -> None:
    send = connector_metadata.SendMeta(
        uuid=1,
        local_block_ids=[0, 1],
        expiration_time=10.0,
        num_tokens=32,
        mamba_state_block_ids=[2],
        group_block_ids=[[0, 1]],
    )
    load = connector_metadata.LoadMeta(
        uuid=2,
        local_block_ids=[3, 4],
        remote_block_ids=[0, 1],
        remote_host="127.0.0.1",
        remote_port=5555,
        remote_side_channel_port=5556,
        report_completion=False,
        fail_only=True,
    )
    meta = connector_metadata.TPUConnectorMetadata(
        reqs_to_send={"r1": send},
        reqs_to_load={"r2": load},
    )
    assert meta.reqs_to_send["r1"].uuid == 1
    assert meta.reqs_to_load["r2"].fail_only is True


def test_tags_and_telemetry_utils() -> None:
    assert tags.dsv4_group_tag(tags.TAG_DSV4_SWA, 3) == "dsv4.swa.g3"
    assert tags.layer_tag(tags.TAG_FA, 0) == "fa.l0"
    with pytest.raises(ValueError, match="non-negative"):
        tags.layer_tag(tags.TAG_FA, -1)
    assert tags.split_layer_tag("fa.l12") == ("fa", 12)
    assert tags.split_layer_tag("fa.lxyz") == ("fa.lxyz", None)
    assert tags.split_layer_tag("fa") == ("fa", None)
    assert tags.class_tag("fa.l12") == "fa"

    assert telemetry_utils.parse_raiden_metric_key("  invalid-key!  ") == (
        "invalid-key!",
        {},
    )
    assert telemetry_utils.parse_raiden_metric_key("tpu_raiden_ops") == (
        "tpu_raiden_ops",
        {},
    )
    assert telemetry_utils.parse_raiden_metric_key("tpu_raiden_ops{}") == (
        "tpu_raiden_ops",
        {},
    )
    base, parsed = telemetry_utils.parse_raiden_metric_key(
        r'tpu_raiden_ops{a="x\ny\tz\r\"\\"}'
    )
    assert base == "tpu_raiden_ops"
    assert parsed == {"a": 'x\ny\tz\r"\\'}
    assert (
        telemetry_utils.normalize_raiden_metric_name("sent_bytes")
        == "tpu_raiden_sent_bytes"
    )
    assert (
        telemetry_utils.normalize_raiden_metric_name("tpu_raiden_sent_bytes")
        == "tpu_raiden_sent_bytes"
    )


def test_tpu_connector_stats_and_prom_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    prepopulated = stats_mod.TpuKVConnectorStats(data={"d2h_transfer_time": [1.5]})
    assert prepopulated.data["d2h_transfer_time"] == [1.5]

    s = stats_mod.TpuKVConnectorStats()
    assert s.is_empty() is True
    s.record_d2h_transfer(2.0)
    s.record_h2d_transfer(3.0)
    s.record_kv_pull(4.0)
    s.record_mb_transferred(8.0)
    s.record_failed_transfer()
    s.record_prefill_queue_length(5)
    s.record_decode_queue_length(6)
    s.record_runner_timings([("0", "issue", 0.01)], [("0", "dev", 0.02)])
    assert s.is_empty() is False

    empty_other = stats_mod.TpuKVConnectorStats()
    s.aggregate(empty_other)

    custom_other = stats_mod.TpuKVConnectorStats(
        data={"custom_metric": [10.0], "mb_transferred": [4.0]}
    )
    s.aggregate(custom_other)
    assert s.data["custom_metric"] == [10.0]
    reduced = s.reduce()
    assert reduced["Num successful transfers"] == 2
    assert reduced["Prefill queue length"] == 5
    assert reduced["Decode queue length"] == 6

    cloned = s.clone_and_reset()
    assert cloned.num_successful_transfers == 2
    assert s.is_empty() is True
    empty_reduced = s.reduce()
    assert empty_reduced["Avg D2H transfer time (ms)"] == 0.0

    s_no_dict = stats_mod.TpuKVConnectorStats()
    s_no_dict.data = {}
    assert s_no_dict.is_empty() is True

    # Test TpuKVConnectorPromMetrics edge cases
    class _DummyMetric:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._labelnames = kwargs.get("labelnames")
            self.observed = []
            self.incremented = []
            self.set_vals = []

        def labels(self, *args, **kwargs):
            return self

        def observe(self, val):
            self.observed.append(val)

        def inc(self, val=1):
            self.incremented.append(val)

        def set(self, val):
            self.set_vals.append(val)

    def _fake_init(self, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        self._histogram_cls = _DummyMetric
        self._counter_cls = _DummyMetric
        self._gauge_cls = _DummyMetric
        self.per_engine_labelvalues = per_engine_labelvalues
        self.labelnames = labelnames

    monkeypatch.setattr(stats_mod.KVConnectorPromMetrics, "__init__", _fake_init)
    monkeypatch.setattr(
        stats_mod,
        "create_metric_per_engine",
        lambda metric_obj, per_engine: dict.fromkeys(per_engine or [0], metric_obj),
    )

    descriptors = [
        SimpleNamespace(
            name="hist_metric",
            description="h",
            type=SimpleNamespace(name="MetricType.HISTOGRAM"),
            label_names=["direction"],
            buckets=[],
        ),
        SimpleNamespace(
            name="count_metric",
            description="c",
            type=SimpleNamespace(name="MetricType.COUNTER"),
            label_names=[],
            buckets=None,
        ),
        SimpleNamespace(
            name="gauge_metric",
            description="g",
            type=SimpleNamespace(name="MetricType.GAUGE"),
            label_names=[],
            buckets=None,
        ),
        SimpleNamespace(
            name="unknown_metric",
            description="u",
            type="OTHER",
            label_names=[],
            buckets=None,
        ),
    ]
    monkeypatch.setattr(
        stats_mod.dist_utils, "configure_raiden_telemetry", lambda: None
    )
    monkeypatch.setattr(
        stats_mod.dist_utils,
        "get_raiden_telemetry_module",
        lambda: SimpleNamespace(get_metric_metadata=lambda: descriptors),
    )

    prom = stats_mod.TpuKVConnectorPromMetrics(
        vllm_config=MagicMock(),
        metric_types={},
        labelnames=["engine"],
        per_engine_labelvalues={0: ["e0"]},
    )
    # Observe once while _labelnames is a list (covers 465->467), then with None
    prom.observe(
        {'tpu_raiden_hist_metric{direction="push"}': [2.5]},
        engine_idx=0,
    )
    prom.dynamic_metrics["tpu_raiden_hist_metric"][1]._labelnames = None
    # Register an unknown mtype and a None target_obj entry in dynamic_metrics
    prom.dynamic_metrics["tpu_raiden_none_target"] = ("counter", {}, [])
    prom.dynamic_metrics["tpu_raiden_weird_type"] = ("weird", _DummyMetric(), [])

    prom.observe(
        {
            "runner_cpu_issue_seconds": [("0", "issue", 0.01)],
            "runner_device_elapsed_seconds": [("0", "dev", 0.02)],
            "d2h_transfer_time": [1.0],
            "h2d_transfer_time": [2.0],
            "kv_pull_time": [3.0],
            "mb_transferred": [4.0],
            "num_failed_transfers": [1],
            "prefill_queue_length": [2],
            "decode_queue_length": [3],
            "empty_list": [],
            "unregistered_metric": [1.0],
            'tpu_raiden_hist_metric{direction="push"}': [5.0],
            "tpu_raiden_count_metric": [2],
            "tpu_raiden_gauge_metric": [7],
            "tpu_raiden_none_target": [1],
            "tpu_raiden_weird_type": [9],
        },
        engine_idx=0,
    )

    # Also cover _get_raiden_metric_definitions with labelnames=None & no base_labels
    prom.labelnames = None
    prom._labelnames = None
    prom.per_engine_labelvalues = {}
    defs_no_labels = prom._get_raiden_metric_definitions(labelnames=None)
    assert "tpu_raiden_count_metric" in defs_no_labels

    # Branch where get_raiden_telemetry_module returns None (341->347)
    monkeypatch.setattr(
        stats_mod.dist_utils, "get_raiden_telemetry_module", lambda: None
    )
    assert prom._get_raiden_metric_definitions(labelnames=[]) == {}

    # Exception branch in _get_raiden_metric_definitions
    def _raise_err():
        raise RuntimeError("boom")

    monkeypatch.setattr(stats_mod.dist_utils, "configure_raiden_telemetry", _raise_err)
    assert prom._get_raiden_metric_definitions(labelnames=[]) == {}


def test_host_kv_shm_pool_full_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = host_kv_shm.PoolSpec(
        num_slots=2,
        tp_size=2,
        num_layers=2,
        max_blocks=4,
        layer_shard_shape=(4, 2, 4),
        dtype=torch.float16,
    )
    assert spec.per_layer_bytes == 4 * 2 * 4 * 2
    assert spec.per_rank_bytes == 2 * spec.per_layer_bytes
    assert spec.per_slot_bytes == 2 * spec.per_rank_bytes
    assert spec.total_bytes == 2 * spec.per_slot_bytes

    shm_name = f"test_td8109_shm_{id(spec) & 0xFFFF}"
    # Create once, then create again without closing to cover stale unlink branch
    pool_stale = host_kv_shm.HostKVShmPool.create(spec, shm_name)
    pool = host_kv_shm.HostKVShmPool.create(spec, shm_name)
    try:
        assert host_kv_shm._shm_base_addr(pool._shm) > 0

        # Attach with matching spec and mismatched larger spec
        attached = host_kv_shm.HostKVShmPool.attach(spec, shm_name)
        bigger_spec = host_kv_shm.PoolSpec(
            num_slots=4,
            tp_size=2,
            num_layers=2,
            max_blocks=4,
            layer_shard_shape=(4, 2, 4),
            dtype=torch.float16,
        )
        with pytest.raises(RuntimeError, match="shm size"):
            host_kv_shm.HostKVShmPool.attach(bigger_spec, shm_name)

        # Slot acquisition and idempotent release
        s0 = pool.acquire_slot(timeout=1.0)
        assert s0 == 0
        pool.release_slot(s0)
        pool.release_slot(s0)  # idempotent double release

        # Layer views, pack_rank, rank_layer_views, unpack_rank_layers, unpack_rank
        v0 = pool.layer_view(0, 0, 0, num_blocks=2)
        v1 = pool.layer_view(0, 0, 1, num_blocks=2)
        v0.fill_(1.0)
        v1.fill_(2.0)

        packed = pool.pack_rank(0, 0, num_blocks=2)
        views = pool.rank_layer_views(0, 0, num_blocks=2)
        assert len(views) == 2

        with pytest.raises(RuntimeError, match="got 1 layers"):
            pool.unpack_rank_layers(0, 1, num_blocks=2, layer_buffers=[views[0]])
        with pytest.raises(RuntimeError, match="layer 0 size"):
            pool.unpack_rank_layers(
                0, 1, num_blocks=2, layer_buffers=[b"short", b"short"]
            )

        # Test _copy_into_shm normal (_memmove), _memmove=None, and np=None
        pool.unpack_rank_layers(
            0, 1, num_blocks=2, layer_buffers=[bytes(v) for v in views]
        )
        monkeypatch.setattr(host_kv_shm, "_memmove", None)
        pool.unpack_rank_layers(
            0, 1, num_blocks=2, layer_buffers=[bytes(v) for v in views]
        )
        monkeypatch.setattr(host_kv_shm, "np", None)
        pool.unpack_rank_layers(
            0, 1, num_blocks=2, layer_buffers=[bytes(v) for v in views]
        )
        del views

        with pytest.raises(RuntimeError, match="payload size"):
            pool.unpack_rank(1, 0, num_blocks=2, payload=b"bad")
        pool.unpack_rank(1, 0, num_blocks=2, payload=packed)
        assert torch.equal(pool.layer_view(1, 0, 0, num_blocks=2), v0)
        del v0, v1

        # Test mlock_async / wait_mlock / _mlock_blocking / _munlock_pool branches
        fake_libc = SimpleNamespace(
            mlock=MagicMock(return_value=0),
            munlock=MagicMock(return_value=0),
        )
        monkeypatch.setattr(host_kv_shm, "_libc_for_mlock", fake_libc)
        pool._mlock_blocking(my_rank=None)
        pool._munlock_pool()
        t = pool.mlock_async(my_rank=None)
        assert pool.mlock_async(my_rank=None) is not None
        t.join(timeout=2.0)
        assert pool.wait_mlock(timeout=1.0) is True
        pool._munlock_pool()
        assert fake_libc.munlock.called

        # Repeat with alive thread short-circuit
        alive_thread = MagicMock()
        alive_thread.is_alive.return_value = True
        pool._mlock_thread = alive_thread
        assert pool.mlock_async(my_rank=0) is alive_thread
        pool._mlock_thread = None

        # _mlock_blocking with my_rank=0 and mlock failure (rc != 0)
        fake_libc_fail = SimpleNamespace(
            mlock=MagicMock(return_value=-1),
            munlock=MagicMock(return_value=0),
        )
        monkeypatch.setattr(host_kv_shm, "_libc_for_mlock", fake_libc_fail)
        pool._mlock_blocking(my_rank=0)

        # _mlock_blocking when _shm_base_addr raises
        monkeypatch.setattr(
            host_kv_shm,
            "_shm_base_addr",
            MagicMock(side_effect=RuntimeError("no addr")),
        )
        pool._mlock_blocking(my_rank=None)

        # _mlock_blocking and _munlock_pool when _libc_for_mlock is None
        monkeypatch.setattr(host_kv_shm, "_libc_for_mlock", None)
        pool._mlock_blocking(my_rank=None)
        pool._munlock_pool()

        # Close attached (owner=False)
        attached.close()
    finally:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                pool_stale._shm,
                "close",
                MagicMock(side_effect=RuntimeError("close error")),
            )
            pool_stale.close()
        pool.close()


def test_kv_scatter_full_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    assert kv_scatter.scatter_available() is True

    # 1. Directly test _scatter_kernel_body with mocked Pallas primitives
    started = []
    waited = []

    class _FakeCopy:
        def start(self):
            started.append(True)

        def wait(self):
            waited.append(True)

    class _FakeRef:
        def __init__(self):
            self.at = self

        def __getitem__(self, idx):
            return 2 if idx == 0 else self

    monkeypatch.setattr(
        kv_scatter.pl,
        "loop",
        lambda lo, hi: lambda fn: [fn(i) for i in range(lo, hi)],
    )
    monkeypatch.setattr(kv_scatter.pl, "ds", lambda start, sz: (start, sz))
    monkeypatch.setattr(
        kv_scatter.pl, "run_scoped", lambda body_fn, sem_type: body_fn("sem")
    )
    monkeypatch.setattr(
        kv_scatter.pltpu, "make_async_copy", lambda src, dst, sem: _FakeCopy()
    )
    ref = _FakeRef()
    kv_scatter._scatter_kernel_body(ref, ref, ref, ref, ref, ref)
    assert len(started) == 2 and len(waited) == 2

    # 2. Test _make_multi_layer_scatter_kernel and inner _jax_multi_layer_scatter
    kv_scatter._make_multi_layer_scatter_kernel.cache_clear()
    monkeypatch.setattr(
        kv_scatter.pl,
        "pallas_call",
        lambda body, out_shape, grid_spec, input_output_aliases: (
            lambda nc, so, do, s, d: d
        ),
    )
    monkeypatch.setattr(
        kv_scatter.ttpu_pallas,
        "custom_jax_kernel",
        lambda fn, **kwargs: fn,
    )
    wrapped = kv_scatter._make_multi_layer_scatter_kernel(2)
    with pytest.raises(ValueError, match="expected 4 src/dst tensors"):
        wrapped(1, 2, 3, torch.zeros(1))
    dummy = SimpleNamespace(shape=(2, 4), dtype="float32")
    res = wrapped(1, 2, 3, dummy, dummy, dummy, dummy)
    assert isinstance(res, tuple) and len(res) == 2
    kv_scatter._make_multi_layer_scatter_kernel.cache_clear()

    # 3. Test multi_layer_scatter_into branches
    t_src = torch.ones((2, 4), dtype=torch.float32)
    t_dst = torch.zeros((4, 4), dtype=torch.float32)
    blocks = torch.tensor([1, 3], dtype=torch.int32)
    prebuilt = kv_scatter.prepare_scatter_args(blocks, torch.device("cpu"))

    assert kv_scatter.multi_layer_scatter_into([], [], blocks) == []
    assert kv_scatter.multi_layer_scatter_into(
        [t_src], [t_dst], torch.zeros((0,), dtype=torch.int32)
    ) == [t_dst]
    with pytest.raises(ValueError, match="len\\(srcs\\)"):
        kv_scatter.multi_layer_scatter_into([t_src], [], blocks)

    # Tuple return vs single-tensor return with and without prebuilt_args
    monkeypatch.setattr(
        kv_scatter,
        "_make_multi_layer_scatter_kernel",
        lambda n: lambda nc, so, do, *args: (t_dst,),
    )
    out_tuple = kv_scatter.multi_layer_scatter_into(
        [t_src], [t_dst], blocks, prebuilt_args=prebuilt
    )
    assert len(out_tuple) == 1

    monkeypatch.setattr(
        kv_scatter,
        "_make_multi_layer_scatter_kernel",
        lambda n: lambda nc, so, do, *args: t_dst,
    )
    out_single = kv_scatter.multi_layer_scatter_into([t_src], [t_dst], blocks)
    assert len(out_single) == 1

    # 4. Test smoke_test_multi_layer_scatter branches (match, mismatch, exception)
    def _fake_scatter_match(srcs, dsts, local_blocks, **kwargs):
        outs = []
        for s, d in zip(srcs, dsts):
            out = d.clone()
            for i, idx in enumerate(local_blocks.tolist()):
                out[idx] = s[i]
            outs.append(out)
        return outs

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kv_scatter, "multi_layer_scatter_into", _fake_scatter_match)
        assert (
            kv_scatter.smoke_test_multi_layer_scatter(
                torch.device("cpu"), num_layers=2, num_blocks=2, trailing_shape=(4,)
            )
            is True
        )

        mp.setattr(
            kv_scatter, "multi_layer_scatter_into", lambda s, d, b, **kw: list(d)
        )
        assert (
            kv_scatter.smoke_test_multi_layer_scatter(
                torch.device("cpu"), num_layers=1, num_blocks=2, trailing_shape=(4,)
            )
            is False
        )

        def _raise_scatter(*args, **kwargs):
            raise RuntimeError("scatter failed")

        mp.setattr(kv_scatter, "multi_layer_scatter_into", _raise_scatter)
        assert (
            kv_scatter.smoke_test_multi_layer_scatter(
                torch.device("cpu"), num_layers=1, num_blocks=2, trailing_shape=(4,)
            )
            is False
        )

    # 5. Test _jax_import_error is not None branches
    monkeypatch.setattr(kv_scatter, "_jax_import_error", ImportError("no jax"))
    assert kv_scatter.scatter_available() is False
    with pytest.raises(RuntimeError, match="KV scatter unavailable"):
        kv_scatter.multi_layer_scatter_into([t_src], [t_dst], blocks)
    assert kv_scatter.smoke_test_multi_layer_scatter(torch.device("cpu")) is False


def test_byte_spans_validation_and_edges() -> None:
    # owned_token_ranges errors
    with pytest.raises(ValueError, match="num_tokens must be positive"):
        byte_spans.owned_token_ranges(
            num_tokens=0, transfer_rank=0, parallelism=1, interleave_tokens=1
        )
    with pytest.raises(ValueError, match="parallelism must be positive"):
        byte_spans.owned_token_ranges(
            num_tokens=4, transfer_rank=0, parallelism=0, interleave_tokens=1
        )
    with pytest.raises(ValueError, match="outside parallelism"):
        byte_spans.owned_token_ranges(
            num_tokens=4, transfer_rank=2, parallelism=2, interleave_tokens=1
        )
    with pytest.raises(ValueError, match="interleave_tokens must be positive"):
        byte_spans.owned_token_ranges(
            num_tokens=4, transfer_rank=0, parallelism=2, interleave_tokens=0
        )

    # lower_fa_spans errors
    with pytest.raises(ValueError, match="page_tokens must be positive"):
        byte_spans.lower_fa_spans(
            num_tokens=4,
            transfer_rank=0,
            parallelism=1,
            interleave_tokens=1,
            page_tokens=0,
            token_bytes=16,
            block_ids=[0],
        )
    with pytest.raises(ValueError, match="token_bytes must be positive"):
        byte_spans.lower_fa_spans(
            num_tokens=4,
            transfer_rank=0,
            parallelism=1,
            interleave_tokens=1,
            page_tokens=4,
            token_bytes=0,
            block_ids=[0],
        )
    with pytest.raises(ValueError, match="block_ids do not match"):
        byte_spans.lower_fa_spans(
            num_tokens=4,
            transfer_rank=0,
            parallelism=1,
            interleave_tokens=1,
            page_tokens=4,
            token_bytes=16,
            block_ids=[0, 1],
        )
    fa_reg = byte_spans.lower_fa_spans(
        num_tokens=6,
        transfer_rank=0,
        parallelism=1,
        interleave_tokens=2,
        page_tokens=4,
        token_bytes=16,
        block_ids=[0, 1],
    )
    assert fa_reg.declared_bytes == 6 * 16
    assert len(fa_reg.spans) == 2

    # lower_gdn_state_shard_spans errors
    with pytest.raises(ValueError, match="must not be empty"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="", block_id=0, transfer_rank=0, parallelism=1, regions=[]
        )
    with pytest.raises(ValueError, match="physical granule must be positive"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.ssm",
            block_id=0,
            transfer_rank=0,
            parallelism=1,
            regions=[],
            physical_granule_bytes=0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.ssm", block_id=-1, transfer_rank=0, parallelism=1, regions=[]
        )
    with pytest.raises(ValueError, match="parallelism must be positive"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.ssm", block_id=0, transfer_rank=0, parallelism=0, regions=[]
        )
    with pytest.raises(ValueError, match="outside parallelism"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.ssm", block_id=0, transfer_rank=1, parallelism=1, regions=[]
        )
    with pytest.raises(ValueError, match="unsupported GDN state tag"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="fa", block_id=0, transfer_rank=0, parallelism=1, regions=[]
        )

    ssm_reg = {
        "name": "gdn_ssm",
        "offset_bytes": 0,
        "stride_bytes": 1024,
        "unit_bytes": 1024,
        "num_units": 1,
        "units_per_stride": 1,
    }
    with pytest.raises(ValueError, match="duplicate GDN state region"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.ssm",
            block_id=0,
            transfer_rank=0,
            parallelism=1,
            regions=[ssm_reg, ssm_reg],
        )
    with pytest.raises(ValueError, match="invalid GDN state region"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.ssm",
            block_id=0,
            transfer_rank=0,
            parallelism=1,
            regions=[{**ssm_reg, "unit_bytes": 0}],
        )
    with pytest.raises(ValueError, match="requires exactly gdn_ssm"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.ssm",
            block_id=0,
            transfer_rank=0,
            parallelism=1,
            regions=[],
        )
    with pytest.raises(ValueError, match="must be dense and head-major"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.ssm",
            block_id=0,
            transfer_rank=0,
            parallelism=1,
            regions=[{**ssm_reg, "offset_bytes": 64}],
        )
    with pytest.raises(ValueError, match="whole physical tokens"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.ssm",
            block_id=0,
            transfer_rank=0,
            parallelism=1,
            regions=[{**ssm_reg, "unit_bytes": 512, "stride_bytes": 512}],
            physical_granule_bytes=1024,
        )
    with pytest.raises(ValueError, match="destination must be token aligned"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.ssm",
            block_id=0,
            transfer_rank=0.5,  # type: ignore[arg-type]
            parallelism=2,
            regions=[ssm_reg],
            physical_granule_bytes=1024,
        )

    # Conv region errors in lower_gdn_state_shard_spans
    qk_reg = {
        "name": "gdn_conv_qk",
        "offset_bytes": 0,
        "stride_bytes": 2048,
        "unit_bytes": 1024,
        "num_units": 3,
        "units_per_stride": 1,
    }
    v_reg = {
        "name": "gdn_conv_v",
        "offset_bytes": 1024,
        "stride_bytes": 2048,
        "unit_bytes": 1024,
        "num_units": 3,
        "units_per_stride": 1,
    }
    with pytest.raises(ValueError, match="qk/v regions disagree on tap count"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.conv",
            block_id=0,
            transfer_rank=0,
            parallelism=1,
            regions=[qk_reg, {**v_reg, "num_units": 2}],
        )
    q_reg = {**qk_reg, "name": "gdn_conv_q", "stride_bytes": 3072}
    k_reg = {
        **qk_reg,
        "name": "gdn_conv_k",
        "offset_bytes": 1024,
        "stride_bytes": 3072,
    }
    v3_reg = {**v_reg, "offset_bytes": 2048, "stride_bytes": 3072}
    with pytest.raises(ValueError, match="q/k/v regions disagree on tap count"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.conv",
            block_id=0,
            transfer_rank=0,
            parallelism=1,
            regions=[q_reg, {**k_reg, "num_units": 2}, v3_reg],
        )
    with pytest.raises(ValueError, match="must be dense tap-major"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.conv",
            block_id=0,
            transfer_rank=0,
            parallelism=1,
            regions=[qk_reg, {**v_reg, "offset_bytes": 512}],
        )
    with pytest.raises(ValueError, match="share the tap row stride"):
        byte_spans.lower_gdn_state_shard_spans(
            tag="gdn.conv",
            block_id=0,
            transfer_rank=0,
            parallelism=1,
            regions=[qk_reg, {**v_reg, "stride_bytes": 4096}],
        )

    # lower_kda_state_shard_spans errors
    with pytest.raises(ValueError, match="must not be empty"):
        byte_spans.lower_kda_state_shard_spans(
            tag="",
            block_id=0,
            transfer_rank=0,
            parallelism=1,
            dst_shards=1,
            regions=[],
        )
    with pytest.raises(ValueError, match="non-negative"):
        byte_spans.lower_kda_state_shard_spans(
            tag="gdn.conv",
            block_id=-1,
            transfer_rank=0,
            parallelism=1,
            dst_shards=1,
            regions=[],
        )
    with pytest.raises(ValueError, match="outside parallelism"):
        byte_spans.lower_kda_state_shard_spans(
            tag="gdn.conv",
            block_id=0,
            transfer_rank=2,
            parallelism=2,
            dst_shards=1,
            regions=[],
        )
    with pytest.raises(ValueError, match="divide source parallelism"):
        byte_spans.lower_kda_state_shard_spans(
            tag="gdn.conv",
            block_id=0,
            transfer_rank=0,
            parallelism=2,
            dst_shards=3,
            regions=[],
        )
    with pytest.raises(ValueError, match="one rank-blocked region"):
        byte_spans.lower_kda_state_shard_spans(
            tag="gdn.conv",
            block_id=0,
            transfer_rank=0,
            parallelism=2,
            dst_shards=1,
            regions=[],
        )
    with pytest.raises(ValueError, match="aligned rank-blocked rows"):
        byte_spans.lower_kda_state_shard_spans(
            tag="gdn.conv",
            block_id=0,
            transfer_rank=0,
            parallelism=2,
            dst_shards=1,
            regions=[
                {
                    "name": "bad",
                    "offset_bytes": 0,
                    "stride_bytes": 8,
                    "unit_bytes": 8,
                    "num_units": 1,
                    "units_per_stride": 1,
                }
            ],
        )
    with pytest.raises(ValueError, match="unsupported KDA state tag"):
        byte_spans.lower_kda_state_shard_spans(
            tag="fa",
            block_id=0,
            transfer_rank=0,
            parallelism=2,
            dst_shards=1,
            regions=[],
        )

    # lower_glm_row_spans and lower_dsv4_row_spans errors
    for fn in (byte_spans.lower_glm_row_spans, byte_spans.lower_dsv4_row_spans):
        with pytest.raises(ValueError, match="must not be empty"):
            fn(
                tag="",
                num_tokens=4,
                transfer_rank=0,
                parallelism=1,
                page_tokens=4,
                live_bytes_per_block=16,
                row_bytes=4,
                block_ids=[0],
            )
        with pytest.raises(ValueError, match="num_tokens must be positive"):
            fn(
                tag="t",
                num_tokens=0,
                transfer_rank=0,
                parallelism=1,
                page_tokens=4,
                live_bytes_per_block=16,
                row_bytes=4,
                block_ids=[0],
            )
        with pytest.raises(ValueError, match="page_tokens must be positive"):
            fn(
                tag="t",
                num_tokens=4,
                transfer_rank=0,
                parallelism=1,
                page_tokens=0,
                live_bytes_per_block=16,
                row_bytes=4,
                block_ids=[0],
            )
        with pytest.raises(ValueError, match="parallelism must be positive"):
            fn(
                tag="t",
                num_tokens=4,
                transfer_rank=0,
                parallelism=0,
                page_tokens=4,
                live_bytes_per_block=16,
                row_bytes=4,
                block_ids=[0],
            )
        with pytest.raises(ValueError, match="outside parallelism"):
            fn(
                tag="t",
                num_tokens=4,
                transfer_rank=1,
                parallelism=1,
                page_tokens=4,
                live_bytes_per_block=16,
                row_bytes=4,
                block_ids=[0],
            )
        with pytest.raises(ValueError, match="must be positive"):
            fn(
                tag="t",
                num_tokens=4,
                transfer_rank=0,
                parallelism=1,
                page_tokens=4,
                live_bytes_per_block=16,
                row_bytes=0,
                block_ids=[0],
            )
        with pytest.raises(ValueError, match="whole rows"):
            fn(
                tag="t",
                num_tokens=4,
                transfer_rank=0,
                parallelism=1,
                page_tokens=4,
                live_bytes_per_block=15,
                row_bytes=4,
                block_ids=[0],
            )
        with pytest.raises(ValueError, match="complete request page set"):
            fn(
                tag="t",
                num_tokens=4,
                transfer_rank=0,
                parallelism=1,
                page_tokens=4,
                live_bytes_per_block=16,
                row_bytes=4,
                block_ids=[0, 1],
            )


def test_layout_fingerprint_validation_and_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert layout_fp.canonical_layout_fingerprint("  abc  ") == "abc"
    with pytest.raises(ValueError, match="must not be empty"):
        layout_fp.canonical_layout_fingerprint("   ")
    with pytest.raises(TypeError, match="string or mapping"):
        layout_fp.canonical_layout_fingerprint(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="PoolManifest"):
        layout_fp.fa_page_tokens("not_manifest")  # type: ignore[arg-type]

    r_valid = RegionSpec("fa", 0, 1024, 16, 16, 1)
    r_zero = RegionSpec("fa", 0, 1024, 16, 16, 1)
    object.__setattr__(r_zero, "num_units", 0)

    # Non-FA pool only -> ValueError
    m_no_fa = _mk_manifest(
        pools=(_mk_pool("gdn.ssm", "l0", 0, 1024, 4, (r_valid,)),),
        storages=(torch.zeros((4, 1024), dtype=torch.uint8),),
    )
    with pytest.raises(ValueError, match="requires an admitted FA pool"):
        layout_fp.fa_page_tokens(m_no_fa)

    # Multi-region FA pool -> ValueError
    m_two_reg = _mk_manifest(
        pools=(_mk_pool("fa", "l0", 0, 2048, 4, (r_valid, r_valid)),),
        storages=(torch.zeros((4, 2048), dtype=torch.uint8),),
    )
    with pytest.raises(ValueError, match="one live region"):
        layout_fp.fa_page_tokens(m_two_reg)

    # Zero tokens FA pool -> ValueError
    m_zero_tok = _mk_manifest(
        pools=(_mk_pool("fa", "l0", 0, 1024, 4, (r_zero,)),),
        storages=(torch.zeros((4, 1024), dtype=torch.uint8),),
    )
    with pytest.raises(ValueError, match="must be positive"):
        layout_fp.fa_page_tokens(m_zero_tok)

    # _default_layout_getter via monkeypatch
    from torch_tpu._internal.compile import tpu_torch_compile

    monkeypatch.setattr(
        tpu_torch_compile,
        "get_device_layout_if_materialized",
        lambda t: ((4, 3, 2, 1, 0), ((4, 128), (4, 1)), 8),
    )
    assert layout_fp._default_layout_getter(None) == (
        (4, 3, 2, 1, 0),
        ((4, 128), (4, 1)),
        8,
    )

    with pytest.raises(TypeError, match="PoolManifest"):
        layout_fp.measured_fa_layout_fingerprint("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires an admitted FA pool"):
        layout_fp.measured_fa_layout_fingerprint(m_no_fa)

    m_bad_idx = _mk_manifest(
        pools=(_mk_pool("fa", "l0", 5, 1024, 4, (r_valid,)),),
        storages=(),
    )
    with pytest.raises(ValueError, match="outside the manifest"):
        layout_fp.measured_fa_layout_fingerprint(m_bad_idx)

    m_fa = _mk_manifest(
        pools=(_mk_pool("fa", "l0", 0, 1024, 4, (r_valid,)),),
        storages=(torch.zeros((4, 16, 1, 4, 128), dtype=torch.uint8),),
    )
    with pytest.raises(RuntimeError, match="no materialized TPU layout"):
        layout_fp.measured_fa_layout_fingerprint(m_fa, layout_getter=lambda _: None)

    # Kimi K3 validation errors
    with pytest.raises(ValueError, match="state_fragment_heads must be positive"):
        layout_fp.measured_kimi_k3_layout_fingerprint(
            m_fa, page_tokens=16, state_fragment_heads=0, state_fragments=1
        )
    with pytest.raises(ValueError, match="state_fragments must be positive"):
        layout_fp.measured_kimi_k3_layout_fingerprint(
            m_fa, page_tokens=16, state_fragment_heads=1, state_fragments=0
        )
    with pytest.raises(ValueError, match="requires an FA pool"):
        layout_fp.measured_kimi_k3_layout_fingerprint(
            m_no_fa, page_tokens=16, state_fragment_heads=1, state_fragments=1
        )

    t4 = torch.zeros((4, 4, 4, 128), dtype=torch.uint8)

    def getter_4d(_: object):
        return ((3, 2, 1, 0), ((4, 128), (4, 1)), 8)

    r_state_odd = RegionSpec("s", 0, 10, 10, 1, 1)
    m_kimi_unaligned = _mk_manifest(
        pools=(
            _mk_pool("fa", "l0", 0, 2048, 4, (r_valid,)),
            _mk_pool("gdn.conv", "l0", 0, 2048, 4, (r_state_odd,)),
        ),
        storages=(t4,),
    )
    with pytest.raises(ValueError, match="divide into source-sized fragments"):
        layout_fp.measured_kimi_k3_layout_fingerprint(
            m_kimi_unaligned,
            page_tokens=16,
            state_fragment_heads=1,
            state_fragments=3,
            layout_getter=getter_4d,
            package_version=lambda _: "1.0",
        )

    r_state_16 = RegionSpec("s", 0, 16, 16, 1, 1)
    r_state_32 = RegionSpec("s", 0, 32, 32, 1, 1)
    m_kimi_disagree = _mk_manifest(
        pools=(
            _mk_pool("fa", "l0", 0, 2048, 4, (r_valid,)),
            _mk_pool("gdn.conv.l0", "l0", 0, 2048, 4, (r_state_16,)),
            _mk_pool("gdn.conv.l1", "l1", 0, 2048, 4, (r_state_32,)),
        ),
        storages=(t4,),
    )
    with pytest.raises(ValueError, match="disagree on live state"):
        layout_fp.measured_kimi_k3_layout_fingerprint(
            m_kimi_disagree,
            page_tokens=16,
            state_fragment_heads=1,
            state_fragments=2,
            layout_getter=getter_4d,
            package_version=lambda _: "1.0",
        )

    m_kimi_missing_ssm = _mk_manifest(
        pools=(
            _mk_pool("fa", "l0", 0, 2048, 4, (r_valid,)),
            _mk_pool("gdn.conv", "l0", 0, 2048, 4, (r_state_16,)),
        ),
        storages=(t4,),
    )
    with pytest.raises(ValueError, match="requires gdn.conv and gdn.ssm"):
        layout_fp.measured_kimi_k3_layout_fingerprint(
            m_kimi_missing_ssm,
            page_tokens=16,
            state_fragment_heads=1,
            state_fragments=2,
            layout_getter=getter_4d,
            package_version=lambda _: "1.0",
        )

    # _measure_packed_row_layouts errors
    with pytest.raises(TypeError, match="PoolManifest"):
        layout_fp._measure_packed_row_layouts(
            "bad",  # type: ignore[arg-type]
            page_tokens=16,
            layout_getter=getter_4d,
            include_page_shape=True,
        )
    with pytest.raises(ValueError, match="page_tokens must be positive"):
        layout_fp._measure_packed_row_layouts(
            m_fa, page_tokens=0, layout_getter=getter_4d, include_page_shape=True
        )
    with pytest.raises(RuntimeError, match="must be rank-4"):
        layout_fp._measure_packed_row_layouts(
            m_fa, page_tokens=16, layout_getter=getter_4d, include_page_shape=True
        )


def test_seq_on_lane_validation_and_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    geom = seq_on_lane.SolFaGeometry(kv_rows=4, packed_rows=32, page_tokens=128)
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="must divide"):
        geom.shard_bytes(0)

    with pytest.raises(seq_on_lane.SeqOnLaneError, match="empty dim"):
        seq_on_lane.sol_fa_geometry_from_shape((0, 4, 32, 4, 128))

    layout_hook = seq_on_lane.SolFaLayout()
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="kv_rows disagree"):
        layout_hook.regions(
            block_size_tokens=128,
            shape=(4, 4, 32, 4, 128),
            nbytes=4 * 4 * 32 * 4 * 128,
            num_kv_heads=4,
            head_size=128,
            itemsize=1,
        )
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="head_dim is smaller"):
        layout_hook.regions(
            block_size_tokens=128,
            shape=(4, 4, 8, 4, 128),
            nbytes=4 * 4 * 8 * 4 * 128,
            num_kv_heads=2,
            head_size=64,
            itemsize=1,
        )
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="byte size disagrees"):
        layout_hook.regions(
            block_size_tokens=128,
            shape=(4, 4, 32, 4, 128),
            nbytes=16,
            num_kv_heads=2,
            head_size=128,
            itemsize=1,
        )

    r_sol = seq_on_lane.sol_fa_regions(block_size_tokens=128, geometry=geom)[0]
    t_sol = torch.zeros((4, 4, 32, 4, 128), dtype=torch.uint8)
    m_no_fa = _mk_manifest(
        pools=(_mk_pool("gdn.ssm", "l0", 0, 1024, 4, (r_sol,)),),
        storages=(t_sol,),
    )
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="no FA pool"):
        seq_on_lane.sol_fa_geometry_from_manifest(m_no_fa)

    r_bad_name = RegionSpec("fa", 0, r_sol.stride_bytes, r_sol.unit_bytes, 1, 2)
    m_bad_name = _mk_manifest(
        pools=(_mk_pool("fa", "l0", 0, r_sol.stride_bytes, 4, (r_bad_name,)),),
        storages=(t_sol,),
    )
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="not admitted with"):
        seq_on_lane.sol_fa_geometry_from_manifest(m_bad_name)

    r_bad_stride = RegionSpec(
        seq_on_lane.SOL_REGION_NAME, 0, r_sol.stride_bytes * 2, r_sol.unit_bytes, 1, 2
    )
    m_bad_stride = _mk_manifest(
        pools=(_mk_pool("fa", "l0", 0, r_sol.stride_bytes, 4, (r_bad_stride,)),),
        storages=(t_sol,),
    )
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="regions disagree"):
        seq_on_lane.sol_fa_geometry_from_manifest(m_bad_stride)

    r_sol_2 = RegionSpec(
        seq_on_lane.SOL_REGION_NAME,
        0,
        r_sol.stride_bytes,
        r_sol.unit_bytes,
        2,
        r_sol.units_per_stride,
    )
    m_diff_pages = _mk_manifest(
        pools=(
            _mk_pool("fa.l0", "l0", 0, r_sol.stride_bytes, 4, (r_sol,)),
            _mk_pool("fa.l1", "l1", 0, r_sol.stride_bytes * 2, 4, (r_sol_2,)),
        ),
        storages=(t_sol,),
    )
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="differs across pools"):
        seq_on_lane.sol_fa_page_tokens(m_diff_pages)

    assert seq_on_lane.sol_fa_skip_bytes(0, geometry=geom, dst_shards=1) == 0

    with pytest.raises(seq_on_lane.SeqOnLaneError, match="must be positive"):
        seq_on_lane._tp_head_range(0, 1, 0)
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="outside degree"):
        seq_on_lane._tp_head_range(2, 2, 2)

    # lower_fa_spans_sol validation errors
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="num_tokens must be positive"):
        seq_on_lane.lower_fa_spans_sol(
            num_tokens=0,
            transfer_rank=0,
            parallelism=1,
            interleave_tokens=128,
            page_tokens=128,
            geometry=geom,
            dst_shards=1,
            block_ids=[0],
        )
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="outside parallelism"):
        seq_on_lane.lower_fa_spans_sol(
            num_tokens=128,
            transfer_rank=1,
            parallelism=1,
            interleave_tokens=128,
            page_tokens=128,
            geometry=geom,
            dst_shards=1,
            block_ids=[0],
        )
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="total_kv_heads is required"):
        seq_on_lane.lower_fa_spans_sol(
            num_tokens=128,
            transfer_rank=0,
            parallelism=1,
            interleave_tokens=128,
            page_tokens=128,
            geometry=geom,
            dst_shards=1,
            block_ids=[0],
            producer_tp_size=2,
            total_kv_heads=None,
        )
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="disagrees with TP ownership"):
        seq_on_lane.lower_fa_spans_sol(
            num_tokens=128,
            transfer_rank=0,
            parallelism=1,
            interleave_tokens=128,
            page_tokens=128,
            geometry=geom,
            dst_shards=1,
            block_ids=[0],
            producer_tp_size=2,
            total_kv_heads=2,
        )
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="DCP size must divide"):
        seq_on_lane.lower_fa_spans_sol(
            num_tokens=128,
            transfer_rank=0,
            parallelism=1,
            interleave_tokens=128,
            page_tokens=128,
            geometry=geom,
            dst_shards=2,
            block_ids=[0],
            dst_dcp_size=3,
        )
    with pytest.raises(
        seq_on_lane.SeqOnLaneError, match="DCP group ranks must own identical"
    ):
        seq_on_lane.lower_fa_spans_sol(
            num_tokens=128,
            transfer_rank=0,
            parallelism=1,
            interleave_tokens=128,
            page_tokens=128,
            geometry=geom,
            dst_shards=2,
            block_ids=[0],
            dst_dcp_size=2,
        )

    # Trigger start % kp_tokens != 0 via monkeypatched owned_token_ranges
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(seq_on_lane, "owned_token_ranges", lambda **kw: [(1, 128)])
        with pytest.raises(seq_on_lane.SeqOnLaneError, match="kernel page boundary"):
            seq_on_lane.lower_fa_spans_sol(
                num_tokens=128,
                transfer_rank=0,
                parallelism=1,
                interleave_tokens=128,
                page_tokens=128,
                geometry=geom,
                dst_shards=1,
                block_ids=[0],
            )

    # lower_gdn_state_shard_spans_sol and _lower_gdn_state_split_spans errors
    with pytest.raises(seq_on_lane.SeqOnLaneError, match="invalid producer"):
        seq_on_lane.lower_gdn_state_shard_spans_sol(
            tag="gdn.ssm",
            block_id=0,
            transfer_rank=-1,
            parallelism=1,
            regions=[],
            dst_shards=1,
        )
    with pytest.raises(
        seq_on_lane.SeqOnLaneError, match="destination shards must be positive"
    ):
        seq_on_lane.lower_gdn_state_shard_spans_sol(
            tag="gdn.ssm",
            block_id=0,
            transfer_rank=0,
            parallelism=1,
            regions=[],
            dst_shards=0,
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            seq_on_lane,
            "lower_gdn_state_shard_spans",
            lambda **kw: byte_spans.PoolSpanRegistration(
                tag="gdn.ssm", block_ids=(0,), spans=(), declared_bytes=0
            ),
        )
        with pytest.raises(seq_on_lane.SeqOnLaneError, match="no state spans"):
            seq_on_lane._lower_gdn_state_split_spans(
                tag="gdn.ssm", block_id=0, regions=[], dst_shards=2
            )

        mp.setattr(
            seq_on_lane,
            "lower_gdn_state_shard_spans",
            lambda **kw: byte_spans.PoolSpanRegistration(
                tag="gdn.ssm",
                block_ids=(0,),
                spans=(byte_spans.PoolByteSpan(0, 0, 0, 0, 1024),),
                declared_bytes=1024,
            ),
        )
        with pytest.raises(seq_on_lane.SeqOnLaneError, match="does not split"):
            seq_on_lane._lower_gdn_state_split_spans(
                tag="gdn.ssm", block_id=0, regions=[], dst_shards=3
            )

    # _default_layout_getter and measured_sol_layout_fingerprint errors
    from torch_tpu._internal.compile import tpu_torch_compile

    monkeypatch.setattr(
        tpu_torch_compile,
        "get_device_layout_if_materialized",
        lambda t: ((4, 3, 2, 1, 0), ((4, 128), (4, 1)), 8),
    )
    assert seq_on_lane._default_layout_getter(None) == (
        (4, 3, 2, 1, 0),
        ((4, 128), (4, 1)),
        8,
    )

    m_sol = _mk_manifest(
        pools=(_mk_pool("fa", "l0", 0, r_sol.stride_bytes, 4, (r_sol,)),),
        storages=(t_sol,),
    )
    with pytest.raises(RuntimeError, match="no materialized TPU layout"):
        seq_on_lane.measured_sol_layout_fingerprint(m_sol, layout_getter=lambda _: None)
    with pytest.raises(RuntimeError, match="minor-to-major gate"):
        seq_on_lane.measured_sol_layout_fingerprint(
            m_sol,
            layout_getter=lambda _: ((0, 1, 2, 3, 4), ((4, 128), (4, 1)), 8),
        )
    with pytest.raises(RuntimeError, match="element-size gate"):
        seq_on_lane.measured_sol_layout_fingerprint(
            m_sol,
            layout_getter=lambda _: ((4, 3, 2, 1, 0), ((4, 128), (4, 1)), 16),
        )
