# SPDX-License-Identifier: Apache-2.0
"""NaN round results survive transfer idleness and reach injected metrics."""

import logging
from functools import partial
from types import SimpleNamespace

import msgspec
import pytest
import torch
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.parser import text_string_to_metric_families
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorStats,
)

from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorPromMetrics,
    TpuKVConnectorStats,
)
from vllm_torchtpu.runner import model_nan_check, tpu_runner
from vllm_torchtpu.runner.model_nan_check import ModelNaNChecker

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def cpu_logger(monkeypatch, model_nan_cpu):
    monkeypatch.setattr(model_nan_check, "logger", logging.getLogger(__name__))


def test_zero_round_is_nonempty_and_survives_serialization_and_reset():
    stats = TpuKVConnectorStats()
    stats.record_model_nan(False)
    snapshot = stats.clone_and_reset()
    restored = TpuKVConnectorStats(
        data=msgspec.msgpack.decode(msgspec.msgpack.encode(snapshot.data))
    )
    assert not restored.is_empty()
    assert restored.data["model_nan_detected"] == [0]
    stats.record_model_nan(True)
    assert snapshot.data["model_nan_detected"] == [0]


@pytest.mark.parametrize("has_nan", [False, True])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_result_preserves_existing_samples_and_creates_missing_stats(
    has_nan, multi, existing
):
    path = ("TPURaidenConnector",) if multi else ()
    check = ModelNaNChecker(rank=1, stats_path=path)
    other = KVConnectorStats(data={"other": [7]})
    leaf = TpuKVConnectorStats() if existing else None
    if leaf is not None:
        leaf.record_kv_pull(2.5)
    original: KVConnectorStats | None = leaf
    if multi:
        original = MultiKVConnectorStats(data={"OtherConnector": other})
        if leaf is not None:
            original.data["TPURaidenConnector"] = leaf
    check.check_logits(torch.tensor([[float("nan") if has_nan else 1.0]]), 1)
    result = check.attach_stats(original)
    if multi:
        assert result is original
        assert result.data["OtherConnector"] is other
        result = result.data["TPURaidenConnector"]
    assert isinstance(result, TpuKVConnectorStats)
    assert result.data["model_nan_detected"] == [int(has_nan)]
    if leaf is not None:
        assert result is leaf
        assert result.data["kv_pull_time"] == [2.5]
    assert check.attach_stats(original) is original


def test_runner_collects_results_even_without_transfer_samples(monkeypatch):
    check = ModelNaNChecker(rank=1, stats_path=())
    check.check_logits(torch.tensor([[float("nan")]]), 1)
    connector = SimpleNamespace(
        get_finished=lambda _: (None, None),
        get_block_ids_with_load_errors=lambda: set(),
        build_connector_worker_meta=lambda: None,
        get_kv_connector_stats=lambda: None,
        clear_connector_metadata=lambda: None,
    )
    runner = SimpleNamespace(
        _nan_checker=check,
        _stage_timing=SimpleNamespace(take_samples=lambda: ([], [])),
    )
    monkeypatch.setattr(tpu_runner, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(tpu_runner, "get_kv_transfer_group", lambda: connector)
    output = tpu_runner.TPUModelRunner.get_finished_kv_transfers(
        runner, SimpleNamespace(finished_req_ids=set())
    )
    assert output[-1].data["model_nan_detected"] == [1]
    assert check.take_result() is None


def test_single_counter_counts_anomalous_rounds_once_per_engine(monkeypatch):
    registry = CollectorRegistry()
    monkeypatch.setattr(
        TpuKVConnectorPromMetrics, "_get_raiden_metric_definitions", lambda *_: {}
    )
    metrics = TpuKVConnectorPromMetrics(
        SimpleNamespace(kv_transfer_config=None),
        {
            Gauge: partial(Gauge, registry=registry),
            Counter: partial(Counter, registry=registry),
            Histogram: partial(Histogram, registry=registry),
        },
        ["model_name", "engine"],
        {0: ["model", "0"], 1: ["model", "1"]},
    )
    for flags in ([0, 0], [0, 1, 0], [0], [], [1, 1]):
        metrics.observe({"model_nan_detected": flags}, engine_idx=0)
    metrics.observe({}, engine_idx=0)
    metrics.observe({"model_nan_detected": [0]}, engine_idx=1)
    values = {
        (sample.name, sample.labels["engine"]): sample.value
        for family in text_string_to_metric_families(generate_latest(registry).decode())
        for sample in family.samples
        if sample.name
        in {
            "vllm:tpu_model_nan_rounds_total",
            "vllm:tpu_model_checked_rounds_total",
        }
    }
    assert values == {
        ("vllm:tpu_model_nan_rounds_total", "0"): 2.0,
        ("vllm:tpu_model_nan_rounds_total", "1"): 0.0,
    }


def test_configuration_uses_multi_connector_class_keys_and_rejects_multiple_sinks(
    monkeypatch,
):
    class TpuLeaf:
        @classmethod
        def build_kv_connector_stats(cls):
            return TpuKVConnectorStats()

    class OtherLeaf:
        @classmethod
        def build_kv_connector_stats(cls):
            return None

    class Multi(MultiConnector):
        children = [TpuLeaf, OtherLeaf]

        @classmethod
        def _get_connector_classes_and_configs(cls, config):
            return [(child, config) for child in cls.children]

    monkeypatch.setattr(KVConnectorFactory, "get_connector_class", lambda _: Multi)
    config = SimpleNamespace(
        kv_transfer_config=object(),
        parallel_config=SimpleNamespace(data_parallel_rank=2),
    )
    check = ModelNaNChecker.from_vllm_config(rank=1, vllm_config=config)
    assert check.engine_idx == 2
    check.check_logits(torch.ones(1, 2), 1)
    result = check.attach_stats(None)
    assert result.data["TpuLeaf"].data["model_nan_detected"] == [0]

    class SecondTpuLeaf(TpuLeaf):
        pass

    Multi.children = [TpuLeaf, SecondTpuLeaf]
    with pytest.raises(ValueError, match="one TPU stats"):
        ModelNaNChecker.from_vllm_config(rank=1, vllm_config=config)
