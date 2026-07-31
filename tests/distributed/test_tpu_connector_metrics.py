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
"""Unit tests for TPU KV Connector Prometheus telemetry metrics."""

from unittest.mock import MagicMock
import pytest

from vllm_torchtpu.distributed.kv_transfer.tpu_connector_stats import (
    TpuKVConnectorStats,
    TpuKVConnectorPromMetrics,
)


def test_tpu_kv_connector_stats_recording():
    """Test recording of P2P, H2D, and D2H events into TpuKVConnectorStats."""
    stats = TpuKVConnectorStats()
    assert stats.is_empty()

    # Record events
    stats.record_p2p_transfer_event(
        bytes_transferred=1024 * 1024 * 16,
        duration_seconds=0.005,
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_rank="0",
        dst_rank="0",
        stream_id="0",
    )
    stats.record_h2d_transfer_event(
        bytes_transferred=1024 * 1024 * 8,
        duration_seconds=0.002,
        host_ip="10.0.0.2",
        device_id="0",
        transfer_kind="kv_cache",
    )
    stats.record_d2h_transfer_event(
        bytes_transferred=1024 * 1024 * 8,
        duration_seconds=0.003,
        host_ip="10.0.0.1",
        device_id="0",
        transfer_kind="kv_cache",
    )

    assert not stats.is_empty()
    assert len(stats.data["p2p_transfer_events"]) == 1
    assert len(stats.data["h2d_transfer_events"]) == 1
    assert len(stats.data["d2h_transfer_events"]) == 1

    p2p_evt = stats.data["p2p_transfer_events"][0]
    assert p2p_evt["src_ip"] == "10.0.0.1"
    assert p2p_evt["dst_ip"] == "10.0.0.2"
    assert p2p_evt["bytes"] == 1024 * 1024 * 16

    # Test clone_and_reset
    cloned = stats.clone_and_reset()
    assert stats.is_empty()
    assert not cloned.is_empty()


def test_tpu_kv_connector_prom_metrics_observe():
    """Test Prometheus counter observation with labeled events."""
    mock_config = MagicMock()
    mock_config.parallel_config.tensor_parallel_size = 1

    # Simple mock Prometheus classes
    class MockCounter:
        def __init__(self, name, documentation, labelnames=None):
            self.name = name
            self.labelnames = labelnames
            self.labeled_counters = {}

        def labels(self, *args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key not in self.labeled_counters:
                self.labeled_counters[key] = MockLabeledCounter()
            return self.labeled_counters[key]

    class MockLabeledCounter:
        def __init__(self):
            self.val = 0

        def inc(self, amount=1):
            self.val += amount

    class MockHistogram:
        def __init__(self, name, documentation, buckets=None, labelnames=None):
            self.name = name
            self.observed = []

        def labels(self, *args, **kwargs):
            return self

        def observe(self, val):
            self.observed.append(val)

    class MockGauge:
        def __init__(self, name, documentation, labelnames=None):
            self.name = name
            self.val = 0

        def labels(self, *args, **kwargs):
            return self

        def set(self, val):
            self.val = val

    labelnames = ["engine_id"]
    per_engine_labelvalues = {0: ["0"]}

    from prometheus_client import Counter, Gauge, Histogram

    prom_metrics = TpuKVConnectorPromMetrics(
        vllm_config=mock_config,
        metric_types={
            Counter: MockCounter,
            Gauge: MockGauge,
            Histogram: MockHistogram,
        },
        labelnames=labelnames,
        per_engine_labelvalues=per_engine_labelvalues,
    )

    stats_data = {
        "p2p_transfer_events": [
            {
                "bytes": 1048576,
                "duration": 0.01,
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.2",
                "src_rank": "0",
                "dst_rank": "1",
                "stream_id": "0",
            }
        ],
        "h2d_transfer_events": [
            {
                "bytes": 524288,
                "duration": 0.005,
                "host_ip": "10.0.0.2",
                "device_id": "0",
                "transfer_kind": "kv_cache",
            }
        ],
        "d2h_transfer_events": [
            {
                "bytes": 524288,
                "duration": 0.004,
                "host_ip": "10.0.0.1",
                "device_id": "0",
                "transfer_kind": "kv_cache",
            }
        ],
    }

    prom_metrics.observe(stats_data, engine_idx=0)

    # Verify P2P counters recorded
    p2p_bytes_counter = prom_metrics.counter_p2p_transfer_bytes_total
    p2p_duration_counter = prom_metrics.counter_p2p_transfer_duration_seconds_total
    
    assert len(p2p_bytes_counter.labeled_counters) == 1
    assert len(p2p_duration_counter.labeled_counters) == 1

    # Verify H2D and D2H counters recorded
    h2d_bytes_counter = prom_metrics.counter_h2d_transfer_bytes_total
    d2h_bytes_counter = prom_metrics.counter_d2h_transfer_bytes_total
    assert len(h2d_bytes_counter.labeled_counters) == 1
    assert len(d2h_bytes_counter.labeled_counters) == 1
