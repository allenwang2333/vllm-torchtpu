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
"""Stats and Prometheus metrics for the TPU KV connector."""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics, KVConnectorStats, PromMetric, PromMetricT)
from vllm.v1.metrics.utils import create_metric_per_engine


@dataclass
class TpuKVConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics"""

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def record_d2h_transfer(self, d2h_transfer_time):
        """Record a D2H transfer operation."""
        self.data["d2h_transfer_time"].append(d2h_transfer_time)

    def record_h2d_transfer(self, h2d_transfer_time):
        """Record a H2D transfer operation."""
        self.data["h2d_transfer_time"].append(h2d_transfer_time)

    def record_kv_pull(self, kv_pull_time):
        """Record a KV pull operation."""
        self.data["kv_pull_time"].append(kv_pull_time)

    def record_mb_transferred(self, mb_transferred):
        """Record a successful KV transfer operation."""
        self.data["mb_transferred"].append(mb_transferred)

    def record_failed_transfer(self):
        """Record a failed KV transfer operation."""
        self.data["num_failed_transfers"].append(1)

    def record_prefill_queue_length(self, prefill_queue_length: int):
        """Record the prefill (send) queue length."""
        self.data["prefill_queue_length"].append(prefill_queue_length)

    def record_decode_queue_length(self, decode_queue_length: int):
        """Record the decode (load) queue length."""
        self.data["decode_queue_length"].append(decode_queue_length)

    def record_p2p_transfer_event(
        self,
        bytes_transferred: float,
        duration_seconds: float,
        src_ip: str,
        dst_ip: str,
        src_rank: str,
        dst_rank: str,
        stream_id: str,
    ):
        """Record a P2P inter-host transfer event with metadata."""
        self.data["p2p_transfer_events"].append({
            "bytes": float(bytes_transferred),
            "duration": float(duration_seconds),
            "src_ip": str(src_ip),
            "dst_ip": str(dst_ip),
            "src_rank": str(src_rank),
            "dst_rank": str(dst_rank),
            "stream_id": str(stream_id),
        })

    def record_h2d_transfer_event(
        self,
        bytes_transferred: float,
        duration_seconds: float,
        host_ip: str,
        device_id: str,
        transfer_kind: str = "kv_cache",
    ):
        """Record a Host-to-Device transfer event with metadata."""
        self.data["h2d_transfer_events"].append({
            "bytes": float(bytes_transferred),
            "duration": float(duration_seconds),
            "host_ip": str(host_ip),
            "device_id": str(device_id),
            "transfer_kind": str(transfer_kind),
        })

    def record_d2h_transfer_event(
        self,
        bytes_transferred: float,
        duration_seconds: float,
        host_ip: str,
        device_id: str,
        transfer_kind: str = "kv_cache",
    ):
        """Record a Device-to-Host transfer event with metadata."""
        self.data["d2h_transfer_events"].append({
            "bytes": float(bytes_transferred),
            "duration": float(duration_seconds),
            "host_ip": str(host_ip),
            "device_id": str(device_id),
            "transfer_kind": str(transfer_kind),
        })

    def reset(self):
        # Must be serializable
        self.data: dict[str, list[Any]] = {
            "d2h_transfer_time": [],
            "h2d_transfer_time": [],
            "kv_pull_time": [],
            "mb_transferred": [],
            "num_failed_transfers": [],
            "prefill_queue_length": [],
            "decode_queue_length": [],
            "p2p_transfer_events": [],
            "h2d_transfer_events": [],
            "d2h_transfer_events": [],
        }

    def clone_and_reset(self) -> "TpuKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if not other.is_empty():
            for k, v in other.data.items():
                if k not in self.data:
                    self.data[k] = []
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        d2h_transfer_time = np.asarray(self.data["d2h_transfer_time"])
        h2d_transfer_time = np.asarray(self.data["h2d_transfer_time"])
        kv_pull_time = np.asarray(self.data["kv_pull_time"])
        mb_transferred = np.asarray(self.data["mb_transferred"])
        prefill_queue_length = np.asarray(
            self.data.get("prefill_queue_length", []))
        decode_queue_length = np.asarray(
            self.data.get("decode_queue_length", []))

        total_mb = mb_transferred.sum()
        avg_mb = total_mb / self.num_successful_transfers if self.num_successful_transfers > 0 else 0

        total_time_seconds = kv_pull_time.sum() / 1e3
        throughput_mb_s = total_mb / total_time_seconds if total_time_seconds > 0 else 0

        return {
            "Avg D2H transfer time (ms)":
            round(d2h_transfer_time.mean(), 3)
            if d2h_transfer_time.size > 0 else 0.0,
            "P90 D2H transfer time (ms)":
            round(np.percentile(d2h_transfer_time, 90).item(), 3)
            if d2h_transfer_time.size > 0 else 0.0,
            "Avg H2D transfer time (ms)":
            round(h2d_transfer_time.mean(), 3)
            if h2d_transfer_time.size > 0 else 0.0,
            "P90 H2D transfer time (ms)":
            round(np.percentile(h2d_transfer_time, 90).item(), 3)
            if h2d_transfer_time.size > 0 else 0.0,
            "Num successful transfers":
            self.num_successful_transfers,
            "Num failed transfers":
            self.data['num_failed_transfers'],
            "Avg KV pull time (ms)":
            round(kv_pull_time.mean(), 3) if kv_pull_time.size > 0 else 0.0,
            "P90 KV pull time (ms)":
            round(np.percentile(kv_pull_time, 90).item(), 3)
            if kv_pull_time.size > 0 else 0.0,
            "Avg MB per transfer":
            round(avg_mb, 3),
            "Throughput (MB/s)":
            round(throughput_mb_s, 3),
            "Prefill queue length":
            int(prefill_queue_length.max())
            if prefill_queue_length.size > 0 else 0,
            "Decode queue length":
            int(decode_queue_length.max())
            if decode_queue_length.size > 0 else 0,
        }

    def is_empty(self) -> bool:
        return (len(self.data["d2h_transfer_time"]) == 0
                and len(self.data["h2d_transfer_time"]) == 0
                and len(self.data["kv_pull_time"]) == 0
                and len(self.data["mb_transferred"]) == 0
                and len(self.data["num_failed_transfers"]) == 0
                and len(self.data["prefill_queue_length"]) == 0
                and len(self.data["decode_queue_length"]) == 0
                and len(self.data.get("p2p_transfer_events", [])) == 0
                and len(self.data.get("h2d_transfer_events", [])) == 0
                and len(self.data.get("d2h_transfer_events", [])) == 0)

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["mb_transferred"])


class TpuKVConnectorPromMetrics(KVConnectorPromMetrics):

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames,
                         per_engine_labelvalues)

        # Buckets for time-based metrics in milliseconds
        buckets = [
            1.0,
            10.0,
            100.0,
            250.0,
            500.0,
            750.0,
            1000.0,
            2500.0,
            5000.0,
            7500.0,
            10000.0,
            25000.0,
            50000.0,
        ]
        tpu_histogram_d2h_transfer_time = self._histogram_cls(
            name="vllm:tpu_d2h_transfer_time_ms",
            documentation=
            "Histogram of D2H transfer duration for TPU KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.tpu_histogram_d2h_transfer_time = create_metric_per_engine(
            tpu_histogram_d2h_transfer_time, self.per_engine_labelvalues)
        tpu_histogram_h2d_transfer_time = self._histogram_cls(
            name="vllm:tpu_h2d_transfer_time_ms",
            documentation=
            "Histogram of H2D transfer duration for TPU KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.tpu_histogram_h2d_transfer_time = create_metric_per_engine(
            tpu_histogram_h2d_transfer_time, self.per_engine_labelvalues)
        tpu_histogram_kv_pull_time = self._histogram_cls(
            name="vllm:tpu_kv_pull_time_ms",
            documentation=
            "Histogram of KV pull duration for TPU KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.tpu_histogram_kv_pull_time = create_metric_per_engine(
            tpu_histogram_kv_pull_time, self.per_engine_labelvalues)
        # Buckets for data transferred
        buckets = [
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
        ]
        tpu_histogram_kv_megabytes_transferred = self._histogram_cls(
            name="vllm:tpu_kv_megabytes_transferred",
            documentation=
            "Histogram of megabytes transferred per TPU KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.tpu_histogram_kv_megabytes_transferred = create_metric_per_engine(
            tpu_histogram_kv_megabytes_transferred,
            self.per_engine_labelvalues)
        counter_tpu_num_failed_transfers = self._counter_cls(
            name="vllm:tpu_num_failed_transfers",
            documentation="Number of failed TPU KV Cache transfers.",
            labelnames=labelnames,
        )
        self.counter_tpu_num_failed_transfers = create_metric_per_engine(
            counter_tpu_num_failed_transfers, self.per_engine_labelvalues)

        gauge_tpu_prefill_queue_length = self._gauge_cls(
            name="vllm:tpu_prefill_kv_queue_length",
            documentation=
            "Current length of the TPU KV transfer prefill queue.",
            labelnames=labelnames,
        )
        self.gauge_tpu_prefill_queue_length = create_metric_per_engine(
            gauge_tpu_prefill_queue_length, self.per_engine_labelvalues)

        gauge_tpu_decode_queue_length = self._gauge_cls(
            name="vllm:tpu_decode_kv_queue_length",
            documentation="Current length of the TPU KV transfer decode queue.",
            labelnames=labelnames,
        )
        self.gauge_tpu_decode_queue_length = create_metric_per_engine(
            gauge_tpu_decode_queue_length, self.per_engine_labelvalues)

        # Telemetry metrics required by P2P Observability Tool Design
        self.counter_p2p_transfer_bytes_total = self._counter_cls(
            name="test_p2p_transfer_bytes_total",
            documentation="Total bytes transferred over P2P network streams.",
            labelnames=labelnames + ["src_ip", "dst_ip", "src_rank", "dst_rank", "stream_id"],
        )

        self.counter_p2p_transfer_duration_seconds_total = self._counter_cls(
            name="test_p2p_transfer_duration_seconds_total",
            documentation="Total active duration (in seconds) spent executing P2P transfers.",
            labelnames=labelnames + ["src_ip", "dst_ip", "stream_id"],
        )

        self.counter_h2d_transfer_bytes_total = self._counter_cls(
            name="test_h2d_transfer_bytes_total",
            documentation="Total bytes copied from Host memory to Accelerator Device memory.",
            labelnames=labelnames + ["host_ip", "device_id", "transfer_kind"],
        )

        self.counter_h2d_transfer_duration_seconds_total = self._counter_cls(
            name="test_h2d_transfer_duration_seconds_total",
            documentation="Total active time spent on H2D transfers.",
            labelnames=labelnames + ["host_ip", "device_id", "transfer_kind"],
        )

        self.counter_d2h_transfer_bytes_total = self._counter_cls(
            name="test_d2h_transfer_bytes_total",
            documentation="Total bytes copied from Accelerator Device memory to Host memory.",
            labelnames=labelnames + ["host_ip", "device_id", "transfer_kind"],
        )

        self.counter_d2h_transfer_duration_seconds_total = self._counter_cls(
            name="test_d2h_transfer_duration_seconds_total",
            documentation="Total active time spent on D2H transfers.",
            labelnames=labelnames + ["host_ip", "device_id", "transfer_kind"],
        )

    def observe(self,
                transfer_stats_data: dict[str, Any],
                engine_idx: int = 0):
        for prom_obj, list_item_key in zip(
            [
                self.tpu_histogram_d2h_transfer_time,
                self.tpu_histogram_h2d_transfer_time,
                self.tpu_histogram_kv_pull_time,
                self.tpu_histogram_kv_megabytes_transferred,
            ],
            [
                "d2h_transfer_time",
                "h2d_transfer_time",
                "kv_pull_time",
                "mb_transferred",
            ],
        ):
            for list_item in transfer_stats_data.get(list_item_key, []):
                prom_obj[engine_idx].observe(list_item)

        for counter_obj, counter_item_key in zip(
            [
                self.counter_tpu_num_failed_transfers,
            ],
            ["num_failed_transfers"],
        ):
            for list_item in transfer_stats_data.get(counter_item_key, []):
                counter_obj[engine_idx].inc(list_item)

        for gauge_obj, gauge_item_key in zip(
            [
                self.gauge_tpu_prefill_queue_length,
                self.gauge_tpu_decode_queue_length,
            ],
            [
                "prefill_queue_length",
                "decode_queue_length",
            ],
        ):
            if gauge_item_key in transfer_stats_data and transfer_stats_data[
                    gauge_item_key]:
                for list_item in transfer_stats_data[gauge_item_key]:
                    gauge_obj[engine_idx].set(list_item)

        engine_labels = (
            dict(zip(self._labelnames, self.per_engine_labelvalues[engine_idx]))
            if engine_idx in self.per_engine_labelvalues else {}
        )

        for event in transfer_stats_data.get("p2p_transfer_events", []):
            self.counter_p2p_transfer_bytes_total.labels(
                **engine_labels,
                src_ip=event["src_ip"],
                dst_ip=event["dst_ip"],
                src_rank=event["src_rank"],
                dst_rank=event["dst_rank"],
                stream_id=event["stream_id"],
            ).inc(event["bytes"])
            self.counter_p2p_transfer_duration_seconds_total.labels(
                **engine_labels,
                src_ip=event["src_ip"],
                dst_ip=event["dst_ip"],
                stream_id=event["stream_id"],
            ).inc(event["duration"])

        for event in transfer_stats_data.get("h2d_transfer_events", []):
            self.counter_h2d_transfer_bytes_total.labels(
                **engine_labels,
                host_ip=event["host_ip"],
                device_id=event["device_id"],
                transfer_kind=event["transfer_kind"],
            ).inc(event["bytes"])
            self.counter_h2d_transfer_duration_seconds_total.labels(
                **engine_labels,
                host_ip=event["host_ip"],
                device_id=event["device_id"],
                transfer_kind=event["transfer_kind"],
            ).inc(event["duration"])

        for event in transfer_stats_data.get("d2h_transfer_events", []):
            self.counter_d2h_transfer_bytes_total.labels(
                **engine_labels,
                host_ip=event["host_ip"],
                device_id=event["device_id"],
                transfer_kind=event["transfer_kind"],
            ).inc(event["bytes"])
            self.counter_d2h_transfer_duration_seconds_total.labels(
                **engine_labels,
                host_ip=event["host_ip"],
                device_id=event["device_id"],
                transfer_kind=event["transfer_kind"],
            ).inc(event["duration"])

