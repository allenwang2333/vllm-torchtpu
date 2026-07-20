#!/usr/bin/env python3
"""Poll TPU tensorcore duty-cycle (%) directly from the libtpu metrics gRPC
endpoint (localhost:8431), fetching ONLY the duty-cycle metric to avoid the
per-chip/per-core length assertion in tpu_info.metrics.get_chip_usage.

Usage: tpu_duty.py [n_samples] [interval_s]
"""
import sys
import time

import grpc
from tpu_info import metrics
from tpu_info.proto import tpu_metric_service_pb2 as tpu_metrics
from tpu_info.proto import tpu_metric_service_pb2_grpc as tpu_metrics_grpc

n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
addr = "localhost:8431"

channel = grpc.secure_channel(addr, grpc.local_channel_credentials())
client = tpu_metrics_grpc.RuntimeMetricServiceStub(channel)


def duty():
    resp = client.GetRuntimeMetric(
        tpu_metrics.MetricRequest(
            metric_name=metrics.MetricName.DUTY_CYCLE_PCT.value))
    ms = sorted(resp.metric.metrics, key=lambda m: m.attribute.value.int_attr)
    return [m.gauge.as_double for m in ms]


rows = []
for i in range(n):
    try:
        d = duty()
        rows.append(d)
        print(f"[{i:02d}] duty%/chip = " + " ".join(f"{x:5.1f}" for x in d)
              + f"   mean={sum(d)/len(d):5.1f}%")
    except Exception as e:
        print(f"[{i:02d}] ERR {type(e).__name__}: {e}")
    time.sleep(interval)

if rows:
    flat = [x for r in rows for x in r]
    per_chip = [sum(c) / len(c) for c in zip(*rows)]
    print("\n=== SUMMARY ===")
    print("per-chip mean duty%: " + " ".join(f"{x:5.1f}" for x in per_chip))
    print(f"overall mean duty% = {sum(flat)/len(flat):.1f}%  "
          f"(max {max(flat):.1f}%, min {min(flat):.1f}%)")
