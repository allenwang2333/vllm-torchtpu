# SPDX-License-Identifier: Apache-2.0
"""Extract device durations from XPlane without starting a TPU runtime."""

import argparse
import json
from pathlib import Path

import numpy as np

from vllm_torchtpu.tools import xplane_pb2


def summary(values):
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def analyze(directory, fused_name):
    (trace,) = directory.rglob("*.xplane.pb")
    space = xplane_pb2.XSpace()
    space.ParseFromString(trace.read_bytes())
    rows = []
    for plane in space.planes:
        if not plane.name.removeprefix("/device:TPU:").isdigit():
            continue
        modules = []
        kernels = []
        for line in plane.lines:
            for event in line.events:
                name = plane.event_metadata[event.metadata_id].name
                start = line.timestamp_ns * 1000 + event.offset_ps
                item = (start, start + event.duration_ps, event.duration_ps / 1e9)
                if line.name == "XLA Modules" and name.startswith("jit_layer("):
                    modules.append(item)
                if line.name == "XLA Ops" and name.startswith("%" + fused_name):
                    kernels.append(item)
        modules.sort()
        kernels.sort()
        assert len(modules) == len(kernels) == 10, (
            plane.name,
            len(modules),
            len(kernels),
        )
        for module, kernel in zip(modules, kernels):
            assert module[0] <= kernel[0] <= kernel[1] <= module[1]
        rows.append(
            {
                "device": plane.name,
                "layer_ms": [m[2] for m in modules],
                "fused_ms": [k[2] for k in kernels],
            }
        )
    assert len(rows) == 8
    metadata = json.loads((directory / "metadata.json").read_text())
    g = metadata["geometry"]
    flops = 6 * metadata["total_input_tokens"] * g["topk"] * g["hidden"] * g["inter"]
    # Eight JAX chiplets correspond to four TPU7x physical chips.
    peak = 4 * 4614e12
    result = {
        "trace": str(trace),
        "devices": rows,
        "useful_ffn_flops": flops,
        "fp8_peak_flops_per_second": peak,
        "peak_source": "https://docs.cloud.google.com/tpu/docs/tpu7x",
        "method": "10 warmed-up device calls; max duration across 8 ranks per step, then median. MFU counts useful GEMM FLOPs only; it is not a hardware MXU busy counter.",
    }
    for name in ("layer_ms", "fused_ms"):
        matrix = np.array([row[name] for row in rows])
        per_step = matrix.max(axis=0)
        result[name] = {
            "per_step_max": per_step.tolist(),
            "max_rank_per_step": summary(per_step),
            "all_device_calls": summary(matrix.ravel()),
        }
    seconds = result["fused_ms"]["max_rank_per_step"]["median"] / 1000
    result["fused_useful_mfu_percent"] = 100 * flops / seconds / peak
    result["fused_effective_tflops"] = flops / seconds / 1e12
    result["ideal_compute_ms"] = 1000 * flops / peak
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fused-name", required=True)
    args = parser.parse_args()
    result = analyze(args.directory, args.fused_name)
    (args.directory / "device_timing.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "devices"}, indent=2))
