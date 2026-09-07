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
"""Smoke test for a multi-host TPU slice provisioned as one Buildkite agent.

Run inside the Ray head container that run_multihost.sh brings up. Schedules
one task per alive Ray node; each task initializes JAX in single-host mode
(so hosts do not need to rendezvous with each other) and reports the chips
it can see. Passes when every host reports the expected number of devices.
On tpu7x each chip has 2 TensorCores and JAX lists each core as a device, so
a 4-chip host reports 8 devices.

Usage (from the head container, repo root):
    python3 .buildkite/scripts/tpu_slice_smoke.py [--hosts 4] [--devices-per-host 8]
"""
import argparse
import sys
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=1)
def probe(devices_per_host: int):
    import os
    import socket

    # Single-host view: each host enumerates only its own chips. Without
    # this, libtpu on a pod slice waits for every host to join a slice-wide
    # rendezvous, which the per-node tasks do not coordinate.
    os.environ["TPU_SKIP_MDS_QUERY"] = "1"
    os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
    os.environ["TPU_WORKER_ID"] = "0"
    os.environ["TPU_WORKER_HOSTNAMES"] = "localhost"
    os.environ.pop("TPU_VISIBLE_CHIPS", None)

    t0 = time.time()
    import jax
    devs = jax.devices()
    return {
        "host": socket.gethostname(),
        "ip": ray.util.get_node_ip_address(),
        "n_devices": len(devs),
        "devices": [str(d) for d in devs],
        "kind": devs[0].device_kind if devs else None,
        "init_s": round(time.time() - t0, 1),
        "ok": len(devs) == devices_per_host,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", type=int, default=4)
    ap.add_argument("--devices-per-host", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    ray.init(address="auto", logging_level="error")
    nodes = [n for n in ray.nodes() if n["Alive"]]
    print(f"[tpu-slice-smoke] ray nodes alive: {len(nodes)} "
          f"{sorted(n['NodeManagerAddress'] for n in nodes)}",
          flush=True)

    refs = [
        probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=n["NodeID"], soft=False)).remote(args.devices_per_host)
        for n in nodes
    ]
    results = ray.get(refs, timeout=args.timeout)

    ok = len(nodes) == args.hosts
    for r in sorted(results, key=lambda r: r["ip"]):
        status = "ok" if r["ok"] else "BAD"
        print(f"[tpu-slice-smoke] {status} host={r['host']} ip={r['ip']} "
              f"kind={r['kind']} jax.devices()={r['n_devices']} "
              f"init={r['init_s']}s", flush=True)
        for d in r["devices"]:
            print(f"    {d}", flush=True)
        ok = ok and r["ok"]

    total = sum(r["n_devices"] for r in results)
    print(f"[tpu-slice-smoke] hosts={len(nodes)}/{args.hosts} "
          f"devices={total}/{args.hosts * args.devices_per_host} "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
