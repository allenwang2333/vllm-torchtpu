# SPDX-License-Identifier: Apache-2.0
"""Device-side op durations of a JAX callable, from a jax.profiler capture.

device_times(fn, args, iters) runs fn(*args) once to compile, then `iters`
times under jax.profiler.trace, and returns {op_name: [duration_us, ...]} from
the "XLA Ops" line of /device:TPU:0. Op names are the HLO op names, so a
pallas_call shows up under its `name=`.
"""

import collections
import glob
import os
import re
import sys
import tempfile

import jax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src", "vllm_torchtpu", "tools"))


def parse_xplane(path):
    import xplane_pb2  # noqa: PLC0415

    xs = xplane_pb2.XSpace()
    with open(path, "rb") as f:
        xs.ParseFromString(f.read())
    out = collections.defaultdict(list)
    for p in xs.planes:
        if not re.match(r"^/device:TPU:0$", p.name):
            continue
        for ln in p.lines:
            if (ln.name or ln.display_name) != "XLA Ops":
                continue
            for e in ln.events:
                md = p.event_metadata.get(e.metadata_id)
                name = (md.display_name or md.name) if md else str(e.metadata_id)
                out[name].append(e.duration_ps / 1e6)
    return out


def device_times(fn, args, iters=10, carry=None):
    """carry: optional fn(out, args) -> new args (for donated buffers)."""
    out = fn(*args)
    jax.block_until_ready(out)
    if carry is not None:
        args = carry(out, args)
    d = tempfile.mkdtemp(prefix="dt_")
    with jax.profiler.trace(d):
        for _ in range(iters):
            out = fn(*args)
            if carry is not None:
                args = carry(out, args)
        jax.block_until_ready(out)
    files = glob.glob(os.path.join(d, "**", "*.xplane.pb"), recursive=True)
    res = collections.defaultdict(list)
    for f in files:
        for k, v in parse_xplane(f).items():
            res[k].extend(v)
    return dict(res), out, args


def summarize(times, pattern):
    """Median per-call us and count of ops whose base name matches `pattern`."""
    import statistics

    vals = []
    for k, v in times.items():
        if re.search(pattern, k):
            vals.extend(v)
    if not vals:
        return None, 0
    return statistics.median(vals), len(vals)
