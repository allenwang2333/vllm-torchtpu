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
"""On-device head-to-head for the **v2** fused EP MoE kernel.

Three MoE programs, one process, one clock: the production GMM path (an XLA
all-gather, a permute, two grouped matmuls and a reduce-scatter), fused-EP v1
(token-stationary), and fused-EP v2 (weight-stationary). CPU gates already cover
compilation (``test_fused_ep_v2_aot.py``) and arithmetic including transport
addressing (``test_fused_ep_v2_numerics.py``, which runs the real Pallas program
under the interpreter), so what only a chip can answer is *time* -- and, for v2,
specifically whether its in-kernel all-to-all beats the XLA collective pair it
replaces.

That is what the wire-bandwidth column is for. v2's transport is two pieces:

* an **XLA all-gather** of this rank's fp8 token rows, so every rank receives
  ``(ep - 1) * t_local * hidden`` bytes; and
* an **in-kernel push** of ``t_local * top_k`` result rows, of which
  ``(ep - 1) / ep`` leave the chip, plus one f32 scale mirror row each.

Both are reported as achieved GB/s against the measured layer time, at every
token count, because that is the number that separates "the transport is the
bottleneck" from "the matmuls are". Today's DP all-gather + reduce-scatter run
at ~140 GB/s effective on this node including straggler wait
(43.0 + 66.2 ms of a 384 ms prefill step, 60 layers).

Run on an 8-chip node:
    pytest tests/kernels/fused_moe/test_fused_ep_v2_device.py -m multichip -s
"""

import functools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME_1D
from vllm_torchtpu.layers.common.fused_moe_ep_v2 import fused_ep_moe_v2_func

from .ep_device_harness import (HIDDEN, INTERMEDIATE, NUM_EXPERTS, SMALL,
                                TOP_K, _gmm_reference, _inputs, _mesh, _rel_l2,
                                _shard, _small_inputs, _small_ref, _weights)

pytestmark = [
    pytest.mark.multichip,
    pytest.mark.skipif(jax.default_backend() != "tpu",
                       reason="fused EP kernel needs a TPU backend"),
]

FP8 = jnp.float8_e4m3fn
# The InferenceX prefill point is 16384 node-wide = 2048 local at ep=8; the rest
# walk down to where the author's own bench-off has v2 losing (below ~512).
TOKEN_LADDER = [2048, 1024, 512, 256, 128, 64]


def _mesh_1d(ep_size):
    """The mesh ``ep_mesh.get_or_create_ep_mesh_1d`` builds."""
    return Mesh(np.asarray(jax.devices()[:ep_size]).reshape(ep_size),
                axis_names=(EP_AXIS_NAME_1D, ),
                axis_types=(AxisType.Explicit, ))


@functools.lru_cache(maxsize=None)
def _v2_fn(mesh):
    """`jax.jit` around v2, ONCE per mesh -- without it this times the compiler.

    `fused_ep_moe_v2` returns `sm(*args)` where `sm` is a bare
    `jax.shard_map` (`kernels/fused_moe/v2/layer.py:474-480`): outside a
    `jax.jit` that executes eagerly and re-traces the whole Pallas program on
    every call. Measured on device, 2048 local tokens: **61,856 ms** per
    iteration unjitted against **1,906 us** for the same shape through a jitted
    harness -- a factor of 32,000, all of it compilation.

    This is the third instance in this repository of one defect: a `jax.jit`
    (or the lack of one) around a callable that is rebuilt per call. It also
    produced `gmm=11102 ms` in the sibling file and the retracted "182x".
    v1 is immune because `fused_ep_moe` is `jax.jit`-decorated at module level.
    """

    @jax.jit
    def run(tokens, logits, w13, s13, w2, s2):
        return fused_ep_moe_v2_func(tokens,
                                    logits,
                                    w13,
                                    w2,
                                    s13,
                                    s2,
                                    mesh=mesh,
                                    top_k=TOP_K,
                                    renormalize=True,
                                    activation="silu")

    return run


def _v2(tokens, logits, w13, s13, w2, s2, mesh):
    return _v2_fn(mesh)(tokens, logits, w13, s13, w2, s2)


def _v2_inputs(mesh, ep_size, local_tokens, seed=0):
    """``_inputs`` on the 1-D mesh, with the scales in the loader's layout."""
    rng = np.random.default_rng(seed)
    tokens = jnp.asarray(
        rng.normal(0, 1, (local_tokens * ep_size, HIDDEN)) / 4, jnp.bfloat16)
    logits = jnp.asarray(
        rng.normal(0, 1, (local_tokens * ep_size, NUM_EXPERTS)), jnp.bfloat16)
    w13, s13, w2, s2 = _weights(seed + 1, NUM_EXPERTS, HIDDEN, INTERMEDIATE)
    spec = P(EP_AXIS_NAME_1D)
    return tuple(
        _shard(x, mesh, spec) for x in (tokens, logits, w13, s13, w2, s2))


def _wire_bytes(local_tokens, ep_size):
    """Bytes each rank moves over ICI per layer, by leg. fp8 rows either way."""
    scale_lanes = HIDDEN // 1024  # host.scale_mirror_lanes
    gather_in = (ep_size - 1) * local_tokens * HIDDEN
    pushed_rows = local_tokens * TOP_K * (ep_size - 1) / ep_size
    push_out = pushed_rows * (HIDDEN + scale_lanes * 4)
    return dict(all_gather_in=int(gather_in),
                kernel_push_out=int(push_out),
                total=int(gather_in + push_out))


def _bench(fn, iters=20):
    out = fn()
    jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / iters * 1e3


def _require_8():
    ep_size = len(jax.devices())
    if ep_size < 8:
        pytest.skip(f"needs 8 devices, saw {ep_size}")
    return ep_size


@pytest.mark.parametrize("local_tokens", [2048])
def test_v2_matches_the_gmm_path_on_device(local_tokens):
    """The transport delivers each row to the rank that owns its token.

    Reported against the GMM path and asserted loosely, for the reason the v1
    module states: the two paths round differently, so their agreement is not
    the bar. What makes this tight rather than hopeful is the CPU numerics gate,
    which pins v2 to its documented arithmetic to 1e-3 with the all-to-all live.
    A large number here therefore means transport, not rounding.
    """
    ep_size = _require_8()
    mesh1d, mesh2d = _mesh_1d(ep_size), _mesh(ep_size)
    got = np.asarray(
        jax.device_get(_v2(*_v2_inputs(mesh1d, ep_size, local_tokens),
                           mesh1d)), np.float32)
    ref = np.asarray(
        jax.device_get(
            _gmm_reference(*_inputs(mesh2d, ep_size, local_tokens), mesh2d,
                           ep_size)), np.float32)
    rel = _rel_l2(got, ref)
    print(f"[v2 device local_tokens={local_tokens}] vs GMM rel_l2={rel:.4g}")
    assert rel < 0.2, "v2 and the GMM path disagree beyond fp8 rounding"


@pytest.mark.parametrize("local_tokens", TOKEN_LADDER)
def test_report_layer_time_and_wire_bandwidth(local_tokens):
    """One clock, three programs, and the achieved wire rate for v2.

    Prints one JSON line per token count so a harness can read the ladder
    without parsing prose.
    """
    ep_size = _require_8()
    mesh1d, mesh2d = _mesh_1d(ep_size), _mesh(ep_size)
    args1d = _v2_inputs(mesh1d, ep_size, local_tokens)
    args2d = _inputs(mesh2d, ep_size, local_tokens)

    v2_ms = _bench(lambda: _v2(*args1d, mesh1d))
    gmm_ms = _bench(lambda: _gmm_reference(*args2d, mesh2d, ep_size))

    wire = _wire_bytes(local_tokens, ep_size)
    row = dict(local_tokens=local_tokens,
               node_tokens=local_tokens * ep_size,
               v2_ms=round(v2_ms, 4),
               gmm_ms=round(gmm_ms, 4),
               v2_over_gmm=round(gmm_ms / v2_ms, 3),
               wire_bytes=wire,
               v2_wire_gbps=round(wire["total"] / (v2_ms * 1e-3) / 1e9, 1))
    print("V2_LADDER " + json.dumps(row))


def test_v2_against_the_fp32_oracle_at_a_small_shape():
    """The third leg: neither Pallas path is the reference.

    Same shape, same inputs and the same ``ref_moe`` oracle the v1 arm was
    scored against, so the two numbers are comparable: **v1 measured 0.141 here
    where the GMM path measured under 0.06, and it was never explained.** v2's
    CPU interpreter run says its whole gap is its documented rounding -- 0.052 --
    so this test asks the chip to confirm the interpreter. A larger number here
    with the interpreter clean localizes the fault to the DMA engine or the
    barrier rather than to the arithmetic, which is a diagnosis v1 never got.
    """
    ep_size = _require_8()
    mesh = _mesh_1d(ep_size)
    (tokens, logits, q_native, s_native, q_concat, s_concat, q2,
     s2) = _small_inputs(ep_size)
    spec = P(EP_AXIS_NAME_1D)
    got = np.asarray(
        jax.device_get(
            fused_ep_moe_v2_func(_shard(tokens, mesh, spec),
                                 _shard(logits, mesh, spec),
                                 _shard(q_concat, mesh, spec),
                                 _shard(q2, mesh, spec),
                                 _shard(s_concat, mesh, spec),
                                 _shard(s2, mesh, spec),
                                 mesh=mesh,
                                 top_k=SMALL["top_k"],
                                 renormalize=True,
                                 activation="silu")), np.float32)
    ref = _small_ref(tokens, logits, q_native, s_native, q2, s2)
    rel = _rel_l2(got, ref)
    print(f"[v2 small] vs ref_moe rel_l2={rel:.4g} "
          "(CPU interpreter says 0.052; v1 measured 0.141 here)")
    assert rel < 0.08, (
        "v2 is further from the fp32 oracle on device than the interpreter "
        "says it should be, which localizes the fault to the chip's transport "
        "rather than to the arithmetic")
