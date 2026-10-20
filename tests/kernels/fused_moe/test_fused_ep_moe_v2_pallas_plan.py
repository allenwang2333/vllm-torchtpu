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
"""The Pallas routing plan reproduces the XLA plan's kernel operands exactly.

Every operand the fused EP kernel and its combine read is compared bit for
bit, on every shard, for decode- and prefill-sized buckets at DeepSeek-V4's
expert layout (384 experts over 16 shards, top-6), with skewed routing that
leaves most runs empty and with rows that route nowhere. The Pallas body runs
in interpret mode on the host.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import host, layer  # noqa: E402

CAPACITY = 128


@pytest.fixture(autouse=True)
def _cpu_backend():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _exchange(ep, t_local, e_total, topk, *, skew, unrouted, seed):
    """The gathered payload, plus each shard's own blocks and histogram."""
    rng = np.random.default_rng(seed)
    block = host.routing_block(t_local, topk)
    layout = layer._exchange_layout(t_local, topk, e_total, True)
    rows, own = [], []
    for _ in range(ep):
        if skew:
            hot = rng.choice(e_total, size=max(4, e_total // 24), replace=False)
            idx = np.stack(
                [rng.choice(hot, size=topk, replace=False) for _ in range(t_local)]
            )
        else:
            idx = np.stack(
                [rng.choice(e_total, size=topk, replace=False) for _ in range(t_local)]
            )
        if unrouted:
            idx[rng.random(t_local) < 0.3] = e_total
        local_blocks = jnp.asarray(idx, jnp.int32).reshape(
            t_local * topk // block, block
        )
        block_hist = host.pair_block_hist(local_blocks, e_total)
        scale = rng.uniform(1e-4, 1.0, t_local).astype(np.float32)
        scale[rng.random(t_local) < 0.1] = 0.0
        bits = jnp.asarray(scale.view(np.int32))
        rows.append(
            layer._pack_routing_exchange(
                local_blocks.reshape(-1), bits, block_hist.sum(axis=0), layout
            )
        )
        own.append((local_blocks, block_hist))
    return jnp.stack(rows), own, layout, block


@pytest.mark.parametrize(
    "t_local,skew,unrouted",
    [
        pytest.param(8, False, False, id="t16_tp2_uniform"),
        pytest.param(8, True, True, id="t16_tp2_skew_unrouted"),
        pytest.param(16, True, False, id="t32_tp2_skew"),
        pytest.param(32, False, True, id="t64_tp2_unrouted"),
        pytest.param(128, True, True, id="t256_tp2_skew_unrouted"),
    ],
)
def test_pallas_plan_matches_xla(t_local, skew, unrouted):
    ep, e_total, topk = 16, 384, 6
    exchange, own, layout, block = _exchange(
        ep, t_local, e_total, topk, skew=skew, unrouted=unrouted, seed=t_local
    )
    T = ep * t_local
    kw = dict(
        layout=layout,
        T=T,
        t_local=t_local,
        topk=topk,
        e_total=e_total,
        ep=ep,
        block=block,
        capacity=CAPACITY,
        ragged_stride=host.ragged_stride_bound(T, topk, e_total, CAPACITY),
    )
    q_g = jnp.zeros((1,), jnp.float32)
    for me in range(ep):
        ref = layer._xla_routing_operands(
            exchange,
            jnp.int32(me),
            *own[me],
            q_g,
            sharded_plan=True,
            quantized_activations=True,
            **kw,
        )
        got = layer._pallas_routing_operands(
            exchange, jnp.full((1, 1), me, jnp.int32), q_g, interpret=True, **kw
        )
        for name in layer.RoutingOperands._fields:
            a, b = getattr(ref, name), getattr(got, name)
            for i, (x, y) in enumerate(
                zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
            ):
                x, y = np.asarray(x), np.asarray(y)
                assert x.shape == y.shape and x.dtype == y.dtype, (
                    me,
                    name,
                    i,
                    x.shape,
                    y.shape,
                )
                np.testing.assert_array_equal(
                    x.view(np.int32) if x.dtype == np.float32 else x,
                    y.view(np.int32) if y.dtype == np.float32 else y,
                    err_msg=f"shard {me} operand {name}[{i}]",
                )


@pytest.mark.multichip
@pytest.mark.parametrize("rows", [16, 32, 256], ids=lambda r: f"t{r}")
@pytest.mark.parametrize("weight_format", ["fp8", "fp4"])
def test_single_host_layer_is_bit_exact_with_the_pallas_plan(rows, weight_format):
    """The whole layer on eight local cores, TP2 replica groups: same bits."""
    from jax.sharding import Mesh

    from vllm_torchtpu.kernels.fused_moe.v2 import AXIS, WeightFormat, fused_ep_moe_v2

    devices = jax.local_devices()
    if len(devices) < 8 or devices[0].platform != "tpu":
        pytest.skip("requires eight local TPU devices")
    from test_fused_ep_moe_v2_tp_tokens import _make_inputs

    mesh = Mesh(np.asarray([devices[i] for i in (3, 0, 6, 1, 7, 4, 2, 5)]), (AXIS,))
    groups = ((6, 1), (4, 3), (0, 7), (2, 5))
    fmt = WeightFormat.FP4 if weight_format == "fp4" else WeightFormat.FP8
    with jax.default_device(devices[0]):
        rank, inputs = _make_inputs(mesh, groups, rows, fmt)
        bias = jnp.linspace(-0.5, 0.5, 2 * mesh.shape[AXIS], dtype=jnp.float32)

        def run(max_tokens, rank_arg, bias_arg, *args):
            return fused_ep_moe_v2(
                *args,
                topk=4,
                renormalize=True,
                mesh=mesh,
                capacity=CAPACITY,
                weight_format=fmt,
                rhs_qb=512 if fmt == WeightFormat.FP4 else None,
                rank=rank_arg,
                sharded_plan=True,
                token_replica_groups=groups,
                scoring_fn="sqrtsoftplus",
                score_bias=bias_arg,
                routed_scaling_factor=2.5,
                pallas_plan_max_tokens=max_tokens,
            )

        outs = [
            np.asarray(jax.jit(run, static_argnums=0)(m, rank, bias, *inputs))
            for m in (0, rows)
        ]
    assert np.isfinite(outs[0].astype(np.float32)).all()
    np.testing.assert_array_equal(outs[0].view(np.uint16), outs[1].view(np.uint16))
