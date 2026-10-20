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
"""Sequence-parallel output projection by all-to-all instead of reduce-scatter.

Under TPU_MOE_SEQUENCE_PARALLEL a decoder layer's attention (or GDN) output
projection leaves a bf16 [T, hidden] TP partial on every rank and the layer
reduce-scatters it to this rank's T / tp rows: 7/8 of T x hidden x 2 bytes on
the wire. The projection's input is already quantized to fp8 rows with one
scale per row, T x (k / tp) bytes a rank. Sending those instead (an
all-to-all: row block d to rank d) moves an eighth of the bytes, and each rank
then projects its own T / tp rows against the whole weight, all-gathered once
after load. The products are the same; the tp segment partials are summed in
float32 and rounded once, where the reduce-scatter sums partials already
rounded to bf16.
"""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from torch_tpu._internal import pallas
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The whole (k, n) projection weight, all-gathered over TP along k.
FULL_WEIGHT_ATTR = "_tpu_sp_a2a_weight_full"
# On a decoder layer: its mixer projects through the all-to-all, so the layer
# skips its reduce-scatter.
LAYER_ATTR = "_tpu_sp_a2a_oproj"


def enabled() -> bool:
    return os.environ.get("TPU_SP_A2A_OPROJ", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def segment_matmul(
    x_q: jax.Array, x_s: jax.Array, w_full: jax.Array, w_s: jax.Array
) -> jax.Array:
    """This rank's rows of the reduced projection.

    `x_q` [tp * t, k] fp8 and `x_s` [tp * t, 1]: block r is this rank's t rows
    as rank r quantized them over its k columns. `w_full` [tp * k, n], rank
    r's k rows at block r; `w_s` [n]. bf16 [t, n].
    """
    k = x_q.shape[1]
    tp = w_full.shape[0] // k
    t = x_q.shape[0] // tp
    acc = None
    for r in range(tp):
        part = jax.lax.dot_general(
            x_q[r * t : (r + 1) * t],
            w_full[r * k : (r + 1) * k],
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ) * x_s[r * t : (r + 1) * t]
        acc = part if acc is None else acc + part
    return (acc * jnp.expand_dims(w_s, 0)).astype(jnp.bfloat16)


# Rows and columns of the output a grid step computes. A step holds its rows
# of every segment and the weight's column block of every segment in VMEM.
SEGMENT_BLOCK_ROWS = 512
SEGMENT_BLOCK_COLS = 512


def _segment_matmul_kernel(x_ref, s_ref, w_ref, ws_ref, o_ref):
    """o = sum_r (x[r] @ w[r]) * s[:, r], times the column scales."""
    acc = None
    for r in range(x_ref.shape[0]):
        part = jnp.dot(
            x_ref[r], w_ref[r], preferred_element_type=jnp.float32
        ) * s_ref[:, r : r + 1]
        acc = part if acc is None else acc + part
    o_ref[...] = (acc * ws_ref[...]).astype(o_ref.dtype)


def segment_matmul_pallas(
    x_q: jax.Array,
    x_s: jax.Array,
    w_full: jax.Array,
    w_s: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """`segment_matmul` as one Pallas kernel: the tp segment partials
    accumulate in float32 in VMEM instead of round-tripping HBM."""
    k = x_q.shape[1]
    tp = w_full.shape[0] // k
    t = x_q.shape[0] // tp
    n = w_full.shape[1]
    bt = min(SEGMENT_BLOCK_ROWS, t)
    bn = min(SEGMENT_BLOCK_COLS, n)
    if t % bt or n % bn or bt % 8 or bn % 128:
        return segment_matmul(x_q, x_s, w_full, w_s)
    x = x_q.reshape(tp, t, k)
    s = x_s.reshape(tp, t).T  # [t, tp]: a row's segment scales along lanes
    w = w_full.reshape(tp, k, n)
    ws = w_s.reshape(1, n).astype(jnp.float32)
    # Rows outer, columns inner: a step's rows stay resident across the
    # column blocks and the weight streams once per row block.
    return pl.pallas_call(
        _segment_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((t, n), jnp.bfloat16),
        grid=(t // bt, n // bn),
        in_specs=[
            pl.BlockSpec((tp, bt, k), lambda i, j: (0, i, 0)),
            pl.BlockSpec((bt, tp), lambda i, j: (i, 0)),
            pl.BlockSpec((tp, k, bn), lambda i, j: (0, 0, j)),
            pl.BlockSpec((1, bn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bt, bn), lambda i, j: (i, j)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=48 * 1024 * 1024,
        ),
        name="sp_a2a_segment_matmul",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(x, s, w, ws)


_segment_matmul_op = None


def _get_segment_matmul_op():
    global _segment_matmul_op
    if _segment_matmul_op is None:
        op = pallas.jax_op("pallas::sp_a2a_segment_matmul", segment_matmul_pallas)

        def _fake(x_q, x_s, w_full, w_s):
            tp = w_full.shape[0] // x_q.shape[1]
            return torch.empty(
                x_q.shape[0] // tp,
                w_full.shape[-1],
                dtype=torch.bfloat16,
                device=x_q.device,
            )

        op.register_fake(_fake)
        _segment_matmul_op = op
    return _segment_matmul_op


def _all_to_all(x: torch.Tensor) -> torch.Tensor:
    """Row block d of `x` goes to rank d; block r of the result came from
    rank r."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(
        x.contiguous(), None, None, get_tp_group().device_group
    )
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out


def has_full_weight(proj: Any) -> bool:
    return isinstance(getattr(proj, FULL_WEIGHT_ATTR, None), torch.Tensor)


def project(x_q: torch.Tensor, x_s: torch.Tensor, proj: Any) -> torch.Tensor:
    """The reduce-scattered projection of the fp8 rows `x_q` [T, k] (scales
    `x_s` [T, 1]): this rank's T / tp rows, bf16."""
    return _get_segment_matmul_op()(
        _all_to_all(x_q),
        _all_to_all(x_s),
        getattr(proj, FULL_WEIGHT_ATTR),
        proj.weight_scale,
    )


def _check(proj: Any, rows: int = 2048) -> float:
    """Relative max difference of the all-to-all path against the
    reduce-scatter path on random rows, both run eagerly on every rank."""
    from vllm_torchtpu.layers.adapter import linear_common

    device = proj.weight.device
    k = proj.weight.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(1234 + get_tp_group().rank_in_group)
    x = torch.randn(rows, k, generator=gen).to(device=device, dtype=torch.bfloat16)
    x_q, x_s = linear_common.quantize_rows(x)
    partial = linear_common._get_prequant_matmul_op()(
        x_q, x_s, proj.weight, proj.weight_scale
    )
    want = tensor_model_parallel_reduce_scatter(partial, 0).float()
    got = project(x_q, x_s, proj).float()
    return float((got - want).abs().max() / want.abs().max().clamp_min(1e-30))


def install(model: Any, layers: list[tuple[Any, Any]]) -> int:
    """Give each (decoder layer, output projection) pair the whole weight and
    mark the layer, after the weights are processed."""
    if not enabled() or get_tensor_model_parallel_world_size() <= 1 or not layers:
        return 0
    from vllm_torchtpu.utils import synchronize_tensors

    full = []
    for layer, proj in layers:
        w = tensor_model_parallel_all_gather(proj.weight.data.contiguous(), 0)
        setattr(proj, FULL_WEIGHT_ATTR, w)
        setattr(layer, LAYER_ATTR, True)
        full.append(w)
    if full[0].device.type == "tpu":
        synchronize_tensors(full)
    err = _check(layers[0][1])
    logger.info(
        "[SP_A2A_OPROJ] %d layers project through an all-to-all of fp8 rows "
        "(+%.2f GiB of whole o_proj weights a rank); self-check rel max diff "
        "vs reduce-scatter %.3e",
        len(layers),
        sum(w.numel() * w.element_size() for w in full) / 2**30,
        err,
    )
    if not err < 2e-2:
        raise RuntimeError(f"TPU_SP_A2A_OPROJ self-check failed: rel diff {err}")
    return len(layers)
