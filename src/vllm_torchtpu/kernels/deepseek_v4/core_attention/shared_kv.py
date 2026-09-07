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
"""Experimental grouped CSA: sorted physical KV union and SparseCore bit masks.

Top-k entries must be unique within a query, with negative padding. Owner bits
are accumulated once per query/key. An overflowing union must use the original
attention path; it must never be truncated to the fixed output capacity.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def prepare_shared_kv_indices(x, *, group_size):
    """Sort physical addresses and return owners, compact slots and full counts."""
    if group_size not in (2, 4, 8) or x.shape[0] % group_size:
        raise ValueError(
            "Shared KV requires complete groups of 2, 4, or 8 queries")
    groups, topk = x.shape[0] // group_size, x.shape[1]
    a = jnp.where(x >= 0, x, jnp.iinfo(jnp.int32).max).reshape(groups, -1)
    owner = jnp.broadcast_to(
        jnp.repeat(jnp.arange(group_size, dtype=jnp.int32), topk), a.shape)
    a, owner = lax.sort_key_val(a, owner, dimension=1, is_stable=False)
    valid = a != jnp.iinfo(jnp.int32).max
    first = jnp.concatenate(
        [jnp.ones((groups, 1), bool), a[:, 1:] != a[:, :-1]], 1) & valid
    slots = jnp.cumsum(first, axis=1, dtype=jnp.int32) - 1
    lengths = jnp.sum(first, axis=1, dtype=jnp.int32)
    return a, owner, slots, lengths


def _pack_kernel(a_ref, code_ref, idx_ref, mask_ref, *, group_size, capacity,
                 topk, lanes, rotate, padding_modulus):
    ncores = lax.axis_size(('core', 'subcore'))
    core = lax.axis_index(('core', 'subcore'))
    width = group_size * topk

    def body(a, code, out, mask):
        group_id = pl.program_id(0) * ncores + core

        def clear(i, _):
            sl = pl.ds(i * lanes, lanes)
            v = ((group_id * capacity + i * lanes +
                  jnp.arange(lanes, dtype=jnp.int32)) *
                 104729) % padding_modulus if rotate else jnp.zeros(
                     (lanes, ), jnp.int32)
            out[sl] = v
            mask[sl] = jnp.zeros((lanes, ), jnp.int32)

        lax.fori_loop(0, capacity // lanes, clear, None)

        def step(i, _):
            sl = pl.ds(i * lanes, lanes)
            addr = a[sl]
            codes = code[sl]
            slots = codes >> 3
            owner = codes & 7
            valid = (addr != jnp.iinfo(jnp.int32).max) & (slots < capacity)
            dst = jnp.clip(slots, 0, capacity - 1)
            if rotate:
                # A multiplier coprime to capacity makes this a permutation.
                dst = (dst *
                       (5 if capacity % 5 else 3 if capacity % 3 else 1) +
                       group_id * 73) % capacity
            count, _ = plsc.scan_count(addr)
            # Duplicate runs crossing a 16-lane boundary rewrite the same address sequentially.
            plsc.store_scatter(out, (dst, ), addr, mask=valid & (count == 1))
            # Each query selects unique keys, hence every owner contributes its bit once.
            plsc.addupdate_scatter(mask, (dst, ),
                                   jnp.left_shift(
                                       jnp.ones((lanes, ), jnp.int32), owner),
                                   mask=valid)

        lax.fori_loop(0, width // lanes, step, None)

    spec = pl.BlockSpec((None, width), lambda b: (b * ncores + core, 0))
    outspec = pl.BlockSpec((None, capacity), lambda b: (b * ncores + core, 0))
    pltpu.emit_pipeline(body,
                        grid=(a_ref.shape[0] // ncores, ),
                        in_specs=[spec, spec],
                        out_specs=[outspec, outspec])(a_ref, code_ref, idx_ref,
                                                      mask_ref)


def pack_shared_kv_indices(a,
                           owner,
                           slots,
                           *,
                           group_size,
                           capacity,
                           rotate=False,
                           padding_modulus=1):
    """Pack each union and its owner bits with native SparseCore indexed stores.

    Rotation permutes slots bijectively and initializes padding to legal spread
    addresses. Callers must reject overflow before consuming these buffers.
    """
    if group_size not in (2, 4, 8):
        raise ValueError("Shared KV requires groups of 2, 4, or 8 queries")
    if capacity <= 0 or capacity % 16:
        raise ValueError(
            "Shared KV capacity must be positive and divisible by 16")
    if rotate and padding_modulus <= 0:
        raise ValueError("Padding addresses require a nonempty cache")
    groups, width = a.shape
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None, "Shared KV packing requires SparseCore"
    padded_width = pl.cdiv(width, info.num_lanes) * info.num_lanes
    topk = padded_width // group_size
    ncores = info.num_cores * info.num_subcores
    padded = pl.cdiv(groups, ncores) * ncores
    codes = (slots << 3) | owner
    inputs = [
        jnp.pad(v, ((0, padded - groups), (0, padded_width - width)),
                constant_values=value)
        for v, value in [(a, jnp.iinfo(jnp.int32).max), (codes, 0)]
    ]
    mesh = plsc.VectorSubcoreMesh(num_cores=info.num_cores,
                                  num_subcores=info.num_subcores,
                                  core_axis_name='core',
                                  subcore_axis_name='subcore')
    out, mask = pl.kernel(
        functools.partial(_pack_kernel,
                          group_size=group_size,
                          capacity=capacity,
                          topk=topk,
                          lanes=info.num_lanes,
                          rotate=rotate,
                          padding_modulus=padding_modulus),
        out_type=(jax.ShapeDtypeStruct((padded, capacity), jnp.int32),
                  jax.ShapeDtypeStruct((padded, capacity), jnp.int32)),
        mesh=mesh,
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False,
                                             use_tc_tiling_on_sc=False),
        name='sc_shared_kv_pack_bits')(*inputs)
    return out[:groups], mask[:groups, None, :]


def build_shared_kv_indices(x, *, group_size, capacity):
    """Return compact indices, expanded membership and full union lengths."""
    *prepared, lengths = prepare_shared_kv_indices(x, group_size=group_size)
    out, bits = pack_shared_kv_indices(*prepared,
                                       group_size=group_size,
                                       capacity=capacity)
    return out, (bits >> jnp.arange(
        group_size, dtype=jnp.int32)[None, :, None]) & 1, lengths


def shared_attention(q, kv, membership, swa_acc, swa_l, swa_m, sinks, *,
                     sm_scale):
    """One shared QK and PV per group, with separate softmaxes for each query.

    q/swa_acc: [tokens, heads, dim]; kv: [capacity, dim]; membership:
    [tokens, capacity]. Invalid gathered KV rows may contain NaNs from overlaid
    caches and must be cleared before either matrix multiplication.
    """
    tokens, heads, dim = q.shape
    # Keep the validity vector on the lane dimension until the transpose.
    # A direct [capacity] -> [capacity, 1] reshape moves lanes to sublanes,
    # which Mosaic cannot infer. Transpose a full 128-lane int32 tile instead.
    present = jnp.max(membership, axis=0, keepdims=True)
    present = jnp.broadcast_to(present, (128, kv.shape[0])).T
    present = jnp.tile(present, (1, dim // 128)) != 0
    kv = jnp.where(present, kv, 0)
    mask = jnp.concatenate([
        jnp.broadcast_to(membership[t:t + 1] != 0, (heads, kv.shape[0]))
        for t in range(tokens)
    ],
                           axis=0)
    scores = jnp.einsum("nd,md->nm",
                        q.reshape(tokens * heads, dim),
                        kv,
                        preferred_element_type=jnp.float32) * sm_scale
    scores = jnp.where(mask, scores, -jnp.inf)

    def head_columns(values):
        # SWA statistics arrive with heads on lanes; softmax statistics have
        # heads on rows. Make that layout change an explicit tiled transpose.
        padded_heads = ((heads + 127) // 128) * 128
        columns = []
        for token in range(tokens):
            row = jnp.pad(values[token:token + 1],
                          ((0, 0), (0, padded_heads - heads)))
            tile = jnp.broadcast_to(row, (128, padded_heads)).T
            columns.append(tile[:heads, :1])
        return jnp.concatenate(columns, axis=0)

    previous_max = head_columns(swa_m)
    sinks = head_columns(jnp.broadcast_to(sinks[None, :], (tokens, heads)))
    maximum = jnp.maximum(
        jnp.maximum(jnp.max(scores, axis=1, keepdims=True), previous_max),
        sinks)
    # Empty sparse + empty SWA + disabled sink has a zero output, not NaN.
    maximum = jnp.where(maximum == -jnp.inf, 0, maximum)
    probabilities = jnp.where(mask, jnp.exp(scores - maximum), 0)
    rescale = jnp.exp(previous_max - maximum)
    denominator = (rescale * head_columns(swa_l) +
                   jnp.sum(probabilities, axis=1, keepdims=True) +
                   jnp.exp(sinks - maximum))
    pv = jnp.einsum("nm,md->nd",
                    probabilities,
                    kv,
                    preferred_element_type=jnp.float32)
    accumulator = rescale * swa_acc.reshape(tokens * heads, dim) + pv
    output = accumulator / jnp.where(denominator > 0, denominator, 1)
    return output.reshape(q.shape).astype(q.dtype)
