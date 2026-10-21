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
"""Random DSv4 dense-chain weights (natural layout) and their ring packing.

Natural layout = what the stock serving ops consume: fp8 ``[k, n]`` with a
per-output-channel f32 scale, bf16 ``[k, n]``, f32 ``hc_*_fn [24, 4h]``. The
packed layout is the ring's: per stream ``[layers, tiles, rows, cols]`` plus
one f32 row of small vectors per layer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils as mhc_utils
from vllm_torchtpu.kernels.deepseek_v4_megakernel.dense_config import (
    CSA,
    HCA,
    DenseChainConfig,
)

_FP8_SPREAD = 64.0  # std of the stored fp8 values; |q| > 448 is ~7 sigma


def _fp8_weight(key, n_layers, k, n):
    """fp8 [L, k, n] + f32 scale [L, n] for a unit-gain projection."""
    kq, ks = jax.random.split(key)

    def one(kk):
        z = jax.random.normal(kk, (k, n), jnp.bfloat16) * _FP8_SPREAD
        return jnp.clip(z, -448, 448).astype(jnp.float8_e4m3fn)

    q = jax.lax.map(one, jax.random.split(kq, n_layers))
    gain = jnp.exp(0.3 * jax.random.normal(ks, (n_layers, n), jnp.float32))
    return q, gain / (_FP8_SPREAD * k**0.5)


def _bf16_weight(key, n_layers, k, n):
    def one(kk):
        return (jax.random.normal(kk, (k, n), jnp.float32) * k**-0.5).astype(
            jnp.bfloat16
        )

    return jax.lax.map(one, jax.random.split(key, n_layers))


def make_weights(cfg: DenseChainConfig, seed: int = 0) -> dict[str, jax.Array]:
    """Natural-layout weights for every layer, stacked per kind."""
    L = cfg.n_layers
    Lc = cfg.layer_types.count(CSA)
    Lh = cfg.layer_types.count(HCA)
    h = cfg.hidden
    keys = iter(jax.random.split(jax.random.key(seed), 64))
    w: dict[str, jax.Array] = {}
    for sub in ("attn", "ffn"):
        w[f"hc_{sub}_fn"] = jax.random.normal(
            next(keys), (L, cfg.n_mix, cfg.hc_hidden), jnp.float32
        ) * (cfg.hc_hidden**-0.5)
        w[f"hc_{sub}_base"] = 0.5 * jax.random.normal(
            next(keys), (L, cfg.n_mix), jnp.float32
        )
        w[f"hc_{sub}_scale"] = 1.0 + 0.2 * jax.random.normal(
            next(keys), (L, 3), jnp.float32
        )
    for name, n in (
        ("attn_norm", h),
        ("ffn_norm", h),
        ("q_norm", cfg.q_lora),
        ("kv_norm", cfg.head_dim),
    ):
        w[name] = (
            1.0 + 0.1 * jax.random.normal(next(keys), (L, n), jnp.float32)
        ).astype(jnp.bfloat16)
    fp8 = {
        "wqa": (L, h, cfg.qkv_cols),
        "wq_b": (L, cfg.q_lora, cfg.heads * cfg.head_dim),
        "idx_wq_b": (Lc, cfg.q_lora, cfg.idx_heads * cfg.idx_head_dim),
        "wo_a": (L, cfg.heads_per_group * cfg.head_dim, cfg.groups * cfg.o_lora),
        "wo_b": (L, cfg.groups * cfg.o_lora, h),
        "w13": (L, h, 2 * cfg.inter),
        "w2": (L, cfg.inter, h),
    }
    for name, shp in fp8.items():
        w[name], w[name + "_scale"] = _fp8_weight(next(keys), *shp)
    bf16 = {
        "wp": (Lc, h, cfg.idx_heads),
        "idx_comp": (Lc, h, 4 * cfg.idx_head_dim),
        "comp_csa": (Lc, h, 4 * cfg.head_dim),
        "comp_hca": (Lh, h, 2 * cfg.head_dim),
        "gate": (L, h, cfg.n_experts),
    }
    for name, shp in bf16.items():
        w[name] = _bf16_weight(next(keys), *shp)
    return w


def make_inputs(cfg: DenseChainConfig, seed: int = 1) -> dict[str, jax.Array]:
    """Chain-start state: residual streams, an identity pending post, rope rows."""
    k1, k2 = jax.random.split(jax.random.key(seed))
    m, h = cfg.rows, cfg.hidden
    res = jax.random.normal(k1, (m, cfg.hc_hidden), jnp.float32).astype(jnp.bfloat16)
    positions = jax.random.randint(k2, (m,), 0, 100_000)
    return {
        "x": jnp.zeros((m, h), jnp.bfloat16),
        "res": res,
        # post = 0, comb = I: the first fused post reproduces `res` exactly, so
        # layer 0 runs the same fused op as every other layer.
        "post": jnp.zeros((m, cfg.hc_mult), jnp.float32),
        "comb": jnp.tile(
            jnp.eye(cfg.hc_mult, dtype=jnp.float32).reshape(1, -1), (m, 1)
        ),
        "cos_sin": inverse_rope_rows(cfg, positions),
    }


def inverse_rope_rows(cfg: DenseChainConfig, positions: jax.Array) -> jax.Array:
    """[M, 256] f32 ``[cos | sin]`` over the last 128 lanes of a head, with the
    identity on the NoPE lanes and the sign of the inverse rotation applied
    (the table ``o_projection.gather_cos_sin(inverse=True)`` builds)."""
    rd = cfg.rope_dim
    inv_freq = 10000.0 ** (-jnp.arange(0, rd, 2, dtype=jnp.float32) / rd)
    ang = positions.astype(jnp.float32)[:, None] * inv_freq[None]
    cos = jnp.repeat(jnp.cos(ang), 2, axis=-1)
    sin = -jnp.repeat(jnp.sin(ang), 2, axis=-1)
    m = positions.shape[0]
    pad = 128 - rd
    cos = jnp.concatenate([jnp.ones((m, pad), jnp.float32), cos], axis=-1)
    sin = jnp.concatenate([jnp.zeros((m, pad), jnp.float32), sin], axis=-1)
    return jnp.concatenate([cos, sin], axis=-1)


# ---- ring packing -----------------------------------------------------------
def _tile_nn(w: jax.Array, rows: int, cols: int) -> jax.Array:
    """[L, k, n] -> [L, n_tiles, rows, cols], n-block outer, k-block inner."""
    L, k, n = w.shape
    w = w.reshape(L, k // rows, rows, n // cols, cols)
    return w.transpose(0, 3, 1, 2, 4).reshape(L, -1, rows, cols)


def hc_chunks(fn: jax.Array, cfg: DenseChainConfig) -> jax.Array:
    """f32 [L, 24, 4h] -> stacked bf16 chunks [L, hc_rows, 4h]."""
    chunks = jax.vmap(mhc_utils.split_fn3)(fn)[: cfg.hc_chunks]
    st = jnp.concatenate(chunks, axis=1)
    return jnp.pad(st, ((0, 0), (0, cfg.hc_rows - st.shape[1]), (0, 0)))


def xproj_csa(w: dict[str, jax.Array], cfg: DenseChainConfig) -> jax.Array:
    wp = jnp.pad(w["wp"], ((0, 0), (0, 0), (0, cfg.wp_cols - cfg.idx_heads)))
    return jnp.concatenate([wp, w["idx_comp"], w["comp_csa"]], axis=-1)


def pack_stream(name: str, w: dict[str, jax.Array], cfg: DenseChainConfig):
    s = cfg.streams[name]
    if not cfg.stream_layers(name):  # never fetched; the kernel still takes it
        return jnp.zeros((1, s.n_tiles, s.tile_rows, s.tile_cols), s.dtype)
    if s.nt:
        st = hc_chunks(w[name + "_fn"], cfg)
        L = st.shape[0]
        return st.reshape(L, s.tile_rows, s.n_k, s.tile_cols).transpose(0, 2, 1, 3)
    swa = jnp.asarray(cfg.stream_layers("wqa_ctx") if "wqa_ctx" in cfg.streams else [])
    src = {
        "xproj_csa": lambda: xproj_csa(w, cfg),
        "xproj_hca": lambda: w["comp_hca"],
        # the kv columns of the SWA layers' fused_wqa_wkv
        "wqa_ctx": lambda: w["wqa"][swa][..., cfg.q_lora :],
    }.get(name, lambda: w[name])()
    return _tile_nn(src, s.tile_rows, s.tile_cols)


def pack_smalls(w: dict[str, jax.Array], cfg: DenseChainConfig) -> jax.Array:
    """[L, rows, 128] f32: norm weights, hc base/scale, per-channel scales."""
    L = cfg.n_layers
    out = jnp.zeros((L, cfg.smalls_rows * 128), jnp.float32)
    csa = jnp.asarray(
        [li for li, t in enumerate(cfg.layer_types) if t == CSA], jnp.int32
    )
    for name, (row, n) in cfg.smalls_layout.items():
        if name == "_total":
            continue
        if name == "wqa_ctx_scale":  # every layer's kv columns (read on SWA ones)
            v = w["wqa_scale"][:, cfg.q_lora :].astype(jnp.float32)
        else:
            v = w[name].astype(jnp.float32)
        if name == "idx_wq_b_scale":
            v = jnp.zeros((L, n), jnp.float32).at[csa].set(v)
        out = out.at[:, row * 128 : row * 128 + n].set(v)
    return out.reshape(L, cfg.smalls_rows, 128)


def pack_weights(w: dict[str, jax.Array], cfg: DenseChainConfig) -> dict:
    return {
        "tiles": {n: pack_stream(n, w, cfg) for n in cfg.stream_names},
        "smalls": pack_smalls(w, cfg),
    }
