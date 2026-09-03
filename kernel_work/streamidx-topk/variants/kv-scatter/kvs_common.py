#!/usr/bin/env python3
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
"""Geometry, inputs, module loading and arms for the KV-scatter relayout A/B.

## The thing under test

`mla_attention_op.py:104`

    cache_kv = cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(
        k_packed)

At prefill this compiles into three HLO instructions, not one:

    reshape   u8[415,256,4,256]{T(4,128)(4,1)} -> u8[424960,256]{T(8,128)(4,1)}
    scatter_fusion                                u8[424960,256]{T(8,128)(4,1)}
    reshape   u8[424960,256]{T(8,128)(4,1)} -> u8[415,256,4,256]{T(4,128)(4,1)}

The two reshapes relayout the entire 108.8 MB indexer KV cache in order to
write 1024 rows into it, and nothing downstream consumes the 2-D form.

## The two layouts

| name   | cache shape         | one token is                       |
| ---    | ---                 | ---                                |
| `base` | `[415,256,4,256]`   | one 256-byte row                   |
| `v128` | `[415,256,8,128]`   | two consecutive 128-byte rows      |

They hold the same bytes in the same row-major order, so `v128` is a candidate
*view* of `base` rather than a different format.  Whether it is a free view on
the device -- i.e. whether the reshape between them is a `bitcast` -- is one of
the questions this harness answers, and it is not the same question as whether
they are row-major-equivalent.  `probe_kvs_hlo.py --mode layout` checks the
byte order; `--mode hlo` checks the bitcast.

## Geometry

Production values, from the GLM-5.2 profile of job `j-9b23bb32-...`:
`TOTAL_PAGES = 415`, `PAGE_SIZE = 1024`, `KV_PACKING = 4`, `LKV_DIM = 256`,
so the cache is `415*256*4*256 = 108.8 MB` -- the same size whose relayout the
profile timed at 349.03 µs + 290.43 µs.  Earlier probes in this exercise used
576 pages; 415 is used here instead so the reshape's byte count matches the
production one exactly and the µs are directly comparable.

`page_indices[j] = perm415[j % 415]`.  With `pages_per_seq = 9` and
`415 % 9 != 0`, every slot's *written* page (`j = 9*i + 8`) is distinct across
all 64 slots -- asserted in `build_inputs`, because two slots scattering to one
(page, slot) would make the decode correctness check nondeterministic.

## Configs

| config    | tokens | what it is |
| ---       | ---    | --- |
| `pre1024` | 1024   | production prefill: one 1024-token chunk, 64-slot bucket. The shape the triad was measured at. |
| `dec64`   | 64     | production decode: 64 slots, one token each. The regression risk -- `base` already scatters cleanly here (8.87 µs, no reshapes). |
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_V128_KERNEL = os.path.join(_HERE, "streamindex_topk_v128.py")
_RUNNER = os.path.abspath(
    os.path.join(_HERE, os.pardir, "option-b-runner", "runner_op.py"))

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Geometry -- the GLM-5.2 nightly deployment, at the production cache size.
# ---------------------------------------------------------------------------
H_IDX = 32
D_IDX = 128
K_IDX = 2048
PAGE_SIZE = 1024
KV_PACKING = 4
LKV_DIM = 256
LKV_USEFUL = D_IDX + 1  # 129: fp8 row + one e8m0 scale byte
MAX_MODEL_LEN = 9216
TOTAL_PAGES = 415  # production: u8[415,256,4,256] = 108.8 MB
INCUMBENT_TUPLE = (2, 2, 2)
NUM_QUERIES_PER_BLOCK = (1, 128, 128)
ROWS_PER_TOKEN = 2  # 256-byte logical row -> two 128-byte rows


def cdiv(a, b):
    return -(-a // b)


def align_to(x, a):
    return cdiv(x, a) * a


PAGES_PER_SEQ = cdiv(MAX_MODEL_LEN, PAGE_SIZE)  # 9
BLOCK_PER_KV_PACKING = PAGE_SIZE // KV_PACKING  # 256

BASE_CACHE_SHAPE = (TOTAL_PAGES, BLOCK_PER_KV_PACKING, KV_PACKING, LKV_DIM)
V128_CACHE_SHAPE = (TOTAL_PAGES, BLOCK_PER_KV_PACKING,
                    KV_PACKING * ROWS_PER_TOKEN, LKV_DIM // ROWS_PER_TOKEN)
CACHE_BYTES = int(np.prod(BASE_CACHE_SHAPE))


@dataclasses.dataclass(frozen=True)
class Cfg:
    name: str
    slots: int  # max_num_seqs: width of seq_lens / cu_q_lens
    q_lens: tuple[int, ...]
    ctx_lens: tuple[int, ...]
    num_decode: int
    num_tokens: int
    note: str = ""

    @property
    def num_seqs(self) -> int:
        return len(self.q_lens)


CONFIGS: tuple[Cfg, ...] = (
    Cfg("pre1024", 64, (1024, ), (9216, ), 0, 1024,
        "production prefill: 1024 rows written, 64-slot bucket"),
    Cfg("dec64", 64, (1, ) * 64, (8704, ) * 64, 64, 64,
        "production decode: 64 padded tokens"),
)
CONFIG_BY_NAME = {c.name: c for c in CONFIGS}

# ---------------------------------------------------------------------------
# Arms.
#
#   base    -- production caller, production kernel, `[415,256,4,256]`.
#   v128    -- 128-minor cache end to end: the v128 caller writes two rows per
#              slot and `streamindex_topk_v128` reads them.  Route (a) proper.
#   v128rt  -- the cache still ARRIVES as `[415,256,4,256]`; the caller
#              reshapes to 128-minor, scatters, and reshapes back for the
#              production kernel.  This is the arm that says whether route (a)
#              removes the relayout or merely moves it.
#   scat256 -- `base`'s scatter alone, no kernel.  The cleanest isolation of
#              the triad, and the arm the HLO is quoted from.
#   scat128 -- `v128`'s scatter alone, no kernel.
# ---------------------------------------------------------------------------
FULL_ARMS = ("base", "v128", "v128rt")
SCAT_ARMS = ("scat256", "scat128", "scat128rt")
ALL_ARMS = FULL_ARMS + SCAT_ARMS
# `scat128rt` is scatter-only with the round trip, i.e. `v128rt` minus the
# kernel.  It exists so the round trip's cost is attributable to the reshapes
# and not to anything the kernel does.

CACHE_LAYOUT = {
    "base": "base",
    "v128": "v128",
    "v128rt": "base",
    "scat256": "base",
    "scat128": "v128",
    "scat128rt": "base",
}


# ---------------------------------------------------------------------------
# Module loading.  Never a bare import: three files define `streamindex_topk`.
# ---------------------------------------------------------------------------
def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_modules() -> dict:
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as incmod
    return {
        "incumbent": incmod,
        "sparsecore": scmod,
        "v128": _load(_V128_KERNEL, "streamindex_topk_v128"),
    }


def md5(path) -> str:
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


# The checkout's values.  A read-back that does not match these means the pod
# imported the image's stale kernels and the job is void -- the image ships a
# different `streamindex_topk.py` and no `sparsecore_topk.py` at all.
EXPECT_MD5 = {
    "streamindex_topk.py": "1e221f10cbf3573aa0d2ae78c3463532",
    "sparsecore_topk.py": "8bc9e25b8701a53a97704564b9fbc658",
}


def report_md5s(mods: dict, log=print) -> bool:
    import vllm_torchtpu
    log(f"MODULE vllm_torchtpu.__file__ = {vllm_torchtpu.__file__}")
    ok = True
    for m in (mods["incumbent"], mods["sparsecore"]):
        h = md5(m.__file__)
        base = os.path.basename(m.__file__)
        want = EXPECT_MD5.get(base)
        good = (want is None or h == want)
        ok = ok and good
        log(f"MODULE {m.__file__} md5={h} expect={want} "
            f"{'OK' if good else 'MISMATCH'}")
    for name in ("streamindex_topk_v128.py", "kvs_common.py", "kvs_op.py"):
        p = os.path.join(_HERE, name)
        if os.path.exists(p):
            log(f"VARIANT {name} {p} md5={md5(p)}")
    log(f"VARIANT runner_op.py {_RUNNER} md5={md5(_RUNNER)}")
    log(f"MD5_READBACK_OK={ok}")
    return ok


# ---------------------------------------------------------------------------
# Inputs.
# ---------------------------------------------------------------------------
def _base_cache(key) -> jnp.ndarray:
    """A cache in the production layout, with plausible fp8 byte values.

    Value construction follows `instruments/bench_streamidx_topk.py`
    (`_make_cache`) so the score distribution is the one every earlier probe in
    this exercise measured.
    """
    raw = jax.random.bits(key, BASE_CACHE_SHAPE, dtype=jnp.uint8)
    exp = jnp.uint8(5) + ((raw >> jnp.uint8(3)) & jnp.uint8(3))
    vals = (raw & jnp.uint8(0x87)) | (exp << jnp.uint8(3))
    lane = jnp.arange(LKV_DIM, dtype=jnp.int32)
    return jnp.where(lane < D_IDX, vals,
                     jnp.where(lane == D_IDX, jnp.uint8(127), jnp.uint8(0)))


# ---------------------------------------------------------------------------
# base <-> v128 sub-row placement.
#
# The scatter writes only the tokens of THIS step.  Everything else in the
# cache is pre-existing KV history that the kernel also reads, so the history
# has to be in whatever placement the kernel expects -- otherwise the A/B
# compares a correct kernel against a cache it cannot read, and every top-k row
# differs.  That is a harness property, not a kernel property, and getting it
# wrong is silent.
#
#   interleaved (`index`, `stride`) -- token k's two 128-byte sub-rows are
#       adjacent, at rows 2k and 2k+1.  Same byte order as the production
#       256-minor cache, so a plain `.reshape()` is the whole conversion.
#   split (`split`) -- all value sub-rows first, all scale sub-rows after.
#       A genuine permutation: rows k and kv_packing+k.
#
# `base_to_v128` and `v128_to_base` are exact inverses in both modes, which is
# what keeps the byte-equality checks meaningful under `split` too.
# ---------------------------------------------------------------------------
DEINTERLEAVE = os.environ.get("KVS_DEINTERLEAVE", "index")


def base_to_v128(arr):
    """`[415,256,4,256]` -> `[415,256,8,128]` in the current placement."""
    nb, bs, kvp, lkv = BASE_CACHE_SHAPE
    rows, lane = ROWS_PER_TOKEN, lkv // ROWS_PER_TOKEN
    if DEINTERLEAVE == "split":
        # [nb,bs,kvp,rows,lane] -> [nb,bs,rows,kvp,lane]: row = r*kvp + k.
        return arr.reshape(nb, bs, kvp, rows,
                           lane).transpose(0, 1, 3, 2,
                                           4).reshape(V128_CACHE_SHAPE)
    return arr.reshape(V128_CACHE_SHAPE)


def v128_to_base(arr):
    """The exact inverse of `base_to_v128`."""
    nb, bs, kvp, lkv = BASE_CACHE_SHAPE
    rows, lane = ROWS_PER_TOKEN, lkv // ROWS_PER_TOKEN
    if DEINTERLEAVE == "split":
        return arr.reshape(nb, bs, rows, kvp,
                           lane).transpose(0, 1, 3, 2,
                                           4).reshape(BASE_CACHE_SHAPE)
    return arr.reshape(BASE_CACHE_SHAPE)


def build_inputs(cfg: Cfg, layout: str = "base", seed: int = 0) -> dict:
    """The eight arrays the caller is invoked with.

    `layout` selects the cache shape.  The v128 cache is produced from the
    *same* base cache through `base_to_v128`, so the two layouts are guaranteed
    to hold the same logical KV history and any difference the A/B shows is the
    scatter's.  In the interleaved modes that conversion is a plain reshape and
    the two are byte-identical; under `split` it is a permutation, so they are
    equal only after `v128_to_base`.
    """
    assert layout in ("base", "v128"), layout
    keys = jax.random.split(jax.random.key(seed), 5)
    n_tok = cfg.num_tokens

    q = jax.random.normal(keys[0], (n_tok, H_IDX, D_IDX),
                          jnp.float32).astype(jnp.float8_e4m3fn)
    # The caller receives q as a uint8 *view* and bitcasts it back.
    q_bytes = jax.lax.bitcast_convert_type(q, jnp.uint8)
    weights = jax.random.uniform(keys[1], (n_tok, H_IDX), jnp.float32, 0.25,
                                 1.75).astype(jnp.bfloat16)

    cache_kv = _base_cache(keys[2])
    if layout == "v128":
        cache_kv = base_to_v128(cache_kv)

    kraw = jax.random.bits(keys[3], (n_tok, LKV_USEFUL), dtype=jnp.uint8)
    kexp = jnp.uint8(5) + ((kraw >> jnp.uint8(3)) & jnp.uint8(3))
    kvals = (kraw & jnp.uint8(0x87)) | (kexp << jnp.uint8(3))
    col = jnp.arange(LKV_USEFUL, dtype=jnp.int32)
    k_packed = jnp.where(col < D_IDX, kvals, jnp.uint8(127))

    perm = np.asarray(
        jax.random.permutation(keys[4],
                               jnp.arange(TOTAL_PAGES, dtype=jnp.int32)))
    n_idx = cfg.slots * PAGES_PER_SEQ
    page_idx_np = perm[np.arange(n_idx) % TOTAL_PAGES]

    # The page each real sequence writes into must be unique, or two scatter
    # updates race for one (page, slot) and the readback is undefined.  Every
    # config here writes each sequence's tokens into a single page, so
    # uniqueness of that page is uniqueness of every (page, slot) pair.
    ctx = np.asarray(cfg.ctx_lens, np.int64)
    q_np = np.asarray(cfg.q_lens, np.int64)
    assert np.all((ctx - q_np) // PAGE_SIZE == (ctx - 1) // PAGE_SIZE), (
        "a sequence spans two pages; the uniqueness check below is too weak")
    written = page_idx_np[np.arange(cfg.num_seqs) * PAGES_PER_SEQ +
                          (ctx - 1) // PAGE_SIZE]
    assert len(set(written.tolist())) == len(written), (
        f"written pages collide: {written.tolist()}")

    seq_lens_np = np.zeros((cfg.slots, ), np.int32)
    seq_lens_np[:cfg.num_seqs] = np.asarray(cfg.ctx_lens, np.int32)
    cu_np = np.zeros((cfg.slots + 1, ), np.int32)
    cu_np[1:cfg.num_seqs + 1] = np.cumsum(np.asarray(cfg.q_lens, np.int32))
    cu_np[cfg.num_seqs + 1:] = cu_np[cfg.num_seqs]

    distribution = jnp.array([cfg.num_decode, cfg.num_decode, cfg.num_seqs],
                             jnp.int32)

    return {
        "q_bytes": jax.block_until_ready(q_bytes),
        "weights": jax.block_until_ready(weights),
        "cache_kv": jax.block_until_ready(cache_kv),
        "k_packed": jax.block_until_ready(k_packed),
        "seq_lens": jnp.asarray(seq_lens_np),
        "page_indices": jnp.asarray(page_idx_np, jnp.int32),
        "cu_q_lens": jnp.asarray(cu_np),
        "distribution": distribution,
    }


ARG_ORDER = ("q_bytes", "weights", "cache_kv", "k_packed", "seq_lens",
             "page_indices", "cu_q_lens", "distribution")


def as_args(inputs: dict) -> tuple:
    return tuple(inputs[n] for n in ARG_ORDER)


def chained_caller(fn, args: tuple):
    """Drive a donating arm: feed each call's returned cache into the next.

    With `donate_argnums=(2,)` -- which is what `pallas.jax_op(...,
    donate_argnums=(2,))` gives the real op -- the input `cache_kv` buffer is
    consumed by the call, so a loop that re-passes the original array raises on
    the second iteration.  Production has the same shape: the cache is a
    persistent buffer each step updates.  Returns `(step, state)`.
    """
    state = list(args)

    def step():
        out = fn(*state)
        state[2] = out[0] if isinstance(out, tuple) else out
        return out

    return step, state
