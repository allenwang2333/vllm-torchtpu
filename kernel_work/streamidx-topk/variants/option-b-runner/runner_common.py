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
"""Configs, caller-level inputs and arms for the Option B runner A/B.

Everything here is at the **caller's** level, not the kernel's.  The unit
under test is `_streamidx_topk_jax` -- KV-cache scatter, fp8 bitcast and
`streamindex_topk` inside one `jax.jit` -- because that is what a runner
change changes, and because the scatter is real work that the kernel-level
harnesses in `variants/option-b-pipeline/` and `variants/bxc/` never charged
anyone for.

## The configs, and why each one exists

| config | slots | real seqs | tokens | what it is for |
| --- | --- | --- | --- | --- |
| `pre_q1024_ctx9216_s1`  | 1  | 1 prefill        | 1024 | the geometry every earlier probe used, so the numbers are comparable |
| `pre_q1024_ctx9216_s64` | 64 | 1 prefill        | 1024 | the same step as vLLM actually presents it: a padded 64-slot bucket. This is the config the static guard has to survive |
| `mix_ragged_s64`        | 64 | 4 prefill, ragged| 1024 | q_lens 300/100/500/124: chunk boundaries fall **inside** sequences at every G. The harness's pseudo-sequence trick cannot express this; the clipped-metadata formula has to |
| `mixdec_s64`            | 64 | 8 decode + 2 prefill | 1024 | decode and prefill in one call, so the "chunks past the first can statically omit the decode launches" claim is tested rather than asserted |
| `dec_b64_ctx8704_s64`   | 64 | 64 decode        | 64   | production decode. `chunk_policy` must return G=1 here; if it does not, B is being applied where `option-c`'s +59.62% lesson says it must not be |

## Reference

Correctness is judged against the **production caller calling the production
kernel**: `runner_op.build_streamidx_topk_jax(src.streamindex_topk,
incumbent=True)`, i.e. `enable_early_exit=True`, `(2,2,2)`, `(1,128,128)` --
`mla_attention_op.py:107-122` verbatim.  Not against the merged module, and
not against a harness.
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
_B = os.path.join(_HERE, "streamindex_topk_b.py")
_MERGED = os.path.abspath(
    os.path.join(_HERE, os.pardir, "merged", "streamindex_topk_merged.py"))

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import runner_op  # noqa: E402

# ---------------------------------------------------------------------------
# Geometry constants -- the GLM-5.2 nightly deployment.  Same values as
# `instruments/bench_streamidx_topk.py` and `option-b-pipeline/pipeline.py`.
# ---------------------------------------------------------------------------
H_IDX = 32
D_IDX = 128
K_IDX = 2048
PAGE_SIZE = 1024
KV_PACKING = 4
LKV_DIM = 256
LKV_USEFUL = D_IDX + 1  # 129: fp8 row + one e8m0 scale byte
MAX_MODEL_LEN = 9216
TOTAL_PAGES = 576  # 64 slots x 9 pages
INCUMBENT_TUPLE = (2, 2, 2)
NUM_QUERIES_PER_BLOCK = (1, 128, 128)


def cdiv(a, b):
    return -(-a // b)


def align_to(x, a):
    return cdiv(x, a) * a


PAGES_PER_SEQ = cdiv(MAX_MODEL_LEN, PAGE_SIZE)  # 9
NUM_SUBLANES_TOTAL = max(
    align_to(PAGES_PER_SEQ, p) * PAGE_SIZE // 128 for p in INCUMBENT_TUPLE)
NUM_POSITIONS = NUM_SUBLANES_TOTAL * 128  # 10240


@dataclasses.dataclass(frozen=True)
class Cfg:
    name: str
    slots: int  # max_num_seqs: the width of seq_lens / cu_q_lens
    q_lens: tuple[int, ...]  # per real sequence, in order
    ctx_lens: tuple[int, ...]  # seq_lens for the real sequences
    num_decode: int  # distribution[0] == distribution[1]
    num_tokens: int  # the compiled token bucket (>= sum(q_lens))
    note: str = ""

    @property
    def num_seqs(self) -> int:
        return len(self.q_lens)


CONFIGS: tuple[Cfg, ...] = (
    Cfg("pre_q1024_ctx9216_s1", 1, (1024, ), (9216, ), 0, 1024,
        "prefill as every earlier probe presented it"),
    Cfg("pre_q1024_ctx9216_s64", 64, (1024, ), (9216, ), 0, 1024,
        "prefill in a padded 64-slot bucket: what vLLM hands the op"),
    Cfg("mix_ragged_s64", 64, (300, 100, 500, 124),
        (9216, 4096, 7168, 2048), 0, 1024,
        "4 ragged prefill seqs; chunk boundaries cut inside sequences"),
    Cfg("mixdec_s64", 64, (1, ) * 8 + (508, 508), (8704, ) * 8 + (9216, 5120),
        8, 1024, "8 decode + 2 prefill in one call"),
    Cfg("dec_b64_ctx8704_s64", 64, (1, ) * 64, (8704, ) * 64, 64, 64,
        "production decode: chunk_policy must return G=1"),
)
CONFIG_BY_NAME = {c.name: c for c in CONFIGS}

# Configs traced for device time.  The ragged one is correctness-only: it is a
# shape vLLM can produce but not the one the nightly runs, and a perf number
# from it would invite being quoted as production.
PERF_CONFIGS = ("pre_q1024_ctx9216_s1", "pre_q1024_ctx9216_s64",
                "mixdec_s64", "dec_b64_ctx8704_s64")

G_SWEEP = (1, 2, 4, 8)


# ---------------------------------------------------------------------------
# Module loading.  Three modules define `streamindex_topk`; never a bare
# import.
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
        "merged": _load(_MERGED, "streamindex_topk_merged"),
        "b": _load(_B, "streamindex_topk_b"),
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
    ok = True
    for m in (mods["incumbent"], mods["sparsecore"]):
        h = md5(m.__file__)
        base = os.path.basename(m.__file__)
        want = EXPECT_MD5.get(base)
        good = (want is None or h == want)
        ok = ok and good
        log(f"MODULE {m.__file__} md5={h} expect={want} "
            f"{'OK' if good else 'MISMATCH'}")
    for name, path in (("merged", _MERGED), ("b", _B),
                       ("runner_op", os.path.join(_HERE, "runner_op.py")),
                       ("runner_common", os.path.join(_HERE,
                                                      "runner_common.py"))):
        log(f"VARIANT {name} {path} md5={md5(path)}")
    log(f"MD5_READBACK_OK={ok}")
    return ok


# ---------------------------------------------------------------------------
# Caller-level inputs.
# ---------------------------------------------------------------------------


def build_inputs(cfg: Cfg, seed: int = 0) -> dict:
    """The eight arrays `_streamidx_topk_jax` is called with.

    Value construction follows `instruments/bench_streamidx_topk.py`
    (`_make_cache`, `build_inputs`) so the score distribution is the one every
    earlier probe measured, and `option-b-pipeline/pipeline.py`'s
    `total_pages = 576` so the DMA is not reading a nine-page toy region.
    """
    keys = jax.random.split(jax.random.key(seed), 5)
    n_tok = cfg.num_tokens

    q = jax.random.normal(keys[0], (n_tok, H_IDX, D_IDX),
                          jnp.float32).astype(jnp.float8_e4m3fn)
    # The caller receives q as a uint8 *view*, and bitcasts it back.
    q_bytes = jax.lax.bitcast_convert_type(q, jnp.uint8)
    weights = jax.random.uniform(keys[1], (n_tok, H_IDX), jnp.float32, 0.25,
                                 1.75).astype(jnp.bfloat16)

    shape = (TOTAL_PAGES, PAGE_SIZE // KV_PACKING, KV_PACKING, LKV_DIM)
    raw = jax.random.bits(keys[2], shape, dtype=jnp.uint8)
    exp = jnp.uint8(5) + ((raw >> jnp.uint8(3)) & jnp.uint8(3))
    vals = (raw & jnp.uint8(0x87)) | (exp << jnp.uint8(3))
    lane = jnp.arange(LKV_DIM, dtype=jnp.int32)
    cache_kv = jnp.where(lane < D_IDX, vals,
                         jnp.where(lane == D_IDX, jnp.uint8(127),
                                   jnp.uint8(0)))

    # This step's K rows, in the caller's packed layout: D fp8 bytes + scale.
    kraw = jax.random.bits(keys[3], (n_tok, LKV_USEFUL), dtype=jnp.uint8)
    kexp = jnp.uint8(5) + ((kraw >> jnp.uint8(3)) & jnp.uint8(3))
    kvals = (kraw & jnp.uint8(0x87)) | (kexp << jnp.uint8(3))
    col = jnp.arange(LKV_USEFUL, dtype=jnp.int32)
    k_packed = jnp.where(col < D_IDX, kvals, jnp.uint8(127))

    perm = np.asarray(
        jax.random.permutation(keys[4],
                               jnp.arange(TOTAL_PAGES, dtype=jnp.int32)))
    page_indices = jnp.asarray(perm[:cfg.slots * PAGES_PER_SEQ], jnp.int32)

    seq_lens_np = np.zeros((cfg.slots, ), np.int32)
    seq_lens_np[:cfg.num_seqs] = np.asarray(cfg.ctx_lens, np.int32)
    # `query_start_loc`: cumulative, then the total repeated across the padded
    # tail.  That is what makes `cu_q_lens[-1]` the real token count, which is
    # what the caller's own validity guard reads.
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
        "page_indices": page_indices,
        "cu_q_lens": jnp.asarray(cu_np),
        "distribution": distribution,
    }


ARG_ORDER = ("q_bytes", "weights", "cache_kv", "k_packed", "seq_lens",
             "page_indices", "cu_q_lens", "distribution")


def as_args(inputs: dict) -> tuple:
    return tuple(inputs[n] for n in ARG_ORDER)


# ---------------------------------------------------------------------------
# Arms.
# ---------------------------------------------------------------------------


def arm_name(kind: str, g: int | None = None) -> str:
    return kind if g is None else f"{kind}G{g}"


def make_arm(mods: dict, kind: str, g: int = 1, donate: bool = False):
    """A jitted caller.  `kind` is `inc` | `b`.

    * `inc`  -- production caller, production kernel, `enable_early_exit=True`.
    * `bG1`  -- the merged module through the same caller, unchunked.  This is
                the baseline every B number is divided by: it isolates B from
                the three changes the merged module already carries.
    * `bG2/4/8` -- the same, chunked.

    `g` is forced rather than left to `chunk_policy` so the sweep can reach
    G values the policy would not choose; `probe_runner.py --arm plan` checks
    separately that the policy picks the intended one on its own.

    `donate` mirrors the real op, which builds its callable with
    `pallas.jax_op(..., donate_argnums=(2,))` -- argument 2 is `cache_kv`, and
    production hands the KV cache in as a donated buffer that the scatter
    updates in place.  Without it XLA has to materialise a second 151 MiB
    cache, which on a traced prefill shows up as a `reshape.N` /
    `reshape.N+2.cache_kv_hbm_ref` pair worth ~864 us and swamps everything
    the op actually does.  Timing arms must set it; `arm_correct` must not,
    because a donated input is invalidated by the call and the reference and
    every G share one input set.
    """
    if kind == "inc":
        fn = runner_op.build_streamidx_topk_jax(
            mods["incumbent"].streamindex_topk,
            topk=K_IDX,
            num_kv_pages_per_block=INCUMBENT_TUPLE,
            num_queries_per_block=NUM_QUERIES_PER_BLOCK,
            incumbent=True,
        )
    elif kind == "b":
        fn = runner_op.build_streamidx_topk_jax(
            mods["b"].streamindex_topk,
            topk=K_IDX,
            num_kv_pages_per_block=INCUMBENT_TUPLE,
            num_queries_per_block=NUM_QUERIES_PER_BLOCK,
            prefill_g=g,
            head_major_prefill=True,
            chunked=(g > 1),
        )
    else:
        raise ValueError(kind)
    return jax.jit(fn, donate_argnums=(2, )) if donate else jax.jit(fn)


def chained_caller(fn, args: tuple):
    """Drive a donating arm: feed each call's returned cache into the next.

    With `donate_argnums=(2,)` the input `cache_kv` buffer is consumed by the
    call, so a loop that re-passes the original array raises on the second
    iteration.  Production has the same shape -- the cache is a persistent
    buffer that each step updates -- so chaining is what the runner does, not
    a workaround.  Returns `(step, state)`; `step()` issues one call without
    blocking and returns its outputs.
    """
    state = list(args)

    def step():
        out = fn(*state)
        state[2] = out[0]
        return out

    return step, state


def perf_arms(mods: dict, cfg: Cfg) -> list[str]:
    """`inc`, then `bG<g>` for every G the plan accepts at this config."""
    out = ["inc"]
    for g in G_SWEEP:
        if g == 1 or mods["b"].plan_chunks(cfg.num_tokens, cfg.slots,
                                           g)["ok"]:
            out.append(f"bG{g}")
    return out
