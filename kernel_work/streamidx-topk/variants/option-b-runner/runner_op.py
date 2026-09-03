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
"""The caller, as it would ship, with Option B turned on.

## Why a copy rather than an edit of `mla_attention_op.py`

The brief allows editing `src/vllm_torchtpu/layers/vllm/custom_ops/
mla_attention_op.py`.  This file is a copy instead, for one reason: **the
incumbent has to stay measurable in the same job.**  Every number in this
variant is a ratio against the unchunked path, and the unchunked path is the
production caller calling the production kernel.  Editing the real file in the
mapped `src/` tree would move the baseline along with the candidate and leave
nothing to divide by.

The copy is only the *jax function* -- `_streamidx_topk_jax`, the thing inside
`_build_streamidx_op` that actually does work.  Everything around it in the
real file (`pallas.jax_op`, `register_fake`, the torch wrapper, `forward_oot`)
is torch/vLLM plumbing that a 4-chip `simple-run-command` pod cannot stand up
and that Option B does not touch.

A copy can drift from its original silently, so `check_fidelity()` reparses
the real file and asserts that the KV-cache scatter prologue -- the half of
the function this change must not touch -- is **character-identical**.  The
delta is printed in full either way.

## What the change to the caller actually is

Three lines, and none of them is a new argument from vLLM:

    max_num_seqs = seq_lens.shape[0]        # static: it is a shape
    num_tokens   = q_bytes.shape[0]         # static: it is a shape
    g            = chunk_policy(num_tokens, max_num_seqs)

`option-b-pipeline/RESULTS.md` section 7 predicted that the chunk plan would
have to be handed in as static metadata, because `cu_q_lens` and
`distribution` are traced.  It does not.  Only `G` needs to be static, and `G`
is a function of two *shapes*, which are static under `jax.jit` by
construction.  Everything the plan needs beyond `G` -- which sequences a chunk
touches, what each one's clipped `seq_len` and `cu_q_len` are -- is elementwise
arithmetic on the traced arrays, done inside the kernel module.

`chunk_policy` is also the **phase gate**.  Phase is traced (it lives in
`distribution`), so nothing can branch on it; `num_tokens` is not, and at this
deployment it separates the two phases cleanly -- decode buckets are at most
`max_num_seqs` tokens and prefill is the 1024-token chunk.  That is the same
static quantity vLLM already compiles a separate program for, so gating on it
costs no extra specialisation.
"""

from __future__ import annotations

import ast
import difflib
import os

import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 4)))
_REL = os.path.join("src", "vllm_torchtpu", "layers", "vllm", "custom_ops",
                    "mla_attention_op.py")


def _find_real_caller() -> str:
    """Locate `mla_attention_op.py` in the tree that is actually importable.

    On the pod the checkout is mapped to `/root/cloud-devkit/vllm-torchtpu`
    while this file arrives under `/root/kw`, so the relative path does not
    hold.  Prefer the installed package's own location -- that is the copy
    whose kernels the run imports, and therefore the only one worth diffing
    against.  Importing the module itself would drag in vLLM, so only the
    package root is imported.
    """
    try:
        import vllm_torchtpu  # noqa: PLC0415
        p = os.path.join(os.path.dirname(vllm_torchtpu.__file__), "layers",
                         "vllm", "custom_ops", "mla_attention_op.py")
        if os.path.exists(p):
            return p
    except Exception:  # noqa: BLE001
        pass
    return os.path.join(_ROOT, _REL)


REAL_CALLER = _find_real_caller()

# `mla_attention_op.py:57`.  Kept as a named constant so the copy reads like
# the original; the merged module deleted the cond this selected, so the B
# path has no use for it and the fidelity diff shows it going away.
STREAMIDX_ENABLE_EARLY_EXIT = True


# ---------------------------------------------------------------------------
# The policy.  Static in, static out.
# ---------------------------------------------------------------------------

# Prefill chunk size at this deployment (`--max-num-batched-tokens=1024`).
PROD_PREFILL_TOKENS = 1024
# The G this variant recommends at that token count.  Swept, not assumed.
DEFAULT_PREFILL_G = 4


def chunk_policy(num_tokens: int,
                 max_num_seqs: int,
                 prefill_g: int = DEFAULT_PREFILL_G) -> int:
    """How many Option B chunks to use, from two static shapes.

    Returns 1 (the unchunked merged path) unless the program is a prefill-sized
    one whose chunks would still be at least `max_num_seqs` tokens wide -- the
    guard that lets chunks past the first statically omit the decode launches.
    See `streamindex_topk_b.plan_chunks`, which is what enforces it.
    """
    g = prefill_g
    while g > 1:
        if num_tokens % g == 0 and num_tokens // g >= max_num_seqs:
            return g
        g -= 1
    return 1


# ---------------------------------------------------------------------------
# The copied jax function.
# ---------------------------------------------------------------------------


def build_streamidx_topk_jax(
    streamindex_topk,
    *,
    topk: int,
    num_kv_pages_per_block=(2, 2, 2),
    num_queries_per_block=(1, 128, 128),
    prefill_g: int = DEFAULT_PREFILL_G,
    head_major_prefill: bool = True,
    chunked: bool = True,
    incumbent: bool = False,
):
    """`_build_streamidx_op`'s inner function, with `streamindex_topk` injected.

    `incumbent=True` passes `enable_early_exit=` and no B/C arguments, i.e. it
    reproduces the production call exactly; that arm is the baseline.
    """

    def _streamidx_topk_jax(
            q_bytes: jax.Array,  # uint8 view of fp8 q [num_tokens, H, D]
            weights: jax.Array,  # [num_tokens, H]
            cache_kv: jax.
        Array,  # uint8 [num_blocks, block_size_per_kv_packing, kv_packing, lkv_dim] where lkv_dim = D + 1 padded to a 128 multiple
            k_packed: jax.
        Array,  # uint8 [num_tokens, D + 1] fp8 k + e8m0 scale
            seq_lens: jax.Array,  # i32 [num_seqs]
            page_indices: jax.Array,  # i32 [num_seqs * pages_per_seq]
            cu_q_lens: jax.Array,  # i32 [num_seqs + 1]
            distribution: jax.Array,  # i32 [3]
    ) -> tuple[jax.Array, jax.Array]:
        _, block_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
        block_size = block_size_per_kv_packing * kv_packing
        # Cache rows are padded to the 128-lane HBM width; pad the packed
        # fp8+scale rows to match before scattering.
        k_packed = jnp.pad(k_packed,
                           ((0, 0), (0, lkv_dim - k_packed.shape[-1])))

        # Insert this step's K rows into the KV cache.
        tok = jnp.arange(k_packed.shape[0], dtype=jnp.int32)
        seq_id = jnp.searchsorted(cu_q_lens[1:], tok,
                                  side="right").astype(jnp.int32)
        q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
        pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
        page = jnp.where(
            tok < cu_q_lens[-1],
            page_indices.reshape(seq_lens.shape[0], -1)[seq_id,
                                                        pos // block_size],
            jnp.int32(2**30))
        slot = pos % block_size
        cache_kv = cache_kv.at[page, slot // kv_packing,
                               slot % kv_packing].set(k_packed)

        q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
        # ---- Option B: the entire caller-side change. -------------------
        # Both operands are SHAPES, so both are static under `jax.jit`; `g`
        # is therefore a Python int by the time `streamindex_topk` sees it.
        g = chunk_policy(q_bytes.shape[0], seq_lens.shape[0],
                         prefill_g) if chunked else 1
        # -----------------------------------------------------------------
        extra = ({
            "enable_early_exit": STREAMIDX_ENABLE_EARLY_EXIT
        } if incumbent else {
            "head_major_prefill": head_major_prefill,
            "num_chunks": g,
        })
        topk_indices = streamindex_topk(
            q,
            weights,
            cache_kv,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            k=topk,
            compression_ratio=1,
            # The following parameters are tuned based on microbenchmark
            # results.
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            **extra,
        )
        return cache_kv, topk_indices

    _streamidx_topk_jax.chunk_g = None  # filled by `resolved_g` below
    return _streamidx_topk_jax


def resolved_g(num_tokens: int, max_num_seqs: int, prefill_g: int,
               chunked: bool) -> int:
    """The `g` the built function will pick, without tracing it."""
    return chunk_policy(num_tokens, max_num_seqs, prefill_g) if chunked else 1


# ---------------------------------------------------------------------------
# Fidelity: the copy against the real file.
# ---------------------------------------------------------------------------

# Everything from here to (and including) the bitcast is the KV-cache scatter.
# Option B does not touch it, so it must match the real file byte for byte.
_PROLOGUE_START = "_, block_size_per_kv_packing, kv_packing, lkv_dim"
_PROLOGUE_END = "q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)"


def _extract(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise LookupError(f"{name} not found")


def _slice_prologue(text: str) -> str:
    i = text.index(_PROLOGUE_START)
    j = text.index(_PROLOGUE_END) + len(_PROLOGUE_END)
    # Normalise indentation only: the copy sits one `def` level shallower.
    return "\n".join(ln.strip() for ln in text[i:j].splitlines())


def check_fidelity(log=print) -> dict:
    """Reparse the real caller and diff the copy against it.

    Returns a dict; `prologue_identical` is the assertion that matters.  The
    full unified diff of the two function bodies is printed regardless, so the
    deliberate delta is on the record rather than merely claimed.
    """
    with open(REAL_CALLER) as fh:
        real_src = fh.read()
    with open(__file__) as fh:
        copy_src = fh.read()

    real = _extract(real_src, "_streamidx_topk_jax")
    copy = _extract(copy_src, "_streamidx_topk_jax")

    real_p = _slice_prologue(real)
    copy_p = _slice_prologue(copy)
    identical = real_p == copy_p

    log(f"FIDELITY real_caller={REAL_CALLER}")
    log(f"FIDELITY prologue_identical={identical} "
        f"prologue_lines={len(real_p.splitlines())}")
    if not identical:
        for ln in difflib.unified_diff(real_p.splitlines(),
                                       copy_p.splitlines(),
                                       "real.prologue",
                                       "copy.prologue",
                                       lineterm=""):
            log(f"FIDELITY-PROLOGUE {ln}")
    log("FIDELITY full function diff (deliberate delta follows):")
    for ln in difflib.unified_diff(
        [l.strip() for l in real.splitlines()],
        [l.strip() for l in copy.splitlines()],
            "mla_attention_op.py::_streamidx_topk_jax",
            "runner_op.py::_streamidx_topk_jax",
            lineterm=""):
        log(f"FIDELITY-DIFF {ln}")
    return {"prologue_identical": identical, "real_caller": REAL_CALLER}


if __name__ == "__main__":
    import sys
    r = check_fidelity()
    sys.exit(0 if r["prologue_identical"] else 1)
