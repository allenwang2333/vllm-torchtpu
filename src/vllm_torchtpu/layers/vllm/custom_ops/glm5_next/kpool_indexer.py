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
"""The fused GLM-5.3-Flash kpool indexer op: write -> score -> top-k -> expand.

One ``pallas.jax_op`` shaped like the GLM-5.2 template at
``mla_attention_op.py:92-135`` and ``:183-200``, donating the pooled indexer K
cache and the two pending-ring tensors, so nothing crosses the torch boundary
mid-pipeline.

WRITE BEFORE SCORE, INSIDE ONE OP
---------------------------------

``kpool_write`` runs first and its *output* cache is what the top-k scores.
That is not an optimisation, it is the correctness property: the pooled record
for a pool completed by this step -- including one completed from a
speculative draft -- must be visible to the scoring that follows it in the same
step, and a rejected draft's record must be overwritten by the next step before
anything can read a stale one. Splitting the two across the torch boundary
would let a graph break, a donation copy or a reordering separate them. The
existing GLM-5.2 indexer already does exactly this
(``mla_attention_op.py:92-121``); this matches it.

``enable_early_exit`` IS FALSE, AND IT IS ASSERTED
--------------------------------------------------

``streamindex_topk``'s fast path triggers on
``jnp.max(seq_lens) // compression_ratio <= k`` (``streamindex_topk.py:1010``).
At GLM's ``compression_ratio = index_kpool = 4`` and ``k = select_k = 512``
that is ``max(seq_lens) <= 2048`` -- i.e. **exactly** the identity regime that
is the current shipped, green, dense-MLA baseline.

The fast path's *output* is correct for GLM: ``_fast_path_max_valid_idx``
routes ``causal_row_offset`` through ``_last_visible_compressed_row``, which
gives exactly ``(p + 1) // index_kpool - 1`` at offset ``index_kpool - 1`` --
the oracle's candidate prefix. It must still be off, for a sharper reason:

``_fast_path`` takes ``seq_lens`` and ``cu_q_lens`` and **not the cache**. It
emits the identity selection without reading a single pooled record. So with
early exit on, every test at or below 2048 tokens -- which is every test that
exists today, because the 2048 cap is what this plan is removing -- would pass
with ``kpool_write`` producing complete garbage, and the port would first
diverge at exactly the length nobody has a baseline for. That is this project's
signature failure mode, so it is a hard asserted property with a message, never
a convention. ``STREAMIDX_ENABLE_EARLY_EXIT`` (``mla_attention_op.py:57``) is
deliberately **not** read from this path either: a module-level toggle another
model may flip must not be able to reach GLM.

NO MESH, NO PARTITION SPECS
---------------------------

Built in the GLM-5.2 form (``mla_attention_op.py:125-128``), not DeepSeek-V4's
head-sharded form (``deepseek_v4_indexer.py:154-172``). The per-head ReLU'd
score sum happens *inside* the Pallas body (``streamindex_topk.py:335``), so
this kernel cannot be partitioned on heads: a rank holding 4 of 32 heads would
select a different pool set than its peers, over a pooled cache they share.
The indexer projections are ``ReplicatedLinear(disable_tp=True)`` for the same
reason (see ``models/vllm/glm5_next/indexer.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
from vllm_torchtpu.kernels.glm5_kpool import (KPOOL_WRITE_DIAG_SIZE,
                                              derive_out_width,
                                              expand_pools_with_tail,
                                              kpool_write)

#: **False for GLM, always.** See the module docstring. Deliberately a distinct
#: constant from ``mla_attention_op.STREAMIDX_ENABLE_EARLY_EXIT`` so that
#: flipping the DeepSeek/GLM-5.2 toggle cannot reach this path.
KPOOL_ENABLE_EARLY_EXIT = False

#: ``streamindex_topk``'s query tiling, the same (decode, prefill, mixed) tuple
#: every current producer passes. Unlike ``num_kv_pages_per_block`` this one is
#: not geometry-constrained, so it is a tuning constant rather than a derived
#: value.
KPOOL_NUM_QUERIES_PER_BLOCK = (1, 128, 128)

_op_cache: dict = {}


@dataclass(frozen=True)
class KpoolIndexerOpConfig:
    """Every static argument of the fused op, each derived or asserted.

    ``causal_row_offset`` and ``out_width`` are **properties**, not fields, so
    a caller cannot pass a value for them at all -- which is the difference
    between "we set it to 3" and "it cannot be anything but 3".
    """

    index_kpool: int
    index_topk: int
    index_head_dim: int
    num_kv_pages_per_block: tuple[int, ...]
    """From ``derive_pooled_indexer_geometry``. NOT DeepSeek-V4's
    tuned ``(3, 2, 2)``: that is legal only at its ``block_size = 1024``."""

    num_queries_per_block: tuple[int, ...] = KPOOL_NUM_QUERIES_PER_BLOCK
    layer_prefix: str = ""

    def __post_init__(self) -> None:
        assert self.index_kpool >= 1, self.index_kpool
        assert self.index_topk % self.index_kpool == 0, (
            f"index_topk={self.index_topk} is not a whole number of pools of "
            f"index_kpool={self.index_kpool}")
        assert self.index_head_dim >= 1, self.index_head_dim
        assert len(self.num_kv_pages_per_block) == 3, (
            "streamindex_topk wants a (decode, prefill, mixed) tuple; got "
            f"{self.num_kv_pages_per_block}")
        assert len(
            self.num_queries_per_block) == 3, (self.num_queries_per_block)
        # Evaluate the derived properties once at construction so a bad
        # geometry raises here rather than inside a traced function.
        _ = self.out_width
        _ = self.causal_row_offset

    @property
    def select_k(self) -> int:
        """``streamindex_topk``'s ``k``, in **pool** space."""
        return self.index_topk // self.index_kpool

    @property
    def causal_row_offset(self) -> int:
        """``index_kpool - 1``: a pool is a candidate only once its **last**
        token is visible (``modeling_glm5_next.py:833-839``). 0 -- the kernel's
        default and DeepSeek-V4's convention -- would make pool ``j`` visible
        as soon as the query entered it, i.e. would let a query attend to its
        own future."""
        return self.index_kpool - 1

    @property
    def out_width(self) -> int:
        """The emitted row width, from the **single home** for that geometry.

        Never recomputed here. Two independent expressions that agree today are
        how the consumer's buffer ends up a few columns short later, clipping
        the always-selected tail -- invisible at ``L % index_kpool == 0`` and
        wrong at the other three residues.
        """
        return derive_out_width(self.index_topk, self.index_kpool)

    def topk_kwargs(self) -> dict:
        """The static arguments handed to ``streamindex_topk``.

        One place, so the guard cannot be bypassed by a caller that assembles
        the kwargs itself -- and so a test that substitutes the kernel still
        sees exactly what the kernel would have seen.

        The guard is a ``ValueError``, not an ``assert``: ``python -O``
        strips ``assert``, and under ``-O`` a bare assert here would silently
        restore the fast path, which is the single worst outcome this file
        has -- see below for why.
        """
        if KPOOL_ENABLE_EARLY_EXIT is not False:
            raise ValueError(
                "glm5_next kpool: enable_early_exit must be False. "
                "streamindex_topk's fast path triggers on "
                "`jnp.max(seq_lens) // compression_ratio <= k` "
                "(streamindex_topk.py:1010), which at compression_ratio="
                f"{self.index_kpool} and k={self.select_k} means "
                f"max(seq_lens) <= {self.select_k * self.index_kpool} -- "
                "exactly the identity regime that is GLM-5.3's whole existing "
                "green baseline. `_fast_path` takes seq_lens and cu_q_lens "
                "and NOT the cache, so it emits the identity selection "
                "without "
                "reading a single pooled record: with it on, every test at or "
                f"below {self.select_k * self.index_kpool} tokens would pass "
                "with kpool_write producing garbage, and the port would first "
                "diverge at the one length nobody has a baseline for.")
        return {
            "k": self.select_k,
            "compression_ratio": self.index_kpool,
            "causal_row_offset": self.causal_row_offset,
            "num_kv_pages_per_block": tuple(self.num_kv_pages_per_block),
            "num_queries_per_block": tuple(self.num_queries_per_block),
            "enable_early_exit": KPOOL_ENABLE_EARLY_EXIT,
        }

    def describe(self) -> str:
        return ("glm5_next kpool fused op: "
                f"prefix={self.layer_prefix!r}, "
                f"index_kpool={self.index_kpool}, "
                f"index_topk={self.index_topk}, "
                f"select_k={self.select_k}, "
                f"causal_row_offset={self.causal_row_offset}, "
                f"out_width={self.out_width}, "
                f"enable_early_exit={KPOOL_ENABLE_EARLY_EXIT}, "
                "num_kv_pages_per_block="
                f"{tuple(self.num_kv_pages_per_block)}, "
                f"num_queries_per_block={tuple(self.num_queries_per_block)}")


def assert_topk_buffer_width(buffer_width: int,
                             config: KpoolIndexerOpConfig) -> None:
    """The width the op emits must equal the width the consumer allocated.

    The indexer sizes ``topk_indices_buffer`` and the sparse MLA consumer
    reads it; if the two numbers ever diverge the symptom is a clipped tail,
    which is
    invisible at ``L % index_kpool == 0``. Both sides go through
    :func:`derive_out_width`, so this only fires if somebody adds a second
    derivation -- which is exactly what it is for.

    ``ValueError``, not ``assert``: two plain ints compared once at op
    construction, and ``python -O`` deletes ``assert``. What it deletes here is
    not a crash but a silently short row.
    """
    if buffer_width != config.out_width:
        raise ValueError(
            f"topk_indices_buffer is {buffer_width} columns wide but the "
            f"kpool op emits {config.out_width} (derive_out_width(index_topk="
            f"{config.index_topk}, index_kpool={config.index_kpool})). The "
            f"columns past index_topk={config.index_topk} are the "
            "always-selected tail -- the (p + 1) % index_kpool most recent "
            "tokens -- so a short buffer drops them, which is invisible at "
            "L % index_kpool == 0 and wrong at the other residues.")


def kpool_request_distribution(request_distribution):
    """``stack([rd[0], rd[0], rd[2]])`` -- ``i == j``, matching every producer.

    ``streamindex_topk``'s docstring describes three bands (decode-only,
    chunked-prefill-only, mixed), but the code runs bands ``[0:i]`` and
    ``[i:j]`` with ``static_q_len=1`` (``streamindex_topk.py:755-791``). A
    genuine three-band distribution would therefore score prefill sequences as
    if their query length were 1 -- silently, and only for prefill. Collapsing
    ``j`` onto ``i`` puts every non-decode sequence in the mixed band, which is
    what ``mla_attention_op.py:177`` and the DeepSeek producers already do.
    """
    rd = request_distribution
    if isinstance(rd, torch.Tensor):
        return torch.stack([rd[0], rd[0], rd[2]])
    return jnp.stack([rd[0], rd[0], rd[2]])


# ---------------------------------------------------------------------------
# The pure-jax pipeline
# ---------------------------------------------------------------------------


def kpool_indexer_stages(
    q_bytes: jax.Array,
    weights: jax.Array,
    pooled_cache: jax.Array,
    pending_kv: jax.Array,
    pending_tag: jax.Array,
    k: jax.Array,
    gate: jax.Array,
    ape: jax.Array,
    positions: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    kpool_state_indices: jax.Array,
    distribution: jax.Array,
    *,
    config: KpoolIndexerOpConfig,
    topk_fn=streamindex_topk,
):
    """``write -> score -> top-k -> expand``, in that order, in one trace.

    Args:
        q_bytes: ``uint8`` view of the fp8 query, ``[T, n_head, head_dim]``.
            Bytes rather than fp8 because torch cannot pass ``float8_e4m3fn``
            through the op boundary; ``mla_attention_op.py:69`` does the same.
        weights: ``[T, n_head]`` with ``q_scale * softmax_scale *
            n_head**-0.5`` already folded in by the query path.
        pooled_cache, pending_kv, pending_tag: the three donated buffers.
        k, gate: ``[T, index_head_dim]`` from ``project_keys``.
        ape: ``[index_kpool, index_head_dim]``.
        positions: ``i32[T]``, absolute.
        page_indices: the pooled cache's flattened block table. The **same**
            array is handed to the writer and to the scorer, so they cannot
            disagree about where a pool lives.
        topk_fn: substituted by the CPU test with an exact top-k. Production
            never passes it; the static arguments come from
            :meth:`KpoolIndexerOpConfig.topk_kwargs` either way, so the
            substitution cannot skip an assertion.

    Returns:
        ``(pooled_cache, pending_kv, pending_tag, topk_indices, diagnostics)``.
    """
    num_tokens = q_bytes.shape[0]
    out_width = config.out_width

    if pooled_cache.shape[0] == 0:
        # Profiling-shape trace: no cache is bound. Unused inputs stay in the
        # compiled signature because torch_tpu jits with keep_unused=True.
        return (pooled_cache, pending_kv, pending_tag,
                jnp.full((num_tokens, out_width), -1, jnp.int32),
                jnp.zeros((KPOOL_WRITE_DIAG_SIZE, ), jnp.int32))

    pooled_cache, pending_kv, pending_tag, diagnostics = kpool_write(
        pooled_cache,
        pending_kv,
        pending_tag,
        k,
        gate,
        ape,
        positions,
        seq_lens,
        cu_q_lens,
        page_indices,
        kpool_state_indices,
        index_kpool=config.index_kpool,
    )

    q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
    pool_ids = topk_fn(
        q,
        weights,
        # The cache *after* this step's writes. See "write before score".
        pooled_cache,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        **config.topk_kwargs(),
    )

    valid_token = jnp.arange(num_tokens, dtype=jnp.int32) < cu_q_lens[-1]
    topk_indices = expand_pools_with_tail(pool_ids,
                                          positions,
                                          valid_token,
                                          index_kpool=config.index_kpool,
                                          out_width=out_width)
    assert topk_indices.shape == (num_tokens, out_width), topk_indices.shape
    return pooled_cache, pending_kv, pending_tag, topk_indices, diagnostics


# ---------------------------------------------------------------------------
# The torch-visible op
# ---------------------------------------------------------------------------


def _op_name_for(config: KpoolIndexerOpConfig) -> str:
    """The registered ``pallas::`` name, from **every** static field.

    Also the key of :data:`_op_cache`. Two configs that differ anywhere must
    get different names, or the second build re-registers a name that already
    exists; two configs that are equal must get the same name, or the cache
    never hits. Both directions are the same requirement, which is why there
    is one expression and not two.
    """
    safe_prefix = config.layer_prefix.replace(".", "_") or "layer"
    pages = "x".join(str(int(b)) for b in config.num_kv_pages_per_block)
    queries = "x".join(str(int(b)) for b in config.num_queries_per_block)
    return (f"pallas::glm5_next_kpool_indexer_{safe_prefix}"
            f"_k{config.select_k}"
            f"_c{config.index_kpool}"
            f"_t{config.index_topk}"
            f"_d{config.index_head_dim}"
            f"_p{pages}"
            f"_q{queries}")


def build_kpool_indexer_op(config: KpoolIndexerOpConfig,
                           topk_indices_buffer_width: int):
    """Build (or reuse) the ``pallas.jax_op`` for one DSA layer's indexer.

    ``donate_argnums=(2, 3, 4)`` -- the pooled cache and both ring tensors are
    in-out operands. If donation does not take, XLA copies the whole pooled
    cache per layer per step; ``pallas.set_buffer_donor_`` is currently
    commented out in ``TpuModelRunner._allocate_kv_cache`` ("TorchTPU does not
    support this right now"), so whether the jax_op aliasing path covers it is
    a **device** question this file cannot answer.

    The returned callable copies each donated output back into its input
    tensor, the idiom ``mla_attention_op.py:137-141`` uses, and returns
    ``(topk_indices, diagnostics)``.
    """
    assert_topk_buffer_width(topk_indices_buffer_width, config)

    # THE CACHE IS KEYED ON THE OP NAME, and the op name spells out every
    # static field. It used to be keyed on a 6-tuple while the name carried
    # only three of those fields, so two configs differing in, say,
    # `num_kv_pages_per_block` missed the cache and then handed
    # `pallas.jax_op` a name that was already registered.
    # `num_kv_pages_per_block` is derived from `max_model_len`, so
    # two engine constructions at different context lengths in one process --
    # a profiling pass, a parametrised test, an engine restart -- collide.
    # Deriving the key *from* the name makes a collision a cache hit by
    # construction: two configs can only share a key if they share a name, and
    # they can only share a name if every field below matches.
    op_name = _op_name_for(config)
    if op_name in _op_cache:
        return _op_cache[op_name]

    # NOT `functools.partial(kpool_indexer_stages, config=config)`.
    # `pallas.jax_op` inspects the callable's signature and refuses any
    # parameter without a type annotation; a partial that binds `config` by
    # keyword leaves BOTH `config` and `topk_fn` in the inspected signature,
    # and `topk_fn` -- the CPU test's exact-top-k substitution hook -- has no
    # annotation, so the op fails to build with
    # "Missing argument type annotation for JAX function". That failure is
    # invisible to every CPU test (they call `kpool_indexer_stages` directly)
    # and to a kernel-level device probe (same), so it first appears at
    # EngineCore startup. An explicit wrapper with exactly the fourteen
    # traced operands, and nothing else, is what the op boundary wants.
    def wrapped(
        q_bytes: jax.Array,
        weights: jax.Array,
        pooled_cache: jax.Array,
        pending_kv: jax.Array,
        pending_tag: jax.Array,
        k: jax.Array,
        gate: jax.Array,
        ape: jax.Array,
        positions: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        kpool_state_indices: jax.Array,
        distribution: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        # The RETURN annotation is as load-bearing as the argument ones:
        # `_verify_signature` also rejects "The return annotation is invalid.
        # Only jax.Arrays and POD types are supported." A bare `tuple`, or
        # none at all, does not build. Five entries, in the order
        # `kpool_indexer_stages` returns them: the three donated buffers, the
        # emitted rows, the diagnostics.
        return kpool_indexer_stages(q_bytes,
                                    weights,
                                    pooled_cache,
                                    pending_kv,
                                    pending_tag,
                                    k,
                                    gate,
                                    ape,
                                    positions,
                                    seq_lens,
                                    page_indices,
                                    cu_q_lens,
                                    kpool_state_indices,
                                    distribution,
                                    config=config)

    wrapped.__name__ = op_name.replace("::", "_")
    jax_op = pallas.jax_op(op_name, wrapped, donate_argnums=(2, 3, 4))

    def _fake(q_bytes, weights, pooled_cache, pending_kv, pending_tag, *args,
              **kwargs):
        return (
            torch.empty_like(pooled_cache),
            torch.empty_like(pending_kv),
            torch.empty_like(pending_tag),
            torch.empty((q_bytes.shape[0], config.out_width),
                        dtype=torch.int32,
                        device=q_bytes.device),
            torch.empty((KPOOL_WRITE_DIAG_SIZE, ),
                        dtype=torch.int32,
                        device=q_bytes.device),
        )

    jax_op.register_fake(_fake)

    def op(*args):
        (new_cache, new_kv, new_tag, topk_indices, diagnostics) = jax_op(*args)
        args[2].copy_(new_cache)
        args[3].copy_(new_kv)
        args[4].copy_(new_tag)
        return topk_indices, diagnostics

    _op_cache[op_name] = op
    return op


__all__ = [
    "KPOOL_ENABLE_EARLY_EXIT",
    "KPOOL_NUM_QUERIES_PER_BLOCK",
    "KpoolIndexerOpConfig",
    "assert_topk_buffer_width",
    "build_kpool_indexer_op",
    "kpool_indexer_stages",
    "kpool_request_distribution",
]
