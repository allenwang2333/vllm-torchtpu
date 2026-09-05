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
"""GLM-5.3-Flash full-attention (DSA) layers on TPU.

GLM-5.3-Flash's 11 non-KDA layers are MLA with two properties this repo's MLA
stack has never been asked for:

1. **NoPE only.** ``qk_rope_head_dim == 0`` -- there is no rotary component at
   all; position information reaches these layers through the KDA recurrence
   and the short convolutions.
2. **A kpool indexer.** ``index_kpool: 4`` pools four consecutive keys, takes
   top-k over pools, and expands back to raw token ids. See
   :class:`~vllm_torchtpu.models.vllm.glm5_next.indexer.Glm5NextKpoolIndexer`.

The RoPE half is still handled by *not* doing it, which is exact rather than
approximate.


THE TWO KNOBS, AND WHY THEY ARE TWO
-----------------------------------

There are two independent construction-time choices here:

1. **Is the kpool indexer built?** (:func:`kpool_indexer_enabled`) -- a kill
   switch, on by default. Off, the layer takes the dense kernel and
   ``model._check_context_within_dense_mla_bound`` applies. This is
   construction-time and not runtime: once the caches are the sparse
   ``(nope, rope)`` pair (``tpu_runner`` allocates from
   ``attn_module.use_sparse``), returning ``None`` from an indexer cannot fall
   back -- ``layers/vllm/attention.py:1654-1659`` asserts instead.
2. **Which MLA cache geometry?** (:func:`resolve_mla_geometry`) -- the width of
   the zero rope slab, hence ``MLAAttention.head_size = kv_lora_rank +
   qk_rope_head_dim``: 128 gives 640, 64 gives 576.

The naive rule "64 iff sparse" makes those one knob, and then the one
configuration an A/B needs -- *the 576 layout with the indexer off*, so that a
dense baseline is measured at the same KV-cache layout as the sparse run --
does not exist, which reintroduces exactly the confound the A/B exists to
remove. So the geometry is its own selector with its own env override, and
``auto`` (the default) is the only value that consults the indexer at all:

======================  ==================  ===========================
indexer                 geometry            expressed as
======================  ==================  ===========================
on                      576 (auto)          ``VLLM_TORCHTPU_GLM5_KPOOL=1``
off                     640 (auto)          ``VLLM_TORCHTPU_GLM5_KPOOL=0``
off                     **576**             ``VLLM_TORCHTPU_GLM5_KPOOL=0
                                            VLLM_TORCHTPU_GLM5_MLA_GEOMETRY=sparse576``
on                      640                 **refused**
======================  ==================  ===========================

See ``docs/models/glm5-3-flash-kpool.md`` for what the indexer computes
and why the dense substitution has a context bound.

The last row is a refusal rather than a silent downgrade because both
consumers of the 64-wide slab assert rather than adapt: the lane-bytes
assertion in ``layers/common/attention_interface.py`` requires
``rope_dim * 2 == dsa_gather.TILE_LANE_BYTES``, and ``get_sparse_kv_cache_specs``
(``layers/vllm/attention.py:1274-1283``) hard-codes ``_DS_MLA_ROPE_HEAD_DIM =
64`` and derives ``nope = head_size - 64``.
"""

import os
import re

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)

from vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op import \
    VllmTPUMultiHeadLatentAttentionWrapper
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.glm5_next.indexer import (
    Glm5NextKpoolIndexer, kpool_out_width, mtp_indexer_needs_pending_state)

logger = init_logger(__name__)

# Width of the all-zeros rotary slab handed to the MLA kernel.
#
# The dense MLA v2 kernel cannot take ``r_dim = 0``. It is parametric in
# multiples of 128, but zero is structurally impossible, not merely untested:
# ``kernels/mla/v2/kernel.py:2408`` and ``:2435`` allocate
# ``pltpu.VMEM((..., r_dim))`` scratch that Mosaic cannot size at zero lanes,
# the prologue at ``:2001`` does ``reshape((2, -1, r_dim))`` which is
# unsolvable when ``r_dim`` is 0, several KV DMAs would become zero-byte copies
# sharing a semaphore with a live one, and ``:562`` contracts a matmul over
# ``d == 0``.
#
# Feeding it a 128-wide slab of zeros instead is numerically exact:
# ``q_pe . k_pe == 0``, so the rotary contribution to the logits at
# ``kernel.py:568`` (``s = s_c + s_pe``) is identically zero, and the value
# path never reads the rotary columns. The cost is a wider KV cache
# (``512 -> 640`` in the minor dim, +25%) and ~25% more QK MXU work; the PV
# matmul that dominates decode is untouched.
#
# The alternative -- teaching the kernel ``r_dim = 0`` -- means editing at
# least five sites inside a 2600-line Pallas kernel where the failure mode is a
# DMA-semaphore hang rather than an error. That belongs in the performance
# phase, not bringup.
_ZERO_ROPE_SLAB = 128

# --- the two knobs --------------------------------------------------------

#: ``0`` builds every DSA layer without an indexer: dense layout, dense
#: kernel, and the dense-MLA context bound back in force. Construction-time --
#: see the module docstring.
KPOOL_INDEXER_ENV = "VLLM_TORCHTPU_GLM5_KPOOL"

#: ``auto`` | ``dense640`` | ``sparse576``. **Not** derived from the indexer
#: switch; see the module docstring's table.
MLA_GEOMETRY_ENV = "VLLM_TORCHTPU_GLM5_MLA_GEOMETRY"

MLA_GEOMETRY_AUTO = "auto"
MLA_GEOMETRY_DENSE_640 = "dense640"
MLA_GEOMETRY_SPARSE_576 = "sparse576"

#: Rope slab width per geometry. The names are the resulting
#: ``MLAAttention.head_size`` at GLM's ``kv_lora_rank = 512``, because that is
#: the number that appears in a KV-cache spec and in an OOM message -- the
#: slab width itself appears nowhere a reader would look.
_ROPE_SLAB_BY_GEOMETRY = {
    MLA_GEOMETRY_DENSE_640: _ZERO_ROPE_SLAB,
    MLA_GEOMETRY_SPARSE_576: 64,
}

#: Built unless something turns it off. The indexer is the supported
#: configuration -- it is what makes ``max_model_len > index_topk`` sound --
#: so the kill switch is opt-out, not opt-in.
KPOOL_INDEXER_DEFAULT = True


def kpool_indexer_enabled(requested: bool | None = None) -> bool:
    """Whether to build the kpool indexer.

    Precedence: explicit argument, then :data:`KPOOL_INDEXER_ENV`, then
    :data:`KPOOL_INDEXER_DEFAULT`.
    """
    if requested is not None:
        return bool(requested)
    raw = os.environ.get(KPOOL_INDEXER_ENV)
    if raw is None or raw.strip() == "":
        return KPOOL_INDEXER_DEFAULT
    normalized = raw.strip().lower()
    if normalized in ("0", "false", "off", "no"):
        return False
    if normalized in ("1", "true", "on", "yes"):
        return True
    raise ValueError(
        f"{KPOOL_INDEXER_ENV}={raw!r} is neither on nor off. Use 0 or 1; a "
        "typo must not silently leave the indexer in whichever state the "
        "default happens to be.")


def resolve_mla_geometry(*,
                         indexer_enabled: bool,
                         requested: str | None = None) -> str:
    """Pick the MLA cache geometry, independently of the indexer.

    Args:
        indexer_enabled: consulted **only** by ``auto``.
        requested: an explicit geometry, or ``None`` to read
            :data:`MLA_GEOMETRY_ENV`, or ``None`` and unset for ``auto``.

    Raises:
        ValueError: for an unknown name, or for the one incoherent pair
            (indexer on at the 640 layout), which the sparse kernel's two
            asserts would otherwise reject deep inside a traced kernel.
    """
    if requested is None:
        requested = os.environ.get(MLA_GEOMETRY_ENV) or MLA_GEOMETRY_AUTO
    geometry = str(requested).strip().lower()
    if geometry == MLA_GEOMETRY_AUTO:
        return (MLA_GEOMETRY_SPARSE_576
                if indexer_enabled else MLA_GEOMETRY_DENSE_640)
    if geometry not in _ROPE_SLAB_BY_GEOMETRY:
        raise ValueError(
            f"glm5_next: unknown MLA geometry {requested!r}. Expected one of "
            f"{[MLA_GEOMETRY_AUTO, *sorted(_ROPE_SLAB_BY_GEOMETRY)]} "
            f"(env {MLA_GEOMETRY_ENV}).")
    if indexer_enabled and geometry == MLA_GEOMETRY_DENSE_640:
        raise ValueError(
            "glm5_next: the kpool indexer is enabled at the "
            f"{MLA_GEOMETRY_DENSE_640} geometry, which cannot work. The "
            "sparse MLA path needs a 64-wide rope slab in two independent "
            "places: attention_interface.py asserts "
            "`rope_dim * 2 == dsa_gather.TILE_LANE_BYTES`, and "
            "get_sparse_kv_cache_specs hard-codes _DS_MLA_ROPE_HEAD_DIM = 64 "
            "and derives nope = head_size - 64 "
            "(layers/vllm/attention.py:1189, :1274-1283). A 128-wide slab "
            f"makes head_size 640 and both fail. Set {MLA_GEOMETRY_ENV}="
            f"{MLA_GEOMETRY_SPARSE_576} (or leave it at {MLA_GEOMETRY_AUTO}), "
            f"or turn the indexer off with {KPOOL_INDEXER_ENV}=0. Note "
            f"{KPOOL_INDEXER_ENV}=0 with {MLA_GEOMETRY_ENV}="
            f"{MLA_GEOMETRY_SPARSE_576} is the *supported* combination for "
            "a dense baseline at the sparse cache layout.")
    return geometry


def rope_slab_width(geometry: str) -> int:
    """Width of the zero rope slab for a resolved geometry."""
    try:
        return _ROPE_SLAB_BY_GEOMETRY[geometry]
    except KeyError:
        raise ValueError(
            f"glm5_next: {geometry!r} is not a resolved geometry; call "
            "resolve_mla_geometry first (it is the only thing that maps "
            f"{MLA_GEOMETRY_AUTO!r}).") from None


def mla_head_size(kv_lora_rank: int, geometry: str) -> int:
    """``MLAAttention.head_size`` at this geometry.

    Spelled out because it is the number every KV-cache spec, page-size
    calculation and OOM message carries, and it is *not* the same quantity as
    the rope slab width that produces it.
    """
    return int(kv_lora_rank) + rope_slab_width(geometry)


def layer_index_from_prefix(prefix: str) -> int:
    """``model.layers.<i>.self_attn`` -> ``i``, or ``-1``.

    Mirrors ``Glm5NextDecoderLayer.__init__``'s regex rather than taking the
    index as an argument, so a layer cannot be told it is one index while its
    weights load from another.
    """
    match = re.search(r"layers\.(\d+)", prefix or "")
    return int(match.group(1)) if match else -1


class Glm5NextMLAAttention(VllmTPUMultiHeadLatentAttentionWrapper):
    """MLA for GLM-5.3-Flash's full-attention layers.

    Built with ``qk_rope_head_dim`` set to the geometry's zero-slab width so
    the KV cache spec and the Pallas kernel see ordinary DeepSeek geometry,
    while ``forward`` is overridden so nothing tries to *split* a rotary
    component out of projections that do not produce one.

    Both knobs are explicit constructor arguments with env-backed defaults;
    see the module docstring for the table of legal combinations.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        *,
        enable_indexer: bool | None = None,
        mla_geometry: str | None = None,
    ) -> None:
        config = vllm_config.model_config.hf_config
        text_config = getattr(config, "text_config", config)
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        # The two knobs, resolved before anything is built, so an incoherent
        # pair raises here rather than inside a traced kernel.
        self.indexer_enabled = kpool_indexer_enabled(enable_indexer)
        self.mla_geometry = resolve_mla_geometry(
            indexer_enabled=self.indexer_enabled, requested=mla_geometry)
        # Named `zero_rope_slab`, not `rope_slab_width`, so it cannot be
        # mistaken for the module-level function of that name.
        self.zero_rope_slab = rope_slab_width(self.mla_geometry)

        from vllm.distributed import get_tensor_model_parallel_world_size

        hidden_size = text_config.hidden_size
        num_heads = text_config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size:
            raise ValueError(
                f"num_attention_heads ({num_heads}) must divide the tensor "
                f"parallel size ({tp_size})")
        # The projections below are built with the *global* head count --
        # Column/RowParallelLinear shard them internally -- but the wrapper and
        # everything downstream of it work per rank, so it gets the local
        # count. vLLM's own MLA callers do the same
        # (`deepseek_v2.py:1206` passes `num_local_heads`). Getting this wrong
        # is invisible at TP=1 and a shape error above it.
        num_local_heads = num_heads // tp_size
        self.actual_qk_rope_head_dim = text_config.qk_rope_head_dim
        if self.actual_qk_rope_head_dim != 0:
            raise NotImplementedError(
                "GLM-5.3-Flash is expected to be NoPE-only "
                f"(qk_rope_head_dim=0); got {self.actual_qk_rope_head_dim}. "
                "A real rotary component needs the stock MLA path, not this "
                "zero-slab one.")

        qk_nope_head_dim = text_config.qk_nope_head_dim
        v_head_dim = text_config.v_head_dim
        q_lora_rank = text_config.q_lora_rank
        kv_lora_rank = text_config.kv_lora_rank
        eps = text_config.rms_norm_eps

        # Scale over the real (NoPE-only) query width. The zero slab must not
        # enter the softmax scale -- it carries no information.
        scale = qk_nope_head_dim**-0.5

        # One projection for q_a and kv_a, as DeepSeek-V4 does; note the kv
        # half is `kv_lora_rank` alone, with no rope tail.
        fused_qkv_a_proj = MergedColumnParallelLinear(
            hidden_size,
            [q_lora_rank, kv_lora_rank],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
            disable_tp=True,
        )
        q_a_layernorm = RMSNorm(q_lora_rank, eps=eps)
        q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * qk_nope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        kv_a_layernorm = RMSNorm(kv_lora_rank, eps=eps)
        kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        from vllm.model_executor.layers.mla import MLAModules

        # --- the indexer, or deliberately not ----------------------------
        #
        # WITHOUT it, this layer is dense causal MLA, which is *exact* for
        # `max_model_len <= index_topk` (2048): a query at position p has
        # floor((p+1)/kpool) candidate pools against a budget of
        # index_topk//kpool == 512, so top-k selects every valid pool,
        # expansion covers [0, 4*floor((p+1)/4) - 1], and the always-selected
        # tail supplies the remaining (p+1) % 4. The union is exactly [0, p].
        # Past that bound it is wrong, which is what
        # `model._check_context_within_dense_mla_bound` refuses.
        #
        # That is the kill switch, and it is CONSTRUCTION-TIME, not runtime:
        # `tpu_runner` allocates the split (nope, rope) sparse caches from
        # `attn_module.use_sparse`, and once the caches are a pair the dense
        # branch at `layers/vllm/attention.py:1677` takes a single tensor, so
        # returning `None` from an indexer at runtime cannot fall back -- the
        # assert at `:1656-1659` fires instead.
        indexer = None
        topk_indices_buffer = None
        if self.indexer_enabled:
            layer_idx = layer_index_from_prefix(prefix)
            num_text_layers = int(
                getattr(text_config, "num_hidden_layers", 0) or 0)
            # Layer 45 is the MTP draft head: it sits past the target stack
            # and carries its own complete seven-tensor indexer block. It
            # needs its own pending ring unless the draft head reuses the
            # target stack's selection, in which case its indexer never runs
            # at all -- see `mtp_indexer_needs_pending_state`.
            is_mtp_layer = (num_text_layers > 0
                            and layer_idx >= num_text_layers)
            indexer = Glm5NextKpoolIndexer(
                text_config,
                quant_config,
                prefix=f"{prefix}.indexer",
                layer_idx=layer_idx,
                vllm_config=vllm_config,
                allocate_pending_state=(
                    mtp_indexer_needs_pending_state(text_config)
                    if is_mtp_layer else True),
            )
            # The same tensor for every layer, and its width is the indexer's
            # `topk_tokens` -- NOT `config.index_topk`, which is short by
            # `index_kpool - 1` and would clip the always-selected tail. The
            # The MLA wrapper asserts the agreement too, but it reads the
            # same attribute rather than re-deriving the width, so this
            # checks it against the derivation.
            topk_indices_buffer = indexer.topk_indices_buffer
            expected_width = kpool_out_width(int(text_config.index_topk),
                                             int(text_config.index_kpool))
            if topk_indices_buffer.shape[-1] != expected_width:
                raise ValueError(
                    f"glm5_next kpool: {prefix} got a topk_indices_buffer "
                    f"{topk_indices_buffer.shape[-1]} columns wide, expected "
                    f"{expected_width} = derive_out_width(index_topk="
                    f"{text_config.index_topk}, index_kpool="
                    f"{text_config.index_kpool}). The columns past "
                    "index_topk are the always-selected tail -- the "
                    "(p + 1) % index_kpool most recent tokens -- so a short "
                    "buffer drops them, which is invisible at "
                    "L % index_kpool == 0.")
            # Every derived quantity, once per layer: the score-buffer and
            # block-utilisation numbers should come out of a run rather than
            # out of a spreadsheet.
            logger.info("%s", indexer.describe())

        mla_modules = MLAModules(
            fused_qkv_a_proj=fused_qkv_a_proj,
            kv_a_proj_with_mqa=None,
            q_a_layernorm=q_a_layernorm,
            q_b_proj=q_b_proj,
            q_proj=None,
            kv_a_layernorm=kv_a_layernorm,
            kv_b_proj=kv_b_proj,
            rotary_emb=None,
            o_proj=o_proj,
            indexer=indexer,
            # NoPE: the reference indexer takes no `position_embeddings` at
            # all (`modeling_glm5_next.py:774-779`).
            indexer_rotary_emb=None,
            is_sparse=self.indexer_enabled,
            topk_indices_buffer=topk_indices_buffer,
        )

        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_local_heads,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=self.zero_rope_slab,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

        # THE REBIND REGRESSION. `VllmTPUMultiHeadLatentAttentionWrapper`
        # retypes an upstream `Indexer` in place via `VllmTPUIndexer.rebind`,
        # which is a `__class__` reassignment and therefore silent on any
        # object. If that guard is ever loosened, this indexer becomes the
        # DeepSeek-V2 lightning indexer -- different forward, different cache
        # class, mutated `k_cache.head_dim` -- and the only symptom is bad
        # retrieval. Checked here as well as in a unit test because this is
        # the site that would actually be running.
        if indexer is not None and self.indexer is not indexer:
            raise ValueError(
                f"glm5_next kpool: the MLA wrapper replaced {prefix}'s "
                "indexer object. Only `VllmTPUIndexer.rebind` does that, and "
                "it must not run on a Glm5NextKpoolIndexer.")
        if indexer is not None and type(indexer) is not Glm5NextKpoolIndexer:
            raise ValueError(
                f"glm5_next kpool: {prefix}'s indexer was retyped to "
                f"{type(indexer).__name__} during wrapper construction. "
                "`VllmTPUIndexer.rebind` reassigns `__class__`, retypes "
                "`k_cache.__class__` and mutates `k_cache.head_dim`; run on "
                "the kpool indexer it silently produces a DeepSeek-V2 "
                "lightning indexer over a ratio-4 pooled cache. Restore the "
                "isinstance guard at mla_attention_op.py.")

        # `use_sparse` is what `tpu_runner` reads to decide between one MLA
        # cache and the split (nope, rope) pair, so a disagreement between it
        # and the geometry is a cache allocated at the wrong layout.
        if bool(getattr(self.mla_attn, "use_sparse",
                        False)) != bool(self.indexer_enabled):
            raise ValueError(f"glm5_next: {prefix} built with indexer_enabled="
                             f"{self.indexer_enabled} but mla_attn.use_sparse="
                             f"{getattr(self.mla_attn, 'use_sparse', None)}.")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Like the base forward, but the rotary halves are synthesised zeros.

        The base implementation splits ``q`` and ``kv_lora`` into NoPE and RoPE
        parts. GLM's projections have no RoPE part to split off, so the split
        widths and the synthesised slab are kept separate here.

        ``positions`` is NOT unused any more. The attention *math* is still
        NoPE -- nothing here rotates anything -- but the kpool indexer is
        addressed by absolute token position: the pending ring by
        ``pos % pending_depth`` and every pooled record by
        ``pos // index_kpool``. It is passed straight through, unmodified.
        """
        del llama_4_scaling  # Not a GLM path.

        num_tokens = hidden_states.shape[0]

        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c = qkv_lora.split([self.q_lora_rank, self.kv_lora_rank],
                                   dim=-1)

        q_c = self.q_a_layernorm(q_c)
        q_nope = self.q_b_proj(q_c)[0].view(-1, self.num_heads,
                                            self.qk_nope_head_dim)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        # The zero slab. `q_pe @ k_pe^T` is exactly 0, so it contributes
        # nothing to the attention logits; it exists only to give the kernel a
        # non-degenerate rotary dimension. Its width is the geometry's, read
        # off the wrapper rather than off `_ZERO_ROPE_SLAB`, so the tensors
        # and the KV-cache spec cannot disagree.
        slab = self.qk_rope_head_dim
        zeros = hidden_states.new_zeros(())
        q_pe = zeros.expand(num_tokens, self.num_heads, slab).contiguous()
        k_pe = zeros.expand(num_tokens, 1, slab).contiguous()

        # The wrapper's own sparse branch never runs -- this override replaces
        # it wholesale -- so the indexer is called here, and its result is
        # what reaches the kernel instead of the hard-coded `None` this method
        # used to pass.
        topk_indices = None
        if self.indexer is not None:
            topk_index = self.indexer(hidden_states, q_c, positions, None)
            topk_indices = topk_index[:num_tokens]

        attn_out = self.mla_attn(
            (q_nope, q_pe),
            kv_c_normed,
            k_pe,
            output_shape=(num_tokens, self.num_heads * self.v_head_dim),
            topk_indices=topk_indices,
        )
        return self.o_proj(attn_out)[0]
