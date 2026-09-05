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
"""GLM-5.3-Flash's kpool lightning indexer: weights and the query path.

Design notes: ``docs/models/glm5-3-flash-kpool.md``.

Everything one DSA layer's sparse indexer needs: the seven checkpoint
tensors and the config contract they only make sense under, the query
projection whose scale folding is invisible in every small test, the pooled K
cache and its geometry, the pending-pool state tensors, and the call into the
fused ``kpool_write -> streamindex_topk -> expand`` op.

**The seven tensors**, read out of ``zai-org/GLM-5.3-Flash``'s
``model.safetensors.index.json`` and the safetensors headers of shards 2 and 5
(not derived from the config) -- 84 = 7 x 12 layers, the 11 DSA layers
``3, 7, ..., 43`` plus the MTP layer 45, every one BF16 with no
``weight_scale_inv`` sibling:

===================================  ======  =============
tensor                               dtype   shape
===================================  ======  =============
``indexer.wq_b.weight``              BF16    ``[4096, 1536]``
``indexer.wk.weight``                BF16    ``[128, 4096]``
``indexer.k_norm.weight``            BF16    ``[128]``
``indexer.k_norm.bias``              BF16    ``[128]``
``indexer.weights_proj.weight``      BF16    ``[32, 4096]``
``indexer.index_kpool_compress_ape``  BF16   ``[4, 128]``
``indexer.index_kpool_compress_gate`` BF16   ``[128, 4096]``
===================================  ======  =============

All 84 are named in ``quantization_config.modules_to_not_convert`` -- as
``...indexer.{wq_b, wk, weights_proj, k_norm, k_norm.bias,
index_kpool_compress_ape, index_kpool_compress_gate}``. Note what that list
does *not* contain: any fused spelling. ``wk`` and ``weights_proj`` are named
separately, so fusing them into one ``wk_weights_proj`` GEMM the way
``VllmTPUIndexer`` does for DeepSeek (``mla_attention_op.py:246-248``) makes
``is_layer_skipped`` miss and hands 11 bf16 layers to ``Fp8LinearMethod`` with
no scales to load. That is not hypothetical: it is exactly the shipped
vision-tower bug (``multimodal.py:_tower_quant_config``), where 24 blocks
computed on unscaled FP8 weights while every loader check reported 298/298
tensors filled. So the three projections stay separate, and -- because a name
match is not proof that the skip took effect -- :class:`Glm5NextKpoolIndexer`
asserts the resulting ``quant_method`` really is unquantized rather than
trusting the spelling.

**Replicated, not head-sharded.** ``streamindex_topk`` sums the per-head ReLU'd
scores *inside* the Pallas body
(``kernels/deepseek_v4/streamindex_topk.py:335``)
and then takes the top-k of that sum, so a rank holding 4 of 32 heads would
select a different key set than its peers -- over a pooled cache they share.
Every projection here is ``ReplicatedLinear(disable_tp=True)``.

**No RoPE, at all.** GLM-5.3 is NoPE (``qk_rope_head_dim == 0``) and
``Glm5NextTextIndexer.forward`` does not even accept ``position_embeddings``
(``modeling_glm5_next.py:774-779``), unlike its base class. This indexer takes
no ``rotary_emb`` and its op will take no ``cos_sin_cache``. ``config``'s
``indexer_rope_interleave`` and ``index_kpool_compress`` are read by no
``glm5_next`` code in transformers 5.16.1 -- inert; implement nothing from
them.

**Which vLLM the pooled cache spec is written against.** The TPU pod
imports vLLM **v0.27.0** -- the ``pyproject.toml:29`` pin, out of the image's
``site-packages``, not the developer checkout on ``PYTHONPATH``. The full
evidence is in ``docs/vllm-kv-cache-api-pin.md`` and the machine check is
``tests/models/vllm/test_vllm_kv_cache_api_pin.py``. What it means here:

* ``MLAAttentionSpec.compress_ratio`` and ``.storage_block_size`` **exist** at
  v0.27.0, so :class:`Glm5NextIndexerKCache` uses them, exactly as
  ``deepseek_v4_indexer.py:113`` and ``TpuModelRunner``'s
  ``_initialize_ds_v4_kv_cache`` already do. Never ``tokens_per_state``,
  which is vLLM ``main``'s replacement and does not exist in the pod.
* The drift points the *other* way from the obvious guess: it is the CPU
  rung that cannot construct these specs, because a recent vLLM dropped
  ``compress_ratio``. :func:`pooled_indexer_kv_cache_spec` therefore raises a
  message naming that document rather than a bare ``TypeError`` when the
  imported vLLM is the wrong one, and every byte of geometry that does *not*
  need the spec class lives in :func:`derive_pooled_indexer_geometry`, which
  is pure Python and fully testable on CPU.
* ``CircularBufferSpec`` and ``SlidingWindowSpec.extra_retained_tokens`` do
  **not** exist at v0.27.0, so the pending-pool state can use neither.

**Where the pending-pool state lives, and what accounts for it.** It is
allocated here; the other two pieces are elsewhere on purpose:

* the **decision** -- why a homemade ring rather than ``CircularBufferSpec``,
  ``SlidingWindowMLASpec`` or a ``compress_ratio=1`` MLA spec, and the seven
  rejected options -- is ``docs/glm5-3-flash-kpool-pending-state.md``;
* the **ring itself** -- ``pending_kv f32[num_slots, depth, 2,
  index_head_dim]`` and ``pending_tag i32[num_slots, depth]``, their geometry
  (``derive_pending_state_geometry``), the depth floor, and the read/write
  split -- is ``kernels/glm5_kpool/pool_write.py``;
* what belongs *here* is **allocating** those two tensors --
  :class:`Glm5NextKpoolIndexer` does, in its constructor, whenever it is
  given a ``vllm_config``.

The ``req_id -> slot`` pool that fills ``AttentionMetadata.kpool_state_indices``
now exists: :class:`vllm_torchtpu.runner.kpool_slots.KpoolSlotPool`, a stride-1
mirror of ``TpuModelRunner._init_mamba_slot_pool`` /
``._build_mamba_state_indices``, driven from ``load_model`` (which sets
``_has_kpool_state`` by walking the module tree for the
``kpool_needs_state_indices`` marker this class sets) and from
``_prepare_inputs``. It is a separate module from ``tpu_runner.py`` on purpose
-- that file is not importable on the CPU rung, so logic left inside it is
logic no test can reach.

:meth:`Glm5NextKpoolIndexer.forward` still raises if the field is absent,
rather than falling back to ``mamba_state_indices``, which under unified-align
mode is an attention-pool block id and would index this module's
``[max_num_reqs + 1, ...]`` ring out of bounds. The refusal is now a
regression guard rather than a handoff note.

The accounting consequence is the part worth stating out loud: **the two ring
tensors are runner-lifetime allocations that no ``KVCacheSpec`` accounts for.**
They are deliberately outside the KV-cache allocator -- their size is flat in
``max_model_len`` (identical at 2048, 128 K and 1 M), which is the whole point
-- so nothing in ``kv_cache_allocation_report`` or
``assert_kv_cache_allocation_within_plan`` below can see them, and they must
exist *before* ``determine_available_memory()`` profiles or the HBM they take
is HBM the profile hands to the block pool twice.
"""

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (ReplicatedLinear,
                                               UnquantizedLinearMethod)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_torchtpu.kernels.glm5_kpool import (PENDING_TAG_EMPTY,
                                              PendingStateGeometry,
                                              derive_out_width,
                                              derive_pending_state_geometry)
from vllm_torchtpu.layers.common.quantization import quantize_tensor
from vllm_torchtpu.utils import align_to

#: Everything in this module is a transcription of
#: ``transformers==5.16.1``'s ``models/glm5_next/modeling_glm5_next.py``.
#: Line numbers in comments refer to that file.
_REFERENCE = "transformers 5.16.1 modeling_glm5_next.py"

#: ``nn.LayerNorm(self.head_dim, eps=1e-6)`` at ``modeling_glm5_next.py:763``.
#: Hard-coded there -- it is *not* ``rms_norm_eps`` (1e-5), and it is
#: LayerNorm-with-bias, not RMSNorm. Both mistakes shift every indexer key by a
#: fraction of a percent, which reads as "the port is slightly off".
K_NORM_EPS = 1e-6

#: The checkpoint values, recorded so a test can pin them and a message can
#: quote them. Read from
#: ``zai-org/GLM-5.3-Flash`` snapshot ``03eb5366`` ``config.json``.
REAL_CHECKPOINT_CONTRACT = {
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_kpool": 4,
    "index_kpool_always_select_tail": True,
    "qk_rope_head_dim": 0,
    "num_hidden_layers": 45,
    # ``indexer_types`` is ``["full"] * 45`` -- indexed by *layer* over all 45
    # layers, not over the 11 DSA layers.
    "indexer_types": "full x 45",
    # True, and inert in the HF reference (layer 45 is dropped by regex at
    # ``modeling_glm5_next.py:1359``). vLLM implements the same-named concept
    # as index reuse across draft iterations, which is not implemented
    # here; see ``KPOOL_MTP_INDEX_REUSE``.
    "index_share_for_mtp_iteration": True,
}


def kpool_out_width(index_topk: int, index_kpool: int) -> int:
    """Width of the emitted ``topk_indices`` row, derived rather than chosen.

    The reference emits ``index_topk + index_kpool - 1`` = 2051 ids: 512 pools
    x 4 raw tokens, plus the 0..3 tokens of the current incomplete pool
    (``modeling_glm5_next.py:866-877``). We round that up to a multiple of 128
    because the width becomes the minor axis of the score tile in the sparse
    MLA kernel's einsum (``kernels/mla/sparse/kernel.py:168-200``), and a
    non-multiple makes Mosaic pad it silently.

    The two asserts are the load-bearing half. Trimming to ``index_topk``
    exactly would drop the tail columns -- which are the ``(p+1) % 4`` *most
    recent* tokens, invisible at ``L % 4 == 0`` and wrong at the other three
    residues.

    **Delegates rather than re-deriving.** The producer's row width and this
    consumer's buffer width must be the same number. Two
    independent expressions that happen to agree today are exactly how the
    buffer ends up short by a few columns later, clipping the tail -- silent at
    ``L % index_kpool == 0`` and wrong at the other three residues. So the
    single home is ``kernels/glm5_kpool/expand.py``, the module that also
    statically asserts the width where the expansion is written.
    """
    return derive_out_width(index_topk, index_kpool)


def dsa_layer_indices(text_config) -> list[int]:
    """Indices of the full-attention (DSA) layers, from ``layer_types``.

    Deliberately *not* ``range(num_hidden_layers)`` filtered by anything else:
    ``layer_types`` is length ``num_hidden_layers`` and is the same list
    ``model.py:is_linear_attention_layer`` consults, so the two cannot drift.
    """
    layer_types = list(getattr(text_config, "layer_types", None) or [])
    if not layer_types:
        raise ValueError(
            "glm5_next: the kpool indexer needs `layer_types` to know which "
            "layers are DSA; the config declares none.")
    return [i for i, t in enumerate(layer_types) if t != "linear_attention"]


def assert_kpool_config_contract(text_config) -> None:
    """Refuse a config whose selection would differ from the reference's.

    Every item here silently changes *which keys the model attends to* while
    leaving every shape valid and every output fluent, which is why they are
    assertions at construction rather than comments.

    Note the two kinds. The structural items must hold for this implementation
    to mean anything. ``index_n_heads`` is **not** among them: it is 32 in the
    real checkpoint but 4 in ``scripts/glm5_next/build_tiny_fixture.py``, so
    pinning it here would make the tiny fixture unbuildable. It is pinned as a
    recorded checkpoint constant in the tests instead.
    """

    def _get(name):
        value = getattr(text_config, name, None)
        if value is None:
            raise ValueError(
                f"glm5_next kpool: config declares no `{name}`. The indexer "
                f"cannot be built without it (see {_REFERENCE}).")
        return value

    index_topk = _get("index_topk")
    index_kpool = _get("index_kpool")
    index_head_dim = _get("index_head_dim")

    if index_topk % index_kpool != 0:
        raise ValueError(
            f"glm5_next kpool: index_topk ({index_topk}) must be a multiple "
            f"of index_kpool ({index_kpool}); select_k = index_topk // "
            "index_kpool and the emitted row is index_topk + index_kpool - 1 "
            "wide. transformers rejects this too "
            "(configuration_glm5_next.py:219).")

    # `index_kpool_always_select_tail` is True in the real checkpoint. False
    # would mean the reference does *not* append the current incomplete pool,
    # so the identity regime shrinks and `out_width` is `index_topk`. Nothing
    # downstream reads the flag today, so a False config would silently get
    # the True behaviour: attending to up to 3 tokens the reference does not.
    always_tail = getattr(text_config, "index_kpool_always_select_tail", None)
    if always_tail is not True:
        raise ValueError(
            "glm5_next kpool: index_kpool_always_select_tail is "
            f"{always_tail!r}, expected True. The tail append at "
            "modeling_glm5_next.py:866-877 is unconditional in this port; a "
            "False config would attend to up to index_kpool-1 tokens the "
            "reference does not, with no shape error to show for it.")

    # `indexer_types[layer] == "shared"` makes the reference skip building an
    # indexer for that layer and reuse an earlier layer's selection
    # (modeling_glm5_next.py:1130-1133). We build one everywhere, so a
    # "shared" entry gives that layer a *legal but different* key set. The
    # field is indexed by layer over all `num_hidden_layers` layers -- NOT
    # over the DSA layers only, which is why the length is checked first.
    indexer_types = getattr(text_config, "indexer_types", None)
    dsa_indices = dsa_layer_indices(text_config)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if indexer_types is None:
        raise ValueError(
            "glm5_next kpool: config declares no `indexer_types`. "
            "transformers synthesises it from index_topk_pattern / "
            "index_topk_freq (configuration_glm5_next.py:177-186); the real "
            "checkpoint ships it explicitly as ['full'] * 45.")
    if len(indexer_types) != num_layers:
        raise ValueError(
            f"glm5_next kpool: indexer_types has length {len(indexer_types)} "
            f"but there are {num_layers} layers. It is indexed by *layer*, so "
            "a per-DSA-layer list would silently read the wrong entries.")
    shared = [i for i in dsa_indices if indexer_types[i] != "full"]
    if shared:
        raise ValueError(
            f"glm5_next kpool: indexer_types is not 'full' on DSA layers "
            f"{shared} (got {[indexer_types[i] for i in shared]}). The "
            "'shared' branch reuses an earlier layer's topk buffer "
            "(the `skip_topk` branch of `mla_attention_op.py`) and is not "
            "wired up here; running "
            "a full indexer there produces a legal but wrong key set on "
            "those layers only.")

    # Values that are load-bearing for the kernel geometry and that hold in
    # both the real checkpoint and the tiny fixture.
    if index_head_dim != 128:
        raise ValueError(
            f"glm5_next kpool: index_head_dim is {index_head_dim}, expected "
            "128. The packed indexer record is `head_dim` fp8 bytes plus one "
            "e8m0 scale byte (streamindex_topk.py:264-271) and the pooled "
            "cache spec is align_to(128 + 1, 128) = 256; neither has been "
            "derived at any other width.")
    if index_kpool != 4:
        raise ValueError(
            f"glm5_next kpool: index_kpool is {index_kpool}, expected 4. The "
            "pooled path runs streamindex_topk at compression_ratio = "
            "index_kpool with causal_row_offset = index_kpool - 1, and only "
            "ratio 4 has a derived num_kv_pages_per_block and a measured "
            "block geometry.")

    # NoPE. `Glm5NextTextIndexer.forward` takes no position_embeddings
    # (modeling_glm5_next.py:774-779) and this port passes no cos_sin_cache.
    rope_dim = getattr(text_config, "qk_rope_head_dim", 0)
    if rope_dim != 0:
        raise ValueError(
            f"glm5_next kpool: qk_rope_head_dim is {rope_dim}, expected 0. "
            "The indexer applies no rotary embedding at all; a checkpoint "
            "with a real rotary component needs the DeepSeek indexer path, "
            "not this one.")


# --- construction-time guards: score buffer, KV dtype, prefix caching -----
#
# Both are construction-time refusals rather than comments, and both live here
# rather than in `attention.py` so they are importable and testable without
# `torch_tpu`. The attention layer calls them; the tests call them directly.

#: Override for :data:`DEFAULT_SCORE_BUFFER_BUDGET_MIB`, in MiB. ``0`` (or a
#: negative value) disables the check entirely -- which is a legitimate thing
#: to want when *measuring* the buffer on device.
SCORE_BUFFER_BUDGET_ENV = "VLLM_TORCHTPU_GLM5_KPOOL_SCORE_BUDGET_MIB"

#: **Calibrated** against device measurements on one tpu7x chip. It is
#: expressed in
#: *predicted* MiB -- the single-buffer model :func:`score_buffer_bytes`
#: computes -- because that is the quantity the rest of this file and its
#: tests speak in; multiply by :data:`SCORE_BUFFER_MEASURED_MULTIPLIER` for
#: the HBM the compiler actually asks for.
#:
#: 4096 is a result rather than a guess: it is the bound that keeps the two
#: interesting configurations on the right sides -- 128 K at
#: ``max_num_batched_tokens`` 8192 predicts
#: 1 GiB / measures 2.98 GiB and must pass; 1 M at 8192 predicts 8 GiB /
#: measures 32.06 GiB and must not -- while leaving 1 M reachable at a
#: 1024-token chunk. Treat a refusal as "this needs a smaller chunk", not as
#: "this cannot run".
DEFAULT_SCORE_BUFFER_BUDGET_MIB = 4096

#: What the compiler really allocates, as a multiple of
#: :func:`score_buffer_bytes`, from ``XlaExecutable.memory_analysis()``'s
#: ``temp_size_in_bytes`` for the ``streamindex_topk`` program alone.
#:
#: **The single-buffer model under-predicts by about 3x, and by 4x past
#: 262144 score columns.** ``scores_init`` is aliased to the kernel's output
#: (``streamindex_topk.py:903-907``), which reads as one buffer; it is not,
#: on device. Measured at ``max_num_batched_tokens = 8192`` except
#: where the context is shorter than that, in which case the chunk is the
#: context:
#:
#: ===============  ==========  ==========  ============  =====
#: ``max_model_len``  columns   predicted   measured temp  ratio
#: ===============  ==========  ==========  ============  =====
#: 2048 (T=2048)          512      4.0 MiB       8.4 MiB   2.09
#: 4096 (T=4096)         1024     16.0 MiB      40.5 MiB   2.53
#: 8192 (T=8192)         2048     64.0 MiB     176.5 MiB   2.76
#: 32768                 8192    256.0 MiB     752.5 MiB   2.94
#: 131072               32768   1024.0 MiB    3056.5 MiB   2.98
#: 262144               65536   2048.0 MiB    6128.5 MiB   2.99
#: 524288              131072   4096.0 MiB   16417.0 MiB   4.01
#: 1048576             262144   8192.0 MiB   32833.0 MiB   4.01
#: ===============  ==========  ==========  ============  =====
#:
#: The step at 524288 is a cliff, not noise: the ratio is flat at ~3.0 for
#: three consecutive doublings below it and flat at 4.01 for two above it.
#: Reported rather than modelled -- the cause is inside XLA's buffer
#: assignment, and a wrong causal story here would be worse than none.
#:
#: Also measured, and worth knowing before tuning ``num_kv_pages_per_block``:
#: at ``bkv_p = 32`` the kernel asks for **104857600 B of scoped VMEM** and
#: XLA lowers it to the 67043328 B maximum ("We are lowering the scoped Vmem
#: for StreamIdxTC-...-bkvp_32-off_3 to 67043328 bytes"), 66 times in one
#: probe run. It still produces exact results, so this is a performance note,
#: not a correctness one.
SCORE_BUFFER_MEASURED_MULTIPLIER = {
    # (upper bound on score_columns, inclusive): multiplier
    262144: 3.0,
}
SCORE_BUFFER_MEASURED_MULTIPLIER_ABOVE = 4.02


def score_buffer_measured_multiplier(score_columns: int) -> float:
    """The measured temp/predicted ratio at this width. See the table above."""
    for limit in sorted(SCORE_BUFFER_MEASURED_MULTIPLIER):
        if score_columns <= limit:
            return SCORE_BUFFER_MEASURED_MULTIPLIER[limit]
    return SCORE_BUFFER_MEASURED_MULTIPLIER_ABOVE


#: ``sparsecore_topk``'s documented envelope (`sparsecore_topk.py:17-19`):
#: exact top-k is claimed for ``N in 2048..256K`` columns per row. At
#: ``index_kpool = 4`` a 1 M context is ``1048576 // 4 == 262144`` pool
#: candidates -- **exactly** 256 Ki, i.e. sitting on the boundary rather than
#: inside it. Quoted in the guard's message because a reader who lowers
#: ``max_num_batched_tokens`` to fit the budget has not thereby made 1 M
#: context safe: the column count does not move with the chunk size.
SPARSECORE_TOPK_MAX_COLUMNS = 256 * 1024

#: **Measured**, and it is sixteen times the documented number. A device
#: probe swept ``sparsecore_topk`` at ``k = 512``
#: over ``N in {2048, 16384, 65536, 131072, 262144, 524288, 1048576, 2097152,
#: 4194304}`` against a numpy ``argpartition`` on the same distinct-valued
#: scores, and **every** size returned ``exact=True`` with all 512 slots
#: filled. Steady-state cost over that 2048x range in N:
#:
#: ===========  ==========
#: ``N``        steady (ms)
#: ===========  ==========
#: 2048               0.53
#: 262144             0.78
#: 1048576            1.07
#: 2097152            1.42
#: 4194304            1.71
#: ===========  ==========
#:
#: So ``sparsecore_topk`` is **not** what stops long context: at
#: ``index_kpool = 4``, ``N = 4194304`` pool candidates is a 16 M-token
#: context, and the thing that fails first is the fp32 score buffer (see
#: :data:`SCORE_BUFFER_MEASURED_MULTIPLIER`). The documented 256 Ki above is
#: kept as the *documented* number and quoted as such; this is the measured
#: one, and neither should be silently replaced by the other -- the docstring
#: at ``sparsecore_topk.py:17-19`` is the contract, and this is only evidence
#: that the contract is conservative at ``k = 512``.
SPARSECORE_TOPK_MEASURED_EXACT_COLUMNS = 4 * 1024 * 1024


def score_buffer_bytes(*, max_num_batched_tokens: int,
                       score_columns: int) -> int:
    """Bytes of the fp32 score buffer ``streamindex_topk`` materialises.

    ``scores_init`` is ``jnp.full((q.shape[0], num_sublanes_total, 128),
    -inf, float32)`` (``streamindex_topk.py:903-907``) and is *aliased* to the
    kernel's output, so it is one buffer and not two. Its minor extent is the
    per-sequence compressed **capacity** -- ``num_sublanes_total`` comes from
    ``pages_per_seq``, which is ``cdiv(max_model_len, block_size)`` -- and not
    from any actual ``seq_len``. A one-token decode of a 40-token request in a
    1 M-context engine still materialises the full 1 MiB row.

    That is what makes ``max_num_batched_tokens`` the binding knob rather than
    the context length: the row width is fixed by ``max_model_len`` and the
    row *count* is the only thing anyone can turn down.
    """
    return int(max_num_batched_tokens) * int(score_columns) * 4


def score_buffer_budget_bytes(override_mib: int | None = None) -> int | None:
    """The budget, from the argument, then the env, then the default.

    ``None`` means "do not check" and is reachable only by an explicit
    ``<= 0``, never by omission.
    """
    if override_mib is None:
        raw = os.environ.get(SCORE_BUFFER_BUDGET_ENV)
        if raw is None or raw.strip() == "":
            override_mib = DEFAULT_SCORE_BUFFER_BUDGET_MIB
        else:
            try:
                override_mib = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{SCORE_BUFFER_BUDGET_ENV}={raw!r} is not an integer "
                    "number of MiB.") from exc
    if override_mib <= 0:
        return None
    return int(override_mib) * 2**20


def describe_score_buffer(*, max_model_len: int, max_num_batched_tokens: int,
                          geometry: "PooledIndexerGeometry",
                          num_indexer_layers: int) -> str:
    """The arithmetic, spelled out, for the log line and the error message."""
    per_call = score_buffer_bytes(
        max_num_batched_tokens=max_num_batched_tokens,
        score_columns=geometry.score_columns)
    return (
        "glm5_next kpool score buffer: "
        f"max_model_len={max_model_len}, index_kpool={geometry.index_kpool}, "
        f"block_size={geometry.block_size}, "
        f"pages_per_seq={geometry.pages_per_seq}, "
        f"num_sublanes_total={geometry.num_sublanes_total} -> "
        f"{geometry.score_columns} fp32 score columns per query token "
        f"({geometry.score_columns * 4} B/token, against "
        f"{geometry.real_pools_per_seq} pools a full-length sequence really "
        f"occupies); max_num_batched_tokens={max_num_batched_tokens} -> "
        f"{per_call} B ({per_call / 2**30:.3f} GiB) per indexer call, "
        f"{per_call * num_indexer_layers / 2**30:.3f} GiB summed over "
        f"{num_indexer_layers} indexer layers"
        # The number a human should compare against HBM. The line above is
        # the single-buffer model; on device the compiler asks for
        # about three times it, and the calls are sequential so the PEAK is
        # one call, not the sum. Both are printed because the first is what
        # the budget is expressed in and the second is what actually OOMs.
        f"; MEASURED on device this is about "
        f"{score_buffer_measured_multiplier(geometry.score_columns):.2f}x -> "
        f"{per_call * score_buffer_measured_multiplier(geometry.score_columns) / 2**30:.3f}"
        " GiB of transient HBM per call (peak, not summed -- the indexer "
        "layers run one at a time). See SCORE_BUFFER_MEASURED_MULTIPLIER.")


def assert_score_buffer_within_budget(
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    geometry: "PooledIndexerGeometry",
    num_indexer_layers: int,
    budget_mib: int | None = None,
) -> str:
    """Refuse a chunk size whose score buffer does not fit, and say why.

    The quantity is not obvious from any single config field --
    it is the product of a width nobody sets directly (``max_model_len //
    index_kpool``, rounded up through the page geometry) and a count that
    looks unrelated (``max_num_batched_tokens``) -- so the failure without
    this guard is an out-of-memory several minutes into a TPU job, attributed
    to the model rather than to the scheduler knob that actually caused it.

    Returns the description on the happy path so the caller can log it.

    Raises:
        ValueError: with the arithmetic, both knobs, and the computed bytes.
    """
    description = describe_score_buffer(
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        geometry=geometry,
        num_indexer_layers=num_indexer_layers)
    budget = score_buffer_budget_bytes(budget_mib)
    if budget is None:
        return description
    per_call = score_buffer_bytes(
        max_num_batched_tokens=max_num_batched_tokens,
        score_columns=geometry.score_columns)
    if per_call <= budget:
        return description

    # The chunk size that would fit, floored to something a human would type.
    fits = max(1, budget // max(1, geometry.score_columns * 4))
    envelope = ""
    if geometry.score_columns > SPARSECORE_TOPK_MAX_COLUMNS:
        envelope = (
            f" Separately: {geometry.score_columns} score columns is past "
            f"sparsecore_topk's documented envelope of "
            f"{SPARSECORE_TOPK_MAX_COLUMNS} (N in 2048..256K, "
            "sparsecore_topk.py:17-19), and lowering "
            "max_num_batched_tokens does NOT fix that -- the column count "
            "follows max_model_len alone.")
    elif geometry.score_columns == SPARSECORE_TOPK_MAX_COLUMNS:
        envelope = (
            f" Separately: {geometry.score_columns} score columns sits "
            "exactly on sparsecore_topk's documented ceiling (N in "
            "2048..256K, sparsecore_topk.py:17-19), i.e. on the boundary "
            "rather than inside it; that is a max_model_len property and "
            "does not move with max_num_batched_tokens. Note the *measured* "
            f"exact-top-k ceiling is {SPARSECORE_TOPK_MEASURED_EXACT_COLUMNS} "
            "columns (measured on device), so the "
            "documented number is conservative and this is not the thing "
            "stopping you -- the score buffer above is.")
    raise ValueError(
        "glm5_next kpool: the indexer's fp32 score buffer does not fit. "
        f"{description}. Budget is {budget} B ({budget / 2**30:.3f} GiB) per "
        f"call; override with {SCORE_BUFFER_BUDGET_ENV}=<MiB> (0 disables the "
        "check). The buffer is sized by the per-sequence compressed CAPACITY "
        "(max_model_len // index_kpool, via pages_per_seq), NOT by the actual "
        "seq_len, so a short request in a long-context engine costs the same "
        "as a full one -- which makes max_num_batched_tokens the binding "
        f"knob, not max_model_len. Lower max_num_batched_tokens to about "
        f"{fits} (from {max_num_batched_tokens}), or lower max_model_len."
        f"{envelope} This bound is uncalibrated: see "
        "DEFAULT_SCORE_BUFFER_BUDGET_MIB.")


#: ``cache_config.cache_dtype`` values that mean "fp8 KV". vLLM spells the
#: same thing three ways depending on the flag.
_FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc")


def assert_fp8_kv_cache(cache_config, *, prefix: str) -> None:
    """Refuse a bf16 KV cache while the sparse MLA path is on.

    ``update_sparse_mla_kv_cache`` asserts ``kv_c_normed.dtype ==
    jnp.float8_e4m3fn`` (``kernels/mla/kv_cache_utils.py:123``) -- it bitcasts
    the latents to uint8 and pads to the spec width, so a bf16 latent makes
    the pad negative. Without this check the failure is a bare
    ``AssertionError: sparse MLA kernel requires --kv-cache-dtype fp8 (got
    bfloat16)`` raised inside a jitted kernel on the *first generate step*,
    which on the real checkpoint is several minutes and a 328 GB load after
    the point where it was decidable -- measured at 2m14s to reach on an
    86 MB fixture, and far longer on a real checkpoint.

    Unknown or absent ``cache_dtype`` passes. That is deliberate rather than
    lax: the attribute is absent in the lightweight configs the unit tests
    build, and a guard that refused an unknown value would be asserting about
    a field it cannot read rather than about the cache.
    """
    dtype = getattr(cache_config, "cache_dtype", None)
    if dtype is None:
        return
    name = str(dtype).lower()
    if name in _FP8_CACHE_DTYPES or "fp8" in name:
        return
    raise ValueError(
        f"glm5_next kpool: the indexer at {prefix!r} is built, so the "
        "full-attention layers take the SPARSE MLA path, which requires an "
        f"fp8 KV cache -- but cache_config.cache_dtype is {dtype!r}. "
        "`update_sparse_mla_kv_cache` bitcasts the latents to uint8 and pads "
        "to the spec width (kernels/mla/kv_cache_utils.py:82-98), so a bf16 "
        "latent makes that pad negative; the kernel asserts at "
        "kv_cache_utils.py:123. Pass --kv-cache-dtype fp8. (Or turn the "
        "indexer off with VLLM_TORCHTPU_GLM5_KPOOL=0, which takes the dense "
        "path and the dense-MLA context bound with it.) Refused here rather "
        "than in the kernel because here is before the weights load.")


def assert_prefix_caching_disabled(cache_config, *, prefix: str) -> None:
    """Refuse prefix caching while the kpool indexer is on.

    ``platforms/tpu_platform.py`` gates prefix caching on
    ``mamba_cache_mode`` and on the unified KV layout, **not** on the indexer,
    so ``enable_prefix_caching=True`` is reachable for this model with
    ``mamba_cache_mode='align'`` and the unified pool -- and it is silently
    wrong rather than loud.

    Why: on a hit vLLM restores the blocks of the *MLA* group. The pooled
    indexer K cache (:class:`Glm5NextIndexerKCache`) is a separate KV-cache
    group that participates in no hashing and no reuse, and the pending ring
    is not a KV-cache group at all. An unwritten pooled page is
    zero-initialised; a zero pooled key scores exactly ``0`` through the ReLU
    at ``streamindex_topk.py:333`` -- finite, not NaN, not ``-inf`` -- so it
    is a perfectly legitimate candidate that always ranks *last*. A request
    hitting a 100 K cached prefix therefore scores every prefix pool at 0 and
    degrades to a recency window: fluent output with no long-range retrieval,
    which is indistinguishable from "the model does not retrieve well".

    The pending ring is named only as the second-order concern it is: local
    hits are block-aligned and ``derive_pooled_indexer_geometry`` already
    requires ``block_size % index_kpool == 0``, so a restored prefix lands on
    a pool boundary and the first completing pool after the hit is entirely
    in-step. See section 5 of
    ``docs/glm5-3-flash-kpool-pending-state.md``.
    """
    if cache_config is None:
        return
    if not getattr(cache_config, "enable_prefix_caching", False):
        return
    raise ValueError(
        "glm5_next kpool: prefix caching is enabled and the kpool indexer is "
        f"built ({prefix}). Refused, because the failure is silent rather "
        "than loud. On a prefix-cache hit vLLM restores the MLA group's "
        "blocks; the POOLED INDEXER K CACHE is a separate KV-cache group "
        "that participates in no hashing and no reuse, so its records for "
        "the restored prefix are never written. An unwritten pooled record "
        "is byte-zero, a zero pooled key scores exactly 0 through the ReLU "
        "(streamindex_topk.py:333) -- finite, not NaN -- so it is a legal "
        "candidate that always ranks last, and the request degrades to a "
        "recency window with fluent output and no long-range retrieval. The "
        "pending-pool ring is a runner-lifetime buffer outside the "
        "allocator and is safe under block-aligned hits "
        "(docs/glm5-3-flash-kpool-pending-state.md section 5), so the "
        "pooled cache is the blocker. tpu_platform.py gates prefix caching "
        "on mamba_cache_mode and the unified KV layout, not on the indexer, "
        "which is why this refusal has to live here. Run with "
        "--no-enable-prefix-caching, or disable the indexer.")


# --- the shared topk_indices buffer ---------------------------------------

#: Where the one shared ``topk_indices`` buffer is stashed. ``VllmConfig`` is
#: copied per submodule by ``as_text_vllm_config`` (``dataclasses.replace``
#: after a ``deepcopy`` of ``model_config`` alone), so the *config* object is
#: not shared between layers -- but ``compilation_config`` is the same object
#: in every copy, and it is already the repo-wide home for per-engine model
#: state (``static_forward_context``). Keying on the config object itself
#: would give every layer its own buffer while looking like it shared one.
_SHARED_TOPK_BUFFER_ATTR = "_glm5_next_kpool_topk_indices_buffer"


def shared_topk_indices_buffer(vllm_config, *, out_width: int):
    """The engine's single ``i32[max_num_batched_tokens, out_width]`` buffer.

    Pre-filled with ``-1``, which is the padding value the sparse MLA
    consumer counts with (``kv_lens = sum(topk_indices != -1)`` at
    ``kernels/mla/sparse/kernel.py:622``).

    Raises:
        ValueError: if a second caller asks for a different width. Two widths
            in one engine means two ``derive_out_width`` answers, and the
            symptom of the smaller one winning is a clipped always-selected
            tail -- invisible at ``L % index_kpool == 0``.
    """
    host = vllm_config.compilation_config
    max_num_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
    existing = getattr(host, _SHARED_TOPK_BUFFER_ATTR, None)
    if existing is not None:
        if tuple(existing.shape) != (max_num_tokens, out_width):
            raise ValueError(
                "glm5_next kpool: two different topk_indices buffer shapes "
                f"in one engine -- {tuple(existing.shape)} already allocated, "
                f"{(max_num_tokens, out_width)} requested. Every DSA layer "
                "and the MTP draft head must agree on out_width "
                "(`VllmTPUMultiHeadLatentAttentionWrapper.__init__` copies it "
                "onto the wrapper), "
                "and a short buffer drops the always-selected tail silently.")
        return existing
    buffer = torch.full((max_num_tokens, out_width), -1, dtype=torch.int32)
    setattr(host, _SHARED_TOPK_BUFFER_ATTR, buffer)
    return buffer


# --- the kpool_write diagnostic counters ----------------------------------

#: Set to ``"1"`` to sync and check :func:`kpool_write`'s counters after every
#: indexer call. **Off by default, and bring-up probes are expected to set
#: it** -- see ``docs/models/glm5-3-flash-kpool.md``.
#:
#: The counters are always *computed* on device -- one int32 compare per
#: carried row -- so a violation is always detected whether or not anybody
#: looks. What costs something is *reading* them: a host round trip per DSA
#: layer per step, eleven of them in every decode step at the real
#: checkpoint. That is why this is a switch and not unconditional.
#:
#: It is read **once, at construction**, and the answer is frozen for the life
#: of the module. That is a requirement, not a micro-optimisation: the flag
#: gates a ``.cpu()`` sync inside ``forward``, so anything that can change its
#: answer between two steps changes the program the backend traces for those
#: steps, across all eleven DSA layers at once.
#:
#: The rejected alternative was a bounded budget -- check the first eight
#: forwards of each layer's life, then stop -- and it was wrong twice over.
#: Warmup spends it: ``_precompile_backbone`` runs one to two ``_dummy_run``s
#: per token bucket and the profile run adds more, so the budget is gone
#: before the first real request and the counters only ever see compilation
#: shapes. And the branch then flips mid-run, re-keying the traced decode step
#: for a check that has already stopped happening.
#:
#: **Set it in probes.** It is what caught the two real defects of this kind:
#: two live requests handed the same pending-ring slot, and sixteen live
#: requests handed the null sentinel slot by a dummy run. Neither raises,
#: neither produces a NaN, and both look exactly like "this model does not
#: retrieve well".
KPOOL_CHECK_DIAGNOSTICS_ENV = "VLLM_TORCHTPU_GLM5_KPOOL_CHECK_DIAGNOSTICS"


def kpool_check_diagnostics_enabled(environ=None) -> bool:
    """Whether to read :func:`kpool_write`'s counters. Decided once.

    Anything other than ``"1"``, ``"0"`` or unset raises. A typo in a
    debugging switch that silently means "off" wastes the run it was set for,
    and this one gets set precisely when something is already suspected.
    """
    env = os.environ if environ is None else environ
    raw = env.get(KPOOL_CHECK_DIAGNOSTICS_ENV)
    if raw is None or raw == "":
        return False
    if raw not in ("0", "1"):
        raise ValueError(
            f"{KPOOL_CHECK_DIAGNOSTICS_ENV}={raw!r} is not understood. Use "
            '"1" to read the kpool pending-state counters after every '
            'indexer call, or "0"/unset to leave them computed but unread.')
    return raw == "1"


def assert_kpool_diagnostics_clean(diagnostics, *, prefix: str) -> None:
    """Refuse a step in which the ring or the slot pool misbehaved.

    The counters are *always computed* inside ``kpool_write`` -- one int32
    compare per carried row -- so a violation is always detected; this
    function only decides whether to look. See ``DIAG_*`` in
    ``kernels/glm5_kpool/pool_write.py`` for what each one means and, in
    particular, why the two slot counters are not redundant with the per-row
    position tag: the tag is an absolute position and absolute positions
    collide across requests, so two sequences sharing a slot read each other's
    rows with tags that match.
    """
    from vllm_torchtpu.kernels.glm5_kpool import DIAG_NAMES

    values = [int(v) for v in torch.as_tensor(diagnostics).reshape(-1).cpu()]
    if not any(values):
        return
    detail = ", ".join(f"{n}={v}" for n, v in zip(DIAG_NAMES, values))
    raise ValueError(
        f"glm5_next kpool: {prefix} reported non-zero pending-state "
        f"diagnostics: {detail}. A tag violation means the ring depth is too "
        "small for the rollback this step performed, or a slot was reused "
        "without being cleared; a slot counter means kpool_state_indices is "
        "out of range or two live requests share a slot. Both produce a "
        "wrong pooled key with no NaN and no shape error. Depth is derived "
        "by derive_pending_state_geometry from max_concurrent_batches * "
        "num_speculative_tokens -- check those, not the constant.")


# --- the MTP draft head's own pending ring ---------------------------------

#: Whether the draft head reuses the target stack's selection instead of
#: running its own indexer. ``False``: not implemented here.
#:
#: This is an **interlock**, not a to-do marker, and it is why
#: :func:`mtp_indexer_needs_pending_state` does not simply read the config.
#: ``index_share_for_mtp_iteration`` is ``True`` in the real checkpoint, and
#: vLLM implements the same-named concept as ``set_skip_topk`` plus a shared
#: ``topk_indices_buffer`` (see ``VllmTPUIndexer.set_skip_topk`` and its
#: consumer in ``mla_attention_op.py``). Reading the config directly would
#: therefore give layer 45 no pending ring while its indexer still runs, so
#: its partial pools would have nowhere to be carried.
#:
#: Index reuse and ring elision must land together, so one constant gates
#: both. When they do, layer 45's indexer stops running and its ring becomes
#: dead weight -- ~7 KiB per request, small, but also a buffer nothing writes
#: and nothing reads, which is what later confuses an accounting assertion.
KPOOL_MTP_INDEX_REUSE = False


def mtp_indexer_needs_pending_state(text_config) -> bool:
    """Whether the MTP draft head's indexer needs its own pending ring.

    ``True`` today for every config: the ring is needed unless the draft head
    reuses the target stack's selection, and nothing implements that yet.
    """
    if not KPOOL_MTP_INDEX_REUSE:
        return True
    return not bool(
        getattr(text_config, "index_share_for_mtp_iteration", False))


class Glm5NextKpoolIndexer(nn.Module):
    """The seven indexer tensors, the key path, and the query path.

    Constructed with the *checkpoint's* submodule names -- ``wq_b``, ``wk``,
    ``k_norm``, ``weights_proj``, ``index_kpool_compress_ape``,
    ``index_kpool_compress_gate`` -- under the layer's own prefix, because
    ``modules_to_not_convert`` is a list of checkpoint paths and
    ``is_layer_skipped`` matches it against module prefixes exactly.
    """

    def __init__(
        self,
        text_config,
        quant_config=None,
        *,
        prefix: str,
        layer_idx: int,
        params_dtype: torch.dtype | None = None,
        vllm_config=None,
        allocate_pending_state: bool = True,
    ) -> None:
        """Build the indexer.

        Args:
            vllm_config: ``None`` builds a **weights-only** indexer: the seven
                tensors, :meth:`project_keys` and :meth:`project_query`, and
                nothing else. No pooled K cache, no pending ring, no engine
                guards, and :meth:`forward` refuses. That is what the
                weight-loading and query-path unit tests want, and it is the
                only reason the argument
                is optional -- a production indexer without an engine config
                would be an indexer whose pooled cache is never allocated,
                which is the byte-zero-scores-0 failure this whole file is
                written around.
            allocate_pending_state: ``False`` only for an indexer whose
                selection is reused rather than computed. See
                :func:`mtp_indexer_needs_pending_state`, which is the only
                thing that can produce ``False`` today.
        """
        super().__init__()
        assert_kpool_config_contract(text_config)

        self.layer_idx = layer_idx
        self.prefix = prefix
        self.hidden_size = int(text_config.hidden_size)
        self.q_lora_rank = int(text_config.q_lora_rank)
        self.n_head = int(text_config.index_n_heads)
        self.head_dim = int(text_config.index_head_dim)
        self.index_topk = int(text_config.index_topk)
        self.index_kpool = int(text_config.index_kpool)
        #: ``head_dim ** -0.5`` at ``modeling_glm5_next.py:765``. Applied
        #: *inside* the reference's ReLU and folded into the per-head weights
        #: here; see :meth:`project_query`.
        self.softmax_scale = self.head_dim**-0.5

        #: The width of the emitted row. The MLA wrapper copies this onto
        #: itself (see ``VllmTPUMultiHeadLatentAttentionWrapper``) and
        #: ``topk_indices_buffer`` is sized from it -- the *output* width, not
        #: ``index_topk``.
        self.topk_tokens = kpool_out_width(self.index_topk, self.index_kpool)
        self.select_k = self.index_topk // self.index_kpool

        params_dtype = params_dtype or torch.get_default_dtype()

        # `disable_tp=True`: these are replicated by design (see the module
        # docstring), and it also keeps every BasevLLMParameter's tp_rank at 0
        # so no rank can narrow a replicated weight at TP=8.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_head * self.head_dim,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            disable_tp=True,
        )
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
            disable_tp=True,
        )
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_head,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.weights_proj",
            disable_tp=True,
        )
        # LayerNorm with bias at eps 1e-6, not RMSNorm and not `rms_norm_eps`.
        self.k_norm = nn.LayerNorm(self.head_dim,
                                   eps=K_NORM_EPS,
                                   bias=True,
                                   dtype=params_dtype)
        # Bare parameters applied with `F.linear` / broadcast addition, exactly
        # as the reference does (`modeling_glm5_next.py:770-771`, `:800`,
        # `:962`). They are also the only two tensors here that vLLM's
        # weights-not-loaded net can see, since `UnquantizedLinearMethod`
        # defines `process_weights_after_loading` and whitelists every Linear
        # param -- which is why `assert_indexer_weights_loaded` exists.
        self.index_kpool_compress_ape = nn.Parameter(torch.empty(
            self.index_kpool, self.head_dim, dtype=params_dtype),
                                                     requires_grad=False)
        self.index_kpool_compress_gate = nn.Parameter(torch.empty(
            self.head_dim, self.hidden_size, dtype=params_dtype),
                                                      requires_grad=False)

        self._assert_projections_are_unquantized(quant_config)

        # --- everything below needs an engine config ----------------------
        self.vllm_config = vllm_config
        self.k_cache = None
        self.pending_geometry: PendingStateGeometry | None = None
        self.indexer_op = None
        self._op_config = None
        if vllm_config is None:
            # Weights-only. Every attribute `forward` would read exists and is
            # None, so the refusal below is a clear message rather than an
            # AttributeError from `nn.Module.__getattr__`.
            self.pending_kv = None
            self.pending_tag = None
            self.topk_indices_buffer = None
            self._check_diagnostics = False
            self.kpool_needs_state_indices = False
            return

        cache_config = vllm_config.cache_config
        # Before anything is allocated: refusals that would otherwise
        # fire only after a 328 GB load, having cost the very queue slot they
        # exist to save.
        assert_prefix_caching_disabled(cache_config, prefix=prefix)
        assert_fp8_kv_cache(cache_config, prefix=prefix)

        self.k_cache = Glm5NextIndexerKCache(
            index_kpool=self.index_kpool,
            index_head_dim=self.head_dim,
            cache_config=cache_config,
            prefix=f"{prefix}.k_cache",
            max_model_len=int(vllm_config.model_config.max_model_len),
        )

        # The score buffer is not allocated by us -- XLA
        # materialises it inside the kernel -- so nothing here can measure it;
        # the guard is the only thing that reports it before the device does.
        self.score_buffer_description = assert_score_buffer_within_budget(
            max_model_len=int(vllm_config.model_config.max_model_len),
            max_num_batched_tokens=int(
                vllm_config.scheduler_config.max_num_batched_tokens),
            geometry=self.k_cache.geometry,
            num_indexer_layers=len(dsa_layer_indices(text_config)) + 1,
        )

        # The two ring tensors, RUNNER-LIFETIME and accounted by
        # no `KVCacheSpec` -- see the module docstring. They are allocated
        # *here*, in the model's construction path, because that is what puts
        # them inside the resident footprint `determine_available_memory()`
        # measures. Allocated in `initialize_kv_cache` instead, they would be
        # HBM the profile has already promised to the block pool.
        if allocate_pending_state:
            self.pending_geometry = derive_pending_state_geometry(
                index_kpool=self.index_kpool,
                index_head_dim=self.head_dim,
                max_num_reqs=int(vllm_config.scheduler_config.max_num_seqs),
                # Both are first-class `VllmConfig` properties at v0.27.0.
                # Never reconstructed from `speculative_config` here: two
                # expressions for the ring depth is how a depth that is green
                # at the probe's default `num_speculative_tokens=1` ships.
                max_concurrent_batches=int(vllm_config.max_concurrent_batches),
                num_speculative_tokens=int(vllm_config.num_speculative_tokens),
            )
            geo = self.pending_geometry
            # Non-persistent buffers: they are state, not weights, so they
            # must move with `.to(device)` and must never appear in a
            # `state_dict` (`assert_indexer_weights_loaded` walks parameters,
            # not buffers, but a checkpoint-shaped name here would still
            # confuse every loader report).
            self.register_buffer("pending_kv",
                                 torch.zeros(geo.kv_shape,
                                             dtype=torch.float32),
                                 persistent=False)
            self.register_buffer("pending_tag",
                                 torch.full(geo.tag_shape,
                                            PENDING_TAG_EMPTY,
                                            dtype=torch.int32),
                                 persistent=False)
            # The runner's marker. `TpuModelRunner._detect_kpool_state` walks
            # the module tree for this attribute rather than importing this
            # file (which would pull the GLM config into every model's import
            # path) and rather than an isinstance tuple that the MTP draft
            # head would have to be remembered into. Set only on the branch
            # that actually allocated a ring, because it is precisely "this
            # module needs `AttentionMetadata.kpool_state_indices`" -- an
            # indexer built with `allocate_pending_state=False` does not.
            self.kpool_needs_state_indices = True
        else:
            self.pending_kv = None
            self.pending_tag = None
            self.kpool_needs_state_indices = False

        # One `-1`-filled buffer for the whole model, not one per layer. Each
        # layer overwrites it and its own sparse-MLA consumer reads it before
        # the next layer runs, which is exactly what upstream vLLM does for
        # DeepSeek (`deepseek_v2.py` builds it once on the model and passes
        # the same tensor to every layer) and what the `skip_topk` reuse
        # `skip_topk` branch of `mla_attention_op.py` assumes. Twelve private
        # copies at `max_num_batched_tokens=8192` would be ~850 MB of HBM
        # that no spec accounts for.
        self.topk_indices_buffer = shared_topk_indices_buffer(
            vllm_config, out_width=self.topk_tokens)

        # Once, here. See KPOOL_CHECK_DIAGNOSTICS_ENV for why this must
        # not be re-decided per step.
        self._check_diagnostics = kpool_check_diagnostics_enabled()
        self.indexer_op = self._build_indexer_op()

    def describe(self) -> str:
        """Every derived quantity this indexer was built at, on one line."""
        parts = [
            f"glm5_next kpool indexer {self.prefix}: "
            f"layer_idx={self.layer_idx}, n_head={self.n_head}, "
            f"head_dim={self.head_dim}, index_topk={self.index_topk}, "
            f"index_kpool={self.index_kpool}, select_k={self.select_k}, "
            f"topk_tokens={self.topk_tokens}"
        ]
        if self.k_cache is not None:
            parts.append(self.k_cache.geometry.describe())
            parts.append(self.score_buffer_description)
        if self.pending_geometry is not None:
            parts.append(self.pending_geometry.describe())
        else:
            parts.append("glm5_next kpool pending state: NOT ALLOCATED "
                         "(selection is reused, see "
                         "mtp_indexer_needs_pending_state)")
        return "\n  ".join(parts)

    # --- construction-time checks ----------------------------------------

    #: The three projections, in the order their absence would bite.
    _PROJECTIONS = ("wq_b", "wk", "weights_proj")

    def _assert_projections_are_unquantized(self, quant_config) -> None:
        """The skip must have *taken effect*, not merely matched a name.

        All 84 indexer tensors are bf16 and all 84 are in the checkpoint's
        ``modules_to_not_convert``, so every projection here must end up on an
        unquantized linear method. Checking the resulting ``quant_method``
        rather than re-running ``is_layer_skipped`` is the point: it is
        immune to how the ignore list is spelled, to ``match_mode``, and to a
        submodule accidentally being constructed at a prefix the list does not
        name -- which is the failure that shipped in the vision tower.
        """
        if quant_config is None:
            return
        quantized = [
            name for name in self._PROJECTIONS if not isinstance(
                getattr(self, name).quant_method, UnquantizedLinearMethod)
        ]
        if not quantized:
            return
        raise ValueError(
            f"glm5_next kpool: indexer projections {quantized} at prefix "
            f"{self.prefix!r} were built on a quantized linear method, but "
            "every indexer tensor in zai-org/GLM-5.3-Flash is BF16 with no "
            "weight_scale_inv sibling. Either the checkpoint's "
            "modules_to_not_convert does not name "
            f"{[f'{self.prefix}.{n}' for n in quantized]} exactly, or this "
            "module was built at a prefix it does not name. Loading bf16 "
            "weights through an FP8 method does not raise -- it computes on "
            "unscaled values while every loader check reports success (see "
            "glm5_next/multimodal.py:_tower_quant_config for the shipped "
            "instance of this bug).")

    # --- the key path ------------------------------------------------------

    def project_keys(
            self,
            hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(k, gate)`` for every token: the two halves a pool is built from.

        ``k = LayerNorm(wk(h))`` and ``gate = index_kpool_compress_gate @ h``
        (``modeling_glm5_next.py:798`` and ``:800``). The pooled key is
        ``sum_j softmax_j(gate_j + ape_j) * k_j`` over the four tokens of a
        pool, per channel -- ``kernels/glm5_kpool.pool_window``, which
        consumes exactly this pair.

        The LayerNorm runs here, in torch, rather than inside a kernel: it
        keeps the "head dim derived from the norm weight's shape" trap
        (``kernels/deepseek_v4/compress_and_store/kernel.py:331``) out of the
        pooling path entirely.
        """
        k, _ = self.wk(hidden_states)
        k = self.k_norm(k)
        gate = F.linear(hidden_states.to(self.index_kpool_compress_gate.dtype),
                        self.index_kpool_compress_gate)
        return k, gate

    # --- the query path ---------------------------------------------------

    def project_query(self, hidden_states: torch.Tensor,
                      q_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(q_fp8[T, n_head, head_dim], weights[T, n_head])`` for the kernel.

        Modeled line-for-line on ``mla_attention_op.py:266-280`` with the rope
        half deleted (§2.3: this indexer applies no rotary embedding), and on
        ``deepseek_v4_indexer.py:75-81``, which folds the same three factors.

        Why the folding is exact, and why it is not optional:

        The reference computes ``relu(q @ pool_keys^T * softmax_scale)`` and
        only then applies the per-head weights and sums over heads
        (``modeling_glm5_next.py:824-830``). ``streamindex_topk`` applies the
        per-head weight *after* its own ReLU (``streamindex_topk.py:333-335``).
        Those agree because ``relu(a*x) == a*relu(x)`` for ``a > 0``, and all
        three folded factors are strictly positive: ``softmax_scale``,
        ``n_head ** -0.5`` and the per-row fp8 scale.

        Dropping ``softmax_scale`` or ``n_head ** -0.5`` changes nothing --
        they are global positive constants and top-k is invariant to a
        positive scalar. **``q_scale`` is not.** ``quantize_tensor(axis=-1)``
        yields one scale per ``(token, head)``, so omitting it reweights heads
        against one another per token; the head-weighted sum then ranks pools
        by a different functional than the reference. Nothing raises, the
        output stays fluent, and the only symptom is degraded long-context
        retrieval. ``tests/models/vllm/glm5_next/test_kpool_query.py`` pins it
        with the cheap per-head-scaling test: scale head ``h`` of ``q`` by
        ``c_h`` and the reference's selection moves, so a folded producer's
        must move with it. An unfolded producer's does not move at all --
        ``q_fp8`` is bit-identical under the scaling and its weights never saw
        ``q_scale`` -- which is what makes the test discriminating. Note that
        "the selected set is unchanged" is the signature of the *bug* here,
        not of the fix; see the test's docstring.
        """
        q, _ = self.wq_b(q_c)
        # Quantize per (token, head): axis -1 over a `head_dim`-wide row.
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = quantize_tensor(q,
                                         torch.float8_e4m3fn,
                                         axis=-1,
                                         use_ue8m0=False)
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights, _ = self.weights_proj(hidden_states)
        # `q_scale` is fp32, so the product promotes; cast back, because the
        # kernel expects the weights in the activation dtype.
        weights_dtype = weights.dtype
        weights = (weights.unsqueeze(-1) * q_scale * self.softmax_scale *
                   self.n_head**-0.5)
        weights = weights.squeeze(-1).to(weights_dtype)
        return q_fp8, weights

    # --- the fused op, and the call into it -------------------------------

    def _build_indexer_op(self):
        """Build the ``pallas.jax_op`` eagerly, at construction.

        Eagerly because ``pallas.jax_op`` runs ``inspect.signature()``, which
        Dynamo cannot trace if the op is first built inside a compiled
        forward -- the same reason ``VllmTPUSparseAttnIndexer.__init__``
        builds its op up front (``mla_attention_op.py:64-67``).
        """
        # Imported here rather than at module scope so the weights-only path
        # (`vllm_config=None`) does not need the op module at all.
        from vllm_torchtpu.layers.vllm.custom_ops.glm5_next.kpool_indexer import (
            KpoolIndexerOpConfig, build_kpool_indexer_op)
        config = KpoolIndexerOpConfig(
            index_kpool=self.index_kpool,
            index_topk=self.index_topk,
            index_head_dim=self.head_dim,
            # Derived from `block_size` and `max_model_len`; NOT
            # DeepSeek-V4's tuned (3, 2, 2), which cannot build at GLM's
            # page_size.
            num_kv_pages_per_block=self.k_cache.geometry.
            num_kv_pages_per_block,
            layer_prefix=self.prefix,
        )
        self._op_config = config
        return build_kpool_indexer_op(config, self.topk_tokens)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor | None = None,
        rotary_emb=None,
    ) -> torch.Tensor:
        """``write -> score -> top-k -> expand``, into the shared buffer.

        The signature matches the one
        ``VllmTPUMultiHeadLatentAttentionWrapper.forward`` uses for the
        DeepSeek indexer (``mla_attention_op.py:604``) so that the wrapper's
        own sparse branch would work if GLM ever stopped overriding
        ``forward``. ``rotary_emb`` is accepted only to be refused: GLM-5.3 is
        NoPE and ``Glm5NextTextIndexer.forward`` does not even take
        ``position_embeddings`` (``modeling_glm5_next.py:774-779``).

        ``positions`` must be the **absolute** raw token position. The pending
        ring is addressed by ``pos % pending_depth`` and every pooled record
        by ``pos // index_kpool``, so a relative or mrope-style 3-D position
        would put every row in the wrong place while raising nothing.
        """
        if self.k_cache is None:
            raise ValueError(
                f"glm5_next kpool: the indexer at {self.prefix!r} was built "
                "weights-only (vllm_config=None), so it has no pooled K "
                "cache, no pending ring and no op. That construction "
                "exists for the weights-only unit tests; a model that reaches "
                "forward() was built wrong.")
        if rotary_emb is not None:
            raise ValueError(
                "glm5_next kpool: the indexer was handed a rotary embedding. "
                "GLM-5.3 is NoPE (qk_rope_head_dim == 0) and the reference "
                "indexer applies no rotary at all; a caller with one is "
                "confusing this indexer with the DeepSeek lightning indexer.")
        if self.pending_kv is None or self.pending_tag is None:
            raise ValueError(
                f"glm5_next kpool: the indexer at {self.prefix!r} was built "
                "without a pending ring (allocate_pending_state=False) but "
                "forward() ran. That combination is only legal for a draft "
                "head whose selection is reused rather than computed; see "
                "mtp_indexer_needs_pending_state and KPOOL_MTP_INDEX_REUSE.")
        if positions is None:
            raise ValueError(
                "glm5_next kpool: positions is None. The pending ring is "
                "addressed by absolute position; there is no fallback.")

        from vllm.forward_context import get_forward_context

        from vllm_torchtpu.layers.common.attention_metadata import \
            AttentionMetadata
        from vllm_torchtpu.layers.vllm.custom_ops.glm5_next.kpool_indexer import \
            kpool_request_distribution

        attn_metadata = get_forward_context().attn_metadata
        metadata = attn_metadata
        if isinstance(attn_metadata, dict):
            metadata = attn_metadata.get(self.k_cache.prefix)
        if not isinstance(metadata, AttentionMetadata):
            # Profiling / warmup shapes: no metadata for this group yet.
            return self.topk_indices_buffer
        pooled_cache = self.k_cache.kv_cache
        if pooled_cache is None or pooled_cache.numel() == 0:
            return self.topk_indices_buffer

        # HANDOFF, deliberately loud. `kpool_state_indices` is a *dedicated*
        # stride-1 slot pool that the runner owns; it is
        # NOT `mamba_state_indices`. Under unified-align mode that field is an
        # attention-pool block id, which can exceed `max_num_reqs + 1` by
        # orders of magnitude, so reusing it would index this module's
        # `[max_num_reqs+1, ...]` ring out of bounds -- see section 4(vii) of
        # docs/glm5-3-flash-kpool-pending-state.md. There is no defensible
        # fallback, so there is none.
        kpool_state_indices = getattr(metadata, "kpool_state_indices", None)
        if kpool_state_indices is None:
            raise ValueError(
                "glm5_next kpool: AttentionMetadata carries no "
                "`kpool_state_indices`. That field and the stride-1 req_id -> "
                "slot pool that fills it are the runner-side handoff recorded "
                "in docs/glm5-3-flash-kpool-pending-state.md, rejected "
                "option 4(vii) "
                "(layers/common/attention_metadata.py plus a copy of "
                "TpuModelRunner._init_mamba_slot_pool / "
                "_build_mamba_state_indices at stride 1). Reusing "
                "`mamba_state_indices` instead is an out-of-bounds read under "
                "unified-align mode, so this raises rather than guessing.")

        k, gate = self.project_keys(hidden_states)
        q_fp8, weights = self.project_query(hidden_states, q_c)

        # f32 on both sides of the ring, deliberately: the in-step rows and
        # the carried rows must be bit-identical or the same pool produces a
        # different key depending on where the scheduler cut the chunk.
        topk_indices, diagnostics = self.indexer_op(
            q_fp8.view(torch.uint8),
            weights,
            pooled_cache,
            self.pending_kv,
            self.pending_tag,
            k.to(torch.float32),
            gate.to(torch.float32),
            self.index_kpool_compress_ape.to(torch.float32),
            positions.to(torch.int32),
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
            kpool_state_indices,
            kpool_request_distribution(metadata.request_distribution),
        )
        # Constant for the life of the module, so every step emits the
        # same ops -- with the sync or without it, never both. See
        # KPOOL_CHECK_DIAGNOSTICS_ENV.
        if self._check_diagnostics:
            assert_kpool_diagnostics_clean(diagnostics, prefix=self.prefix)

        num_tokens = hidden_states.shape[0]
        self.topk_indices_buffer[:num_tokens] = -1
        num_out = min(topk_indices.shape[0], self.topk_indices_buffer.shape[0])
        self.topk_indices_buffer[:num_out] = topk_indices[:num_out]
        return self.topk_indices_buffer


# --- loader support ------------------------------------------------------


def indexer_parameter_names(model: nn.Module) -> list[str]:
    """Every parameter path under an ``.indexer.`` submodule."""
    return [n for n, _ in model.named_parameters() if ".indexer." in n]


def assert_indexer_weights_loaded(model: nn.Module, loaded: set[str]) -> None:
    """Every indexer parameter in the tree must have received a tensor.

    This is deliberately not ``default_weight_loader``'s missing-weight net,
    which cannot see the failure that matters here: a submodule built under a
    name the checkpoint does not use loads *nothing*, keeps its
    ``torch.empty`` contents, and reports no error anywhere -- while the
    checkpoint tensor it wanted falls through ``params_dict.get(name) is None``
    and is dropped by ``continue``. Both halves are silent; together they are
    a layer computing on uninitialised memory.

    Vacuous while no indexer is built, which is the point: it costs
    nothing then and becomes load-bearing the moment the module tree gains an
    ``indexer``.
    """
    expected = set(indexer_parameter_names(model))
    if not expected:
        return
    missing = sorted(expected - set(loaded))
    if missing:
        raise ValueError(
            f"glm5_next: {len(missing)} of {len(expected)} indexer parameters "
            f"received no checkpoint tensor: {missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''}. The checkpoint spells "
            "them model.language_model.layers.<i>.self_attn.indexer.<name>; a "
            "mismatch here means the module is built under a name the "
            "checkpoint does not use, and it would otherwise run on "
            "uninitialised memory with every loader check reporting success.")


# --- the pooled indexer K cache -------------------------------------------
#
# One 256-byte record per *pool* of `index_kpool` raw tokens: `index_head_dim`
# fp8_e4m3 bytes followed by one e8m0 scale byte at offset `index_head_dim`,
# padded to a 128-multiple. That is the record `streamindex_topk` bitcasts
# (`kernels/deepseek_v4/streamindex_topk.py:264-271`), and the same shape
# DeepSeek-V4's indexer cache already proves end to end
# (`deepseek_v4_indexer.py:106-115`, allocated by
# `TpuModelRunner._initialize_ds_v4_kv_cache`, in the branch keyed on
# `_DS_V4_INDEXER_CACHE_SUFFIX`).
#
# Everything below is *derived* from `block_size`, `index_kpool`,
# `index_head_dim` and `max_model_len`. Copying DSv4's numbers does not work
# and does not fail loudly -- see `derive_pooled_indexer_geometry`.


def pooled_record_bytes(index_head_dim: int) -> int:
    """Bytes per pooled record: fp8 values + one e8m0 scale, 128-aligned.

    ``align_to(128 + 1, 128) == 256`` at the real checkpoint's
    ``index_head_dim``. The +1 is the scale byte, not padding: dropping it
    would overlap the next record's first value, which decodes as a finite
    fp8 number and merely ranks pools wrong.
    """
    return align_to(index_head_dim + 1, 128)


#: Records per minor-axis group in the packed cache. The allocation is
#: ``(num_blocks, page_size // kv_packing, kv_packing, record_bytes)``, which
#: is what ``streamindex_topk`` unpacks at ``:745`` as
#: ``page_size = page_size_per_kv_packing * kv_packing``. It must match
#: ``TPUModelRunner._DS_V4_KV_PACKING``; :func:`assert_kv_packing_matches_runner`
#: is the tie so the two cannot drift.
POOLED_CACHE_KV_PACKING = 4

#: Minor axis of the Pallas score tile, in elements. ``streamindex_topk``
#: rejects a ``page_size * bkv_p`` that is not a multiple of this
#: (``streamindex_topk.py:749-754``).
_SCORE_LANE_WIDTH = 128


@dataclass(frozen=True)
class PooledIndexerGeometry:
    """Everything the pooled cache's shape and the top-k tiling depend on.

    Constructed only by :func:`derive_pooled_indexer_geometry`, which raises
    rather than returning an instance that cannot build.
    """

    block_size: int
    """Raw tokens per KV block (``cache_config.block_size``)."""

    index_kpool: int
    """Raw tokens per pool. Also the spec's ``compress_ratio``."""

    index_head_dim: int

    max_model_len: int

    page_size: int
    """Compressed *records* per page == ``MLAAttentionSpec.storage_block_size``
    == ``block_size // index_kpool``. This is the quantity
    ``streamindex_topk`` calls ``page_size`` (``:746``), **not** the number of
    raw tokens in a block. Conflating the two is exactly how ``(3, 2, 2)``
    got copied from DSv4."""

    pages_per_seq: int
    """``cdiv(max_model_len, block_size)`` -- the block-table width, which the
    kernel recovers as ``page_indices.shape[0] // max_num_seqs`` (``:747``)."""

    num_kv_pages_per_block: tuple[int, ...]
    """The ``streamindex_topk`` tiling, derived rather than copied."""

    num_sublanes_total: int
    """``max(align_to(pages_per_seq, bkv_p) * page_size // 128)`` (``:756``)."""

    record_bytes: int

    page_bytes: int
    """Allocated bytes per KV block for this cache."""

    kv_packing: int

    @property
    def score_columns(self) -> int:
        """Pool-score columns the kernel materialises per query row."""
        return self.num_sublanes_total * _SCORE_LANE_WIDTH

    @property
    def real_pools_per_seq(self) -> int:
        """Pools a full-length sequence actually occupies."""
        return self.max_model_len // self.index_kpool

    @property
    def block_utilisation(self) -> float:
        """Fraction of a padded KV block that holds real pages.

        At ``bkv_p = 32`` a KV block spans 32 pages = 128 compressed rows =
        512 raw tokens, so a short sequence processes a mostly-empty block.
        Printed rather than asserted: it is a cost, not a bug.
        """
        bkv_p = max(self.num_kv_pages_per_block)
        return self.pages_per_seq / align_to(self.pages_per_seq, bkv_p)

    def cache_shape(self, num_blocks: int) -> tuple[int, int, int, int]:
        """``(num_blocks, page_size // kv_packing, kv_packing, record_bytes)``.

        The same 4-D packed layout
        ``TpuModelRunner._initialize_ds_v4_kv_cache`` allocates for
        DeepSeek-V4's indexer cache, in the branch keyed on
        ``_DS_V4_INDEXER_CACHE_SUFFIX``, and the layout ``streamindex_topk``
        unpacks.
        """
        return (num_blocks, self.page_size // self.kv_packing, self.kv_packing,
                self.record_bytes)

    def describe(self) -> str:
        """One line per derived quantity, for the construction-time log.

        Worth printing because block utilisation is not obvious from the
        config: the tiny fixture runs ``align_to(12, 32) = 32`` pages against
        12 real ones, so it is *not* representative of the regime a real
        checkpoint is in and nobody should assume it generalises.
        """
        return ("glm5_next kpool pooled indexer K cache: "
                f"block_size={self.block_size} raw tokens, "
                f"index_kpool={self.index_kpool}, "
                f"page_size={self.page_size} records/page, "
                f"record_bytes={self.record_bytes}, "
                f"page_bytes={self.page_bytes}, "
                f"max_model_len={self.max_model_len}, "
                f"pages_per_seq={self.pages_per_seq}, "
                f"num_kv_pages_per_block={self.num_kv_pages_per_block}, "
                f"num_sublanes_total={self.num_sublanes_total} "
                f"({self.score_columns} score columns vs "
                f"{self.real_pools_per_seq} real pools), "
                f"block_utilisation={self.block_utilisation:.3f}")


def _minimum_legal_bkv_pages(page_size: int) -> int:
    """Smallest ``bkv_p`` with ``page_size * bkv_p % 128 == 0``.

    ``lcm(page_size, 128) // page_size``, i.e. ``128 // gcd(page_size, 128)``.
    Every legal value is a multiple of this one, which is why the check below
    can quote "must be a multiple of N" rather than enumerating.

    At DSv4's ``page_size = 256`` this is 1, so its tuned ``(3, 2, 2)`` is
    legal there. At GLM's ``page_size = 4`` it is 32, so ``(3, 2, 2)`` is not.
    """
    return _SCORE_LANE_WIDTH // math.gcd(page_size, _SCORE_LANE_WIDTH)


def derive_pooled_indexer_geometry(
    *,
    block_size: int,
    index_kpool: int,
    index_head_dim: int,
    max_model_len: int,
    num_kv_pages_per_block: tuple[int, ...] | None = None,
    kv_packing: int = POOLED_CACHE_KV_PACKING,
) -> PooledIndexerGeometry:
    """Derive the pooled cache geometry, or refuse to build.

    ``num_kv_pages_per_block=(3, 2, 2)`` is
    DeepSeek-V4's *measured* tuning and is copied into every new
    ``streamindex_topk`` call site by reflex. It is illegal for GLM, and the
    reason is a quantity the two models do not share: ``streamindex_topk``'s
    ``page_size`` is **compressed records per page**,
    ``block_size // compress_ratio``. DSv4 runs ``block_size = 1024``
    (``tpu_platform.py:1150-1157`` takes the MLA page size for it), so its
    ``page_size`` is 256 and ``256 * bkv_p`` is a multiple of 128 for every
    ``bkv_p``. GLM is hybrid, so ``tpu_platform.py:1159-1163`` leaves
    ``block_size`` at vLLM's default 16 and ``page_size`` is 4 -- and
    ``4 * 3 = 12`` is not a multiple of 128, so the kernel raises at *trace*
    time, inside a TPU job, several minutes into a queue.

    This function moves that failure to engine construction and quotes the
    two numbers that decide it (``block_size`` and ``max_model_len``).

    Args:
        num_kv_pages_per_block: ``None`` derives the smallest legal tuple
            ``(N, N, N)``. Pass an explicit tuple only to *check* one.

    Raises:
        ValueError: on any geometry ``streamindex_topk`` or the packed cache
            allocation cannot honour.
    """
    if block_size <= 0 or index_kpool <= 0:
        raise ValueError(f"glm5_next kpool: block_size={block_size} and "
                         f"index_kpool={index_kpool} must both be positive.")

    # A pool must never straddle two KV blocks: the pooled record for pool i
    # is written into the page of raw token `index_kpool * i`, and a pool
    # split across two pages would have its four tokens land in different
    # blocks with only the first block's page written. A user-supplied
    # `--block-size` is the realistic way to get here.
    if block_size % index_kpool != 0:
        raise ValueError(
            f"glm5_next kpool: block_size ({block_size}) is not a multiple of "
            f"index_kpool ({index_kpool}), so a pool would straddle two KV "
            "blocks and only the first block's page would be written -- an "
            "unwritten pooled record is byte-zero, scores exactly 0 through "
            "the ReLU at streamindex_topk.py:333, and is therefore a legal "
            "candidate that always ranks last. No NaN, no shape error, just "
            f"silently missing keys. (max_model_len={max_model_len}.) Pass a "
            f"--block-size that is a multiple of {index_kpool}.")

    page_size = block_size // index_kpool

    if page_size % kv_packing != 0:
        raise ValueError(
            f"glm5_next kpool: page_size ({block_size} // {index_kpool} = "
            f"{page_size} records/page) is not a multiple of kv_packing "
            f"({kv_packing}). The packed cache is allocated as "
            "(num_blocks, page_size // kv_packing, kv_packing, record_bytes) "
            f"-- at page_size {page_size} that middle dimension floors to "
            f"{page_size // kv_packing}, so the tensor holds "
            f"{page_size // kv_packing * kv_packing} of {page_size} records "
            "per page and torch.zeros allocates it without complaint. "
            f"block_size must be a multiple of {index_kpool * kv_packing}; "
            f"got {block_size} (max_model_len={max_model_len}).")

    if max_model_len <= 0:
        raise ValueError(
            f"glm5_next kpool: max_model_len={max_model_len} must be "
            "positive; pages_per_seq is derived from it.")
    pages_per_seq = -(-max_model_len // block_size)

    quantum = _minimum_legal_bkv_pages(page_size)
    if num_kv_pages_per_block is None:
        num_kv_pages_per_block = (quantum, quantum, quantum)
    num_kv_pages_per_block = tuple(int(b) for b in num_kv_pages_per_block)
    if not num_kv_pages_per_block:
        raise ValueError(
            "glm5_next kpool: num_kv_pages_per_block must be non-empty.")

    for bkv_p in num_kv_pages_per_block:
        if bkv_p <= 0:
            raise ValueError(
                "glm5_next kpool: num_kv_pages_per_block entries must be "
                f"positive; got {num_kv_pages_per_block}.")
        bkv_sz = page_size * bkv_p
        if bkv_sz % _SCORE_LANE_WIDTH != 0:
            raise ValueError(
                "glm5_next kpool: num_kv_pages_per_block="
                f"{num_kv_pages_per_block} cannot build. streamindex_topk "
                f"requires page_size * bkv_p % {_SCORE_LANE_WIDTH} == 0 "
                f"(streamindex_topk.py:749-754); at block_size={block_size} "
                f"and index_kpool={index_kpool} the page holds "
                f"page_size={page_size} compressed records, so "
                f"{page_size} * {bkv_p} = {bkv_sz} fails and every entry must "
                f"be a multiple of {quantum}. DeepSeek-V4's tuned (3, 2, 2) "
                "is legal only at its block_size=1024 / page_size=256; do not "
                f"copy it. (max_model_len={max_model_len}, "
                f"pages_per_seq={pages_per_seq}.)")
        # Free, and it documents the coupling: `align_to` rounds up to a
        # multiple of `bkv_p`, so `align_to(pages_per_seq, bkv_p) * page_size`
        # is a multiple of `bkv_p * page_size`, which the check above just
        # forced to be a multiple of 128. A `bkv_p` that would truncate
        # `num_sublanes_total` is therefore already rejected loudly above,
        # so the silent-truncation case cannot arise. Kept so a future edit to
        # either rule trips here.
        spanned = align_to(pages_per_seq, bkv_p) * page_size
        if spanned % _SCORE_LANE_WIDTH != 0:
            raise ValueError(
                "glm5_next kpool: align_to(pages_per_seq="
                f"{pages_per_seq}, {bkv_p}) * page_size={page_size} = "
                f"{spanned} is not a multiple of {_SCORE_LANE_WIDTH}, so the "
                "floor division at streamindex_topk.py:756-757 would drop "
                "score columns. This should be unreachable given the check "
                "above; if it fires, the two rules have drifted apart.")

    num_sublanes_total = max(
        align_to(pages_per_seq, bkv_p) * page_size // _SCORE_LANE_WIDTH
        for bkv_p in num_kv_pages_per_block)

    record_bytes = pooled_record_bytes(index_head_dim)
    return PooledIndexerGeometry(
        block_size=block_size,
        index_kpool=index_kpool,
        index_head_dim=index_head_dim,
        max_model_len=max_model_len,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_sublanes_total=num_sublanes_total,
        record_bytes=record_bytes,
        page_bytes=page_size * record_bytes,
        kv_packing=kv_packing,
    )


def pooled_indexer_kv_cache_spec(
    *,
    block_size: int,
    index_kpool: int,
    index_head_dim: int,
) -> KVCacheSpec:
    """The pooled cache's ``MLAAttentionSpec``, in the v0.27.0 spelling.

    ``compress_ratio`` is the field the *deployment* has; vLLM ``main``
    replaced it with ``tokens_per_state``. See the module docstring and
    ``docs/vllm-kv-cache-api-pin.md``. The ``TypeError`` from the wrong vLLM
    is caught and re-raised with the document named, because the bare message
    ("unexpected keyword argument 'compress_ratio'") sends the reader looking
    for a typo.

    ``compress_ratio`` is *not* clamped to ``block_size`` the way
    ``deepseek_v4_indexer.py:113`` clamps it. That clamp silently turns a
    ratio-4 cache into a ratio-1 cache when ``block_size < 4``;
    :func:`derive_pooled_indexer_geometry` refuses that geometry outright
    instead, so there is nothing left to clamp.
    """
    try:
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=pooled_record_bytes(index_head_dim),
            dtype=torch.uint8,
            compress_ratio=index_kpool,
            alignment=None,
        )
    except TypeError as exc:  # pragma: no cover - depends on the vLLM import
        raise TypeError(
            "glm5_next kpool: MLAAttentionSpec does not accept "
            "`compress_ratio`, so the imported vLLM is not the one the TPU "
            "pod runs. The pod imports v0.27.0 (pyproject.toml:29), which has "
            "`compress_ratio`/`storage_block_size`; vLLM main replaced both "
            "with `tokens_per_state`/`num_states`. This is the expected "
            "failure on the CPU rung, where PYTHONPATH points at the "
            "developer checkout -- it is not a bug in this module. See "
            "docs/vllm-kv-cache-api-pin.md and "
            f"tests/models/vllm/test_vllm_kv_cache_api_pin.py. ({exc})"
        ) from exc


class Glm5NextIndexerKCache(torch.nn.Module, AttentionLayerBase):
    """The pooled indexer K cache for one DSA layer.

    Deliberately **not** a subclass of ``DeepseekV32IndexerCache`` or
    ``DeepseekV4IndexerCache``, however convenient their ``get_kv_cache_spec``
    would be. ``tpu_runner.is_cache_for_ds_v4`` is an
    ``isinstance`` check over the DSv4 cache classes, and any ``True`` there
    sets ``_is_ds_v4`` (``tpu_runner.py:6147``) and reroutes the *whole model*
    through ``_initialize_ds_v4_kv_cache``, which raises in
    ``_classify_ds_v4_layers`` on GLM's first KDA layer. Inheriting for
    convenience would trade a 30-line branch for a model that cannot boot.

    The module registers itself in ``static_forward_context`` exactly as
    ``DeepseekV32IndexerCache`` does, because that dict is what
    ``get_layers_from_vllm_config`` walks and therefore the only way the
    runner ever sees this cache.
    """

    def __init__(
        self,
        *,
        index_kpool: int,
        index_head_dim: int,
        cache_config,
        prefix: str,
        max_model_len: int,
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.prefix = prefix
        self.cache_config = cache_config
        self.index_kpool = int(index_kpool)
        self.index_head_dim = int(index_head_dim)
        self.max_model_len = int(max_model_len)
        self.dtype = torch.uint8

        # Derive (and therefore validate) at construction, not at trace time.
        self.geometry = self._derive_geometry(cache_config.block_size)

        from vllm.config import get_current_vllm_config
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _derive_geometry(self, block_size: int) -> PooledIndexerGeometry:
        return derive_pooled_indexer_geometry(
            block_size=int(block_size),
            index_kpool=self.index_kpool,
            index_head_dim=self.index_head_dim,
            max_model_len=self.max_model_len,
        )

    # `AttentionLayerBase` API ------------------------------------------

    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
        """The ratio-``index_kpool`` pooled spec, re-derived from the config.

        ``block_size`` is re-read from ``vllm_config`` rather than trusted
        from construction: ``update_block_size_for_backend`` can raise it
        (``tpu_platform.py:1167-1175``) and a stale value here would size the
        page for a block geometry the allocator no longer uses.
        """
        block_size = int(vllm_config.cache_config.block_size)
        max_model_len = int(vllm_config.model_config.max_model_len)
        if (block_size != self.geometry.block_size
                or max_model_len != self.geometry.max_model_len):
            self.max_model_len = max_model_len
            self.geometry = self._derive_geometry(block_size)
        return pooled_indexer_kv_cache_spec(
            block_size=block_size,
            index_kpool=self.index_kpool,
            index_head_dim=self.index_head_dim,
        )

    def get_attn_backend(self):
        """``PallasMLAttentionBackend`` -- the *same* class the MLA sibling
        returns, and that is the whole point.

        ``Platform._find_non_ssm_backend`` returns the backend of the **first
        non-SSM ``AttentionLayerBase`` in registration order**
        (``vllm/platforms/interface.py:589-606``), and its answer decides
        ``cache_config.block_size`` via ``get_min_page_size``
        (``tpu_platform.py:1167-1175``) *and* ``attn_page_size_bytes`` via
        ``TPUModelRunner._update_attention_page_size_padded``
        (``tpu_runner.py:1780-1783``), which is folded into every attention
        spec's ``page_size_padded`` and into the mamba page padding. An
        indexer cache registers before its ``mla_attn`` sibling, so returning
        a backend with a different ``get_min_page_size`` here would repaginate
        the entire model -- KDA layers included -- purely as a side effect of
        the pooled cache existing. Returning the sibling's own backend makes
        the answer independent of registration order.

        ``tpu_runner._assert_pooled_indexer_backend_agrees`` checks that
        property against the model actually built, because this docstring is
        not enforcement.
        """
        from vllm_torchtpu.layers.vllm.attention import \
            PallasMLAttentionBackend
        return PallasMLAttentionBackend

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Keep the 4-D packed buffer as allocated.

        **Not called on the TPU path.** ``tpu_bind_kv_cache`` assigns
        ``forward_context[layer_name].kv_cache = kv_cache`` directly
        (``utils.py:287``); it never looks for a ``bind_kv_cache`` method. So
        this is *not* the protection against the squeeze below -- the
        protection is that nothing squeezes at all. It is here because the
        method is part of the interface ``DeepseekV32IndexerCache`` defines,
        an inherited or vLLM-side caller would find it, and the wrong body
        (see the next paragraph) is silent.

        ``DeepseekV32IndexerCache.bind_kv_cache`` does ``kv_cache.squeeze(1)``
        because its cache is ``[B, H=1, N, C]``. This one is
        ``(num_blocks, page_size // kv_packing, kv_packing, record_bytes)``
        and dimension 1 is a real axis -- squeezing it would silently drop the
        packing axis whenever ``page_size == kv_packing``, which is exactly
        GLM's ``block_size 16 / index_kpool 4`` case, where it is 1.
        """
        self.kv_cache = kv_cache

    def forward(self):  # pragma: no cover - never called
        ...


def is_pooled_indexer_cache(module) -> bool:
    """Whether ``module`` is a GLM-5.3 pooled indexer K cache.

    Isolated here so the runner's predicate has one definition. See
    ``tpu_runner.is_compressed_indexer_cache`` for what it must *not* be
    folded into.
    """
    return isinstance(module, Glm5NextIndexerKCache)


def assert_kv_packing_matches_runner(runner_kv_packing: int) -> None:
    """The packed layout has two owners; refuse to let them disagree.

    The runner allocates DSv4's indexer cache with its own
    ``_DS_V4_KV_PACKING`` and this module derives the pooled shape with
    :data:`POOLED_CACHE_KV_PACKING`. They describe the same physical layout.
    If they ever differ, the pooled cache is allocated at one shape and read
    at another -- an in-range, wrong-offset read, which is fluent-and-wrong
    rather than a crash.
    """
    if runner_kv_packing != POOLED_CACHE_KV_PACKING:
        raise ValueError(
            "glm5_next kpool: the runner's KV packing "
            f"({runner_kv_packing}) and the pooled cache's "
            f"({POOLED_CACHE_KV_PACKING}) disagree. Both describe the "
            "(num_blocks, page_size // packing, packing, record_bytes) "
            "layout that streamindex_topk unpacks at :745; a mismatch is an "
            "in-range read at the wrong offset, not a crash.")


# --- HBM accounting: planned vs allocated ---------------------------------
#
# These live here rather than in the runner for one reason: the runner module
# cannot be imported on the CPU rung at all (it imports
# `ModelOptNvFp4LinearMethod`, which the developer checkout of vLLM no longer
# has), so an assertion written inside it has no test. These are pure
# functions over already-allocated tensors and plain ints; the runner calls
# them and `tests/runner/test_glm5_next_kv_accounting.py` tests them.
#
# What they exist to catch: `uniform_page_size_bytes` in
# `tpu_runner._update_mamba_page_size_padded` was `num_attn_groups * attn_page
# + num_mamba_groups * mamba_unpadded` -- attention and mamba groups **only**,
# with the pooled indexer caches in neither term. On the tiny fixture there is
# slack and it passes; the discrepancy scales with `num_blocks` and surfaces
# as an OOM, or as a silently reduced context, inside the 328 GB TP=8 job.
#
# That formula now carries a third term, `num_pooled_groups * pooled_page`, so
# the budget is *correct* rather than knowingly short -- which matters because
# `assert_kv_cache_allocation_within_plan` is fail-closed and a pooled cache
# was otherwise a guaranteed overrun (11 caches x 1024 B/block; see
# `tests/runner/test_glm5_next_kv_accounting.py`). These helpers stay: the
# budget being right by derivation and the allocation matching it by
# measurement are two different claims, and only the second survives a change
# to either side.
#
# A second overrun was suspected -- the sparse MLA split (nope, rope) cache
# at 10240 bytes against a budgeted 16 x 576 = 9216. Measured, it is not real:
# nothing budgets `block_size * head_size`. The TPU backend pads the
# minor axis to a 128 multiple (576 -> 640) and packs by dtype, so at fp8 both
# numbers are 10240. `tests/runner/test_glm5_next_kv_accounting.py` pins the
# equality so it is noticed if it ever stops holding.


@dataclass(frozen=True)
class AllocatedCacheBytes:
    """Measured, not predicted: what one layer's cache really costs."""

    layer_name: str
    num_blocks: int
    bytes_per_block: int
    shapes: tuple[tuple[int, ...], ...]

    @property
    def total_bytes(self) -> int:
        return self.num_blocks * self.bytes_per_block


def measure_allocated_cache_bytes(layer_name: str,
                                  cache) -> AllocatedCacheBytes:
    """Bytes per block of an allocated cache, however many tensors it is.

    Handles the three shapes the runner produces: a single tensor (dense
    attention / MLA), a tuple of tensors (mamba states, and the sparse MLA
    ``(nope, rope)`` pair), and the 4-D packed pooled cache. The leading
    dimension is the block/slot count in every one of them.

    Raises:
        ValueError: if a tensor's bytes are not divisible by its leading
            dimension, which would make "bytes per block" a lie rather than a
            rounding.
    """
    tensors = tuple(cache) if isinstance(cache, (tuple, list)) else (cache, )
    if not tensors:
        raise ValueError(f"{layer_name}: no tensors allocated.")
    num_blocks = int(tensors[0].shape[0])
    bytes_per_block = 0
    for tensor in tensors:
        if int(tensor.shape[0]) != num_blocks:
            raise ValueError(
                f"{layer_name}: cache tensors disagree about the block count "
                f"({[int(t.shape[0]) for t in tensors]}); bytes-per-block is "
                "not well defined and the block ids of the shorter tensor "
                "would be clipped silently by JAX's indexed writes.")
        total = tensor.numel() * tensor.element_size()
        if num_blocks == 0 or total % num_blocks != 0:
            raise ValueError(
                f"{layer_name}: {total} bytes over {num_blocks} blocks is not "
                "an integer bytes-per-block.")
        bytes_per_block += total // num_blocks
    return AllocatedCacheBytes(
        layer_name=layer_name,
        num_blocks=num_blocks,
        bytes_per_block=bytes_per_block,
        shapes=tuple(tuple(int(d) for d in t.shape) for t in tensors),
    )


def total_allocated_cache_bytes(kv_caches: dict) -> int:
    """Every byte of KV cache actually on the device, summed.

    The per-group comparison below is per *block*, and that is a weaker check
    than it looks: it can only flag a group whose summed bytes/block exceed
    the uniform page, and in the layout GLM will actually get -- one layer from
    each of six groups per ``KVCacheTensor`` -- the uniform page *is* the sum
    over the group, so the comparison is close to vacuous. The absolute total
    is the independent one: it is compared against the bytes vLLM decided to
    hand out (``sum(t.size for t in kv_cache_tensors)``) and is sensitive to
    ``num_blocks`` being wrong, which the per-block view is not.
    """
    return sum(
        measure_allocated_cache_bytes(name, cache).total_bytes
        for name, cache in kv_caches.items())


def kv_cache_allocation_report(
    *,
    kv_caches: dict,
    tensor_groups: list[list[str]],
    planned_page_size_bytes: int | None,
    tensor_sizes: list[int] | None = None,
) -> tuple[str, list[tuple[tuple[str, ...], int]]]:
    """A printable planned-vs-allocated table, plus the over-budget groups.

    Args:
        kv_caches: ``layer_name -> tensor | tuple[tensor, ...]``, after
            allocation.
        tensor_groups: the ``shared_by`` lists of every ``KVCacheTensor``;
            each group's layers share one budget of
            ``planned_page_size_bytes`` per block.
        planned_page_size_bytes: ``_hybrid_uniform_page_size_bytes``, or
            ``None`` when the model is not hybrid and no uniform budget was
            installed (then nothing is compared, only printed).
        tensor_sizes: ``[t.size for t in kv_cache_config.kv_cache_tensors]``.
            When given, the report ends with the **whole-allocation**
            comparison -- total bytes really allocated against total bytes
            vLLM handed out. See :func:`total_allocated_cache_bytes` for why
            the per-group figures above are not a substitute.

    Returns:
        ``(report, overruns)`` where ``overruns`` is one entry per group whose
        summed allocated bytes/block exceed the budget, as
        ``((layer names...), allocated_bytes_per_block)``. The whole-allocation
        comparison is **not** in ``overruns``: compact-mamba sizing makes the
        total legitimately *smaller* than the budget, so only the caller that
        knows the layout can decide what an excess means.
    """
    lines = [
        "glm5_next kpool KV-cache accounting (planned vs allocated, "
        "bytes per block):"
    ]
    overruns: list[tuple[tuple[str, ...], int]] = []
    grouped = set()
    for shared_by in tensor_groups:
        measured = [
            measure_allocated_cache_bytes(name, kv_caches[name])
            for name in shared_by if name in kv_caches
        ]
        grouped.update(m.layer_name for m in measured)
        allocated = sum(m.bytes_per_block for m in measured)
        if planned_page_size_bytes is None:
            verdict = "(no uniform plan)"
        else:
            delta = allocated - planned_page_size_bytes
            verdict = (f"planned={planned_page_size_bytes} "
                       f"delta={delta:+d}")
            if delta > 0:
                overruns.append((tuple(shared_by), allocated))
        lines.append(f"  tensor shared_by={list(shared_by)}: "
                     f"allocated={allocated} {verdict}")
        for m in measured:
            lines.append(f"      {m.layer_name}: {m.bytes_per_block} B/block "
                         f"x {m.num_blocks} blocks, shapes={list(m.shapes)}")
    ungrouped = sorted(set(kv_caches) - grouped)
    for name in ungrouped:
        m = measure_allocated_cache_bytes(name, kv_caches[name])
        lines.append(f"  NOT IN ANY KVCacheTensor: {name}: "
                     f"{m.bytes_per_block} B/block x {m.num_blocks} blocks, "
                     f"shapes={list(m.shapes)}")
    if tensor_sizes is not None:
        allocated_total = total_allocated_cache_bytes(kv_caches)
        planned_total = sum(int(s) for s in tensor_sizes)
        delta = allocated_total - planned_total
        lines.append(
            f"  WHOLE ALLOCATION: allocated={allocated_total} B "
            f"({allocated_total / 2**30:.3f} GiB) over {len(kv_caches)} "
            f"caches vs KVCacheTensor budget={planned_total} B "
            f"({planned_total / 2**30:.3f} GiB) over {len(tensor_sizes)} "
            f"tensors, delta={delta:+d} B ({delta / 2**30:+.3f} GiB)")
    return "\n".join(lines), overruns


def assert_kv_cache_allocation_within_plan(
    *,
    kv_caches: dict,
    tensor_groups: list[list[str]],
    planned_page_size_bytes: int | None,
    tensor_sizes: list[int] | None = None,
) -> str:
    """Refuse an allocation that costs more per block than vLLM budgeted.

    vLLM derives ``num_blocks`` from ``planned_page_size_bytes``. Allocating
    more than that per block multiplies the overrun by ``num_blocks``, which
    is the difference between "fits in 328 GB" and "OOMs, or quietly gets a
    shorter context". Nothing compares the two today.

    Two comparisons, and the second is the one with teeth. The per-group one
    is per *block* and can only fire for a group that is essentially the whole
    model, because the uniform page **is** the per-group sum by construction.
    The whole-allocation one -- ``tensor_sizes``, when the caller passes it --
    compares total bytes on the device against total bytes vLLM handed out and
    is sensitive to ``num_blocks`` being wrong, which the per-block view is
    not. It is one-sided: allocating *less* is legitimate (compact-mamba caps
    mamba at ``max_num_reqs + 1`` slots), allocating more is not.

    Returns the report so the caller can log it on the happy path too.
    """
    report, overruns = kv_cache_allocation_report(
        kv_caches=kv_caches,
        tensor_groups=tensor_groups,
        planned_page_size_bytes=planned_page_size_bytes,
        tensor_sizes=tensor_sizes,
    )
    if tensor_sizes is not None:
        allocated_total = total_allocated_cache_bytes(kv_caches)
        planned_total = sum(int(s) for s in tensor_sizes)
        if allocated_total > planned_total:
            raise ValueError(
                "glm5_next kpool: the KV caches actually allocated total "
                f"{allocated_total} B, more than the {planned_total} B vLLM "
                "handed out across "
                f"{len(tensor_sizes)} KVCacheTensor(s) "
                f"(delta {allocated_total - planned_total:+d} B). vLLM sized "
                "those tensors against the HBM it had left after "
                "`determine_available_memory`, so this is HBM that no budget "
                "reserved -- it lands as an OOM at the first long request, "
                "not here. The per-block table below is not a substitute for "
                "this check: the uniform page is the per-group sum by "
                f"construction, so it can only ever flag a group that is the "
                f"whole model.\n{report}")
    if overruns:
        detail = "; ".join(f"{list(names)} needs {allocated} B/block"
                           for names, allocated in overruns)
        raise ValueError(
            "glm5_next kpool: allocated KV cache exceeds the page size vLLM "
            f"budgeted ({planned_page_size_bytes} B/block), so num_blocks was "
            f"computed against a smaller footprint than we allocate: {detail}."
            " uniform_page_size_bytes in "
            "tpu_runner._update_mamba_page_size_padded budgets "
            "num_attn_groups*attn_page + num_mamba_groups*mamba_unpadded + "
            "num_pooled_groups*pooled_page; if a pooled cache is over budget "
            "here, either its group count or its page bytes disagree with "
            "that term. Widen the budget explicitly or shrink num_blocks; "
            f"do not delete this check.\n{report}")
    return report
