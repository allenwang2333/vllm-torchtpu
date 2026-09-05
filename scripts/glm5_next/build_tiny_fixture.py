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
"""Build a tiny random GLM-5.3-Flash (``glm5_next``) checkpoint for bringup.

The point of this fixture is to iterate on the TPU port without touching the
328 GB real checkpoint. It shrinks *counts* -- layers, heads, experts, vocab --
while holding every *dimension* the Pallas kernels tile over at its production
value:

  * ``linear_head_dim`` 128 -- the KDA kernels tile the key/value dim.
  * ``qk_nope_head_dim`` / ``v_head_dim`` 256 -- the sparse MLA kernel's geometry.
  * ``index_head_dim`` 128 and ``index_kpool`` 4 -- the DSA indexer's.
  * ``linear_conv_kernel_dim`` 4 -- the short convolution's tap count.

A fixture with ``head_dim=16`` would build and run while hiding every tiling
and layout bug that actually matters on device, which is the only class of bug
a fixture this small is still able to catch.

The layer schedule keeps GLM's real 3:1 ratio so both attention kinds and the
boundary between them are exercised: ``[lin, lin, lin, dsa] x 2``.

Two flags change the fixture's *regime* rather than its size, and both default
to today's values so an unflagged build is byte-identical to the shipped
fixture at ``~/data/glm5_next_tiny``:

  * ``--index-topk`` -- drop the indexer budget below the sequence length so
    the kpool top-k stops being the identity. See the long comment on
    ``index_topk`` in ``build_config`` for why the default cannot do that.
  * ``--mla-production-dims`` -- ``kv_lora_rank`` 64 -> 512, the only value the
    sparse-MLA consumer can execute at all.

Usage:
    PYTHONPATH=$GLM5_NEXT_TRANSFORMERS_SRC python scripts/glm5_next/build_tiny_fixture.py \
        --out /tmp/glm5_next_tiny

    # the variant that can actually catch a kpool selection bug
    PYTHONPATH=$GLM5_NEXT_TRANSFORMERS_SRC python scripts/glm5_next/build_tiny_fixture.py \
        --out /tmp/glm5_next_tiny_kpool --index-topk 64 --mla-production-dims
"""

import argparse
import json
import math
import pathlib

import torch
from transformers.models.glm5_next import (Glm5NextConfig,
                                           Glm5NextForConditionalGeneration)

#: ``kv_lora_rank`` the sparse-MLA path is hard-wired to. It is not a tuning
#: knob: ``layers/common/attention_interface.py:805-807`` asserts
#: ``lkv_dim == TILE_SUBROWS * TILE_LANE_BYTES`` = 4 * 128 = 512 before it will
#: build ``sparse_mla_attention``, and ``get_sparse_kv_cache_specs``
#: (``layers/vllm/attention.py:1274-1283``) splits the MLA head into
#: ``head_size - 64`` nope bytes plus a 64-wide rope tail. So the fixture's
#: default 64 cannot execute one line of the sparse consumer.
_PRODUCTION_KV_LORA_RANK = 512

#: Width of the zero rope slab the sparse path forces (`_DS_MLA_ROPE_HEAD_DIM`
#: at ``layers/vllm/attention.py:1189``). GLM is NoPE -- ``qk_rope_head_dim``
#: is 0 -- so ``MLAAttention.head_size`` is ``kv_lora_rank + slab``, and only
#: 512 + 64 = 576 satisfies both asserts above.
_SPARSE_ROPE_SLAB = 64

#: ``index_kpool``: four raw tokens per pooled indexer key. Real checkpoint
#: value, held fixed by the fixture -- the TPU port derives ``select_k``,
#: ``causal_row_offset`` and the pooled cache's compression ratio from it.
_INDEX_KPOOL = 4

#: The six per-layer mHC tensors. The real checkpoint's nextn layer (index 45)
#: carries none of them, so the MTP block runs a plain pre-norm residual
#: instead of the mHC seam; the fixture must reproduce that or it would be
#: testing a draft head the port does not build. Kept identical to
#: layer 45 of the real checkpoint, which carries none of them.
_HC_NAMES = ("hc_attn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_base",
             "hc_ffn_fn", "hc_ffn_scale")

# Repeat unit of GLM-5.3-Flash's attention schedule: three KDA layers then one
# DeepSeek sparse attention layer.
_LAYER_CYCLE = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "deepseek_sparse_attention",
]


def build_config(num_cycles: int = 2,
                 kda_heads: int = 4,
                 attn_heads: int = 4,
                 index_topk: int = 2048,
                 kv_lora_rank: int = 64) -> Glm5NextConfig:
    """A tiny `glm5_next` config that still exercises the real kernel geometry."""
    # bf16 queries are packed two per 32-bit lane in the sparse MLA kernel,
    # which asserts `num_q_heads % q_packing == 0`
    # (`kernels/mla/sparse/kernel.py:176-177`). An odd head count builds a
    # fixture that no sparse run can consume.
    assert attn_heads % 2 == 0, (
        f"attn_heads must be even for bf16 q packing, got {attn_heads}")
    # `select_k = index_topk // index_kpool` and the emitted row is
    # `index_topk + index_kpool - 1` wide; a non-multiple makes both ragged.
    assert index_topk % _INDEX_KPOOL == 0, (
        f"index_topk ({index_topk}) must be a multiple of index_kpool "
        f"({_INDEX_KPOOL})")
    assert index_topk > 0

    num_layers = len(_LAYER_CYCLE) * num_cycles
    layer_types = _LAYER_CYCLE * num_cycles

    # One dense FFN then MoE everywhere else, mirroring `first_k_dense_replace`
    # on the real model (3 of 45) at a ratio that fits 8 layers.
    mlp_layer_types = ["dense"] + ["sparse"] * (num_layers - 1)

    kda_layers = [
        i for i, t in enumerate(layer_types) if t == "linear_attention"
    ]
    full_attn_layers = [
        i for i, t in enumerate(layer_types) if t != "linear_attention"
    ]

    text_config = dict(
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        mlp_layer_types=mlp_layer_types,
        indexer_types=["full"] * num_layers,
        hidden_size=512,
        intermediate_size=256,
        # --- KDA: real dims, few heads ---
        # Spelled as the nested dict the real checkpoint ships, not as the
        # unpacked `linear_*` fields, so the fixture exercises the same config
        # path. `Glm5NextTextConfig.__post_init__` unpacks it either way.
        #
        # Note `kda_layers` here is 0-indexed, as GLM-5.3-Flash's own
        # config.json is -- Kimi-Linear's identically-named field is 1-indexed.
        # Anything consuming this list has to know which convention it is
        # reading, and getting it wrong is silent.
        linear_attn_config=dict(
            num_heads=kda_heads,
            head_dim=128,
            short_conv_kernel_size=4,
            gate_lower_bound=-5.0,
            kda_layers=kda_layers,
            full_attn_layers=full_attn_layers,
        ),
        # --- DSA/MLA: real head dims, NoPE only (qk_rope_head_dim must be 0) ---
        num_attention_heads=attn_heads,
        num_key_value_heads=attn_heads,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        q_lora_rank=128,
        # 64 by default; `--mla-production-dims` raises it to 512. See
        # `_PRODUCTION_KV_LORA_RANK`.
        kv_lora_rank=kv_lora_rank,
        # --- DSA indexer: real head dim and kpool ---
        #
        # `index_topk` must stay >= the sequence length used for validation.
        # The TPU port runs these layers as dense causal MLA, which is exact
        # only while the kpool indexer selects every causal token: a query at
        # position p has floor((p+1)/kpool) candidate pools against a budget of
        # index_topk//kpool. At index_topk=64 that budget is 16 pools = 64
        # tokens, so a 192-token prompt exceeds it and the reference really
        # does attend sparsely -- making the reference disagree with a
        # *correct* dense implementation, on the MLA layers only.
        #
        # Set to the real checkpoint's 2048 rather than something merely large
        # enough. The bound doubles as the model's supported context, so a
        # smaller value here would make the fixture refuse a `max_model_len`
        # that production accepts. Costs nothing: the reference clamps its
        # budget to the pools that exist (48 for a 192-token prompt).
        #
        # --- extension, kpool implementation round ---
        #
        # Everything above is still true, and it is exactly why the default
        # harness *structurally cannot* catch a kpool selection bug. At
        # index_topk 2048 the reference's top-k is the IDENTITY at every length
        # this fixture is ever run at: 192 tokens give 48 candidate pools
        # against a 512-pool budget, so `topk(512)` returns all 48 whatever the
        # scores are. A kpool implementation that pools with the wrong softmax,
        # scores with the wrong query scale, or applies the wrong causal bound
        # still selects the same set, and every test scores green. That is not
        # a hypothetical: it is the regime the default fixture sits in.
        #
        # `--index-topk 64` is the escape. Budget = 64//4 = 16 pools = 64 raw
        # tokens; at 192 tokens there are 48 candidate pools, so two thirds are
        # genuinely dropped and the surviving set depends on every one of the
        # quantities above. Small enough to print by hand.
        #
        # A fixture built that way **will disagree with the current dense
        # port**, and that disagreement is the fixture working, not a bug in
        # it. Below the identity bound dense causal MLA is bit-correct by
        # theorem; at index_topk 64 the bound is 64 + 4 - 1 = 67 tokens and a
        # 192-token prompt is far past it, so a dense implementation attends to
        # 192 keys where the reference attends to at most 64. Do not "fix" that
        # by raising index_topk back up -- that just re-enters the blind
        # regime. The default stays 2048 so that production geometry, the
        # `max_model_len` the guard derives from it, and every existing green
        # run are untouched.
        index_n_heads=attn_heads,
        index_head_dim=128,
        index_topk=index_topk,
        index_kpool=_INDEX_KPOOL,
        index_kpool_always_select_tail=True,
        # --- MoE ---
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=64,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        swiglu_limit=10.0,
        # --- MHC: keep the real stream count and Sinkhorn schedule, it is cheap
        # and the routing is subtle enough to be worth exercising ---
        hc_mult=4,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
        # --- misc ---
        vocab_size=1024,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=[2],
    )

    # The vision tower is out of scope for functional bringup, but
    # `Glm5NextModel` always constructs one. Make it negligible rather than
    # absent so the checkpoint stays loadable by stock transformers.
    vision_config = dict(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=2,
        out_hidden_size=512,
        image_size=56,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        projection_intermediate_size=128,
        rms_norm_eps=1e-5,
        swiglu_limit=10.0,
    )

    return Glm5NextConfig(
        text_config=text_config,
        vision_config=vision_config,
        tie_word_embeddings=False,
    )


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _align_to(x: int, alignment: int) -> int:
    """``streamindex_topk.align_to`` (:35-36), without importing jax."""
    return _cdiv(x, alignment) * alignment


def print_kpool_geometry(index_topk: int, index_kpool: int, kv_lora_rank: int,
                         seq_len: int, block_size: int,
                         max_model_len: int) -> dict:
    """Print (and assert) the derived kpool geometry this fixture implies.

    Every number here is derived from the config the fixture just wrote, using
    the same expressions the device path uses. It is printed rather than
    discovered at trace time three rungs later, because the two ways this
    geometry goes wrong are both quiet:

    * a budget that is not actually a budget -- ``select_k`` at or above the
      candidate-pool count means the top-k is the identity and the fixture
      measures nothing (see the ``index_topk`` comment in ``build_config``);
    * a pooled-cache pagination that cannot be scored -- ``streamindex_topk``
      requires ``page_size * bkv_p % 128 == 0`` (:596-601) and derives
      ``num_sublanes_total`` from ``align_to(pages_per_seq, bkv_p)``
      (:603-605).

    Recorded finding for the shipped geometry -- 192-token fixture,
    ``block_size`` 16 (GLM is hybrid, so ``tpu_platform.py:1159-1163`` leaves
    vLLM's default), ``index_kpool`` 4: ``page_size`` is 4, which forces
    ``bkv_p`` to a multiple of 32; ``pages_per_seq`` 12 pads to 32 (37.5%
    utilisation); ``num_sublanes_total`` is 1, i.e. 128 score columns for 48
    real pools. **Viable.** The fixture does not need a ``block_size`` change
    on this ground, and raising it would repaginate the fixture away from the
    production geometry. What the fixture is *not* is representative of block
    utilisation -- hence the print rather than silence.

    One caveat for whoever writes the device-side assertion: the equality
    ``num_sublanes_total * 128 == ceil(max_model_len / index_kpool)`` holds
    only when ``max_model_len`` is a multiple of ``block_size * bkv_p`` (512
    here). It is true at ``max_model_len`` 2048 and false at 192, where it is
    128 vs 48. The safe form is ``>=``.

    Returns the derived dict so callers can record it.
    """
    select_k = index_topk // index_kpool
    # Identity regime: at query position p the reference selects
    # 4*min(select_k, floor((p+1)/4)) pool tokens plus the (p+1) % 4 tail
    # tokens, which is exactly {0..p} while p <= index_topk + index_kpool - 2.
    identity_len = index_topk + index_kpool - 1
    out_width = _align_to(index_topk + index_kpool - 1, 128)

    # A pool is a candidate only if it is *complete*, so a prompt of `seq_len`
    # offers floor(seq_len / index_kpool) of them at its last query position.
    candidate_pools = seq_len // index_kpool
    dropped = max(0, candidate_pools - select_k)

    # --- pooled indexer K cache pagination ------------------------------
    # `storage_block_size` = pooled records per page = block_size // ratio,
    # and `streamindex_topk` recovers `page_size = page_size_per_kv_packing *
    # kv_packing` from the cache shape (num_blocks, sbs // 4, 4, 256), so
    # page_size == storage_block_size.
    storage_block_size = block_size // index_kpool
    page_size = storage_block_size
    # Smallest bkv_p satisfying `page_size * bkv_p % 128 == 0`.
    min_bkv_p = 128 // math.gcd(128, page_size) if page_size else 0
    bkv_p = min_bkv_p
    pages_per_seq = _cdiv(max_model_len, block_size)
    padded_pages = _align_to(pages_per_seq, bkv_p) if bkv_p else 0
    num_sublanes_total = padded_pages * page_size // 128
    score_columns = num_sublanes_total * 128
    pools_in_context = _cdiv(max_model_len, index_kpool)

    print(f"select_k          {select_k} pools "
          f"({select_k * index_kpool} raw tokens)")
    print(f"identity bound    {identity_len} tokens "
          f"(dense causal MLA is exact at or below this)")
    print(f"out_width         {out_width} "
          f"(align_to({index_topk} + {index_kpool} - 1, 128))")
    sparse_ok = ("" if kv_lora_rank == _PRODUCTION_KV_LORA_RANK else
                 "  <-- sparse consumer asserts lkv_dim == 512; this fixture "
                 "cannot execute it, pass --mla-production-dims")
    print(f"MLA head_size     {kv_lora_rank + _SPARSE_ROPE_SLAB} "
          f"= kv_lora_rank {kv_lora_rank} + zero rope slab "
          f"{_SPARSE_ROPE_SLAB}{sparse_ok}")
    if dropped:
        print(f"selection         at seq_len {seq_len}: {candidate_pools} "
              f"candidate pools, {select_k} kept, {dropped} DROPPED "
              f"-- the top-k does real work")
    else:
        suggested = max(index_kpool, (candidate_pools // 2) * index_kpool)
        print(f"selection         at seq_len {seq_len}: {candidate_pools} "
              f"candidate pools <= budget {select_k}: top-k is the IDENTITY. "
              f"A kpool bug cannot be observed on this fixture; build with "
              f"--index-topk {suggested} or smaller to change that.")
    print(f"pooled pagination block_size {block_size} -> storage_block_size "
          f"{storage_block_size} = page_size; bkv_p must be a multiple of "
          f"{min_bkv_p}")
    print(f"                  max_model_len {max_model_len} -> pages_per_seq "
          f"{pages_per_seq}, padded to {padded_pages} at bkv_p {bkv_p} "
          f"(utilisation {pages_per_seq / padded_pages:.1%})")
    print(f"                  num_sublanes_total {num_sublanes_total} -> "
          f"{score_columns} score columns for {pools_in_context} pools")

    # These two hold *by construction* once `bkv_p` is a multiple of
    # `128 // gcd(128, page_size)`: `align_to` makes the padded page count a
    # multiple of bkv_p, so `padded_pages * page_size` is a multiple of 128 and
    # the `// 128` never truncates, and it is at least `pages_per_seq *
    # page_size >= max_model_len / index_kpool`. They are asserted anyway
    # because they are the regression witness on the derivation itself: pick
    # `bkv_p = 8` at `page_size = 4` -- the value DSv4's `(3,2,2)` is closest
    # to -- and `align_to(12, 8) * 4 // 128` floors to **0** score columns for
    # 48 real pools. Loud here, a `ValueError` at trace time three rungs later.
    assert page_size * bkv_p % 128 == 0, (
        f"bkv_sz ({page_size} * {bkv_p}) must be a multiple of 128 "
        f"(streamindex_topk.py:596-601)")
    assert padded_pages * page_size % 128 == 0, (
        f"align_to({pages_per_seq}, {bkv_p}) * {page_size} must be a multiple "
        f"of 128 or num_sublanes_total truncates "
        f"(streamindex_topk.py:603-605)")
    assert score_columns >= pools_in_context, (
        f"pooled score buffer holds {score_columns} columns but "
        f"max_model_len {max_model_len} needs {pools_in_context} pools")
    assert block_size % index_kpool == 0, (
        f"block_size {block_size} must be a multiple of index_kpool "
        f"{index_kpool} or the pooled cache's ratio-4 pagination is ragged")

    return dict(select_k=select_k,
                identity_len=identity_len,
                out_width=out_width,
                candidate_pools=candidate_pools,
                dropped_pools=dropped,
                storage_block_size=storage_block_size,
                min_bkv_p=min_bkv_p,
                pages_per_seq=pages_per_seq,
                padded_pages=padded_pages,
                num_sublanes_total=num_sublanes_total,
                score_columns=score_columns,
                pools_in_context=pools_in_context)


def _decontaminate_ties(model, gain: float) -> None:
    """Make the fixture's discrete decisions decisive rather than coin flips.

    A default-initialised random model is a bad correctness oracle for a bf16
    device, because both of its discrete choices sit on knife edges:

    * **The argmax over logits.** The top1-top2 margin had a median of 0.078
      against an end-to-end bf16 noise level of ~0.139, so injecting noise of
      that size into the *reference itself* left only ~52% of positions with
      an unchanged argmax. Greedy-token equality therefore could not
      distinguish a correct port from a broken one. Scaling ``lm_head`` fixes
      this: the margin becomes median 0.758, min 0.020.

    * **Expert selection.** The gap between the k-th and (k+1)-th expert score
      was *exactly zero* for 2 of 192 tokens, and under 2e-3 for several more.
      A tie is broken differently on device than on host -- both correct -- but
      KDA is recurrent over tokens, so one flipped token corrupts the state for
      every later one and the disagreement compounds layer over layer.

      Scaling the gate does not help: bigger logits saturate the sigmoid and
      push the top scores *together*, shrinking the margin. Instead give
      ``e_score_correction_bias`` a real spread. It is a genuine model
      parameter that a default init leaves at zero.

      A sweep over spreads, counting selection gaps below 2e-3 across all 7
      MoE layers and 192 tokens: +/-0.5 leaves 20, +/-1.0 leaves 2, and
      +/-2.0 leaves **none**, with a minimum margin of 5.0e-02 -- about 25x
      bf16 resolution -- while still letting the token-dependent gate decide
      the ordering.

    Affects only the synthetic fixture; real checkpoints are trained and
    already decisive.
    """
    n_experts = model.config.text_config.n_routed_experts
    with torch.no_grad():
        model.lm_head.weight.mul_(gain)
        spread = torch.linspace(-2.0, 2.0, n_experts)
        for i, layer in enumerate(model.model.language_model.layers):
            gate = getattr(layer.mlp, "gate", None)
            bias = getattr(gate, "e_score_correction_bias", None)
            if bias is None:
                continue
            # A different permutation per layer, so layers do not all favour
            # the same experts.
            perm = torch.randperm(n_experts,
                                  generator=torch.Generator().manual_seed(i))
            bias.copy_(spread[perm])


def add_nextn_layer(out_dir: pathlib.Path, num_layers: int,
                    layer_types: list[str], seed: int) -> int:
    """Append one synthetic nextn (MTP) layer to a saved fixture.

    ``transformers`` 5.16.1 does not model GLM's MTP layer at all -- it drops
    it on load with ``_keys_to_ignore_on_load_unexpected = [r"layers\\.45\\."]``
    -- so ``Glm5NextForConditionalGeneration`` cannot be asked to build one.
    The tensors are therefore synthesised here, from the fixture's own weights,
    the same way the real checkpoint is shaped: take a
    *full-attention* layer (the nextn layer is DSA/MLA, never KDA), strip the
    six ``hc_*`` tensors it must not have, and add the four wrapper tensors it
    must: ``enorm`` / ``hnorm`` / ``eh_proj`` / ``shared_head.norm``.

    The weights are random, so a draft head loaded from them predicts nothing
    useful -- acceptance rate on this fixture is expected to be near zero. What
    it does exercise is every piece of plumbing between the two: the draft
    architecture resolution, the name mapping, the KDA target's windowed
    verify step, and the state rollback. Acceptance is measured on the real
    checkpoint.

    Returns the index of the layer that was added.
    """
    from safetensors.torch import load_file, save_file

    torch.manual_seed(seed + 1)
    path = out_dir / "model.safetensors"
    shard = load_file(str(path))

    # Full attention, not KDA: layer 45's `self_attn` has q_a_proj/q_b_proj/
    # kv_a_proj_with_mqa/kv_b_proj/o_proj and not one KDA marker. Cloning a
    # `linear_attention` layer here would build a fixture that only the wrong
    # draft head can load.
    source = next(idx for idx, kind in enumerate(layer_types)
                  if kind != "linear_attention")
    src = f"model.language_model.layers.{source}."
    dst = f"model.language_model.layers.{num_layers}."

    added = {}
    for name, tensor in shard.items():
        if not name.startswith(src):
            continue
        tail = name[len(src):]
        if tail in _HC_NAMES:
            continue
        added[dst + tail] = tensor.clone()

    hidden = shard["model.language_model.norm.weight"].shape[0]
    dtype = shard["model.language_model.norm.weight"].dtype
    # `eh_proj` maps [enorm(embeds) ; hnorm(prev_hidden)] back to `hidden`.
    # The real checkpoint's second half is ~1.78*I (a residual pass-through of
    # the previous hidden state); nothing here depends on that, but keeping the
    # shape right is what makes the concat order a *testable* property rather
    # than a silent one.
    added[dst + "eh_proj.weight"] = torch.randn(
        hidden, 2 * hidden, dtype=dtype) * 0.02
    added[dst + "enorm.weight"] = torch.ones(hidden, dtype=dtype)
    added[dst + "hnorm.weight"] = torch.ones(hidden, dtype=dtype)
    added[dst + "shared_head.norm.weight"] = torch.ones(hidden, dtype=dtype)

    shard.update(added)
    save_file(shard, str(path), metadata={"format": "pt"})

    config_path = out_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"]["num_nextn_predict_layers"] = 1
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    print(f"nextn layer       {num_layers} "
          f"(cloned from layer {source}, {layer_types[source]}); "
          f"{len(added)} tensors")
    return num_layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decision-gain",
                        type=float,
                        default=8.0,
                        help="scale on lm_head and MoE gate so expert choice "
                        "and argmax are decisive against bf16 noise")
    parser.add_argument("--cycles",
                        type=int,
                        default=2,
                        help="number of [lin,lin,lin,dsa] repeats")
    parser.add_argument("--kda-heads",
                        type=int,
                        default=4,
                        help="KDA heads. Must be divisible by the TP size the "
                        "fixture will run at; the default 4 cannot be "
                        "sharded 8 ways, which is the only reason a "
                        "TP=8 fixture run needs its own build.")
    parser.add_argument("--attn-heads",
                        type=int,
                        default=4,
                        help="MLA / indexer heads; same divisibility rule")
    parser.add_argument("--nextn",
                        action="store_true",
                        help="also emit a synthetic MTP (nextn) layer at "
                        "index num_hidden_layers, so speculative "
                        "decoding can be exercised on the fixture")
    parser.add_argument("--index-topk",
                        type=int,
                        default=2048,
                        help="DSA indexer budget in raw tokens. Default 2048 "
                        "is the real checkpoint's value, at which the "
                        "top-k is the identity for every length this "
                        "fixture is run at -- so it cannot observe a "
                        "kpool selection bug. Use 64 (a 16-pool budget) "
                        "with --seq-len 192 downstream to drop two "
                        "thirds of the candidate pools. Such a fixture "
                        "*will* disagree with the current dense port; "
                        "that is the fixture working. Must be a "
                        "multiple of index_kpool=4.")
    parser.add_argument("--mla-production-dims",
                        action="store_true",
                        help=f"set kv_lora_rank to "
                        f"{_PRODUCTION_KV_LORA_RANK} instead of 64, "
                        f"keeping qk_nope/v at 256, so MLA head_size is "
                        f"{_PRODUCTION_KV_LORA_RANK}+"
                        f"{_SPARSE_ROPE_SLAB}="
                        f"{_PRODUCTION_KV_LORA_RANK + _SPARSE_ROPE_SLAB}."
                        f" Mandatory for anything that executes the "
                        f"sparse MLA consumer, which asserts lkv_dim == "
                        f"512 (attention_interface.py:805-807).")
    parser.add_argument("--geometry-seq-len",
                        type=int,
                        default=192,
                        help="prompt length used only to report whether the "
                        "top-k actually drops pools at this index_topk; "
                        "matches dump_reference.py --seq-len")
    parser.add_argument("--geometry-block-size",
                        type=int,
                        default=16,
                        help="vLLM KV block size the fixture will run at, "
                        "used only to report the pooled cache's derived "
                        "pagination. GLM is hybrid, so tpu_platform.py:"
                        "1159-1163 leaves this at vLLM's default 16.")
    parser.add_argument("--geometry-max-model-len",
                        type=int,
                        default=None,
                        help="max_model_len used only for the pagination "
                        "report; defaults to --geometry-seq-len")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    kv_lora_rank = (_PRODUCTION_KV_LORA_RANK
                    if args.mla_production_dims else 64)
    config = build_config(args.cycles, args.kda_heads, args.attn_heads,
                          args.index_topk, kv_lora_rank)
    text = config.text_config
    print(f"layers            {text.num_hidden_layers}")
    print(f"layer_types       {text.layer_types}")
    print(f"mlp_layer_types   {text.mlp_layer_types}")
    print(f"hidden_size       {text.hidden_size}")
    print(f"KDA               {text.linear_num_heads} heads x "
          f"{text.linear_head_dim}, conv {text.linear_conv_kernel_dim}")
    print(f"MLA               {text.num_attention_heads} heads, "
          f"qk_nope {text.qk_nope_head_dim}, v {text.v_head_dim}, "
          f"rope {text.qk_rope_head_dim}, "
          f"kv_lora_rank {text.kv_lora_rank}")
    print(f"indexer           {text.index_n_heads} x {text.index_head_dim}, "
          f"topk {text.index_topk}, kpool {text.index_kpool}")
    print(f"MoE               {text.n_routed_experts} experts, "
          f"top-{text.num_experts_per_tok}")
    print(f"MHC               mult {text.hc_mult}, "
          f"sinkhorn {text.hc_sinkhorn_iters}")
    geometry = print_kpool_geometry(
        index_topk=text.index_topk,
        index_kpool=text.index_kpool,
        kv_lora_rank=text.kv_lora_rank,
        seq_len=args.geometry_seq_len,
        block_size=args.geometry_block_size,
        max_model_len=(args.geometry_max_model_len
                       if args.geometry_max_model_len is not None else
                       args.geometry_seq_len),
    )

    # float32 so the reference dump is an exact oracle; the TPU side runs bf16
    # and is compared against this with an explicit tolerance.
    with torch.device("cpu"):
        model = Glm5NextForConditionalGeneration(config)
    model = model.to(torch.float32).eval()
    _decontaminate_ties(model, args.decision_gain)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters        {n_params/1e6:.1f}M")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)

    nextn_layer = None
    if args.nextn:
        nextn_layer = add_nextn_layer(out, text.num_hidden_layers,
                                      list(text.layer_types), args.seed)

    # `index_topk` / `kv_lora_rank` are recorded because a fixture built in
    # the identity regime and one built in the sparse regime are otherwise
    # indistinguishable at a glance, and a test that silently ran against the
    # wrong one would report exactly the false green this flag exists to end.
    (out / "fixture_meta.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "cycles": args.cycles,
                "num_parameters": n_params,
                "nextn_layer": nextn_layer,
                "index_topk": text.index_topk,
                "index_kpool": text.index_kpool,
                "kv_lora_rank": text.kv_lora_rank,
                "mla_production_dims": bool(args.mla_production_dims),
                "kpool_geometry": geometry,
                "built_by": "scripts/glm5_next/build_tiny_fixture.py",
            },
            indent=2,
        ) + "\n")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
