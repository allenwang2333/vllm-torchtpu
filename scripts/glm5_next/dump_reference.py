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
"""Dump a per-layer HF CPU reference ("oracle") for a glm5_next checkpoint.

Every later stage of the TPU port -- the KDA kernel parity tests, the ported
model's forward, the first on-device run -- diffs against this file. That turns
"the output is garbage" into "layer 7 diverges", which is the difference
between a bisect and a debugging session.

Captured per forward:
  * ``input_ids`` and the final ``logits``
  * ``embed`` and each decoder layer's output hidden states
  * for each KDA (``linear_attention``) layer: the module input, the conv-mixed
    ``query``/``key``/``value``, the forget gate ``g``, ``beta`` (post-sigmoid,
    as the HF module passes it to the kernel), the output gate, the kernel's
    ``core_attn_out``, the final recurrent state, and the module output
  * for each DSA (``deepseek_sparse_attention``) layer: module input and
    output, **and** the kpool indexer's own intermediates -- ``q``, ``k``,
    ``gate``, the per-head ``weights``, ``pool_keys``, ``index_scores``, the
    candidate mask, the selected pool ids and the final ``topk_indices``

Why the indexer intermediates are worth their bytes: a wrong causal bound
(``pool_end`` visible vs ``pool_start`` visible) and a wrong pooling softmax
(per-channel over 4 slots vs anything else) both present on device as "DSA
layer 3 diverges", and no amount of staring at layer outputs separates them.
``valid_candidates`` isolates the bound, ``pool_keys`` isolates the pooling,
``index_scores`` isolates the query scale, ``selected_pools`` isolates the
top-k, and ``topk_indices`` isolates the expansion and the tail. That is one
device job instead of three.

Usage:
    PYTHONPATH=$GLM5_NEXT_TRANSFORMERS_SRC python scripts/glm5_next/dump_reference.py \
        --model /path/to/glm5_next_tiny --out /path/to/oracle.npz

    # against a fixture built with --index-topk 64, where the top-k is not
    # the identity and the dump is required to prove it
    PYTHONPATH=$GLM5_NEXT_TRANSFORMERS_SRC python scripts/glm5_next/dump_reference.py \
        --model /path/to/glm5_next_tiny_kpool --out /path/to/oracle.npz \
        --seq-len 192 --assert-sparse
"""

import argparse
import pathlib

import numpy as np
import torch
from transformers.models.glm5_next import Glm5NextForConditionalGeneration
from transformers.models.glm5_next.modeling_glm5_next import \
    Glm5NextTextLinearAttention


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def _npi(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.int32).cpu().numpy()


def _npb(t: torch.Tensor) -> np.ndarray:
    """Booleans as uint8 -- ``np.savez`` keeps bools, consumers want a mask."""
    return t.detach().to(torch.uint8).cpu().numpy()


def _record_indexer(captured: dict, indexer, idx: int, taps: dict,
                    topk_indices: torch.Tensor) -> None:
    """Store one DSA layer's kpool intermediates into ``captured``.

    The score/selection recomputation below is a transcription of
    ``modeling_glm5_next.py:823-846`` operating on the *same tensor objects*
    the reference used, so it is bit-identical rather than merely close -- and
    the assertion at the end proves it against the reference's own return
    value.
    """
    prefix = f"dsa{idx:02d}.idx"
    head_dim = indexer.head_dim
    n_heads = indexer.n_heads
    kpool = indexer.index_kpool

    packed = taps["packed_states"]
    k, gate, valid = torch.split(packed, [head_dim, head_dim, 1], dim=-1)
    valid = valid.bool().squeeze(-1)
    batch, seq_len = k.shape[:2]
    # No padding in this dump: `input_ids` is dense, so the model synthesises
    # an all-ones mask (`modeling_glm5_next.py:1463-1469`) and `first_key` is
    # 0. The recomputation below therefore skips the reference's final
    # `masked_fill(~attention_mask)`; if a padded dump is ever added, that line
    # has to come back with it.
    assert bool(valid.all()), (
        "dump_reference assumes an unpadded prompt; the indexer's valid-key "
        "channel has zeros, so the recomputed selection would miss the "
        "reference's trailing attention-mask fill")

    q = taps["q_flat"].view(batch, seq_len, n_heads, head_dim)
    pool_keys = taps["pool_keys"]
    pool_indices = taps["pool_indices"]
    pool_valid = taps["pool_valid"]
    visible = taps["visible_tokens"]
    kv_len = visible.shape[-1]
    num_pools = pool_keys.shape[1]

    scores = torch.matmul(q.float(),
                          pool_keys.transpose(-1, -2).float().unsqueeze(1))
    scores = torch.relu(scores * indexer.softmax_scale)
    weights = taps["weights_raw"].float() * (n_heads**-0.5)
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

    # A pool is selectable only if it is complete *and* its LAST token is
    # visible -- `floor((p+1)/kpool)` candidates at query position p, not
    # `floor(p/kpool)+1`. This mask is the single thing that separates a wrong
    # `causal_row_offset` from every other kpool bug.
    pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
    pool_visible = visible.gather(dim=-1,
                                  index=pool_end[:, None, :].expand(
                                      batch, seq_len, -1))
    valid_candidates = pool_visible & pool_valid[:, None]
    masked_scores = index_scores.masked_fill(
        ~valid_candidates,
        torch.finfo(index_scores.dtype).min,
    )

    select_k = min(indexer.index_topk // kpool, num_pools)
    selected = masked_scores.topk(select_k, dim=-1).indices
    batch_idx = torch.arange(batch, device=selected.device)[:, None, None]
    selected_valid = valid_candidates.gather(-1, selected)
    selected_indices = pool_indices[batch_idx, selected]
    expanded = selected_indices.flatten(-2).masked_fill(
        ~selected_valid[..., None].expand_as(selected_indices).flatten(-2),
        -1,
    )

    width = expanded.shape[-1]
    if not torch.equal(expanded.to(torch.int32), topk_indices[..., :width].to(
            torch.int32)):
        bad = (expanded.to(torch.int32) != topk_indices[..., :width].to(
            torch.int32)).nonzero()[0].tolist()
        raise AssertionError(
            f"layer {idx}: the recomputed kpool selection disagrees with the "
            f"indexer's own output at {bad} -- transformers' scoring has "
            f"changed under this script, so every captured intermediate is "
            f"suspect. Re-derive them from "
            f"modeling_glm5_next.Glm5NextTextIndexer.forward before trusting "
            f"this oracle.")

    captured[f"{prefix}.q"] = _np(q)
    captured[f"{prefix}.k"] = _np(k)
    captured[f"{prefix}.gate"] = _np(gate)
    # Raw `weights_proj` output: the n_heads**-0.5 factor is applied above and
    # the TPU path folds it (with `softmax_scale` and `q_scale`) into the
    # per-head weights instead, so the unscaled tensor is the portable one.
    captured[f"{prefix}.weights_raw"] = _np(taps["weights_raw"])
    captured[f"{prefix}.pool_keys"] = _np(pool_keys)
    captured[f"{prefix}.pool_indices"] = _npi(pool_indices)
    captured[f"{prefix}.pool_valid"] = _npb(pool_valid)
    captured[f"{prefix}.visible_tokens"] = _npb(visible)
    captured[f"{prefix}.index_scores"] = _np(index_scores)
    captured[f"{prefix}.valid_candidates"] = _npb(valid_candidates)
    captured[f"{prefix}.selected_pools"] = _npi(selected)
    captured[f"{prefix}.topk_indices"] = _npi(topk_indices)
    captured[f"{prefix}.w.ape"] = _np(indexer.index_kpool_compress_ape)
    captured[f"{prefix}.w.gate_proj"] = _np(indexer.index_kpool_compress_gate)
    captured[f"{prefix}.w.k_norm_weight"] = _np(indexer.k_norm.weight)
    captured[f"{prefix}.w.k_norm_bias"] = _np(indexer.k_norm.bias)


def _summarize_indexer(captured: dict, dsa_indices: list[int], index_topk: int,
                       index_kpool: int, assert_sparse: bool) -> None:
    """Print, and check, whether this oracle is in the identity regime.

    Two things are asserted here and neither is a restatement of the code
    above:

    * **The identity theorem.** For every query position `p` at or below
      `index_topk + index_kpool - 2` the selected raw ids must be exactly
      `{0..p}`. That is the claim the shipped 2048 cap rests on
      (`model.py:196`), checked against the actual HF reference rather than
      argued. If it ever fails, the cap is wrong, not conservative.
    * **Sparsity, on request.** ``--assert-sparse`` fails the dump when the
      last query position still selects every causal token, i.e. when the
      oracle cannot distinguish a correct kpool port from a dense one.
    """
    identity_bound = index_topk + index_kpool - 2
    any_sparse = False
    for idx in dsa_indices:
        key = f"dsa{idx:02d}.idx.topk_indices"
        if key not in captured:
            continue
        rows = captured[key][0]
        seq_len = rows.shape[0]
        counts = (rows >= 0).sum(axis=-1)
        last = int(counts[-1])
        sparse = last < seq_len
        any_sparse = any_sparse or sparse
        for p in range(seq_len):
            if p > identity_bound:
                break
            ids = np.sort(rows[p][rows[p] >= 0])
            expected = np.arange(p + 1, dtype=ids.dtype)
            if ids.shape != expected.shape or not np.array_equal(
                    ids, expected):
                raise AssertionError(
                    f"layer {idx}, position {p} is inside the identity regime "
                    f"(p <= {identity_bound}) but the reference selected "
                    f"{ids.shape[0]} ids, not {p + 1}. The dense-MLA "
                    f"equivalence the 2048 cap relies on does not hold.")
        verdict = ("SPARSE"
                   if sparse else "IDENTITY (a kpool bug is invisible here)")
        print(f"  dsa{idx:02d}: row width {rows.shape[-1]}, "
              f"non-(-1) entries min {int(counts.min())} "
              f"max {int(counts.max())}, at the last position "
              f"{last}/{seq_len} -> {verdict}")
        if sparse:
            # Small enough to check by hand, which is the whole point of
            # running at index_topk 64: `sorted(pool_ids)` must be a subset of
            # `range(seq_len // index_kpool)` and its highest member must
            # satisfy `index_kpool * pool + index_kpool - 1 <= p`.
            pools = np.sort(captured[f"dsa{idx:02d}.idx.selected_pools"][0,
                                                                         -1])
            print(f"          selected pools at p={seq_len - 1}: "
                  f"{pools.tolist()}")

    if assert_sparse and not any_sparse:
        raise SystemExit(
            "--assert-sparse: every DSA layer selected all causal tokens at "
            f"the last position. index_topk={index_topk} gives a budget of "
            f"{index_topk // index_kpool} pools = {index_topk} tokens, which "
            "is not below this sequence length. Rebuild the fixture with "
            "build_tiny_fixture.py --index-topk 64 (or lower the budget "
            "another way) -- an oracle in the identity regime measures "
            "dense attention, not kpool.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len",
                        type=int,
                        default=192,
                        help="must span >1 chunk of 64 to exercise the "
                        "chunked kernel's cross-tile state carry")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy-steps", type=int, default=8)
    parser.add_argument("--assert-sparse",
                        action="store_true",
                        help="fail unless the kpool top-k actually drops "
                        "candidate pools at the last query position. Use "
                        "it on a fixture built with --index-topk 64: it "
                        "is the one assertion that proves this oracle "
                        "can see a kpool bug at all.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = Glm5NextForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32)
    model.eval()
    text_config = model.config.text_config

    input_ids = torch.randint(0,
                              text_config.vocab_size, (1, args.seq_len),
                              dtype=torch.long)

    captured: dict[str, np.ndarray] = {}
    handles = []

    text_model = model.model.language_model

    # --- decoder layer boundaries -------------------------------------------
    for idx, layer in enumerate(text_model.layers):

        def layer_hook(_mod, _inp, out, i=idx):
            h = out[0] if isinstance(out, tuple) else out
            captured[f"layer{i:02d}.out"] = _np(h)

        handles.append(layer.register_forward_hook(layer_hook))

        # The FFN boundary, so a device run can separate "the attention is
        # wrong" from "the MoE is wrong".
        def mlp_hook(_mod, inp, out, i=idx):
            if inp:
                captured[f"layer{i:02d}.ffn_in"] = _np(inp[0])
            captured[f"layer{i:02d}.ffn_out"] = _np(
                out[0] if isinstance(out, tuple) else out)

        handles.append(layer.mlp.register_forward_hook(mlp_hook))

        # Router logits and the shared expert, to attribute an MoE divergence
        # to the router, the shared expert, or the routed experts.
        if hasattr(layer.mlp, "gate"):

            def gate_hook(_mod, inp, out, i=idx):
                captured[f"layer{i:02d}.router_logits"] = _np(
                    out[0] if isinstance(out, tuple) else out)

            handles.append(layer.mlp.gate.register_forward_hook(gate_hook))
        if getattr(layer.mlp, "shared_experts", None) is not None:

            def shared_hook(_mod, inp, out, i=idx):
                captured[f"layer{i:02d}.shared_out"] = _np(
                    out[0] if isinstance(out, tuple) else out)

            handles.append(
                layer.mlp.shared_experts.register_forward_hook(shared_hook))

    # --- KDA internals ------------------------------------------------------
    # Wrap the module forward rather than hooking submodules: the quantities
    # that matter (post-conv q/k/v, the kernel's core_attn_out) are locals, not
    # submodule outputs.
    def wrap_kda(mod: Glm5NextTextLinearAttention, idx: int):
        orig = mod.forward

        def patched(hidden_states, *a, **kw):
            captured[f"kda{idx:02d}.hidden_in"] = _np(hidden_states)

            qkv_taps = {}

            def tap(name):

                def hook(_m, _i, out):
                    qkv_taps[name] = out

                return hook

            local = [
                mod.q_proj.register_forward_hook(tap("q")),
                mod.k_proj.register_forward_hook(tap("k")),
                mod.v_proj.register_forward_hook(tap("v")),
                mod.forget_gate.register_forward_hook(tap("g")),
                # The op takes the *raw* f_b output and applies the decay
                # activation itself, so capture both sides of that boundary.
                mod.forget_gate.f_b_proj.register_forward_hook(tap("f_b_out")),
                mod.b_proj.register_forward_hook(tap("beta_raw")),
                mod.g_b_proj.register_forward_hook(tap("out_gate")),
                mod.o_norm.register_forward_hook(tap("o_norm_out")),
            ]

            # o_norm's first input is core_attn_out straight off the kernel.
            def o_norm_pre(_m, inp):
                qkv_taps["core_attn_out"] = inp[0]

            local.append(mod.o_norm.register_forward_pre_hook(o_norm_pre))

            try:
                out = orig(hidden_states, *a, **kw)
            finally:
                for h in local:
                    h.remove()

            for k, v in qkv_taps.items():
                captured[f"kda{idx:02d}.{k}"] = _np(v)
            # beta as the kernel actually receives it
            captured[f"kda{idx:02d}.beta"] = _np(
                torch.sigmoid(qkv_taps["beta_raw"]))
            captured[f"kda{idx:02d}.out"] = _np(out)
            return out

        mod.forward = patched

    # --- DSA kpool indexer internals ----------------------------------------
    def wrap_indexer(attn, idx: int) -> bool:
        """Capture the kpool indexer's intermediates for one DSA layer.

        Everything the indexer computes is a local inside a single ``forward``
        (`modeling_glm5_next.py:773-874`), so there is nothing to hook. Two
        wrappers get almost all of it for free without restating any reference
        math:

        * ``get_pooled_states`` receives ``packed_states = cat([k, gate,
          valid])`` and returns ``(pool_keys, pool_indices, pool_valid)``, so
          wrapping it yields both the pooling inputs and its outputs;
        * ``get_visible_tokens`` returns the causal/padding mask the pool
          candidacy rule is built from.

        Only ``index_scores`` and the selected pool ids are genuinely
        unreachable, and those are recomputed here from the captured tensors
        using the reference's own lines. That recomputation is **checked, not
        trusted**: the selected pools are expanded and asserted equal to the
        pool region of the ``topk_indices`` the reference actually returned.
        If transformers ever changes the scoring, this dump fails loudly
        instead of writing an oracle that quietly disagrees with the model it
        claims to describe.
        """
        indexer = getattr(attn, "indexer", None)
        if indexer is None:
            # `indexer_types[idx] == "shared"`: this layer reuses the previous
            # full layer's selection. The real checkpoint is ["full"] * 45, so
            # this branch should never fire; say so rather than skip silently.
            print(f"  layer {idx}: no indexer (indexer_types is 'shared') -- "
                  "no kpool intermediates captured")
            return False

        taps: dict[str, torch.Tensor] = {}
        orig_forward = indexer.forward
        orig_pooled = indexer.get_pooled_states
        orig_visible = indexer.get_visible_tokens

        def patched_pooled(packed_states, *a, **kw):
            out = orig_pooled(packed_states, *a, **kw)
            taps["packed_states"] = packed_states
            taps["pool_keys"], taps["pool_indices"], taps["pool_valid"] = out
            return out

        def patched_visible(*a, **kw):
            out = orig_visible(*a, **kw)
            taps["visible_tokens"] = out
            return out

        def patched_forward(*a, **kw):
            local = [
                indexer.wq_b.register_forward_hook(
                    lambda _m, _i, out: taps.__setitem__("q_flat", out)),
                indexer.weights_proj.register_forward_hook(
                    lambda _m, _i, out: taps.__setitem__("weights_raw", out)),
            ]
            indexer.get_pooled_states = patched_pooled
            indexer.get_visible_tokens = patched_visible
            try:
                topk_indices = orig_forward(*a, **kw)
            finally:
                for h in local:
                    h.remove()
                del indexer.get_pooled_states
                del indexer.get_visible_tokens

            _record_indexer(captured, indexer, idx, taps, topk_indices)
            return topk_indices

        indexer.forward = patched_forward
        return True

    kda_indices, dsa_indices, wrapped_indexers = [], [], []
    for idx, layer in enumerate(text_model.layers):
        attn = layer.self_attn
        if isinstance(attn, Glm5NextTextLinearAttention):
            kda_indices.append(idx)
            wrap_kda(attn, idx)
        else:
            dsa_indices.append(idx)

            # The decoder layer calls `self_attn` with keyword arguments, so
            # `args` is empty; ask for kwargs too.
            def dsa_hook(_mod, args, kwargs, out, i=idx):
                h_in = args[0] if args else kwargs.get("hidden_states")
                if h_in is not None:
                    captured[f"dsa{i:02d}.hidden_in"] = _np(h_in)
                h = out[0] if isinstance(out, tuple) else out
                captured[f"dsa{i:02d}.out"] = _np(h)

            handles.append(
                attn.register_forward_hook(dsa_hook, with_kwargs=True))
            if wrap_indexer(attn, idx):
                wrapped_indexers.append(idx)

    print(f"KDA layers: {kda_indices}")
    print(f"DSA layers: {dsa_indices}")

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    # Detach every capture point before anything else runs the model. The
    # greedy rollout below feeds progressively longer sequences, and leaving
    # these live would overwrite each tensor with the last (longer) step --
    # silently changing the captured sequence length out from under the
    # consumer.
    for h in handles:
        h.remove()
    for idx in kda_indices:
        attn = text_model.layers[idx].self_attn
        if hasattr(attn, "__dict__") and "forward" in attn.__dict__:
            del attn.__dict__["forward"]
    # Same reason as the KDA unwrap above: the greedy rollout below runs the
    # model on progressively longer sequences, and a live indexer wrapper would
    # overwrite every captured intermediate with the last (longer) step.
    for idx in wrapped_indexers:
        indexer = text_model.layers[idx].self_attn.indexer
        indexer.__dict__.pop("forward", None)

    if wrapped_indexers:
        print(f"kpool indexer intermediates captured for layers "
              f"{wrapped_indexers}")
        _summarize_indexer(captured, wrapped_indexers,
                           int(text_config.index_topk),
                           int(text_config.index_kpool), args.assert_sparse)

    captured["input_ids"] = input_ids.cpu().numpy()
    captured["logits"] = _np(out.logits)
    # The embedding itself, to separate "the input is wrong" from "the mHC is
    # wrong" when a device run disagrees at layer 0.
    with torch.no_grad():
        captured["embed.out"] = _np(text_model.embed_tokens(input_ids))
    captured["kda_layers"] = np.asarray(kda_indices, dtype=np.int32)
    captured["dsa_layers"] = np.asarray(dsa_indices, dtype=np.int32)

    # Per-KDA-layer weights, so the parity test needs only this one file.
    # `conv1d.weight` is [3*H*D, 1, K] over the concatenated q,k,v channels;
    # the TPU op wants it as a single fused [K, 3, H, D] buffer.
    for idx in kda_indices:
        attn = text_model.layers[idx].self_attn
        captured[f"kda{idx:02d}.w.conv1d"] = _np(attn.conv1d.weight)
        captured[f"kda{idx:02d}.w.A_log"] = _np(attn.forget_gate.A_log)
        captured[f"kda{idx:02d}.w.dt_bias"] = _np(attn.forget_gate.dt_bias)
        captured[f"kda{idx:02d}.w.o_norm"] = _np(attn.o_norm.weight)

    # A greedy continuation, so a device run can be checked token-for-token
    # without depending on `prompt_logprobs` -- which the TPU path returns only
    # one entry of, whatever the prompt length.
    greedy = []
    ids = input_ids
    with torch.no_grad():
        for _ in range(args.greedy_steps):
            step_logits = model(input_ids=ids, use_cache=False).logits
            nxt = int(torch.argmax(step_logits[0, -1]))
            greedy.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long)],
                            dim=1)
    captured["greedy_tokens"] = np.asarray(greedy, dtype=np.int32)
    print(f"greedy continuation: {greedy}")

    for idx, layer in enumerate(text_model.layers):
        captured[f"layer{idx:02d}.attn_norm_w"] = _np(
            layer.input_layernorm.weight)
        captured[f"layer{idx:02d}.ffn_norm_w"] = _np(
            layer.post_attention_layernorm.weight)

    # So a device run can turn its own final hidden state into logits and be
    # scored on all prompt positions, not just the generated tail.
    captured["lm_head.weight"] = _np(model.lm_head.weight)

    captured["cfg.linear_num_heads"] = np.int32(text_config.linear_num_heads)
    captured["cfg.linear_head_dim"] = np.int32(text_config.linear_head_dim)
    captured["cfg.conv_kernel"] = np.int32(text_config.linear_conv_kernel_dim)
    captured["cfg.lower_bound"] = np.float32(text_config.linear_lower_bound)
    captured["cfg.rms_norm_eps"] = np.float32(text_config.rms_norm_eps)

    # kpool geometry, so a consumer can tell which regime this oracle was
    # produced in instead of assuming the production one. An oracle built at
    # index_topk 2048 and one built at 64 look identical apart from these.
    captured["cfg.index_topk"] = np.int32(text_config.index_topk)
    captured["cfg.index_kpool"] = np.int32(text_config.index_kpool)
    captured["cfg.index_n_heads"] = np.int32(text_config.index_n_heads)
    captured["cfg.index_head_dim"] = np.int32(text_config.index_head_dim)
    captured["cfg.kv_lora_rank"] = np.int32(text_config.kv_lora_rank)
    captured["cfg.index_kpool_always_select_tail"] = np.int32(
        bool(text_config.index_kpool_always_select_tail))
    captured["cfg.select_k"] = np.int32(text_config.index_topk //
                                        text_config.index_kpool)
    # Two names because they are two different quantities, and writing one
    # under the other's name is the count-versus-index confusion this whole
    # area keeps producing. ``identity_len`` is a
    # token COUNT -- the largest prompt for which dense causal MLA is bit-exact
    # (2051). ``identity_max_position`` is a 0-based query POSITION (2050), the
    # last p for which ``set(row[p]) == range(p+1)`` holds, and it is what
    # ``_summarize_indexer`` above and ``KpoolParams.identity_bound`` both
    # mean. A step-16 consumer that ranges over the count walks one position
    # past the regime, gets a legitimately sparse row, and reads it as a kpool
    # bug.
    identity_len = int(text_config.index_topk + text_config.index_kpool - 1)
    captured["cfg.identity_len"] = np.int32(identity_len)
    captured["cfg.identity_max_position"] = np.int32(identity_len - 1)

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, **captured)

    print(f"\nlogits {captured['logits'].shape} "
          f"mean={captured['logits'].mean():.6f} "
          f"std={captured['logits'].std():.6f}")
    print(f"captured {len(captured)} tensors -> {outp}")
    for k in sorted(captured):
        if k.startswith(("kda00.", "dsa03.")) or k in ("logits", ):
            print(f"  {k:28s} {captured[k].shape}")


if __name__ == "__main__":
    main()
