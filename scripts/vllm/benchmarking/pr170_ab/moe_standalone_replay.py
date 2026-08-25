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
"""Standalone JAX replay of the one-hot MoE permute divergence.

Reproduces, outside the vLLM server, the accuracy collapse traced to
ONEHOT_MOE_PERMUTE_THRESHOLD=32768 on Qwen3.5-397B-A17B-FP8: at decode
num_tokens=32 the one-hot permute/unpermute path in
``layers/common/fused_moe_gmm.py`` produces NaN-grade logits while the
SparseCore ragged-gather path does not.

The two paths are selected purely by the static ``onehot_moe_permute_threshold``
argument of ``fused_moe_func``:

  * threshold=0      -> ragged_gather_v2 / ragged_gather_reduce_v2 (gather)
  * threshold=32768  -> jax.nn.one_hot + matmul                    (one-hot)

both under ``use_ep=True, use_sparse_core=True``. Everything else -- weights,
routing, dtypes -- is held identical, so any divergence between them is the
compiled one-hot path.

Expert parallelism here needs no device mesh: ``fused_moe_func`` is a plain
``jax.jit`` with no shard_map and no cross-device collectives. EP is expressed
entirely by giving each rank a *slice* of the experts plus a traced
``experts_start`` offset, which the function subtracts to remap global expert
ids to local ones and masks the rest. This script therefore emulates EP rank 0
of an 8-way split on a single device.

For each num_tokens it reports:
  a_vs_b  gather (jitted) vs one-hot (jitted)     -- the suspected divergence
  b_vs_c  one-hot (jitted) vs one-hot (unjitted)  -- compiled vs eager truth

and, for any shape where the one-hot path diverges or goes non-finite, splits
the computation at its natural seams (permute -> gmm1 -> gmm2 -> unpermute) to
localize the first bad tensor.

Run inside the CI docker image on a tpu7x-8 agent. Verdicts are data, not
failures: the process exits 0 whenever it ran, and 1 only on infrastructure
errors (imports, no devices).
"""

import os
import sys
import traceback

# HF cache is mounted read-only in CI; never hit the network for the config.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_ID = "Qwen/Qwen3.5-397B-A17B-FP8"

# TP=8 with --enable-expert-parallel: experts are split 8 ways, and this
# process replays rank 0's slice.
EP_SIZE = 8
EP_RANK = 0

# Decode shapes to sweep. 32 is the shape the server garbles.
NUM_TOKENS_SWEEP = (16, 32, 64, 128)

# fp8 weights make exact equality meaningless; 1e-2 relative is the
# "same answer" band, far tighter than the NaN-grade failure we are hunting.
RTOL = 1e-2

# Fallback MoE dims if the config is unreadable (offline cache miss).
FALLBACK_DIMS = {
    "hidden_size": 4096,
    "moe_intermediate_size": 1536,
    "num_experts": 128,
    "topk": 8,
}


class Section:
    """Run a block, report and swallow any exception.

    One bad seam must not take down the rest of the report.
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            print(
                f"=== REPLAY SECTION ERROR name={self.name} "
                f"exc={exc_type.__name__}: {exc} ===",
                flush=True)
            traceback.print_exception(exc_type, exc, tb)
            return True
        return False


def _first_attr(obj, names):
    """Return (name, value) for the first attribute present and truthy."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return name, value
    return None, None


def load_moe_dims():
    """Read MoE dims from the HF config, reporting exactly what it found.

    Qwen3.5 may nest the language config under ``text_config`` (the server runs
    with --language-model-only), so search the top level and the nested configs
    and use whichever actually carries ``hidden_size``.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    candidates = [("<top>", cfg)]
    for nested in ("text_config", "llm_config", "language_config"):
        sub = getattr(cfg, nested, None)
        if sub is not None:
            candidates.append((nested, sub))

    for where, node in candidates:
        keys = sorted(k for k in vars(node) if not k.startswith("_"))
        print(f"[config] {where}: {keys}", flush=True)

    for where, node in candidates:
        if getattr(node, "hidden_size", None) is None:
            continue
        hidden_name, hidden = "hidden_size", node.hidden_size
        inter_name, inter = _first_attr(
            node, ("moe_intermediate_size", "intermediate_size"))
        experts_name, experts = _first_attr(
            node, ("num_experts", "n_routed_experts", "num_local_experts",
                   "moe_num_experts"))
        topk_name, topk = _first_attr(node, ("num_experts_per_tok", "moe_topk",
                                             "num_experts_per_token", "top_k"))
        if None in (inter, experts, topk):
            continue
        print(
            f"[config] using {where}: {hidden_name}={hidden} "
            f"{inter_name}={inter} {experts_name}={experts} "
            f"{topk_name}={topk}",
            flush=True)
        return {
            "hidden_size": int(hidden),
            "moe_intermediate_size": int(inter),
            "num_experts": int(experts),
            "topk": int(topk),
            "source": f"config:{where}",
        }

    raise ValueError("no config node carried all four MoE dims")


def align128(value):
    return (value + 127) // 128 * 128


def quantize_kmajor_fp8(w_float, jnp, quantize_tensor):
    """Quantize a checkpoint-layout MoE weight the way serving does.

    Serving requantizes MoE weights to float8_e4m3fn with
    MOE_REQUANTIZE_BLOCK_SIZE unset, i.e. one scale per (expert, output row)
    taken over the whole contracting axis -- see
    ``_quantize_and_format_single_moe_weight`` in
    ``layers/vllm/quantization/fp8.py``. Input is ``[E, N_out, K_in]``; output
    is gmm_v2's K-major ``[E, K, N]`` plus its fp32 scale ``[E, 1, 1, N]``.
    """
    w_q, scale = quantize_tensor(w_float,
                                 jnp.float8_e4m3fn,
                                 dim=-1,
                                 block_size=None)
    w_kmajor = jnp.swapaxes(w_q, 1, 2)
    scale = jnp.swapaxes(scale.astype(jnp.float32), 1, 2)
    return w_kmajor, jnp.expand_dims(scale, 2)


def build_weights(jax, jnp, quantize_tensor, dims, local_experts):
    """One MoE layer's worth of random fp8 weights for this EP rank."""
    hidden = dims["hidden_size"]
    inter = align128(dims["moe_intermediate_size"])
    key = jax.random.PRNGKey(0)
    k13, k2 = jax.random.split(key)

    # w13 is [E, 2*I, H] in checkpoint layout (gate and up stacked).
    w13_f = jax.random.normal(k13, (local_experts, 2 * inter, hidden),
                              dtype=jnp.float32) * 0.02
    w2_f = jax.random.normal(k2, (local_experts, hidden, inter),
                             dtype=jnp.float32) * 0.02

    w13, w13_scale = quantize_kmajor_fp8(w13_f, jnp, quantize_tensor)
    w2, w2_scale = quantize_kmajor_fp8(w2_f, jnp, quantize_tensor)
    del w13_f, w2_f
    return w13, w13_scale, w2, w2_scale


def build_inputs(jax, jnp, num_tokens, dims):
    """Realistic router output: softmax probs, renormalized top-k.

    Expert ids are GLOBAL, so most of them fall outside this rank's slice and
    get masked -- the same sparse valid_mask the server sees under EP.
    """
    key = jax.random.PRNGKey(1234 + num_tokens)
    k_hidden, k_router = jax.random.split(key)
    hidden = jax.random.normal(k_hidden, (num_tokens, dims["hidden_size"]),
                               dtype=jnp.float32) * 0.5
    logits = jax.random.normal(k_router, (num_tokens, dims["num_experts"]),
                               dtype=jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    weights, ids = jax.lax.top_k(probs, dims["topk"])
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return (hidden.astype(jnp.bfloat16), weights.astype(jnp.bfloat16),
            ids.astype(jnp.int32))


def tensor_stats(jnp, arr):
    finite = jnp.isfinite(arr)
    bad = int(arr.size - jnp.sum(finite))
    magnitude = float(jnp.max(jnp.abs(jnp.where(finite, arr, 0)))) if arr.size \
        else 0.0
    return bad, magnitude


def compare(jnp, ref, cand):
    """Return (maxdiff, meandiff, bad_ref, bad_cand, verdict)."""
    ref32 = ref.astype(jnp.float32)
    cand32 = cand.astype(jnp.float32)
    bad_ref, scale = tensor_stats(jnp, ref32)
    bad_cand, _ = tensor_stats(jnp, cand32)
    diff = jnp.abs(ref32 - cand32)
    # NaN-vs-NaN would silently vanish from max(); count it separately above.
    finite_diff = jnp.where(jnp.isfinite(diff), diff, 0.0)
    maxdiff = float(jnp.max(finite_diff)) if diff.size else 0.0
    meandiff = float(jnp.mean(finite_diff)) if diff.size else 0.0
    rel = maxdiff / (scale + 1e-6)
    verdict = "PASS" if (bad_ref == 0 and bad_cand == 0
                         and rel <= RTOL) else "FAIL"
    return maxdiff, meandiff, bad_ref, bad_cand, verdict


def run_path(fused_moe_func, jnp, weights, inputs, dims, threshold,
             experts_start):
    w13, w13_scale, w2, w2_scale = weights
    hidden, topk_weights, topk_ids = inputs
    return fused_moe_func(
        hidden,
        w13,
        w2,
        w13_scale,
        w2_scale,
        None,
        None,
        topk_weights,
        topk_ids,
        experts_start,
        topk=dims["topk"],
        activation="silu",
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=threshold,
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )


def seam_report(jax, jnp, mods, weights, inputs, dims, threshold,
                experts_start, num_tokens):
    """Re-run the computation stage by stage and report each seam.

    Mirrors ``fused_moe_func`` / ``moe_gmm`` verbatim (the internals are not
    separately callable), so the returned dict names the first tensor that
    goes bad: permute -> gmm1 -> gmm2 -> unpermute.
    """
    prepare_routed_gmm_inputs = mods["prepare_routed_gmm_inputs"]
    gmm_wrapper = mods["gmm_wrapper"]
    get_packing_factor = mods["get_packing_factor"]

    w1, w1_scale, w2, w2_scale = weights
    hidden_states, topk_weights, topk_ids = inputs
    topk = dims["topk"]

    def body(hidden_states, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids,
             experts_start):
        out = {}
        num_tokens_local, hidden_size = hidden_states.shape
        _, padded_hidden_size, _ = w1.shape
        padded_hidden_size *= get_packing_factor(w1.dtype, None)

        topk_weights = topk_weights.astype(hidden_states.dtype)
        local_ids = topk_ids - experts_start
        valid = (local_ids >= 0) & (local_ids < w1.shape[0])
        topk_weights = jnp.where(valid, topk_weights,
                                 jnp.zeros_like(topk_weights))
        topk_ids = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))

        (x, group_sizes, argsort_revert_indices, topk_weights_flat,
         valid_mask) = prepare_routed_gmm_inputs(
             hidden_states,
             topk_ids,
             topk_weights,
             local_num_experts=w1.shape[0],
             topk=topk,
             use_ep=True,
             use_sparse_core=True,
             onehot_moe_permute_threshold=threshold,
             skip_padded_tokens=False,
         )
        out["permute"] = x
        x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))

        group_offset = jnp.array([0], dtype=jnp.int32)
        gmm1_res = gmm_wrapper(x,
                               w1,
                               w1_scale,
                               None,
                               group_sizes,
                               group_offset,
                               zero_initialize=False,
                               fuse_act="silu",
                               rhs_quant_dtype=None)
        packing_factor = get_packing_factor(w2.dtype, None)
        gmm1_res = gmm1_res[:, :w2.shape[1] * packing_factor]
        out["gmm1"] = gmm1_res

        gmm2_res = gmm_wrapper(gmm1_res,
                               w2,
                               w2_scale,
                               None,
                               group_sizes,
                               group_offset,
                               zero_initialize=False,
                               preferred_element_type=x.dtype,
                               rhs_quant_dtype=None)
        out["gmm2"] = gmm2_res

        topk_weights_2d = topk_weights_flat.reshape((num_tokens_local, topk))
        valid_mask_2d = valid_mask.reshape((num_tokens_local, topk))
        if argsort_revert_indices.size <= threshold:
            revert_indices = argsort_revert_indices.reshape(
                num_tokens_local, topk)
            onehot = jax.nn.one_hot(revert_indices,
                                    argsort_revert_indices.size,
                                    dtype=gmm2_res.dtype)
            combine = (onehot * topk_weights_2d[..., None] *
                       valid_mask_2d[..., None]).sum(axis=1)
            unpermute = (combine @ gmm2_res).astype(hidden_states.dtype)
        else:
            token_hidden = gmm2_res[argsort_revert_indices]
            token_topk_hidden = token_hidden.reshape(
                (num_tokens_local, topk, gmm2_res.shape[-1]))
            token_topk_hidden = token_topk_hidden * jnp.expand_dims(
                topk_weights_2d, axis=-1)
            token_topk_hidden = jnp.where(valid_mask_2d[:, :, None],
                                          token_topk_hidden, 0)
            unpermute = token_topk_hidden.sum(axis=1).astype(
                hidden_states.dtype)
        out["unpermute"] = unpermute[:num_tokens_local, :hidden_size]
        return out

    seams = jax.jit(body)(hidden_states, w1, w1_scale, w2, w2_scale,
                          topk_weights, topk_ids, experts_start)
    for name in ("permute", "gmm1", "gmm2", "unpermute"):
        arr = seams.get(name)
        if arr is None:
            continue
        bad, magnitude = tensor_stats(jnp, arr.astype(jnp.float32))
        print(
            f"=== REPLAY SEAM num_tokens={num_tokens} seam={name} "
            f"threshold={threshold} shape={tuple(arr.shape)} "
            f"maxabs={magnitude:.6g} nan={bad} ===",
            flush=True)
    return seams


def main():
    try:
        import jax
        from jax import numpy as jnp
    except Exception as exc:  # noqa: BLE001 - infrastructure failure
        print(f"=== REPLAY INFRA ERROR importing jax: {exc} ===", flush=True)
        return 1

    try:
        devices = jax.devices()
    except Exception as exc:  # noqa: BLE001 - infrastructure failure
        print(f"=== REPLAY INFRA ERROR no devices: {exc} ===", flush=True)
        return 1

    print(
        f"=== REPLAY START jax={jax.__version__} devices={len(devices)} "
        f"kind={devices[0].device_kind if devices else 'none'} ===",
        flush=True)

    try:
        from vllm_torchtpu.kernels.megablox.gmm_v2 import get_packing_factor
        from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor
        from vllm_torchtpu.layers.common.fused_moe_gmm import (
            fused_moe_func, gmm_wrapper, prepare_routed_gmm_inputs)
    except Exception as exc:  # noqa: BLE001 - infrastructure failure
        print(f"=== REPLAY INFRA ERROR importing vllm_torchtpu: {exc} ===",
              flush=True)
        traceback.print_exc()
        return 1

    mods = {
        "prepare_routed_gmm_inputs": prepare_routed_gmm_inputs,
        "gmm_wrapper": gmm_wrapper,
        "get_packing_factor": get_packing_factor,
    }

    dims = dict(FALLBACK_DIMS, source="fallback")
    with Section("load_config"):
        dims = load_moe_dims()
    print(
        f"=== REPLAY CONFIG source={dims['source']} "
        f"hidden={dims['hidden_size']} "
        f"inter={dims['moe_intermediate_size']} "
        f"experts={dims['num_experts']} topk={dims['topk']} ===",
        flush=True)

    local_experts = max(1, dims["num_experts"] // EP_SIZE)
    cap = int(os.environ.get("MOE_REPLAY_MAX_LOCAL_EXPERTS", "0"))
    if cap > 0 and local_experts > cap:
        print(
            f"[replay] capping local experts {local_experts} -> {cap} "
            f"(MOE_REPLAY_MAX_LOCAL_EXPERTS)",
            flush=True)
        local_experts = cap
    inter_aligned = align128(dims["moe_intermediate_size"])
    approx_bytes = local_experts * (2 * inter_aligned * dims["hidden_size"] +
                                    dims["hidden_size"] * inter_aligned)
    print(
        f"=== REPLAY EP ep_size={EP_SIZE} ep_rank={EP_RANK} "
        f"local_experts={local_experts} inter_aligned={inter_aligned} "
        f"weight_bytes~{approx_bytes} ===",
        flush=True)

    weights = None
    with Section("build_weights"):
        weights = build_weights(jax, jnp, quantize_tensor, dims, local_experts)
        print(
            f"[replay] w13={tuple(weights[0].shape)}/{weights[0].dtype} "
            f"w13_scale={tuple(weights[1].shape)} "
            f"w2={tuple(weights[2].shape)}/{weights[2].dtype} "
            f"w2_scale={tuple(weights[3].shape)}",
            flush=True)
    if weights is None:
        print("=== REPLAY INFRA ERROR weights unavailable ===", flush=True)
        return 1

    # experts_start must be traced data, not a Python int -- see the
    # fused_moe_func docstring on rank-uniform compiled programs.
    experts_start = jnp.asarray(EP_RANK * local_experts, dtype=jnp.int32)

    for num_tokens in NUM_TOKENS_SWEEP:
        inputs = None
        with Section(f"inputs_T{num_tokens}"):
            inputs = build_inputs(jax, jnp, num_tokens, dims)
        if inputs is None:
            continue

        out_gather = out_onehot = out_eager = None

        with Section(f"gather_T{num_tokens}"):
            out_gather = run_path(fused_moe_func, jnp, weights, inputs, dims,
                                  0, experts_start)
        with Section(f"onehot_T{num_tokens}"):
            out_onehot = run_path(fused_moe_func, jnp, weights, inputs, dims,
                                  32768, experts_start)
        with Section(f"onehot_eager_T{num_tokens}"):
            # Ground truth: same one-hot path with fusion turned off. Pallas
            # kernels can refuse to run outside jit; that is reported, not fatal.
            with jax.disable_jit():
                out_eager = run_path(fused_moe_func, jnp, weights, inputs,
                                     dims, 32768, experts_start)

        onehot_bad = True
        if out_gather is not None and out_onehot is not None:
            with Section(f"compare_a_vs_b_T{num_tokens}"):
                maxdiff, meandiff, bad_a, bad_b, verdict = compare(
                    jnp, out_gather, out_onehot)
                onehot_bad = verdict == "FAIL"
                print(
                    f"=== REPLAY num_tokens={num_tokens} compare=a_vs_b "
                    f"maxdiff={maxdiff:.6g} nan={bad_b} verdict={verdict} ===",
                    flush=True)
                print(
                    f"[replay] T={num_tokens} a_vs_b meandiff={meandiff:.6g} "
                    f"nan_gather={bad_a} nan_onehot={bad_b}",
                    flush=True)

        if out_onehot is not None and out_eager is not None:
            with Section(f"compare_b_vs_c_T{num_tokens}"):
                maxdiff, meandiff, bad_b, bad_c, verdict = compare(
                    jnp, out_eager, out_onehot)
                print(
                    f"=== REPLAY num_tokens={num_tokens} compare=b_vs_c "
                    f"maxdiff={maxdiff:.6g} nan={bad_b} verdict={verdict} ===",
                    flush=True)
                print(
                    f"[replay] T={num_tokens} b_vs_c meandiff={meandiff:.6g} "
                    f"nan_eager={bad_c} nan_onehot={bad_b}",
                    flush=True)
        else:
            print(
                f"=== REPLAY num_tokens={num_tokens} compare=b_vs_c "
                f"maxdiff=nan nan=-1 verdict=SKIPPED ===",
                flush=True)

        if onehot_bad and inputs is not None:
            # Only worth the extra compiles on shapes that actually diverge.
            with Section(f"seams_onehot_T{num_tokens}"):
                seam_report(jax, jnp, mods, weights, inputs, dims, 32768,
                            experts_start, num_tokens)
            with Section(f"seams_gather_T{num_tokens}"):
                seam_report(jax, jnp, mods, weights, inputs, dims, 0,
                            experts_start, num_tokens)

    print("=== REPLAY DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
