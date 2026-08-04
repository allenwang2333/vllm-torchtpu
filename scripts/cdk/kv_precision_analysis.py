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
"""Cost of each MLA KV-cache quantization scheme, measured on real tensors.

Consumes the `.npz` written by `kv_precision_probe.py` and answers the
question that decides whether the `fp8_ds_mla` packed-layout work is worth
doing: how much accuracy does the TPU sparse path's *per-tensor fp8 for
everything* actually give up against the GPU reference's *per-128-block fp8
nope + bf16 rope*?

Four schemes are compared, all against the unquantized bf16 tensors:

  tpu_pertensor_k1    what the branch does today -- one scalar `k_scale`
                      (1.0 unless the checkpoint ships KV scales) applied to
                      both the 512-d latent and the 64-d RoPE tail.
  tpu_pertensor_cal   same, but with `k_scale = amax/448`. Isolates "is the
                      1.0 fallback the problem" from "is per-tensor fp8 the
                      problem".
  gpu_ds_mla          the GPU reference: latent quantized per 128-element
                      block with an fp32 scale, RoPE tail left in bf16.
  block_only          per-128-block latent but RoPE still fp8. Separates the
                      per-block win from the bf16-RoPE win.

Element-wise error is reported, but the metric that actually matters is the
attention *logit* error: MLA scores are q . concat(latent, rope), so a
uniform relative error on the latent partly cancels in the dot product while
an error on the 64-d RoPE tail does not. Top-k agreement is reported for the
same reason -- a logit perturbation only costs accuracy if it reorders keys.
"""

import argparse
import json
import sys

import numpy as np

try:
    import ml_dtypes
except ImportError:  # pragma: no cover - dependency is declared in the runbook
    sys.exit("ml_dtypes is required: uv pip install ml_dtypes")

FP8 = ml_dtypes.float8_e4m3fn
FP8_MAX = float(ml_dtypes.finfo(FP8).max)  # 448.0
BF16 = ml_dtypes.bfloat16


def _round_trip(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Quantize to e4m3 with `scale`, then dequantize. Mirrors the TPU path.

    `scale` broadcasts against `x`. Matches
    `static_per_tensor_quantize_tensor` / `quantize_tensor` in
    `layers/common/quantization.py`: divide, clamp to the dtype range, cast,
    then multiply back.
    """
    scaled = np.clip(x.astype(np.float32) / scale, -FP8_MAX, FP8_MAX)
    return scaled.astype(FP8).astype(np.float32) * scale


def per_tensor_fp8(x: np.ndarray, k_scale: float | None) -> np.ndarray:
    """Single scalar scale for the whole tensor (the TPU sparse path)."""
    if k_scale is None:
        k_scale = float(np.abs(x).max()) / FP8_MAX
    return _round_trip(x, np.float32(k_scale))


def per_block_fp8(x: np.ndarray, block: int = 128) -> np.ndarray:
    """Per-`block` fp32 scale along the last axis (the GPU ds_mla nope path)."""
    *lead, d = x.shape
    if d % block:
        raise ValueError(f"last dim {d} not divisible by block {block}")
    blocked = x.astype(np.float32).reshape(*lead, d // block, block)
    amax = np.abs(blocked).max(axis=-1, keepdims=True)
    # Zero blocks would give 0/0; keep them exactly zero, as _safe_inverse_scale does.
    scale = np.where(amax == 0, 1.0, amax / FP8_MAX).astype(np.float32)
    return _round_trip(blocked, scale).reshape(*lead, d)


def to_bf16(x: np.ndarray) -> np.ndarray:
    """bf16 round trip (the GPU ds_mla rope path, and the `auto` KV cache)."""
    return x.astype(BF16).astype(np.float32)


def elementwise_stats(exact: np.ndarray, approx: np.ndarray) -> dict:
    exact = exact.astype(np.float32)
    err = approx - exact
    denom = np.linalg.norm(exact)
    # Relative error per element, guarded on the near-zero entries that would
    # otherwise dominate the percentiles.
    scale = np.abs(exact)
    floor = np.percentile(scale, 50) * 1e-3
    rel = np.abs(err) / np.maximum(scale, floor)
    return {
        "rel_frobenius": float(np.linalg.norm(err) / denom) if denom else 0.0,
        "rel_p50": float(np.percentile(rel, 50)),
        "rel_p99": float(np.percentile(rel, 99)),
        "rel_max": float(rel.max()),
        "frac_below_fp8_normal": float((scale < 2**-6).mean()),
        "frac_flushed_to_zero": float(((exact != 0) & (approx == 0)).mean()),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def logit_stats(q: np.ndarray, k_exact: np.ndarray, k_approx: np.ndarray,
                v_exact: np.ndarray, v_approx: np.ndarray,
                q_token_idx: np.ndarray, sm_scale: float, topk: int) -> dict:
    """Score error, top-k reordering, and the resulting attention-output error.

    `q` is [Tq, H, D], `k` is [S, D] (MLA is MQA over the latent, so one key
    row serves every head), and the MLA "value" is the latent itself, so
    `v` is the first `kv_lora_rank` columns of `k`.

    The causal mask matters: without it every query would appear to attend to
    the full sequence, which both flattens the softmax and hides the fact that
    early tokens average over very few keys (where a per-key error cancels
    least).
    """
    tq, h, d = q.shape
    s = k_exact.shape[0]
    assert k_exact.shape == (s, d), (k_exact.shape, (s, d))

    qs = q.reshape(-1, d).astype(np.float32)
    # Each query row is repeated H times in `qs`, so the mask must be too.
    pos = np.repeat(q_token_idx.astype(np.int64), h)
    causal = np.arange(s)[None, :] <= pos[:, None]
    neg_inf = np.float32(-3.0e38)

    exact = (qs @ k_exact.astype(np.float32).T) * sm_scale
    approx = (qs @ k_approx.astype(np.float32).T) * sm_scale
    err = (approx - exact)[causal]
    spread = exact[causal].std()

    exact_m = np.where(causal, exact, neg_inf)
    approx_m = np.where(causal, approx, neg_inf)

    p_exact = _softmax(exact_m)
    p_approx = _softmax(approx_m)
    out_exact = p_exact @ v_exact.astype(np.float32)
    out_approx = p_approx @ v_approx.astype(np.float32)
    out_den = np.linalg.norm(out_exact)

    # Top-k overlap is only meaningful where the row has more than k causal
    # candidates; shorter rows select everything and would inflate it to 100%.
    k_eff = min(topk, s)
    eligible = pos >= k_eff
    if eligible.any():
        ex = np.where(causal, exact, neg_inf)[eligible]
        ap = np.where(causal, approx, neg_inf)[eligible]
        top_e = np.argpartition(-ex, k_eff - 1, axis=-1)[:, :k_eff]
        top_a = np.argpartition(-ap, k_eff - 1, axis=-1)[:, :k_eff]
        pe = p_exact[eligible]
        overlap = float(np.mean([
            len(set(a.tolist()) & set(b.tolist())) / k_eff
            for a, b in zip(top_e, top_a)
        ]))
        kept = float(np.mean([pe[i, top_a[i]].sum() for i in range(len(top_a))]))
    else:
        overlap = float("nan")
        kept = float("nan")

    return {
        "logit_rmse": float(np.sqrt((err**2).mean())),
        "logit_rmse_over_std": float(np.sqrt((err**2).mean()) / spread),
        "logit_max_abs_err": float(np.abs(err).max()),
        "attn_out_rel_frobenius": float(
            np.linalg.norm(out_approx - out_exact) / out_den) if out_den else 0.0,
        f"top{k_eff}_overlap": overlap,
        f"top{k_eff}_prob_mass_kept": kept,
        "topk_eligible_rows": int(eligible.sum()),
    }


def build_variants(latent: np.ndarray, rope: np.ndarray,
                   k_scale: float) -> dict:
    """The four cache schemes, each returned as (latent_hat, rope_hat)."""
    # Ordered as a progression, so each row shows the marginal value of one
    # more change on top of the row above it.
    return {
        # What the branch ships today.
        "tpu_pertensor_k1": (
            per_tensor_fp8(latent, k_scale),
            per_tensor_fp8(rope, k_scale),
        ),
        # + a calibrated scalar instead of the 1.0 fallback. Costs nothing to
        # implement, so if this is where the win is, the kernel work is moot.
        "tpu_pertensor_cal": (
            per_tensor_fp8(latent, None),
            per_tensor_fp8(rope, None),
        ),
        # + RoPE tail in bf16, latent still per-tensor fp8. The cheapest change
        # that needs a cache-layout edit, and the one that only affects logits.
        "rope_bf16_only": (
            per_tensor_fp8(latent, None),
            to_bf16(rope),
        ),
        # + per-128-block latent scales, RoPE back to fp8. Isolates the
        # per-block win, which is the part that moves the attention output.
        "block_only": (
            per_block_fp8(latent),
            per_tensor_fp8(rope, None),
        ),
        # Both: the full GPU fp8_ds_mla reference.
        "gpu_ds_mla": (
            per_block_fp8(latent),
            to_bf16(rope),
        ),
    }


def analyze(data: dict, sm_scale: float, topk: int) -> dict:
    latent = data["kv_c_normed"].astype(np.float32)
    rope = data["k_pe"].astype(np.float32)
    k_scale = float(data.get("k_scale", 1.0))

    report = {
        "shapes": {"latent": list(latent.shape), "rope": list(rope.shape)},
        "k_scale_from_checkpoint": k_scale,
        "latent_amax": float(np.abs(latent).max()),
        "latent_rms": float(np.sqrt((latent**2).mean())),
        "rope_amax": float(np.abs(rope).max()),
        "rope_rms": float(np.sqrt((rope**2).mean())),
        "variants": {},
    }

    have_q = "ql_nope" in data and "q_pe" in data
    if have_q:
        ql_nope = data["ql_nope"].astype(np.float32)
        q_pe = data["q_pe"].astype(np.float32)
        q_full = np.concatenate([ql_nope, q_pe], axis=-1)
        k_exact = np.concatenate([latent, rope], axis=-1)
        idx = data.get("q_token_idx")
        if idx is None:
            # Older dumps: assume the query rows were an even stride over the
            # sequence, which is how the probe samples them.
            stride = max(1, latent.shape[0] // ql_nope.shape[0])
            idx = np.arange(ql_nope.shape[0], dtype=np.int32) * stride
        idx = np.asarray(idx)

    for name, (lat_hat, rope_hat) in build_variants(latent, rope,
                                                    k_scale).items():
        entry = {
            "latent": elementwise_stats(latent, lat_hat),
            "rope": elementwise_stats(rope, rope_hat),
        }
        if have_q:
            entry["logits"] = logit_stats(
                q_full, k_exact,
                np.concatenate([lat_hat, rope_hat], axis=-1),
                latent, lat_hat, idx, sm_scale, topk)
        report["variants"][name] = entry
    return report


def _fmt(report: dict) -> str:
    out = []
    out.append(f"latent {report['shapes']['latent']}  "
               f"amax={report['latent_amax']:.4g} rms={report['latent_rms']:.4g}")
    out.append(f"rope   {report['shapes']['rope']}  "
               f"amax={report['rope_amax']:.4g} rms={report['rope_rms']:.4g}")
    out.append(f"checkpoint k_scale = {report['k_scale_from_checkpoint']}")
    out.append("")

    header = f"{'scheme':<20} {'latent relF':>12} {'rope relF':>11}"
    first = next(iter(report["variants"].values()))
    has_logits = "logits" in first
    if has_logits:
        lk = [k for k in first["logits"] if k.endswith("_overlap")][0]
        header += (f" {'logit rmse/sd':>14} {'attn out relF':>14}"
                   f" {lk:>16} {'prob kept':>10}")
    out.append(header)
    out.append("-" * len(header))

    for name, v in report["variants"].items():
        row = (f"{name:<20} {v['latent']['rel_frobenius']:>12.4%} "
               f"{v['rope']['rel_frobenius']:>11.4%}")
        if has_logits:
            lg = v["logits"]
            pk = [k for k in lg if k.endswith("_prob_mass_kept")][0]
            row += (f" {lg['logit_rmse_over_std']:>14.4%} "
                    f"{lg['attn_out_rel_frobenius']:>14.4%} "
                    f"{lg[lk]:>16.4%} {lg[pk]:>10.4%}")
        out.append(row)

    out.append("")
    out.append("frac of latent elements below fp8 min-normal (2^-6): "
               f"{first['latent']['frac_below_fp8_normal']:.3%}")
    return "\n".join(out)


def scale_sweep(x: np.ndarray, scales: list) -> list:
    """Round-trip error of `x` across candidate per-tensor scales.

    Dividing by a power of two is exact in binary floating point -- it only
    shifts the exponent -- so every power-of-two scale produces bit-identical
    fp8 output. `k_scale = 1.0` is therefore not a neutral default but one
    sample of the power-of-two case.

    A non-power-of-two scale moves values within their binades, and e4m3's
    relative step is 1/8 of a binade: worst just above a power of two, best
    just below the next one. Choosing the scale so the distribution lands in
    the favourable part of the binade is free precision.
    """
    out = []
    for s in scales:
        hat = per_tensor_fp8(x, s)
        out.append((s, float(np.linalg.norm(hat - x) / np.linalg.norm(x))))
    return out


def synthetic(num_tokens: int = 512, seed: int = 0) -> dict:
    """Stand-in tensors for validating the harness without a device.

    Shapes and rough statistics of a DeepSeek/GLM MLA layer: an RMS-normed
    512-d latent (so per-dimension scale is set by the norm weight, which is
    what makes per-block scaling interesting) and a 64-d RoPE tail whose
    entries are bounded by the pre-rotation key magnitude.
    """
    rng = np.random.default_rng(seed)
    # A norm weight with real per-dimension spread -- a flat weight would make
    # per-block and per-tensor scaling look identical by construction.
    norm_w = np.exp(rng.normal(0.0, 0.6, size=512)).astype(np.float32)
    x = rng.standard_normal((num_tokens, 512)).astype(np.float32)
    latent = x / np.sqrt((x**2).mean(axis=-1, keepdims=True)) * norm_w

    rope = rng.standard_normal((num_tokens, 64)).astype(np.float32) * 0.7
    q_nope = rng.standard_normal((num_tokens, 8, 512)).astype(np.float32) * 0.1
    q_pe = rng.standard_normal((num_tokens, 8, 64)).astype(np.float32) * 0.3
    return {
        "kv_c_normed": latent,
        "k_pe": rope,
        "ql_nope": q_nope,
        "q_pe": q_pe,
        "k_scale": np.float32(1.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", nargs="?",
                    help="dump from kv_precision_probe.py; omit for --synthetic")
    ap.add_argument("--synthetic", action="store_true",
                    help="validate the harness on generated tensors")
    ap.add_argument("--sm-scale", type=float, default=576**-0.5,
                    help="MLA softmax scale (layer.scale); default 1/sqrt(576)")
    ap.add_argument("--topk", type=int, default=2048,
                    help="index_topk, for the selection-overlap metric")
    ap.add_argument("--json", help="also write the full report here")
    args = ap.parse_args()

    if args.synthetic:
        data = synthetic()
    elif args.npz:
        data = dict(np.load(args.npz))
    else:
        ap.error("give an npz path or --synthetic")

    report = analyze(data, args.sm_scale, args.topk)
    print(_fmt(report))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nfull report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
