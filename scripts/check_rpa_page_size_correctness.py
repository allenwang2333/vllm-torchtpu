# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness of RPA v3 at the KV page sizes `get_page_size` picks.

Why this exists. `PallasAttentionBackend.get_page_size` used to hard-return 16
for any `max_model_len > 8192`; removing that guard moves the serving config
for Qwen3-Coder-30B-A3B-Instruct-FP8 (max_model_len 9216) from page size 16 to
256, which the operator sweep in `benchmark_rpa_decode_page_size.py` measures at
1.87x faster. A speedup is not a result until the kernel is shown to still be
correct at the new page size, and `tests/kernels/test_rpa_skip_kv_update.py`
hardcodes `page_size = 16` everywhere -- nothing in the tree exercises RPA v3 at
256.

What it checks, per page size, against `ref_ragged_paged_attention` (the pure
JAX reference that ships beside the Pallas kernel):

  1. Attention output matches the reference.
  2. The KV cache *after* the call matches the reference's, which is the check
     that the kernel's KV writer puts the new tokens at the right page and
     offset -- the part of the paging path a same-page-size output comparison
     cannot see.
  3. A negative control per case: perturbing one input must move the output by
     much more than the tolerance, so a silently-zeroed kernel cannot pass.
  4. That the reference itself has signal, because at long context with an fp8
     cache it does not -- see REF_MIN_RMS. Run the reference comparison with
     `--kv-cache-dtype bf16`; the fp8 arm still carries checks 1-3 and the
     cross-page-size comparison, which do not involve the reference.

Cases cover the three `distribution` regimes the serving path produces: pure
decode, chunked prefill with history, and a batch mixing both.

Only the JAX kernel is imported, not the vLLM serving stack, so this runs on a
bare TPU pod. The `get_page_size` mapping itself is covered by
`tests/platforms/test_tpu_platform.py`; this file covers the kernel underneath
it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# The serving shape: 32 q heads / TP8, and vLLM's get_num_kv_heads gives
# max(1, 4 // 8) == 1 kv head per rank.
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 128
MAX_MODEL_LEN = 9216  # what makes the old code return 16 and the new one 256

# bf16 output of a long softmax-weighted sum: the reference accumulates in a
# different order than the kernel's blocked flash pass, so exact equality is not
# the bar. Job j-c82635ab measured the worst pallas-vs-ref gap at 0.0703 on
# `prefill_fresh` (mean 0.0054 over 2M elements, ~18 bf16 ulps) -- and measured
# it *identically at page 16 and page 256*, so it is a property of the reference
# comparison, not of paging. 0.1 covers it with margin. The page-size claim
# itself does not rest on this number: it rests on `cross_page_size`, which is
# an exact equality.
ATOL = 1e-1
# The control perturbs a query vector and requires the output to move. Sized off
# measurement, not guesswork: a single-element bump moved it 0.15, and the
# whole-vector bump below moves it much further. 10x ATOL leaves room for both.
CONTROL_MIN_DIFF = 1.0

# The reference must itself have signal, or agreeing with it proves nothing.
#
# This is not hypothetical. `ref_ragged_paged_attention` does
# `jax.nn.softmax(attn, axis=-1).astype(v.dtype)` (v3/kernel.py:200), casting the
# attention weights to the KV dtype. With an fp8 cache that is float8_e4m3fn,
# whose smallest subnormal is 2^-9 = 0.001953125 -- so a decode over 8192 keys,
# whose near-uniform weights are ~1/8192 = 1.2e-4, underflows *every* weight to
# zero and the reference returns an exactly-zero tensor. Job j-9037b4cc read
# `baseline_abs_max: 0.0` on decode and prefill_with_history, and exactly one
# fp8 subnormal (0.001953125) on mixed_batch, for precisely this reason. The
# Pallas kernel does not round its weights that way and returned the correct
# ~0.02 magnitude, so those three cases were scoring the kernel against zeros
# and calling the agreement a pass.
#
# Run with --kv-cache-dtype bf16 to get a reference that means something at long
# context; the fp8 arm remains meaningful for cross_page_size, which compares
# the kernel against itself.
REF_MIN_RMS = 1e-3


def _load_kernel(repo_root: Path):
    """Import the v3 RPA kernel without pulling in the vLLM serving stack.

    Falls back to the installed wheel when ``repo_root`` has no ``src`` tree,
    so the file can be dropped anywhere on a pod that has vllm_torchtpu
    installed rather than only run from inside a checkout.
    """
    src = repo_root / "src"
    rpa_dir = src / "vllm_torchtpu" / "kernels" / "ragged_paged_attention" / "v3"
    if not rpa_dir.is_dir():
        import importlib
        return importlib.import_module(
            "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel")
    package_paths = {
        "vllm_torchtpu": src / "vllm_torchtpu",
        "vllm_torchtpu.kernels": src / "vllm_torchtpu" / "kernels",
        "vllm_torchtpu.kernels.ragged_paged_attention":
        src / "vllm_torchtpu" / "kernels" / "ragged_paged_attention",
        "vllm_torchtpu.kernels.ragged_paged_attention.v3": rpa_dir,
    }
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _load("vllm_torchtpu.kernels.ragged_paged_attention.v3.util",
          rpa_dir / "util.py")
    _load(
        "vllm_torchtpu.kernels.ragged_paged_attention.v3.tuned_block_sizes",
        rpa_dir / "tuned_block_sizes.py",
    )
    return _load("vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel",
                 rpa_dir / "kernel.py")


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# (name, per-sequence (q_len, kv_len), distribution). Sequences must be ordered
# decode-only, then chunked-prefill-only, then mixed -- `distribution` is read
# as index boundaries into the batch, not as counts of a sorted list.
CASES = [
    ("decode", [(1, 8192)] * 8, (8, 8, 8)),
    ("prefill_fresh", [(1024, 1024)] * 4, (0, 0, 4)),
    ("prefill_with_history", [(512, 8192)] * 4, (0, 0, 4)),
    ("mixed_batch", [(1, 8192)] * 4 + [(256, 4096)] * 2, (4, 4, 6)),
]


def _logical_kv(seq_shapes, seed: int):
    """Per-sequence K/V for every position, page-size independent and float32.

    float32 so both page sizes quantize from bit-identical source values; if
    each arm rounded its own way the cross-arm comparison would measure the
    rounding, not the paging.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _, kv_len in seq_shapes:
        shape = (kv_len, NUM_KV_HEADS, HEAD_DIM)
        out.append((
            (rng.standard_normal(shape) * 0.5).astype(np.float32),
            (rng.standard_normal(shape) * 0.5).astype(np.float32),
        ))
    return out


def _build(kernel_mod, *, page_size, seq_shapes, logical, kv_dtype, seed):
    """Lay the same logical K/V into a cache paged at ``page_size``."""
    num_seqs = len(seq_shapes)
    pages_per_seq = cdiv(MAX_MODEL_LEN, page_size)
    total_num_pages = num_seqs * pages_per_seq + 1

    cache_shape = kernel_mod.get_kv_cache_shape(total_num_pages, page_size,
                                                NUM_KV_HEADS, HEAD_DIM,
                                                kv_dtype)
    _, _, x2_per_packing, kv_packing, padded_head_dim = cache_shape
    merged_heads = x2_per_packing * kv_packing

    rng = np.random.default_rng(seed)
    # Scattered, not compact: a real block allocator hands out whatever pages
    # are free, and a bug in the page-table indexing hides under an identity
    # mapping.
    page_indices = rng.permutation(total_num_pages)[:num_seqs *
                                                    pages_per_seq].astype(
                                                        np.int32)

    cache = np.zeros(cache_shape, dtype=np.float32)
    queries, new_k, new_v, cu = [], [], [], [0]
    q_rng = np.random.default_rng(seed + 1)
    for seq, (q_len, kv_len) in enumerate(seq_shapes):
        k_all, v_all = logical[seq]
        # Positions [0, kv_len - q_len) are already in the cache; the last
        # q_len are handed to the kernel and must be written by it.
        span = kv_len - q_len
        merged = np.zeros((pages_per_seq * page_size, merged_heads,
                           padded_head_dim),
                          dtype=np.float32)
        for head in range(NUM_KV_HEADS):
            merged[:span, 2 * head, :HEAD_DIM] = k_all[:span, head]
            merged[:span, 2 * head + 1, :HEAD_DIM] = v_all[:span, head]
        paged = merged.reshape(pages_per_seq, page_size, x2_per_packing,
                               kv_packing, padded_head_dim)
        cache[page_indices[seq * pages_per_seq:(seq + 1) *
                           pages_per_seq]] = paged

        queries.append((q_rng.standard_normal(
            (q_len, NUM_Q_HEADS, HEAD_DIM)) * 0.5).astype(np.float32))
        new_k.append(k_all[span:kv_len])
        new_v.append(v_all[span:kv_len])
        cu.append(cu[-1] + q_len)

    return {
        "queries": np.concatenate(queries, axis=0),
        "keys": np.concatenate(new_k, axis=0),
        "values": np.concatenate(new_v, axis=0),
        "kv_cache": cache,
        "kv_lens": np.asarray([kv for _, kv in seq_shapes], np.int32),
        "page_indices": page_indices,
        "cu_q_lens": np.asarray(cu, np.int32),
    }


def _call(fn, built, distribution, *, q_dtype, kv_dtype, perturb=False):
    """Run one implementation on fresh device copies of ``built``.

    Fresh every time: the kernel donates its cache, so a second call would see
    a buffer the first one already mutated.

    ``perturb`` drives the negative control. It moves the **last** query token
    of every sequence, and moves the whole 128-wide vector rather than one
    element. Both details are load-bearing, and job j-c82635ab is why: the first
    version bumped `queries[0, 0, 0]`, and on a fresh prefill (q_len == kv_len)
    the causal mask leaves token 0 attending to exactly one key, so its softmax
    is 1.0 whatever the logit and the output provably cannot move. That arm read
    `control_diff: 0.0` -- a correct kernel making the control vacuous. The last
    token of a sequence always attends to its full context, so no masking regime
    can silence it.
    """
    queries = built["queries"]
    if perturb:
        queries = queries.copy()
        for end in built["cu_q_lens"][1:]:
            queries[int(end) - 1, :, :] += 50.0
    out, cache = fn(
        jnp.asarray(queries, q_dtype),
        jnp.asarray(built["keys"], kv_dtype),
        jnp.asarray(built["values"], kv_dtype),
        jnp.asarray(built["kv_cache"], kv_dtype),
        jnp.asarray(built["kv_lens"]),
        jnp.asarray(built["page_indices"]),
        jnp.asarray(built["cu_q_lens"]),
        jnp.asarray(distribution, jnp.int32),
        sm_scale=HEAD_DIM**-0.5,
    )
    return (np.asarray(out.astype(jnp.float32)),
            np.asarray(cache.astype(jnp.float32)))


def _diff(a: np.ndarray, b: np.ndarray) -> dict:
    """``b`` is the baseline, so the scale columns describe ``b``.

    An absolute tolerance says nothing without the magnitude it applies to, and
    the first version of this file shipped an atol chosen before anyone had seen
    one. Reporting the baseline's scale means the next reader can judge the
    tolerance from the record instead of re-deriving it.
    """
    delta = np.abs(a - b)
    return {
        "max_abs_diff": float(delta.max(initial=0.0)),
        "mean_abs_diff": float(delta.mean()),
        "baseline_abs_max": float(np.abs(b).max(initial=0.0)),
        "baseline_rms": float(np.sqrt(np.mean(np.square(b, dtype=np.float64)))),
        "finite": bool(np.isfinite(a).all()),
    }


def _emit(path: Path, record: dict) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-sizes", type=str, default="16,256")
    parser.add_argument("--kv-cache-dtype",
                        choices=["fp8", "bf16"],
                        default="fp8")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--repo-root",
                        type=Path,
                        default=None,
                        help="checkout to import the kernel from; defaults to "
                        "this file's repo, then to the installed wheel")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)

    repo_root = args.repo_root or Path(__file__).resolve().parents[1]
    kernel_mod = _load_kernel(repo_root)
    kv_dtype =(jnp.float8_e4m3fn
                if args.kv_cache_dtype == "fp8" else jnp.bfloat16)
    page_sizes = [int(x) for x in args.page_sizes.split(",") if x.strip()]

    _emit(
        args.output, {
            "event": "environment",
            "jax_version": jax.__version__,
            "devices": [str(d) for d in jax.devices()],
            "check": "rpa_v3_page_size_correctness",
            "page_sizes": page_sizes,
            "kv_cache_dtype": args.kv_cache_dtype,
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "max_model_len": MAX_MODEL_LEN,
            "atol": ATOL,
        })

    failures = 0
    for name, seq_shapes, distribution in CASES:
        logical = _logical_kv(seq_shapes, args.seed)
        ref_outputs: dict[int, np.ndarray] = {}
        for page_size in page_sizes:
            built = _build(kernel_mod,
                           page_size=page_size,
                           seq_shapes=seq_shapes,
                           logical=logical,
                           kv_dtype=kv_dtype,
                           seed=args.seed)
            record = {
                "event": "check",
                "case": name,
                "page_size": page_size,
                "pages_per_seq": cdiv(MAX_MODEL_LEN, page_size),
                "distribution": list(distribution),
            }
            try:
                pal_out, pal_cache = _call(kernel_mod.ragged_paged_attention,
                                           built,
                                           distribution,
                                           q_dtype=jnp.bfloat16,
                                           kv_dtype=kv_dtype)
                ref_out, ref_cache = _call(kernel_mod.ref_ragged_paged_attention,
                                           built,
                                           distribution,
                                           q_dtype=jnp.bfloat16,
                                           kv_dtype=kv_dtype)
                ctrl_out, _ = _call(kernel_mod.ragged_paged_attention,
                                    built,
                                    distribution,
                                    q_dtype=jnp.bfloat16,
                                    kv_dtype=kv_dtype,
                                    perturb=True)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["passed"] = False
                failures += 1
                _emit(args.output, record)
                continue

            record["output_vs_ref"] = _diff(pal_out, ref_out)
            record["cache_vs_ref"] = _diff(pal_cache, ref_cache)
            record["control_diff"] = float(
                np.abs(ctrl_out - pal_out).max(initial=0.0))
            # A reference that returned zeros agrees with anything small. Fail
            # loudly instead of banking the agreement -- see REF_MIN_RMS.
            record["ref_degenerate"] = bool(
                record["output_vs_ref"]["baseline_rms"] < REF_MIN_RMS)
            record["passed"] = bool(
                record["output_vs_ref"]["finite"]
                and not record["ref_degenerate"]
                and record["output_vs_ref"]["max_abs_diff"] <= ATOL
                and record["cache_vs_ref"]["max_abs_diff"] <= ATOL
                and record["control_diff"] >= CONTROL_MIN_DIFF)
            ref_outputs[page_size] = pal_out
            if not record["passed"]:
                failures += 1
            _emit(args.output, record)

        # Same logical content, different paging: the outputs must agree.
        base = page_sizes[0]
        for page_size in page_sizes[1:]:
            if base not in ref_outputs or page_size not in ref_outputs:
                continue
            cross = _diff(ref_outputs[page_size], ref_outputs[base])
            cross_passed = bool(cross["finite"]
                                and cross["max_abs_diff"] <= ATOL)
            if not cross_passed:
                failures += 1
            _emit(
                args.output, {
                    "event": "cross_page_size",
                    "case": name,
                    "page_size": page_size,
                    "vs_page_size": base,
                    "passed": cross_passed,
                    **cross,
                })

    _emit(args.output, {"event": "done", "failures": failures})
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
