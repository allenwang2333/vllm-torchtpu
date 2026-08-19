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
"""MoE microbenchmark at Qwen3.5-397B-A17B-FP8 EP8 shapes (one TPU chip).

Standalone JAX: no vLLM, no torch, no checkpoint. It calls the *same* kernels
the server calls -- ``gmm_v2`` through ``fused_moe_gmm.gmm_wrapper``, the
SparseCore ``ragged_gather_v2`` / ``ragged_gather_reduce`` kernels, and
``fused_moe_gmm.fused_moe_func`` end to end -- on synthetic weights and
routing, so a MoE perf question costs a minute instead of a 60-minute server
run.

Row semantics (the two counts are different and both are printed):

* ``tokens``  -- ``num_tokens``, the rows of ``hidden_states`` entering the
  layer. This is what the scheduler calls a batched-token count.
* ``rows``    -- ``num_tokens * topk``, the routed rows: the row count of the
  GMM lhs, of the permute output, and of the gather/reduce inputs.
* ``work``    -- rows that carry real expert work, i.e. ``sum(group_sizes)``.
  Under EP the lhs is allocated for all ``rows`` but only the locally-owned
  experts' rows are computed, so ``work`` is what TFLOP/s is derived from.

``--m`` is a list of *routed-row* counts by default (``--m-semantics rows``);
pass ``--m-semantics tokens`` to read it as token counts instead. Either way
the effective ``(tokens, rows)`` pair is snapped to ``rows = tokens * topk``
and printed, so every variant in a table row runs at identical shapes.

Variants:

  gmm1              gmm_wrapper(lhs[rows, hidden], w13[E, hidden, 2*inter],
                    fuse_act=<activation>) -> [rows, inter]
  gmm2              gmm_wrapper(lhs[rows, inter], w2[E, inter, hidden])
                    -> [rows, hidden]
  permute_onehot    the one-hot permute matmul from prepare_routed_gmm_inputs
  permute_sc        ragged_gather_v2, the SparseCore alternative
  prepare_onehot    the whole prepare_routed_gmm_inputs (argsort + permute)
  prepare_sc        ... with the SparseCore branch
  combine_onehot    the one-hot unpermute/reduce matmul from moe_gmm
  combine_sc_v{1,2,3}  ragged_gather_reduce, all three versions
  e2e_onehot        fused_moe_func with onehot_moe_permute_threshold=32768
  e2e_sc            fused_moe_func with onehot_moe_permute_threshold=0

TFLOP/s and GB/s are derived, not measured: FLOPs count the matmuls only and
HBM bytes count the logical operands (lhs + out + the weights of every
non-empty group + fp32 scales), assuming each weight is streamed exactly once.
Both are upper-bound models -- treat them as a sanity check on the microsecond
column, which is the measured quantity.

Example:

  python scripts/benchmarking/kernels/bench_moe_gmm.py
  python scripts/benchmarking/kernels/bench_moe_gmm.py --m 640,5120 --skew 0.8
  python scripts/benchmarking/kernels/bench_moe_gmm.py --only gmm1,gmm2

Needs a free TPU chip: it allocates ~0.8 GiB of weights plus activations, and
the profiler needs the device to itself for the numbers to mean anything.
"""

import argparse
import dataclasses
import functools
import importlib.util
import os
import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from device_time import DEVICE, PIPELINED, WALL, measure  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/jax_microbench")

# 397B/EP8 defaults: 512 experts over 8 ranks, hidden 4096, per-expert
# intermediate 1024, top-10 routing, FP8 weights with BF16 activations.
_DEFAULT_M = "384,480,640,1280,2560,3840,5120,20480,40960"
_PROD_ONEHOT_THRESHOLD = 32768  # ONEHOT_MOE_PERMUTE_THRESHOLD in the 397B cfg

# Every row build_variants can emit, in table order (used to check --only).
_VARIANT_NAMES = (
    "gmm1",
    "gmm2",
    "permute_onehot",
    "permute_sc",
    "prepare_onehot",
    "prepare_sc",
    "combine_onehot",
    "combine_sc_v1",
    "combine_sc_v2",
    "combine_sc_v3",
    "e2e_onehot",
    "e2e_sc",
)


def _configure_compilation_cache(path: str) -> None:
    """Persist compiled programs so repeat runs skip the Pallas compiles."""
    os.makedirs(path, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = path
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)


def _load_repo_api() -> SimpleNamespace:
    """Import the MoE path without importing ``vllm_torchtpu.layers``.

    ``vllm_torchtpu/layers/__init__.py`` registers the vLLM layers and pulls in
    torch (and with it a torch_tpu client) -- which this benchmark must not do.
    ``layers/common/fused_moe_gmm.py`` itself only imports ``vllm_torchtpu.envs``
    and the Pallas kernels, so load that file directly under a private module
    name and take the kernels from ``vllm_torchtpu.kernels`` as usual.
    """
    if importlib.util.find_spec("vllm_torchtpu") is None:
        sys.path.insert(0, str(_SRC))

    name = "_bench_fused_moe_gmm"
    module = sys.modules.get(name)
    if module is None:
        path = _SRC / "vllm_torchtpu" / "layers" / "common" / "fused_moe_gmm.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

    import vllm_torchtpu.envs as envs
    from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce import \
        ragged_gather_reduce as reduce_v1
    from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper import \
        ragged_gather_reduce_v2 as reduce_v2
    from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import \
        ragged_gather_reduce as reduce_v3
    from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import \
        ragged_gather_v2

    return SimpleNamespace(
        gmm_wrapper=module.gmm_wrapper,
        prepare_routed_gmm_inputs=module.prepare_routed_gmm_inputs,
        fused_moe_func=module.fused_moe_func,
        ragged_gather_v2=ragged_gather_v2,
        reduce_versions={
            "v1": reduce_v1,
            "v2": reduce_v2,
            "v3": reduce_v3
        },
        selected_reduce_version=envs.RAGGED_GATHER_REDUCE_VERSION,
    )


@dataclasses.dataclass(frozen=True)
class Case:
    """One (tokens, routed rows) point of the sweep."""

    requested_m: int
    num_tokens: int
    routed_rows: int


@dataclasses.dataclass
class Variant:
    """A benchable callable plus the roofline model for its printed row."""

    name: str
    fn: Callable[..., Any] | None  # None = skipped, `note` says why
    args: tuple
    work_rows: int
    flops: float = 0.0
    hbm_bytes: float = 0.0
    note: str = ""


def _expert_ramp(num_groups: int, skew: float) -> np.ndarray:
    """Relative expert load: flat for skew=0, (1+s)x..(1-s)x mean for skew=s."""
    if num_groups == 1 or not skew:
        return np.ones(num_groups, dtype=np.float64)
    ramp = np.linspace(1.0, -1.0, num_groups)
    weights = np.clip(1.0 + skew * ramp, 0.0, None)
    if weights.sum() <= 0:
        return np.ones(num_groups, dtype=np.float64)
    return weights


def make_group_sizes(rows: int, num_groups: int, skew: float) -> np.ndarray:
    """Rows per local expert, summing to exactly ``rows`` (largest-remainder)."""
    weights = _expert_ramp(num_groups, skew)
    exact = weights / weights.sum() * rows
    sizes = np.floor(exact).astype(np.int64)
    leftover = rows - int(sizes.sum())
    if leftover > 0:
        order = np.argsort(-(exact - sizes))
        sizes[order[:leftover]] += 1
    return sizes.astype(np.int32)


def make_local_routing(num_tokens: int, topk: int, group_sizes: np.ndarray,
                       rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Sorted/revert indices built exactly as prepare_routed_gmm_inputs does.

    ``group_sizes`` fixes how many routed rows each local expert owns; the
    remaining rows are marked non-local (id ``-1``) and sort after the valid
    prefix, which is the EP layout the kernels see. Simplification: expert ids
    are shuffled over the flat (token, k) slots, so one token can draw the same
    local expert twice. Real top-k routing cannot, but nothing benchmarked here
    depends on it -- the shapes, the group sizes, and the index scatter pattern
    are all unchanged.
    """
    num_local_experts = int(group_sizes.shape[0])
    total = num_tokens * topk
    valid = int(group_sizes.sum())
    if valid > total:
        raise ValueError(f"{valid=} routed rows exceeds {total=}")

    experts = np.concatenate([
        np.repeat(np.arange(num_local_experts, dtype=np.int32), group_sizes),
        np.full(total - valid, -1, dtype=np.int32),
    ])
    rng.shuffle(experts)
    token_indices_flat = np.repeat(np.arange(num_tokens, dtype=np.int32), topk)

    valid_mask = experts >= 0
    sort_keys = np.where(valid_mask, experts, num_local_experts)
    sorted_indices = np.argsort(sort_keys, kind="stable")
    topk_weights_flat = rng.random(total, dtype=np.float32)
    return {
        "topk_ids_local":
        experts.reshape(num_tokens, topk),
        "token_indices_sorted":
        token_indices_flat[sorted_indices],
        "argsort_revert_indices":
        np.argsort(sorted_indices, kind="stable").astype(np.int32),
        "valid_mask_flat":
        valid_mask,
        "topk_weights_flat":
        topk_weights_flat,
    }


def make_global_routing(num_tokens: int, topk: int, num_experts: int,
                        num_local_experts: int, skew: float,
                        rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Global expert ids (distinct per token) for the end-to-end EP path.

    Gumbel top-k over the skew ramp, tiled across ranks, so every rank owns the
    same load shape and rank 0 (``experts_start=0``) is representative.
    """
    ramp = _expert_ramp(num_local_experts, skew)
    probs = np.tile(ramp, num_experts // num_local_experts)
    gumbel = -np.log(-np.log(
        rng.random((num_tokens, num_experts), dtype=np.float32) + 1e-20) +
                     1e-20)
    scores = np.log(probs + 1e-9).astype(np.float32) + gumbel
    ids = np.argpartition(-scores, topk - 1, axis=1)[:, :topk].astype(np.int32)
    weights = rng.random((num_tokens, topk), dtype=np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    return {"topk_ids": ids, "topk_weights": weights}


@functools.partial(jax.jit, static_argnums=(1, 2))
def _random_array(key: jax.Array, shape: tuple[int, ...],
                  dtype: Any) -> jax.Array:
    """Random values cast to ``dtype`` in one fused kernel (no wide temporary)."""
    return jax.random.normal(key, shape, jnp.bfloat16).astype(jnp.dtype(dtype))


def make_weights(cfg: argparse.Namespace, wdtype: Any,
                 quantized: bool) -> dict[str, Any]:
    """w13/w2 in gmm_v2's K-major layout plus their fp32 scales.

    Layouts match ``fp8._quantize_and_format_single_moe_weight``:
    ``w13[E, hidden, 2*inter]``, ``w2[E, inter, hidden]``, scales
    ``[E, num_blocks, 1, N]`` with ``num_blocks = K // quant_block_size``
    (one block, i.e. per-output-channel, when ``--quant-block-size 0``).
    """
    experts = cfg.num_local_experts
    hidden, inter = cfg.hidden, cfg.moe_intermediate
    w13 = _random_array(jax.random.key(0), (experts, hidden, 2 * inter),
                        wdtype)
    w2 = _random_array(jax.random.key(1), (experts, inter, hidden), wdtype)
    if not quantized:
        return {"w13": w13, "w13_scale": None, "w2": w2, "w2_scale": None}

    def _scale(num_blocks: int, size_n: int, seed: int) -> jax.Array:
        raw = jax.random.uniform(jax.random.key(seed),
                                 (experts, num_blocks, 1, size_n),
                                 jnp.float32,
                                 minval=0.5,
                                 maxval=1.5)
        return raw / 448.0  # rough fp8_e4m3fn amax scale

    block = cfg.quant_block_size
    w13_blocks = hidden // block if block else 1
    w2_blocks = inter // block if block else 1
    return {
        "w13": w13,
        "w13_scale": _scale(w13_blocks, 2 * inter, 2),
        "w2": w2,
        "w2_scale": _scale(w2_blocks, hidden, 3),
    }


def _gmm_bytes(rows: int, size_k: int, size_n: int, out_n: int,
               nonempty_groups: int, act_bytes: int, w_bits: int,
               scale_blocks: int) -> float:
    """lhs + out + one full pass over every non-empty group's weight/scale."""
    weights = nonempty_groups * size_k * size_n * w_bits / 8
    scales = nonempty_groups * scale_blocks * size_n * 4 if scale_blocks else 0
    return rows * size_k * act_bytes + rows * out_n * act_bytes + weights + scales


def build_variants(api: SimpleNamespace, cfg: argparse.Namespace, case: Case,
                   weights: dict[str, Any], act_dtype: Any,
                   w_bits: int) -> list[Variant]:
    """All benchable pieces for one (tokens, rows) case, at shared shapes."""
    rng = np.random.default_rng(cfg.seed)
    tokens, rows, topk = case.num_tokens, case.routed_rows, cfg.topk
    hidden, inter = cfg.hidden, cfg.moe_intermediate
    experts = cfg.num_local_experts
    act_bytes = jnp.dtype(act_dtype).itemsize

    work_rows = int(round(rows * cfg.gmm_valid_frac))
    group_sizes_np = make_group_sizes(work_rows, experts, cfg.skew)
    nonempty = int((group_sizes_np > 0).sum())
    local = make_local_routing(tokens, topk, group_sizes_np, rng)

    group_sizes = jnp.asarray(group_sizes_np)
    group_offset = jnp.zeros((1, ), jnp.int32)
    hidden_states = _random_array(jax.random.key(10), (tokens, hidden),
                                  act_dtype)
    lhs1 = _random_array(jax.random.key(11), (rows, hidden), act_dtype)
    lhs2 = _random_array(jax.random.key(12), (rows, inter), act_dtype)
    gmm2_out = _random_array(jax.random.key(13), (rows, hidden), act_dtype)
    token_indices_sorted = jnp.asarray(local["token_indices_sorted"])
    revert_indices = jnp.asarray(local["argsort_revert_indices"])
    valid_mask_flat = jnp.asarray(local["valid_mask_flat"])
    topk_weights_flat = jnp.asarray(local["topk_weights_flat"],
                                    dtype=act_dtype)
    topk_ids_local = jnp.asarray(local["topk_ids_local"])
    topk_weights_2d = topk_weights_flat.reshape(tokens, topk)
    valid_mask_2d = valid_mask_flat.reshape(tokens, topk)

    w13, w2 = weights["w13"], weights["w2"]
    w13_scale, w2_scale = weights["w13_scale"], weights["w2_scale"]
    w13_blocks = 0 if w13_scale is None else w13_scale.shape[1]
    w2_blocks = 0 if w2_scale is None else w2_scale.shape[1]

    variants: list[Variant] = []

    # --- Grouped GEMMs ------------------------------------------------------
    gmm1 = jax.jit(
        functools.partial(api.gmm_wrapper,
                          zero_initialize=False,
                          fuse_act=cfg.activation))
    variants.append(
        Variant(
            name="gmm1",
            fn=gmm1,
            args=(lhs1, w13, w13_scale, None, group_sizes, group_offset),
            work_rows=work_rows,
            flops=2.0 * work_rows * hidden * 2 * inter,
            hbm_bytes=_gmm_bytes(rows, hidden, 2 * inter, inter, nonempty,
                                 act_bytes, w_bits, w13_blocks),
        ))
    gmm2 = jax.jit(
        functools.partial(api.gmm_wrapper,
                          zero_initialize=False,
                          preferred_element_type=jnp.dtype(act_dtype)))
    variants.append(
        Variant(
            name="gmm2",
            fn=gmm2,
            args=(lhs2, w2, w2_scale, None, group_sizes, group_offset),
            work_rows=work_rows,
            flops=2.0 * work_rows * inter * hidden,
            hbm_bytes=_gmm_bytes(rows, inter, hidden, hidden, nonempty,
                                 act_bytes, w_bits, w2_blocks),
        ))

    # --- Permute ------------------------------------------------------------
    # One-hot permute: [rows, tokens] @ [tokens, hidden]. Mirrors the
    # `token_indices_sorted.shape[0] <= onehot_moe_permute_threshold` branch of
    # prepare_routed_gmm_inputs.
    onehot_permute_bytes = rows * tokens * act_bytes
    onehot_permute_skip = (
        f"skipped: one-hot is {onehot_permute_bytes / 2**30:.1f} GiB > "
        f"--max-onehot-bytes"
        if onehot_permute_bytes > cfg.max_onehot_bytes else "")

    def _permute_onehot(states: jax.Array, indices: jax.Array) -> jax.Array:
        onehot = jax.nn.one_hot(indices, tokens, dtype=states.dtype)
        return onehot @ states

    variants.append(
        Variant(
            name="permute_onehot",
            fn=None if onehot_permute_skip else jax.jit(_permute_onehot),
            args=(hidden_states, token_indices_sorted),
            work_rows=rows,
            flops=2.0 * rows * tokens * hidden,
            hbm_bytes=(tokens * hidden * act_bytes +
                       rows * hidden * act_bytes + onehot_permute_bytes),
            note=onehot_permute_skip,
        ))
    variants.append(
        Variant(
            name="permute_sc",
            fn=jax.jit(api.ragged_gather_v2),
            args=(hidden_states, token_indices_sorted,
                  jnp.zeros((1, ),
                            jnp.int32), jnp.asarray([work_rows], jnp.int32)),
            work_rows=work_rows,
            hbm_bytes=(work_rows * hidden * act_bytes * 2 + rows * 4),
        ))

    # Whole prepare step (mask + argsort + group_sizes + permute), which is
    # what the layer actually pays; the two rows differ only in the branch.
    for label, threshold, skip in (("prepare_onehot", rows,
                                    onehot_permute_skip), ("prepare_sc", 0,
                                                           "")):
        variants.append(
            Variant(
                name=label,
                fn=None if skip else jax.jit(
                    functools.partial(
                        api.prepare_routed_gmm_inputs,
                        local_num_experts=experts,
                        topk=topk,
                        use_ep=True,
                        use_sparse_core=True,
                        onehot_moe_permute_threshold=threshold,
                        skip_padded_tokens=cfg.skip_padded_tokens)),
                args=(hidden_states, topk_ids_local, topk_weights_2d),
                work_rows=work_rows,
                hbm_bytes=(tokens * hidden * act_bytes +
                           rows * hidden * act_bytes),
                note=skip or "argsort + permute",
            ))

    # --- Combine ------------------------------------------------------------
    # One-hot combine: build [tokens, rows] from the revert permutation and the
    # topk weights, then [tokens, rows] @ [rows, hidden]. Mirrors the
    # `argsort_revert_indices.size <= onehot_moe_permute_threshold` branch of
    # moe_gmm, including the f32 accumulation.
    onehot_combine_bytes = tokens * topk * rows * act_bytes
    onehot_combine_skip = (
        f"skipped: one-hot is {onehot_combine_bytes / 2**30:.1f} GiB > "
        f"--max-onehot-bytes"
        if onehot_combine_bytes > cfg.max_onehot_bytes else "")

    def _combine_onehot(gmm_out: jax.Array, revert: jax.Array,
                        weights_2d: jax.Array,
                        mask_2d: jax.Array) -> jax.Array:
        revert = revert.reshape(tokens, topk)
        onehot = jax.nn.one_hot(revert, rows, dtype=gmm_out.dtype)
        combine = (onehot * weights_2d[..., None] *
                   mask_2d[..., None]).sum(axis=1)
        return jnp.matmul(combine, gmm_out,
                          preferred_element_type=jnp.float32).astype(
                              gmm_out.dtype)

    variants.append(
        Variant(
            name="combine_onehot",
            fn=None if onehot_combine_skip else jax.jit(_combine_onehot),
            args=(gmm2_out, revert_indices, topk_weights_2d, valid_mask_2d),
            work_rows=rows,
            flops=2.0 * tokens * rows * hidden,
            hbm_bytes=(rows * hidden * act_bytes +
                       tokens * hidden * act_bytes +
                       tokens * rows * act_bytes),
            note=onehot_combine_skip,
        ))

    for version, reduce_fn in sorted(api.reduce_versions.items()):
        selected = version == api.selected_reduce_version
        variants.append(
            Variant(
                name=f"combine_sc_{version}",
                fn=jax.jit(functools.partial(reduce_fn,
                                             reduce_group_size=topk)),
                args=(gmm2_out, revert_indices, topk_weights_flat,
                      valid_mask_flat),
                work_rows=rows,
                hbm_bytes=(rows * hidden * act_bytes +
                           tokens * hidden * act_bytes + rows * 9),
                note="RAGGED_GATHER_REDUCE_VERSION" if selected else "",
            ))

    # --- End to end ---------------------------------------------------------
    global_routing = make_global_routing(tokens, topk, cfg.num_experts,
                                         experts, cfg.skew, rng)
    topk_ids = jnp.asarray(global_routing["topk_ids"])
    topk_weights = jnp.asarray(global_routing["topk_weights"])
    experts_start = jnp.asarray(0, jnp.int32)
    e2e_work = int((global_routing["topk_ids"] < experts).sum())
    e2e_flops = 2.0 * e2e_work * hidden * 2 * inter + 2.0 * e2e_work * inter * hidden
    e2e_bytes = (_gmm_bytes(rows, hidden, 2 * inter, inter, experts, act_bytes,
                            w_bits, w13_blocks) +
                 _gmm_bytes(rows, inter, hidden, hidden, experts, act_bytes,
                            w_bits, w2_blocks) +
                 2 * tokens * hidden * act_bytes)
    for label, threshold in (("e2e_onehot", cfg.onehot_threshold), ("e2e_sc",
                                                                    0)):
        note = ("routing decides work rows"
                if label == "e2e_sc" else f"threshold={threshold}")
        if label == "e2e_onehot" and rows > threshold:
            note = f"threshold={threshold} < rows: same program as e2e_sc"
        variants.append(
            Variant(
                name=label,
                fn=functools.partial(
                    api.fused_moe_func,
                    topk=topk,
                    activation=cfg.activation,
                    use_ep=True,
                    use_sparse_core=True,
                    onehot_moe_permute_threshold=threshold,
                    rhs_quant_dtype=None,
                    skip_padded_tokens=cfg.skip_padded_tokens),
                args=(hidden_states, w13, w2, w13_scale, w2_scale, None, None,
                      topk_weights, topk_ids, experts_start),
                work_rows=e2e_work,
                flops=e2e_flops,
                hbm_bytes=e2e_bytes,
                note=note,
            ))
    return variants


def _cases(cfg: argparse.Namespace) -> list[Case]:
    cases = []
    for requested in [int(v) for v in cfg.m.split(",") if v.strip()]:
        if cfg.m_semantics == "tokens":
            tokens = requested
        else:
            tokens = max(1, int(round(requested / cfg.topk)))
        cases.append(Case(requested, tokens, tokens * cfg.topk))
    return cases


_HEADER = (f"{'tokens':>7} {'rows':>7} {'work':>7}  {'variant':<18} "
           f"{'us':>10} {'TFLOP/s':>9} {'GB/s':>9}  {'src':<4} note")

# `src` column: which clock produced the microseconds.
_SRC_LABEL = {DEVICE: "dev", WALL: "wall", PIPELINED: "pipe"}


def _print_row(case: Case, variant: Variant, micros: float | None, method: str,
               note: str) -> None:
    """One table line; `-` for anything the row does not have a number for."""
    if micros is None:
        us = tflops = gbs = "-"
    else:
        seconds = micros * 1e-6
        us = f"{micros:.2f}"
        tflops = (f"{variant.flops / seconds / 1e12:.2f}"
                  if variant.flops else "-")
        gbs = (f"{variant.hbm_bytes / seconds / 1e9:.1f}"
               if variant.hbm_bytes else "-")
    print(
        f"{case.num_tokens:>7} {case.routed_rows:>7} {variant.work_rows:>7}  "
        f"{variant.name:<18} {us:>10} {tflops:>9} {gbs:>9}  "
        f"{_SRC_LABEL.get(method, '-'):<4} {note}".rstrip())


def _validate(cfg: argparse.Namespace) -> None:
    """Fail before touching the device on combinations that cannot work."""
    if cfg.num_local_experts < 1 or cfg.num_experts % cfg.num_local_experts:
        raise SystemExit(f"--num-experts ({cfg.num_experts}) must be a "
                         f"multiple of --num-local-experts "
                         f"({cfg.num_local_experts})")
    if not 0.0 < cfg.gmm_valid_frac <= 1.0:
        raise SystemExit("--gmm-valid-frac must be in (0, 1]")
    block = cfg.quant_block_size
    if block and (cfg.hidden % block or cfg.moe_intermediate % block):
        raise SystemExit(f"--quant-block-size {block} must divide both "
                         f"--hidden and --moe-intermediate")
    if cfg.topk < 1:
        raise SystemExit("--topk must be >= 1")
    unknown = sorted(
        set(n.strip()
            for n in cfg.only.split(",") if n.strip()) - set(_VARIANT_NAMES))
    if unknown:
        raise SystemExit(f"unknown --only variant(s) {unknown}; "
                         f"choose from {list(_VARIANT_NAMES)}")


def run(cfg: argparse.Namespace) -> None:
    _validate(cfg)
    _configure_compilation_cache(cfg.cache_dir)
    api = _load_repo_api()

    act_dtype = jnp.dtype(cfg.dtype)
    wdtype = jnp.dtype(cfg.wdtype)
    quantized = wdtype != act_dtype
    w_bits = jax.dtypes.itemsize_bits(wdtype)
    selected = [n.strip() for n in cfg.only.split(",") if n.strip()]

    device = jax.devices()[0]
    print(f"device={device.device_kind} ({device.platform}), "
          f"hidden={cfg.hidden}, moe_intermediate={cfg.moe_intermediate}, "
          f"topk={cfg.topk}, local_experts={cfg.num_local_experts}/"
          f"{cfg.num_experts}, act={act_dtype.name}, w={wdtype.name}, "
          f"activation={cfg.activation!r}, skew={cfg.skew}, "
          f"iters={cfg.iters}, warmup={cfg.warmup}")
    print(f"m list read as {cfg.m_semantics}; rows = tokens * topk; "
          f"work = sum(group_sizes); gmm_valid_frac={cfg.gmm_valid_frac}; "
          f"RAGGED_GATHER_REDUCE_VERSION={api.selected_reduce_version}")
    print("combine_sc_v2/v3 fall back to a plain-JAX reduce for shapes their "
          "kernels reject; a suspiciously flat row is that fallback.")
    print(_HEADER)

    weights = make_weights(cfg, wdtype, quantized)
    jax.block_until_ready((weights["w13"], weights["w2"]))

    for case in _cases(cfg):
        note_prefix = ("" if case.routed_rows == case.requested_m
                       or cfg.m_semantics == "tokens" else
                       f"(m={case.requested_m} snapped) ")
        variants = build_variants(api, cfg, case, weights, act_dtype, w_bits)
        for variant in variants:
            if selected and variant.name not in selected:
                continue
            if variant.fn is None:
                _print_row(case, variant, None, "-",
                           note_prefix + variant.note)
                continue
            trace_dir = (os.path.join(cfg.trace_dir,
                                      f"{variant.name}_m{case.routed_rows}")
                         if cfg.trace_dir else None)
            try:
                result = measure(variant.name,
                                 variant.fn,
                                 *variant.args,
                                 num_iters=cfg.iters,
                                 warmup=cfg.warmup,
                                 trace_dir=trace_dir,
                                 prefer_device=cfg.device_time)
            except Exception as err:  # noqa: BLE001 - report and keep sweeping
                _print_row(
                    case, variant, None, "-",
                    f"{note_prefix}FAILED: {type(err).__name__}: "
                    f"{str(err)[:120]}")
                continue
            note = note_prefix + variant.note
            if result.method != DEVICE:
                note = (note + " " if note else "") + "(host clock)"
            _print_row(case, variant, result.us, result.method, note)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--m",
        default=_DEFAULT_M,
        help="comma-separated sweep points (see --m-semantics)")
    parser.add_argument("--m-semantics",
                        default="rows",
                        choices=("rows", "tokens"),
                        help="read --m as routed rows (num_tokens*topk) or as "
                        "token counts; rows are always snapped to tokens*topk")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--num-local-experts", type=int, default=64)
    parser.add_argument("--num-experts",
                        type=int,
                        default=512,
                        help="global expert count; only the end-to-end path "
                        "sees it (it drives how many rows are local)")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--moe-intermediate", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16", help="activation dtype")
    parser.add_argument("--wdtype",
                        default="float8_e4m3fn",
                        help="weight dtype")
    parser.add_argument("--quant-block-size",
                        type=int,
                        default=0,
                        help="weight quant block along K; 0 = per-channel, "
                        "which is what MOE_REQUANTIZE_BLOCK_SIZE unset gives")
    parser.add_argument("--activation",
                        default="silu",
                        help="gmm_v2 fuse_act string for gmm1")
    parser.add_argument("--skew",
                        type=float,
                        default=0.0,
                        help="expert-load imbalance: 0 = uniform, s ramps the "
                        "per-expert load from (1+s)x to (1-s)x the mean")
    parser.add_argument(
        "--gmm-valid-frac",
        type=float,
        default=1.0,
        help="fraction of the routed rows that carry expert "
        "work in the standalone GMM benches; the EP end-to-end "
        "path effectively runs at 1/ep_size")
    parser.add_argument("--onehot-threshold",
                        type=int,
                        default=_PROD_ONEHOT_THRESHOLD,
                        help="onehot_moe_permute_threshold for e2e_onehot")
    parser.add_argument("--skip-padded-tokens", action="store_true")
    parser.add_argument("--only",
                        default="",
                        help="comma-separated subset of "
                        f"{','.join(_VARIANT_NAMES)}")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-onehot-bytes",
                        type=float,
                        default=8 * 2**30,
                        help="skip one-hot variants whose one-hot matrix is "
                        "larger than this")
    parser.add_argument(
        "--device-time",
        action="store_true",
        help="use jax.profiler device time instead of the "
        "pipelined host clock. Only on hosts whose jaxlib and "
        "libtpu agree on the profiler plugin ABI -- where they "
        "do not, starting a trace SIGSEGVs the process")
    parser.add_argument("--trace-dir",
                        default=None,
                        help="keep every xprof trace under this dir "
                        "(<variant>_m<rows>) instead of a temp dir")
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
