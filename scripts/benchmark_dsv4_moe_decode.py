# SPDX-License-Identifier: Apache-2.0
"""Standalone routed-MoE benchmark; see benchmark_dsv4_moe_decode.md."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import glob
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EP = 8
MESH_AXIS = "d"  # Checked against the v2 kernel's exported AXIS at import.
BLOCK = 512
ACTIVATION = "silu_and_mul_with_clamp"
# Existing FP4+FP8-wire acceptance bands, not fitted to these measurements:
# tests/kernels/fused_moe/test_fused_ep_moe_v2.py, test_fp4_block512_... .
REL_BOUND = 0.06
TOKEN_BOUND = 0.075


@dataclasses.dataclass(frozen=True)
class Shape:
    hidden: int = 4096
    inter: int = 2048
    experts: int = 256
    topk: int = 6


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def statistics(values: list[float]) -> dict:
    a = np.asarray(values, np.float64)
    if not a.size or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("Timings must be nonempty, finite, positive milliseconds")
    return dict(
        count=len(a),
        mean_ms=float(a.mean()),
        std_ms=float(a.std()),
        p50_ms=float(np.percentile(a, 50)),
        p90_ms=float(np.percentile(a, 90)),
        p99_ms=float(np.percentile(a, 99)),
    )


def errors(actual: np.ndarray, reference: np.ndarray) -> dict:
    a, r = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if a.shape != r.shape or a.ndim != 2:
        raise ValueError("Expected matching [tokens, hidden] outputs")
    if not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError("Non-finite MoE output")
    delta = a - r
    norms = np.linalg.norm(r, axis=1)
    token = np.linalg.norm(delta, axis=1) / np.where(norms == 0, 1, norms)
    norm = np.linalg.norm(r)
    rel = float(np.linalg.norm(delta) / (norm if norm else 1))
    return dict(
        relative_l2=rel,
        max_abs=float(np.max(np.abs(delta))),
        worst_token_relative_l2=float(token.max()),
        per_token_relative_l2=token.tolist(),
        per_token_max_abs=np.max(np.abs(delta), axis=1).tolist(),
        passed=rel < REL_BOUND and float(token.max()) < TOKEN_BOUND,
        relative_bound=REL_BOUND,
        token_bound=TOKEN_BOUND,
    )


def cpu_routes(logits: np.ndarray, bias: np.ndarray, topk: int) -> tuple:
    """Independent FP32 DSV4 routing oracle, checked against moe_routing.route."""
    scores = np.sqrt(np.logaddexp(np.float32(0), logits.astype(np.float32)))
    ids = np.argsort(-(scores + bias), axis=1, kind="stable")[:, :topk]
    weights = np.take_along_axis(scores, ids, axis=1)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-20)
    return ids.astype(np.int32), weights * np.float32(1.5)


def make_inputs(
    shape: Shape, tokens: int, bias: np.ndarray, seed: int, distribution: str
) -> tuple:
    """Synthetic hidden and logits; bias-aware choices keep load reproducible."""
    rng = np.random.default_rng(seed)
    hidden = rng.normal(0, 1, (tokens, shape.hidden)).astype(np.float32)
    if distribution == "balanced":
        permutation = rng.permutation(shape.experts)
        chosen = permutation[
            np.arange(tokens * shape.topk).reshape(tokens, shape.topk) % shape.experts
        ]
    elif distribution == "skewed":
        # All tokens choose distinct experts on rank 0: no dropped/capped routes.
        chosen = np.stack(
            [
                rng.choice(shape.experts // EP, shape.topk, replace=False)
                for _ in range(tokens)
            ]
        )
    else:
        raise ValueError(distribution)
    selection = rng.uniform(0, 0.02, (tokens, shape.experts)).astype(np.float32)
    selection[np.arange(tokens)[:, None], chosen] += 0.5
    scores = selection + np.max(bias) - bias + 1.0
    squared = scores * scores
    logits = squared + np.log(-np.expm1(-squared))
    return hidden, logits.astype(np.float32)


class TensorReader:
    """Read only named tensor byte ranges, never materialize an entire shard."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.weight_map = json.loads(
            (directory / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.headers = {}
        self.records = {}

    def read(self, key: str) -> np.ndarray:
        filename = self.weight_map[key]
        with (self.directory / filename).open("rb") as stream:
            if filename not in self.headers:
                size = struct.unpack("<Q", stream.read(8))[0]
                self.headers[filename] = (8 + size, json.loads(stream.read(size)))
            start, header = self.headers[filename]
            item = header[key]
            lo, hi = item["data_offsets"]
            stream.seek(start + lo)
            raw = stream.read(hi - lo)
        if len(raw) != hi - lo:
            raise ValueError(f"Truncated tensor: {key}")
        dtype = {
            "I8": np.uint8,
            "U8": np.uint8,
            "F8_E8M0": np.uint8,
            "F32": np.dtype("<f4"),
        }[item["dtype"]]
        data = np.frombuffer(raw, dtype=dtype).reshape(item["shape"]).copy()
        self.records[key] = dict(file=filename, **item, sha256=digest(raw))
        return data


def configuration(model: Path | None, smoke: bool) -> tuple:
    if model is None:
        config = dict(
            hidden_size=4096,
            moe_intermediate_size=2048,
            n_routed_experts=256,
            num_experts_per_tok=6,
            scoring_func="sqrtsoftplus",
            norm_topk_prob=True,
            routed_scaling_factor=1.5,
            swiglu_limit=10.0,
            num_hash_layers=3,
        )
    else:
        config = json.loads((model / "config.json").read_text())
    expected = dict(
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        hidden_size=4096,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        num_experts_per_tok=6,
    )
    for key, value in expected.items():
        if config[key] != value:
            raise NotImplementedError(f"Unsupported config {key}={config[key]!r}")
    # EP8 needs >= topk local experts to exercise rank-skew without ties.
    shape = Shape(1024, 512, 64, 6) if smoke else Shape()
    return shape, config


def source_version() -> dict:
    manifest_path = ROOT / "BENCH_SOURCE.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, sha in manifest["files"].items():
            if digest((ROOT / name).read_bytes()) != sha:
                raise ValueError(f"Source snapshot changed: {name}")
        return manifest
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = list((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    return dict(
        base_commit=revision,
        files={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sorted(paths)},
    )


def preflight() -> dict:
    """No JAX/torch imports here: do not claim devices to inspect occupancy."""
    nodes = sorted(glob.glob("/dev/vfio/[0-9]*") + glob.glob("/dev/accel*"))
    if not nodes:
        raise RuntimeError("No TPU device nodes found; hardware run refused")
    result = subprocess.run(
        ["sudo", "-n", "fuser", "-v", *nodes], text=True, capture_output=True
    )
    if result.stdout.strip() or result.returncode == 0:
        raise RuntimeError(
            "TPUs occupied; nothing stopped: " + result.stdout + result.stderr
        )
    if result.returncode != 1 or result.stderr.strip():
        raise RuntimeError("Cannot reliably check device ownership: " + result.stderr)
    packages = {}
    for name in (
        "jax",
        "jaxlib",
        "libtpu",
        "torch",
        "torch-tpu",
        "vllm",
        "numpy",
        "tpu-sync-torch",
    ):
        packages[name] = importlib.metadata.version(name)
    for name, version in {
        "jax": "0.11.2",
        "jaxlib": "0.11.2",
        "libtpu": "0.0.47",
    }.items():
        if packages[name] != version:
            raise RuntimeError(f"Expected {name}=={version}, got {packages[name]}")
    return dict(
        host=platform.node(),
        platform=platform.platform(),
        python=sys.version,
        packages=packages,
        device_nodes=nodes,
        occupancy_stdout=result.stdout,
        occupancy_stderr=result.stderr,
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                (
                    "TPU_",
                    "TORCH_TPU_",
                    "XLA_",
                    "JAX_",
                    "LIBTPU_",
                    "MOE_",
                    "USE_MOE_",
                    "ONEHOT_",
                    "RAGGED_",
                    "VLLM_MOE_",
                )
            )
        },
    )


def load_runtime():
    """Import exactly the supplied checkout, with unrelated MoE paths disabled."""
    os.environ["TPU_MOE_DENSE_EXPERT_THRESHOLD"] = "0"
    os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = ""
    os.environ["MOE_REQUANTIZE_BLOCK_SIZE"] = str(BLOCK)
    sys.path.insert(0, str(ROOT / "src"))
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.fused_moe.v2 import AXIS
    from vllm_torchtpu.kernels.fused_moe.v2 import layer as ep_layer
    from vllm_torchtpu.layers.core import fused_moe_gmm as gmm

    for module in (ep_layer, gmm):
        if not Path(module.__file__).resolve().is_relative_to(ROOT / "src"):
            raise RuntimeError(f"Mixed source versions: {module.__file__}")
    devices = jax.devices()
    if len(devices) != EP or any(d.platform != "tpu" for d in devices):
        raise RuntimeError(f"Expected exactly {EP} TPU devices, got {devices}")
    if AXIS != MESH_AXIS:
        raise RuntimeError("Benchmark mesh axis disagrees with fused kernel")
    mesh = jax.sharding.Mesh(np.asarray(devices), (MESH_AXIS,))
    return jax, jnp, ep_layer, gmm, mesh


def weights(shape: Shape, mesh, model: Path | None, layer: int, output: Path):
    import jax
    import jax.numpy as jnp
    import torch
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.layers.core.fused_moe_gmm import quantize_to_native_fp4_kmajor
    from vllm_torchtpu.layers.core.quantization import dequantize_mxfp4_packed

    torch.set_num_threads(4)
    reader = TensorReader(model) if model is not None else None
    rng = np.random.default_rng(20261009)
    prefix = f"layers.{layer}.ffn"
    bias = (
        reader.read(prefix + ".gate.bias")
        if reader
        else rng.normal(0, 0.02, shape.experts).astype(np.float32)
    )
    if bias.shape != (shape.experts,):
        raise ValueError("Score-bias shape mismatch")
    local_experts = shape.experts // EP
    pieces = [[] for _ in range(6)]

    @jax.jit
    def prepare(w1, w2):
        native1, scale1 = quantize_to_native_fp4_kmajor(w1, BLOCK, pack=False)
        native2, scale2 = quantize_to_native_fp4_kmajor(w2, BLOCK, pack=False)
        packed1, _ = quantize_to_native_fp4_kmajor(w1, BLOCK, pack=True)
        packed2, _ = quantize_to_native_fp4_kmajor(w2, BLOCK, pack=True)
        return native1, native2, packed1, packed2, scale1, scale2

    for rank, device in enumerate(mesh.devices.flat):
        w13, w2 = [], []
        for expert in range(rank * local_experts, (rank + 1) * local_experts):
            matrices = []
            for name, dims in (
                ("w1", (shape.inter, shape.hidden)),
                ("w3", (shape.inter, shape.hidden)),
                ("w2", (shape.hidden, shape.inter)),
            ):
                if reader:
                    key = f"{prefix}.experts.{expert}.{name}"
                    packed, scale = (
                        reader.read(key + ".weight"),
                        reader.read(key + ".scale"),
                    )
                    tensor = dequantize_mxfp4_packed(
                        torch.from_numpy(packed),
                        torch.from_numpy(scale),
                        axis=1,
                        out_dtype=torch.bfloat16,
                    )
                    if tuple(tensor.shape) != dims:
                        raise ValueError(f"Unexpected expert shape: {key}")
                else:
                    tensor = torch.from_numpy(
                        rng.normal(0, 0.02, dims).astype(np.float32)
                    ).bfloat16()
                matrices.append(tensor)
            # Match VllmDeepseekV4Mxfp4MoEMethod._dequantize_and_pad exactly.
            w13.append(torch.cat(matrices[:2], dim=0).T.contiguous())
            w2.append(matrices[2].T.contiguous())
        one = torch.stack(w13).float().numpy()
        two = torch.stack(w2).float().numpy()
        prepared = prepare(
            jax.device_put(one, device).astype(jnp.bfloat16),
            jax.device_put(two, device).astype(jnp.bfloat16),
        )
        jax.block_until_ready(prepared)
        for slot, item in zip(pieces, prepared):
            slot.append(item)
        print(f"Prepared rank {rank}: {local_experts} experts", flush=True)
    arrays = tuple(
        jax.make_array_from_single_device_arrays(
            (shape.experts, *parts[0].shape[1:]),
            NamedSharding(mesh, P(MESH_AXIS)),
            parts,
        )
        for parts in pieces
    )
    save_json(
        output / "weights.json",
        dict(
            source="checkpoint" if reader else "synthetic",
            layer=layer,
            checkpoint_directory=str(model),
            tensors=reader.records if reader else {},
            requantization="checkpoint MXFP4 -> BF16 -> common FP4 block512",
            layouts=[
                dict(shape=list(a.shape), dtype=str(a.dtype), bytes=a.nbytes)
                for a in arrays
            ],
        ),
    )
    return arrays, bias


def build_paths(shape: Shape, mesh, ep_layer, gmm):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.fused_moe.v2 import AXIS, WeightFormat

    threshold = gmm.resolve_onehot_permute_threshold()

    def route(logits, bias):
        scores = ep_layer._apply_scoring_fn(logits, "sqrtsoftplus")
        _, ids = lax.top_k(scores + bias, shape.topk)
        values = jnp.take_along_axis(scores, ids, axis=1)
        values /= jnp.maximum(values.sum(axis=1, keepdims=True), 1e-20)
        return ids, values * jnp.float32(1.5)

    def grouped_local(x, logits, w1, w2, s1, s2, bias):
        # Same dispatch/combine as TpuDeviceCommunicator, expressed in JAX
        # so both paths run in the same single-controller EP8 execution model.
        hidden = lax.all_gather(x, AXIS, axis=0, tiled=True)
        gating = lax.all_gather(logits, AXIS, axis=0, tiled=True)
        ids, values = route(gating, bias)
        mine = gmm.fused_moe_func(
            hidden,
            w1,
            w2,
            s1,
            s2,
            None,
            None,
            values,
            ids,
            experts_start=lax.axis_index(AXIS) * (shape.experts // EP),
            topk=shape.topk,
            activation=ACTIVATION,
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=threshold,
            rhs_quant_dtype=jnp.float4_e2m1fn,
        )
        return lax.psum_scatter(mine, AXIS, scatter_dimension=0, tiled=True)

    grouped = jax.jit(
        jax.shard_map(
            grouped_local,
            mesh=mesh,
            in_specs=(P(AXIS),) * 6 + (P(),),
            out_specs=P(AXIS),
            check_vma=False,
        )
    )

    def fused(x, logits, w1, w2, s1, s2, bias):
        return ep_layer.fused_ep_moe_v2(
            x,
            w1,
            w2,
            s1.squeeze(2),
            s2.squeeze(2),
            logits.astype(jnp.float32),
            topk=shape.topk,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            weight_format=WeightFormat.FP4,
            rhs_qb=BLOCK,
            act_fn=ACTIVATION,
            scoring_fn="sqrtsoftplus",
            score_bias=bias,
            routed_scaling_factor=1.5,
            sharded_plan=False,
        )

    return {"grouped": grouped, "fused": jax.jit(fused)}, threshold


def build_reference(shape: Shape, mesh):
    """Independent dense FP32 per-expert oracle on the common decoded weights.

    Follows the existing v2 device test's sharded dense reference; adds the
    DSV4 routing operands and uses the repository's clamped activation.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.megablox.gmm_v2 import silu_and_mul_with_clamp

    def local(x, ids, scores, w1, w2, s1, s2):
        x = lax.all_gather(x, MESH_AXIS, axis=0, tiled=True).astype(jnp.float32)
        ids = lax.all_gather(ids, MESH_AXIS, axis=0, tiled=True)
        scores = lax.all_gather(scores, MESH_AXIS, axis=0, tiled=True)
        start = lax.axis_index(MESH_AXIS) * (shape.experts // EP)

        def expert(e, out):
            def decode(w, s):
                blocks = w.shape[0] // BLOCK
                return (
                    w.astype(jnp.float32).reshape(blocks, BLOCK, -1)
                    * s.astype(jnp.float32)
                ).reshape(w.shape)

            one, two = decode(w1[e], s1[e]), decode(w2[e], s2[e])
            a = jnp.matmul(x, one, precision=lax.Precision.HIGHEST)
            mid = silu_and_mul_with_clamp(a[:, : shape.inter], a[:, shape.inter :])
            y = jnp.matmul(mid, two, precision=lax.Precision.HIGHEST)
            scale = jnp.where(ids == e + start, scores, 0).sum(axis=1)
            return out + y * scale[:, None]

        mine = lax.fori_loop(0, shape.experts // EP, expert, jnp.zeros_like(x))
        return lax.psum_scatter(mine, MESH_AXIS, scatter_dimension=0, tiled=True)

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(MESH_AXIS),) * 7,
            out_specs=P(MESH_AXIS),
            check_vma=False,
        )
    )


def verify_router(hidden, logits, bias, topk: int) -> tuple:
    """Verify the standalone router oracle against the real Torch adapter."""
    from types import SimpleNamespace

    import torch

    from vllm_torchtpu.layers.adapter.moe_routing import select_experts

    ids, values = cpu_routes(logits, bias, topk)
    layer = SimpleNamespace(
        hash_indices_table=None,
        use_grouped_topk=False,
        e_score_correction_bias=torch.from_numpy(bias),
        routed_scaling_factor=1.5,
    )
    w, i = select_experts(
        torch.from_numpy(hidden).bfloat16(),
        torch.from_numpy(logits),
        topk=topk,
        renormalize=True,
        scoring_fn="sqrtsoftplus",
        layer=layer,
    )
    np.testing.assert_array_equal(i.numpy(), ids)
    np.testing.assert_allclose(w.float().numpy(), values, rtol=0.004, atol=0.002)
    return ids, values


def compile_path(fn, operands, name: str, directory: Path):
    lowered = fn.lower(*operands)
    lowered_text = lowered.as_text()
    required = "fused_ep_moe_v2" if name == "fused" else "gmm"
    if required not in lowered_text:
        raise RuntimeError(f"Required {required} kernel missing from lowered {name}")
    if "adaptive_fused_moe" in lowered_text or "dense_expert_moe" in lowered_text:
        raise RuntimeError("Unexpected algorithm selected")
    if "all_gather" not in lowered_text and "all-gather" not in lowered_text:
        raise RuntimeError(f"No all-gather in complete {name} path")
    if name == "grouped" and "reduce_scatter" not in lowered_text:
        raise RuntimeError("Grouped path omitted reduce-scatter")
    (directory / f"{name}.stablehlo.txt").write_text(lowered_text)
    executable = lowered.compile()
    (directory / f"{name}.optimized-hlo.txt").write_text(executable.as_text())
    return executable


def measure(executable, operands, warmup: int, iterations: int) -> list[float]:
    import jax

    for _ in range(warmup):
        jax.block_until_ready(executable(*operands))
    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = executable(*operands)
        jax.block_until_ready(result)
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return elapsed


def run_case(args, shape, arrays, bias, mesh, paths, reference, tokens, distribution):
    import jax
    import ml_dtypes
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    directory = args.output / f"{distribution}-dp{tokens}"
    directory.mkdir(parents=True, exist_ok=False)
    shard = NamedSharding(mesh, P(MESH_AXIS))
    bias_device = jax.device_put(bias, NamedSharding(mesh, P()))
    native1, native2, packed1, packed2, s1, s2 = arrays
    weight_args = {
        "grouped": (packed1, packed2, s1, s2),
        "fused": (native1, native2, s1, s2),
    }
    bank, routes = [], []
    for round_id in range(args.rounds):
        x, gate = make_inputs(shape, tokens * EP, bias, 100 + round_id, distribution)
        x = x.astype(ml_dtypes.bfloat16).astype(np.float32)
        gate = gate.astype(ml_dtypes.bfloat16).astype(np.float32)
        ids, values = verify_router(x, gate, bias, shape.topk)
        histogram = np.bincount(ids.ravel(), minlength=shape.experts)
        np.savez(
            directory / f"input-{round_id}.npz",
            hidden=x,
            logits=gate,
            ids=ids,
            weights=values,
            score_bias=bias,
        )
        routes.append(
            dict(
                round=round_id,
                expert_counts=histogram.tolist(),
                rank_counts=histogram.reshape(EP, -1).sum(1).tolist(),
            )
        )
        bank.append(
            tuple(
                jax.device_put(a, shard)
                for a in (
                    x.astype(ml_dtypes.bfloat16),
                    gate.astype(ml_dtypes.bfloat16),
                    ids,
                    values,
                )
            )
        )
    jax.block_until_ready(bank)
    save_json(directory / "routing.json", routes)

    def operands(which, n):
        return (*bank[n][:2], *weight_args[which], bias_device)

    compiled = {
        name: compile_path(fn, operands(name, 0), name, directory)
        for name, fn in paths.items()
    }
    validations = []
    for n, (x, _, ids, values) in enumerate(bank):
        want = np.asarray(reference(x, ids, values, native1, native2, s1, s2))
        np.save(directory / f"reference-{n}.npy", want)
        entry = {"round": n}
        for name in paths:
            actual = np.asarray(compiled[name](*operands(name, n)), dtype=np.float32)
            np.save(directory / f"{name}-output-{n}.npy", actual)
            entry[name] = errors(actual, want)
        validations.append(entry)
    save_json(directory / "correctness.json", validations)
    if not all(v[name]["passed"] for v in validations for name in paths):
        raise AssertionError(
            f"Correctness failed; see {directory / 'correctness.json'}"
        )
    # Each separately dispatched executable consumes runtime arguments. Three
    # banks change across rounds; no Python constants/loop get compiled away.
    rounds = []
    for n in range(args.rounds):
        order = ("grouped", "fused") if n % 2 == 0 else ("fused", "grouped")
        row = dict(round=n, order=order)
        for name in order:
            raw = measure(
                compiled[name], operands(name, n), args.warmup, args.iterations
            )
            row[name] = dict(raw_ms=raw, **statistics(raw))
        row["speedup_grouped_over_fused"] = (
            row["grouped"]["p50_ms"] / row["fused"]["p50_ms"]
        )
        rounds.append(row)
        save_json(directory / "rounds.json", rounds)
        print(
            f"{distribution} DP={tokens} round={n} "
            f"GMM={row['grouped']['p50_ms']:.3f}ms "
            f"EPv2={row['fused']['p50_ms']:.3f}ms",
            flush=True,
        )
    summary = dict(
        tokens_per_dp=tokens,
        global_tokens=tokens * EP,
        distribution=distribution,
        rounds=rounds,
    )
    for name in paths:
        summary[name] = statistics([v for r in rounds for v in r[name]["raw_ms"]])
    summary["speedup_grouped_over_fused"] = (
        summary["grouped"]["p50_ms"] / summary["fused"]["p50_ms"]
    )
    save_json(directory / "summary.json", summary)
    if args.trace:
        for name in paths:
            with jax.profiler.trace(str(directory / f"trace-{name}")):
                for step in range(5):
                    with jax.profiler.StepTraceAnnotation(name, step_num=step):
                        jax.block_until_ready(compiled[name](*operands(name, 0)))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "scan"), required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--synthetic-weights", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=("balanced", "skewed"),
        default=["balanced", "skewed"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.warmup < 20 or args.iterations < 100 or args.rounds < 3:
        parser.error("Require >=20 warmups, >=100 measurements, >=3 rounds")
    if args.mode == "scan" and bool(args.model_dir) == args.synthetic_weights:
        parser.error("Choose --model-dir or explicitly --synthetic-weights")
    if any(t not in (8, 16, 32, 64) for t in args.tokens):
        parser.error("Decode tokens per DP must be 8, 16, 32, or 64")
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "source.json", source_version())
    save_json(
        args.output / "arguments.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    # Cooperative lock plus a fresh privileged occupancy check before importing
    # any runtime. Never kill a device owner or change service/topology state.
    with open("/tmp/dsv4-moe-benchmark.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = preflight()
        save_json(args.output / "environment.json", environment)
        if args.mode == "preflight":
            print(json.dumps(environment, indent=2))
            return
        shape, config = configuration(args.model_dir, args.mode == "smoke")
        save_json(args.output / "model-config.json", config)
        jax, _, ep_layer, gmm, mesh = load_runtime()
        save_json(
            args.output / "hardware.json",
            [
                dict(
                    id=d.id,
                    kind=d.device_kind,
                    coords=list(d.coords),
                    core_on_chip=d.core_on_chip,
                    process_index=d.process_index,
                )
                for d in mesh.devices.flat
            ],
        )
        save_json(
            args.output / "effective-environment.json",
            {
                k: v
                for k, v in os.environ.items()
                if k.startswith(
                    ("TPU_", "XLA_", "JAX_", "LIBTPU_", "MOE_", "VLLM_MOE_")
                )
            },
        )
        model = None if args.mode == "smoke" else args.model_dir
        arrays, bias = weights(
            shape, mesh, model, config["num_hash_layers"], args.output
        )
        paths, threshold = build_paths(shape, mesh, ep_layer, gmm)
        save_json(
            args.output / "contract.json",
            dict(
                shape=dataclasses.asdict(shape),
                dp=EP,
                tp=1,
                ep=EP,
                onehot_threshold=threshold,
                chunk_pipeline=False,
                sharded_plan=False,
                hidden_logits="synthetic, BF16; not captured from model",
                weights="real single non-hash layer" if model else "synthetic",
                gate_projection_included=False,
                shared_expert_included=False,
                scope="one routed MoE layer, local hidden/logits to local output",
                implementation="single-controller JAX shard_map; no vLLM engine",
                grouped=dict(
                    weights="packed U8 carrying FP4 block512; FP32 scales",
                    activations="dynamic FP8, block512; BF16 GMM accumulation/output",
                    communication="BF16 hidden/logits all-gather; BF16 reduce-scatter",
                    routing="FP32 scoring; BF16 router weights in combine",
                ),
                fused=dict(
                    weights="native FP4 block512; same logical values/scales",
                    activations="rowwise FP8 input/intermediate; FP32 accumulation",
                    communication=(
                        "FP8 activation all-gather; I32 metadata "
                        "all-gather; FP8 return DMA"
                    ),
                    routing="FP32 scoring/weights/final weighted sum; BF16 output",
                ),
                tolerance_source="test_fused_ep_moe_v2.py FP4 transport bands",
                timing=(
                    "host wall time with block_until_ready every dispatch; "
                    "includes dispatch overhead"
                ),
                cache=(
                    "same resident weights/buffers, three input banks; not model TPOT"
                ),
            ),
        )
        reference = build_reference(shape, mesh)
        summaries = []
        tokens = [8] if args.mode == "smoke" else args.tokens
        for distribution in args.distributions:
            for count in tokens:
                summaries.append(
                    run_case(
                        args,
                        shape,
                        arrays,
                        bias,
                        mesh,
                        paths,
                        reference,
                        count,
                        distribution,
                    )
                )
                save_json(args.output / "results.json", summaries)
        jax.block_until_ready(arrays)
        save_json(args.output / "complete.json", dict(status="complete"))


if __name__ == "__main__":
    main()
