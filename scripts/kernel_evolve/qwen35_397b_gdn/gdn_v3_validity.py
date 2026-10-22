r"""Validity adapter for the fused conv1d + GDN v3 kernel at Qwen3.5-397B-A17B-FP8 serving shapes.

Run it through perfkit, never directly:

    perfkit validity run --adapter scripts/kernel_evolve/qwen35_397b_gdn/gdn_v3_validity.py \
        --arm <checkout> --python <a python with jax 0.10.2 and libtpu 0.0.47>

`perfkit/validity_arm.py` defines the hooks and the two checks. Nothing here
uses a TPU: compiles are ahead of time for the tpu7x topology on the CPU, and
trip counts come from Pallas interpret mode on the CPU.

What the server sends
---------------------
The recipe serves TP1 x DP8 with `--max-num-seqs=64`,
`--max-num-batched-tokens=1024`, `--max-model-len=10240`, fp8 KV,
`--block-size 256`, the CUSTOM attention backend, no prefix caching and no
speculative decoding. Each engine calls the GDN op once per GDN layer per step
through `run_jax_gdn_attention_pooled_local` (layers/core/gdn_attention.py),
which builds the unified-pool state plan and calls `fused_conv1d_gdn`; this
adapter calls that same function, so the state plan, the tiling and every
default are the server's.

  * Token axis: every forward is padded to a bucket of
    `_get_token_paddings(16, 1024)` (platforms/tpu_platform.py), so
    16, 32, ..., 512, 1024.
  * Sequence axis: `seq_lens`, `query_start_loc` and the state indices are
    sized `num_reqs_max_model_len` = 64 (`--max-num-seqs`) for real steps,
    warmup, the step profile and idle-DP dummies alike
    (runner/tpu_runner.py `_prepare_inputs`, `_dummy_run` callers), so the
    request-padding ladder never reaches this op: one shape per token bucket.
  * State: the unified pool, `(blocks, 128, 1, 4, 256)` fp8. Each GDN slot
    needs 4096 pool tokens of fp32 SSM state plus 72 of bf16 conv state,
    rounded up to 128, so the manager block is 4224 tokens, which the CUSTOM
    backend's kernel block sizes (128, 256) split into 33 blocks of 128.

The kernel runs two pallas_calls per op, BATCHED over the leading decodes and
PER_SEQ over the rest, each with one `emit_pipeline` over a grid read from
run-time metadata.

What it checks
--------------
`compile` lowers that call for tpu7x and compiles it; `limit` refuses a program
with no Mosaic kernel, or one that asks for more scoped VMEM than a tpu7x core
has.

`trip_counts` runs the same call in interpret mode, with every
`emit_pipeline` replaced by a loop that only evaluates the grid and each input
spec's index map and every DMA replaced by a no-op, and reads each pipeline's
grid at run time and the buffer count and lookahead of each input pipe from the
buffered refs the kernel hands its pipeline. Under EP the runner pads every
engine's forward to the largest bucket any DP engine needs
(layers/core/sequence_layout.py), so a bucket T can hold any token count up to
T. The runtime inputs, per bucket:

  * warmup and idle-DP dummy: min(T, 64) one-token sequences, all decodes;
  * n decodes alone, n = 1..min(T, 64);
  * n decodes beside a prefill chunk that fills the bucket, n = 1..min(T-1, 63);
  * one prefill chunk of every length p = 1..T;
  * the bucket split evenly over k prefills, k = 2..min(T, 64).

Pipeline bodies are not run. A DMA a kernel starts outside its pipelines is
counted, and an input that starts any while no pipe has more than two buffers
raises, since nothing there would have been checked.

The pipelines' output write-back (copy_out / wait_out) is not modelled.

What it was calibrated on
-------------------------
With jax 0.10.2 and libtpu 0.0.47, on throwaway copies of the kernel at
79bd30e0a:

  * the kernel as it is passes both checks; every pipe has 2 buffers.
  * `mixed_tile_size` 512 for 64 fails `aot_all_shapes` from t128 up, out of
    scoped VMEM, and compiles at t16..t64, where the tile clamps to the bucket.
  * `pl.Buffered(4)` on the three activation input pipes (state and output
    refs cannot take more than 2) fails `pipeline_trip_count` wherever a
    pipeline has 2 steps: 5..8 decodes on BATCHED in every bucket, and on
    PER_SEQ a prefill of 65..128 tokens, or a bucket of 16..64 over 2 prefills.
  * `pl.Buffered(3)` on the same pipes passes both.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import math
import os
import re
import sys
import types
from pathlib import Path

KERNEL_DIR = "src/vllm_torchtpu/kernels/gdn/v3"
TOPOLOGY = "tpu7x:2x2x1"

#: config.json's linear-attention geometry; TP1 keeps every head local.
N_KQ, N_V, D_K, D_V, CONV_KERNEL = 16, 64, 128, 128, 4
CONV_DIM = 2 * N_KQ * D_K + N_V * D_V

#: `num_reqs_max_model_len`: the sequence axis of every call.
NUM_SEQS = 64
TOKEN_PADDINGS = (16, 32, 64, 128, 256, 512, 1024)

#: Unified pool: 2 KV heads x K/V, fp8 packed 4 per row, head 256; a manager
#: block of 4224 tokens stored as kernel blocks of 128.
POOL_ROW = (1, 4, 256)
POOL_KERNEL_BLOCK = 128
POOL_BLOCK_TOKENS = 4224
POOL_SPLIT = POOL_BLOCK_TOKENS // POOL_KERNEL_BLOCK
POOL_BLOCKS = (NUM_SEQS + 1) * POOL_SPLIT

#: Scoped VMEM of one tpu7x TensorCore.
TPU7X_VMEM_BYTES = 64 * 1024 * 1024

SHAPES = {f"t{tokens}_s{NUM_SEQS}": {"tokens": tokens, "num_seqs": NUM_SEQS}
          for tokens in TOKEN_PADDINGS}


def _load_kernel(root: Path):
    """Import the serving GDN op from the arm root without the packages' `__init__`s.

    Those of `vllm_torchtpu` and `vllm_torchtpu.layers` import torch and the
    serving stack, which a compile-only environment does not have, so they are
    replaced by bare packages; every module the op runs is the arm's own file.
    `env_override` is loaded because it sets the server's LIBTPU_INIT_ARGS and
    applies its import-time Pallas patches, so the program compiled here is the
    one the server compiles.
    """
    pkg = root / "src" / "vllm_torchtpu"

    def package(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
        return mod

    # The plan reads one vLLM setting, the KV layout; the recipe leaves it at
    # vLLM's default, which is not HND.
    vllm = types.ModuleType("vllm")
    vllm.envs = types.ModuleType("vllm.envs")
    vllm.envs.VLLM_KV_CACHE_LAYOUT = None
    sys.modules["vllm"], sys.modules["vllm.envs"] = vllm, vllm.envs

    package("vllm_torchtpu", pkg)
    importlib.import_module("vllm_torchtpu.env_override")
    package("vllm_torchtpu.kernels", pkg / "kernels")
    package("vllm_torchtpu.layers", pkg / "layers")
    package("vllm_torchtpu.layers.core", pkg / "layers" / "core")
    # gdn_attention takes one helper from the torch-side utils, for its sharded
    # entry points; the per-engine function called here does not use it.
    utils = types.ModuleType("vllm_torchtpu.utils")

    def get_mesh_shape_product(*args, **kwargs):
        raise NotImplementedError("the validity adapter calls no sharded entry point")

    utils.get_mesh_shape_product = get_mesh_shape_product
    sys.modules[utils.__name__] = utils
    return (importlib.import_module("vllm_torchtpu.layers.core.gdn_attention"),
            importlib.import_module("vllm_torchtpu.kernels.gdn.v3.wrapper"))


def setup(root):
    # CPU only: jax must never open the host's TPU, and the compile-only
    # topology does not need the libtpu lock.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ["TPU_ACCELERATOR_TYPE"] = "tpu7x-4"
    # The recipe's server leaves the conv state in its plain pool layout.
    os.environ.pop("TPU_GDN_CONV_QK_PAIR_LAYOUT", None)
    sys.dont_write_bytecode = True
    op, wrapper = _load_kernel(Path(root).resolve())

    import jax
    import jax.numpy as jnp
    import numpy as np

    return types.SimpleNamespace(op=op, wrapper=wrapper, jax=jax, jnp=jnp, np=np,
                                 device=None)


def origin(handle) -> str:
    return handle.wrapper.__file__


def _entry(handle):
    """The op as one engine's GDN layer calls it (TP1, so shard_map is the identity)."""
    jnp = handle.jnp

    def call(qkv, b, a, pool, conv_weight, a_log, dt_bias, query_start_loc,
             state_indices, distribution, seq_lens):
        return handle.op.run_jax_gdn_attention_pooled_local(
            qkv, b, a, pool, conv_weight, None, a_log, dt_bias, query_start_loc,
            state_indices, distribution, seq_lens,
            n_kq=N_KQ, n_v=N_V, d_k=D_K, d_v=D_V, kernel_size=CONV_KERNEL,
            pool_block_tokens=POOL_BLOCK_TOKENS, recurrent_state_dtype=jnp.float32,
            num_spec_tokens=0)

    return call


def _arg_specs(handle, params, pool_blocks=POOL_BLOCKS):
    """Serving dtypes: bf16 activations and conv weight (no bias), fp32 A_log, bf16 dt_bias."""
    jnp = handle.jnp
    t, s = params["tokens"], params["num_seqs"]
    return [((t, CONV_DIM), jnp.bfloat16), ((t, N_V), jnp.bfloat16), ((t, N_V), jnp.bfloat16),
            ((pool_blocks, POOL_KERNEL_BLOCK, *POOL_ROW), jnp.float8_e4m3fn),
            ((CONV_DIM, 1, CONV_KERNEL), jnp.bfloat16), ((N_V,), jnp.float32),
            ((N_V,), jnp.bfloat16), ((s + 1,), jnp.int32), ((s,), jnp.int32),
            ((3,), jnp.int32), ((s,), jnp.int32)]


def _tpu7x_mesh():
    """Trace as for one tpu7x core, so the kernel's chip queries answer for it."""
    from jax._src import mesh as mesh_lib

    return mesh_lib.use_abstract_mesh(mesh_lib.AbstractMesh(
        (), (), abstract_device=mesh_lib.AbstractDevice(
            device_kind="TPU7x", num_cores=1, platform="tpu")))


def compile(handle, name, params):
    jax = handle.jax
    if handle.device is None:
        from jax.experimental import topologies

        handle.device = topologies.get_topology_desc(TOPOLOGY, platform="tpu").devices[0]
    sharding = jax.sharding.SingleDeviceSharding(handle.device)
    args = [jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
            for shape, dtype in _arg_specs(handle, params)]
    with _tpu7x_mesh():
        lowered = jax.jit(_entry(handle), donate_argnums=(3,)).trace(*args).lower(
            lowering_platforms=("tpu",))
    mem = lowered.compile().memory_analysis()
    text = lowered.as_text()
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "mosaic": text.count("tpu_custom_call"),
        "kernels": ",".join(sorted(set(re.findall(r'kernel_name = "([^"]+)"', text)))),
        # A kernel's vmem_limit_bytes is its scoped VMEM config (memory space 1).
        "vmem_limit_bytes": max(map(int, re.findall(
            r'"memory_space":\s*1,[^}]*"size":\s*(\d+)', text.replace("\\22", '"'))),
            default=0),
    }


def limit(name, result):
    if not result["mosaic"]:
        return "the compiled program launches no Mosaic kernel, so the compile checked nothing"
    if result["vmem_limit_bytes"] > TPU7X_VMEM_BYTES:
        return (f"a kernel asks for {result['vmem_limit_bytes']} bytes of scoped VMEM, more "
                f"than the {TPU7X_VMEM_BYTES} a tpu7x core has")
    return None


class _NoCopy:
    """A DMA that copies nothing and reports each run-time start to `on_start`."""

    def __init__(self, jax, on_start):
        self._jax, self._on_start = jax, on_start

    def start(self, *args, **kwargs):
        self._jax.debug.callback(self._on_start)

    def wait(self, *args, **kwargs):
        pass


def _input_pipes(jax_pipeline, in_specs, allocations, num_stages):
    """`(label, buffer_count, lookahead, index_map)` for every multiple-buffered input.

    Read from the buffered refs the pipeline rotates when the kernel passes its
    own, as `emit_pipeline` does, else from the specs as it would build them.
    """
    import jax

    specs = jax.tree.leaves(in_specs)
    found = []
    if allocations is None:
        for i, spec in enumerate(specs):
            mode = spec.pipeline_mode
            found.append((f"in[{i}]", mode.buffer_count if mode else 2,
                          bool(mode and mode.use_lookahead), spec.index_map))
        return found
    brefs = jax.tree.leaves(allocations,
                            is_leaf=lambda x: isinstance(x, jax_pipeline.BufferedRef))
    for i, bref in enumerate(brefs[:len(specs)]):
        if not bref.is_input or not bref.is_buffered or bref.is_trivial_windowing:
            continue
        if bref.buffer_count > num_stages:
            # The prologue issues only num_stages - 1 fetches, a schedule the
            # trip-count rule does not describe.
            raise RuntimeError(f"input {i} rotates {bref.buffer_count} buffers but its "
                               f"pipeline's specs ask for at most {num_stages}")
        found.append((f"in[{i}]", bref.buffer_count, bref.use_lookahead,
                      bref.compute_index))
    return found


@contextlib.contextmanager
def _interpret(handle):
    """Run the op in interpret mode and record its pipelines instead of running them.

    Yields a namespace collecting, per `emit_pipeline` traced, the kernel it is
    in and its input pipes, and at run time its grid and the block index each
    pipe reads at each step. `copies` counts the DMAs started outside any
    pipeline, since pipeline bodies are not run.
    """
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax._src.pallas import core as pallas_core
    from jax._src.pallas.mosaic import pipeline as jax_pipeline
    from jax._src.pallas.mosaic import tpu_info

    jax, jnp = handle.jax, handle.jnp
    seen = types.SimpleNamespace(kernel=None, pipes=[], grids=[], blocks={}, copies=0)
    chip = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    real_pallas_call = pl.pallas_call

    def pallas_call(kernel, *args, name=None, **kwargs):
        kwargs.pop("interpret", None)
        run = real_pallas_call(kernel, *args, name=name, interpret=True, **kwargs)

        def call(*operands):
            seen.kernel = name or "pallas_call"
            return run(*operands)

        return call

    def emit_pipeline(body, *, grid, in_specs=(), out_specs=(), **kwargs):
        index = len(seen.pipes)
        seen.pipes.append((seen.kernel, []))
        num_stages = max((2, *(s.pipeline_mode.buffer_count
                               for s in jax.tree.leaves((in_specs, out_specs))
                               if s.pipeline_mode is not None)))

        def on_grid(*dims):
            seen.grids.append((index, tuple(int(d) for d in dims)))

        def on_step(sizes, step, *flat):
            flat, blocks = [int(x) for x in flat], []
            for size in sizes:
                blocks.append(tuple(flat[:size]))
                flat = flat[size:]
            seen.blocks.setdefault(index, {})[int(step)] = blocks

        def pipeline(*refs, scratches=None, allocations=None, **kw):
            pipes = _input_pipes(jax_pipeline, in_specs, allocations, num_stages)
            seen.pipes[index] = (seen.pipes[index][0],
                                 [(label, b, lookahead) for label, b, lookahead, _ in pipes])

            def step(i, indices):
                # The pipeline's own step order and index arithmetic
                # (`_emit_pipeline`'s loop body).
                indices = jax_pipeline._filter_indices(indices, grid)
                with pallas_core.grid_env(list(map(pallas_core.GridAxis, indices, grid))):
                    blocks = [[jnp.asarray(x, jnp.int32)
                               for x in jax.tree.leaves(index_map(*indices))]
                              for _, _, _, index_map in pipes]
                jax.debug.callback(functools.partial(on_step, [len(b) for b in blocks]),
                                   i, *[x for b in blocks for x in b])
                return jax_pipeline._next_index(indices, grid)

            dims = [jnp.asarray(g, jnp.int32) for g in grid]
            jax.debug.callback(on_grid, *dims)
            lax.fori_loop(0, math.prod(dims), step, (0,) * len(grid))

        return pipeline

    def on_copy():
        seen.copies += 1

    def make_async_copy(*args, **kwargs):
        return _NoCopy(jax, on_copy)

    def async_copy(*args, **kwargs):
        copy = _NoCopy(jax, on_copy)
        copy.start()
        return copy

    # Patched on the shared modules, so every kernel module sees them.
    replaced = {(pl, "pallas_call"): pallas_call,
                (pltpu, "emit_pipeline"): emit_pipeline,
                (pltpu, "make_async_copy"): make_async_copy,
                (pltpu, "async_copy"): async_copy,
                (pltpu, "get_tpu_info"): lambda: chip,
                (tpu_info, "get_tpu_info"): lambda: chip}
    saved = {key: getattr(*key) for key in replaced}
    for (mod, attr), value in replaced.items():
        setattr(mod, attr, value)
    try:
        yield seen
    finally:
        for (mod, attr), value in saved.items():
            setattr(mod, attr, value)


def _runtime_inputs(tokens):
    """`(description, query lens, decodes, warmup)` for every runtime input in a bucket.

    Decodes come first, as `_reorder_batch_for_rpa` orders a batch. Under EP
    every engine pads to the largest bucket any DP engine needs, so a bucket
    holds any token count up to its size.
    """
    most = min(tokens, NUM_SEQS)
    yield f"warmup / idle-DP dummy: {most} one-token requests", [1] * most, most, True
    for n in range(1, most + 1):
        yield f"{n} decodes alone", [1] * n, n, False
    for n in range(1, min(tokens - 1, NUM_SEQS - 1) + 1):
        yield (f"{n} decodes beside a {tokens - n}-token prefill chunk",
               [1] * n + [tokens - n], n, False)
    for p in range(1, tokens + 1):
        yield f"a {p}-token prefill chunk alone", [p], 0, False
    for k in range(2, most + 1):
        base, extra = divmod(tokens, k)
        yield (f"{tokens} tokens over {k} prefills",
               [base + 1] * extra + [base] * (k - extra), 0, False)


def _metadata(np, query_lens, decodes, warmup):
    """query_start_loc, state_indices, distribution and seq_lens as the runner builds them."""
    live = len(query_lens)
    q = np.zeros(NUM_SEQS, np.int32)
    q[:live] = query_lens
    qsl = np.concatenate([[0], np.cumsum(q)]).astype(np.int32)
    state = np.zeros(NUM_SEQS, np.int32)
    state[:live] = np.arange(1, live + 1)
    if warmup:
        seq_lens = np.ones(NUM_SEQS, np.int32)
        state[:] = 0
    else:
        # A request with history resumes from its state; decodes have one.
        seq_lens = q + (np.arange(NUM_SEQS) < decodes) * 128
    return qsl, state, np.array([decodes, decodes, live], np.int32), seq_lens.astype(np.int32)


def pipelines_at(handle, params, inputs):
    """`[(description, [(pipe, buffer_count, lookahead, trip_count, blocks), ...])]`."""
    jax, jnp, np = handle.jax, handle.jnp, handle.np
    out = []
    # Interpret mode moves no state, so a pool of two slots serves every input.
    specs = _arg_specs(handle, params, pool_blocks=2 * POOL_SPLIT)
    with _interpret(handle) as seen, _tpu7x_mesh():
        run = jax.jit(_entry(handle))
        dense = [jnp.zeros(s, d) for s, d in specs[:7]]
        for description, query_lens, decodes, warmup in inputs:
            assert sum(query_lens) <= params["tokens"], description
            seen.grids.clear()
            seen.blocks.clear()
            seen.copies = 0
            meta = [jnp.asarray(x) for x in _metadata(np, query_lens, decodes, warmup)]
            jax.block_until_ready(run(*dense, *meta))
            jax.effects_barrier()
            if not seen.grids:
                raise RuntimeError("the kernel ran no emit_pipeline this adapter can see, so "
                                   "its trip counts are unknown")
            if seen.copies and not any(b > 2 for i, _ in seen.grids
                                       for _, b, _ in seen.pipes[i][1]):
                raise RuntimeError(
                    f"the kernel started {seen.copies} DMAs of its own and pipelines no "
                    f"input with more than two buffers, so nothing here was checked")
            found = []
            for i, grid in seen.grids:
                n = math.prod(grid)
                steps = seen.blocks.get(i, {})
                if sorted(steps) != list(range(n)):
                    raise RuntimeError(f"pipeline {i} reported block indices for "
                                       f"{len(steps)} of its {n} steps")
                kernel, pipes = seen.pipes[i]
                for j, (pipe, b, lookahead) in enumerate(pipes):
                    found.append((f"{kernel}:{pipe}", b, lookahead, n,
                                  [steps[s][j] for s in range(n)]))
            out.append((description, found))
    return out


def trip_counts(handle, name, params):
    found = pipelines_at(handle, params, _runtime_inputs(params["tokens"]))
    return [{"pipe": pipe, "buffer_count": b, "trip_count": n, "lookahead": lookahead,
             "blocks": blocks, "input": description}
            for description, pipes in found for pipe, b, lookahead, n, blocks in pipes]
