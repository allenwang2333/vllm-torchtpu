r"""Validity adapter for megablox gmm_v2 at Qwen3.5-397B-A17B-FP8 TP8+EP8 serving shapes.

Run it through perfkit, never directly (the campaign's frozen `validity` block):

    perfkit validity run --adapter scripts/evolve/gmm_v2_397b/validity.py --arm <checkout> \
        --python <a python with jax 0.10.2 and libtpu 0.0.47>

`perfkit/validity_arm.py` defines the hooks and the two checks. Nothing here uses
a TPU: compiles are ahead of time for the tpu7x topology on the CPU, and trip
counts come from Pallas interpret mode on the CPU with DMAs and pipeline bodies
replaced by no-ops. Adapted from forge's `scorers/gmm_v2_validity.py` (the 35B
campaign's), with this recipe's geometry; that file's docstring says what the
trip-count check was calibrated on (a deep weight pipe that halted the server in
warmup, `skill:tpu-silent-correctness-traps` trap 7).

What it checks
--------------
`SHAPES` is every shape this server sends to `gmm_v2`: the 15 padded token
counts of the recipe (`TPU_TOKEN_BUCKET_EXTRA=4,8,48,3072` on top of the
default ladder; the served log's "Using token paddings"), times top-10, for both
projections. The scorer freezes 4 of these 30.

`compile` builds each shape through the kernel's own `gmm_v2` entry point and
compiles it for tpu7x; `limit` refuses a program that launches no Mosaic kernel.

`trip_counts` runs the kernel body in interpret mode for each runtime input and
reads every `emit_pipeline`: its run-time grid, each input pipe's buffer count
and lookahead, and the block each pipe reads per step. The runtime inputs, per
shape: 0 to 8 active experts sharing the rank's rows (`size_m / 8`; at decode
that is 5 rows, and warmup's dummy batches reach 1 to 4 experts); one expert
spanning 2 to 8 gm tiles; every token routed to one expert.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib.util
import math
import os
import sys
import types
from pathlib import Path

KERNEL_DIR = "src/vllm_torchtpu/kernels/megablox"
TOPOLOGY = "tpu7x:2x2x1"

#: The model's MoE geometry under the serving recipe: 512 experts over EP=8 is
#: 64 on a rank, and each token picks 10.
NUM_GROUPS = 64
EP = 8
TOPK = 10
TOKEN_PADDINGS = (4, 8, 16, 32, 48, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192, 16384)
LEGS = {
    "gate_up": {"size_k": 4096, "size_n": 2048, "fuse_act": "silu", "out_dtype": None},
    "down": {"size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": "bfloat16"},
}
SHAPES = {f"{leg}_m{TOPK * tokens}": {"leg": leg, "size_m": TOPK * tokens, **LEGS[leg]}
          for leg in LEGS for tokens in TOKEN_PADDINGS}

MAX_ACTIVE = 8
MAX_SPAN = 8


def _load_kernel(root: Path):
    """Import gmm_v2 from the arm root without the package's own `__init__`.

    That `__init__` imports torch and the serving stack, which a compile-only
    environment does not have. `env_override` is loaded because it sets the
    server's LIBTPU_INIT_ARGS and applies its import-time Pallas patches, so
    the program compiled here is the one the server compiles.
    """
    pkg = root / "src" / "vllm_torchtpu"

    def package(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
        return mod

    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    top = package("vllm_torchtpu", pkg)
    for name in ("envs", "logger", "patch_registry"):
        setattr(top, name, module(f"vllm_torchtpu.{name}", pkg / f"{name}.py"))
    module("vllm_torchtpu.env_override", pkg / "env_override.py")
    package("vllm_torchtpu.kernels", pkg / "kernels")
    package("vllm_torchtpu.kernels.megablox", pkg / "kernels" / "megablox")
    return module("vllm_torchtpu.kernels.megablox.gmm_v2",
                  pkg / "kernels" / "megablox" / "gmm_v2.py")


def setup(root):
    # CPU only: jax must never open the host's TPU, and the compile-only
    # topology does not need the libtpu lock.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-4")
    sys.dont_write_bytecode = True
    kernel = _load_kernel(Path(root).resolve())

    import jax
    import jax.numpy as jnp
    import numpy as np

    return types.SimpleNamespace(kernel=kernel, jax=jax, jnp=jnp, np=np, device=None)


def origin(handle) -> str:
    return handle.kernel.__file__


def _entry(handle, params):
    """The kernel called as the server's MoE layer calls it, at one shape."""
    gmm = handle.kernel.gmm_v2
    gmm = getattr(gmm, "__wrapped__", gmm)
    out_dtype = getattr(handle.jnp, params["out_dtype"]) if params["out_dtype"] else None

    def call(lhs, rhs, group_sizes, rhs_scale, group_offset):
        return gmm(lhs, rhs, group_sizes, rhs_scale, None, group_offset,
                   zero_initialize=False, fuse_act=params["fuse_act"],
                   preferred_element_type=out_dtype, maybe_quantize_lhs=True,
                   rhs_quant_dtype=None)

    return call


def _arg_specs(handle, params):
    jnp = handle.jnp
    m, k, n = params["size_m"], params["size_k"], params["size_n"]
    return [((m, k), jnp.bfloat16), ((NUM_GROUPS, k, n), jnp.float8_e4m3fn),
            ((NUM_GROUPS,), jnp.int32), ((NUM_GROUPS, 1, 1, n), jnp.float32),
            ((1,), jnp.int32)]


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
        lowered = jax.jit(_entry(handle, params)).trace(*args).lower(
            lowering_platforms=("tpu",))
    mem = lowered.compile().memory_analysis()
    text = lowered.as_text()
    start = text.find("gmm_v2-g_")
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "mosaic": "tpu_custom_call" in text,
        "kernel": text[start:start + 200].split('"')[0] if start >= 0 else "",
    }


def limit(name, result):
    if not result["mosaic"]:
        return "the compiled program launches no Mosaic kernel, so the compile checked nothing"
    return None


class _NoCopy:
    """A DMA that copies nothing and reports each run-time start to `on_start`."""

    def __init__(self, jax, on_start):
        self._jax, self._on_start = jax, on_start

    def start(self, *args, **kwargs):
        self._jax.debug.callback(self._on_start)

    def wait(self, *args, **kwargs):
        pass


class _Override:
    """A module with some attributes replaced."""

    def __init__(self, real, **replaced):
        self._real = real
        self.__dict__.update(replaced)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _input_pipes(specs, label="in"):
    """`(label, spec)` for every input spec with a pipeline mode.

    A spec without one is double-buffered, and two buffers have no unsafe trip
    count. The walk follows lists, tuples and dataclasses, because the kernel
    wraps its specs in its own weight-ref dataclasses.
    """
    found = []
    if getattr(getattr(specs, "pipeline_mode", None), "buffer_count", None):
        found.append((label, specs))
    if isinstance(specs, (list, tuple)):
        for i, s in enumerate(specs):
            found += _input_pipes(s, f"{label}[{i}]")
    elif dataclasses.is_dataclass(specs) and not isinstance(specs, type):
        for f in dataclasses.fields(specs):
            found += _input_pipes(getattr(specs, f.name), f"{label}.{f.name}")
    return found


@contextlib.contextmanager
def _interpret(handle):
    """Run the kernel in interpret mode and record its pipelines instead of running them.

    Yields a namespace that collects, per `emit_pipeline` call, its input pipes
    at trace time, and at run time its grid and the block index each pipe reads
    at each step, plus each `GmmConfigs` built. `copies` counts the DMAs the
    kernel starts itself at run time, outside any pipeline body, since those
    bodies are not run.
    """
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax._src.pallas import core as pallas_core
    from jax._src.pallas.mosaic import pipeline as jax_pipeline
    from jax._src.pallas.mosaic import tpu_info

    jax, jnp, k = handle.jax, handle.jnp, handle.kernel
    seen = types.SimpleNamespace(pipes=[], grids=[], blocks={}, cfgs=[], copies=0)
    chip = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)

    def emit_pipeline(body, *, grid, in_specs=(), out_specs=(), **kwargs):
        index = len(seen.pipes)
        pipes = _input_pipes(in_specs)
        seen.pipes.append([(label, int(spec.pipeline_mode.buffer_count),
                            bool(spec.pipeline_mode.use_lookahead)) for label, spec in pipes])

        def on_grid(*dims):
            seen.grids.append((index, tuple(int(d) for d in dims)))

        def on_step(sizes, step, *flat):
            flat, blocks = [int(x) for x in flat], []
            for size in sizes:
                blocks.append(tuple(flat[:size]))
                flat = flat[size:]
            seen.blocks.setdefault(index, {})[int(step)] = blocks

        def step(i, indices):
            # The pipeline's own step order and index arithmetic
            # (`_emit_pipeline`'s loop body), and its block comparison reads
            # the same leaves (`_tuples_differ`).
            indices = jax_pipeline._filter_indices(indices, grid)
            with pallas_core.grid_env(list(map(pallas_core.GridAxis, indices, grid))):
                blocks = [[jnp.asarray(x, jnp.int32)
                           for x in jax.tree.leaves(spec.index_map(*indices))]
                          for _, spec in pipes]
            jax.debug.callback(functools.partial(on_step, [len(b) for b in blocks]),
                               i, *[x for b in blocks for x in b])
            return jax_pipeline._next_index(indices, grid)

        def pipeline(*refs, **kwargs):
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

    def make_gmm_configs(*args, **kwargs):
        cfgs = real_configs(*args, **kwargs)
        seen.cfgs.append(cfgs)
        return cfgs

    real_pl, real_pltpu, real_configs = k.pl, k.pltpu, k.make_gmm_configs
    real_info = (tpu_info.get_tpu_info, pltpu.get_tpu_info)
    k.pl = _Override(pl, pallas_call=functools.partial(pl.pallas_call, interpret=True))
    k.pltpu = _Override(pltpu, emit_pipeline=emit_pipeline, get_tpu_info=lambda: chip,
                        make_async_copy=make_async_copy, async_copy=async_copy)
    k.make_gmm_configs = make_gmm_configs
    tpu_info.get_tpu_info = pltpu.get_tpu_info = lambda: chip
    try:
        yield seen
    finally:
        k.pl, k.pltpu, k.make_gmm_configs = real_pl, real_pltpu, real_configs
        tpu_info.get_tpu_info, pltpu.get_tpu_info = real_info


def _runtime_inputs(size_m, tile_m):
    """`(description, nonzero group sizes)` for every runtime input checked."""
    rows, tokens = size_m // EP, size_m // TOPK
    for active in range(MAX_ACTIVE + 1):
        base, extra = divmod(rows, active) if active else (0, 0)
        yield (f"{active} active experts sharing {rows} rows",
               [base + 1] * extra + [base] * (active - extra))
    for span in range(2, MAX_SPAN + 1):
        size = (span - 1) * tile_m + 1
        if size > size_m:
            break
        yield f"one expert of {size} rows spanning {span} tiles", [size]
    yield f"all {tokens} tokens routed to one expert", [tokens]


def pipelines_at(handle, params, inputs):
    """`[(description, [(pipe, buffer_count, lookahead, trip_count, blocks), ...])]`.

    `inputs` is a callable from the kernel's `tile_m` to `(description, group
    sizes)` pairs, so callers can check any group-size vectors at a shape.
    `blocks` is the block index the pipe reads at each step, in step order.
    """
    jax, jnp, np = handle.jax, handle.jnp, handle.np
    out = []
    with _interpret(handle) as seen, _tpu7x_mesh():
        run = jax.jit(_entry(handle, params))
        lhs, rhs, sizes, scale, offset = (jnp.zeros(s, d) for s, d in _arg_specs(handle, params))
        scale = jnp.ones_like(scale)

        def trip_counts_for(group_sizes):
            vec = np.zeros(NUM_GROUPS, np.int32)
            vec[:len(group_sizes)] = group_sizes
            seen.grids.clear()
            seen.blocks.clear()
            seen.copies = 0
            jax.block_until_ready(run(lhs, rhs, jnp.asarray(vec), scale, offset))
            jax.effects_barrier()
            if not seen.grids:
                raise RuntimeError("the kernel ran no emit_pipeline this adapter can see, so "
                                   "its trip counts are unknown")
            if seen.copies and not any(seen.pipes[i] for i, _ in seen.grids):
                raise RuntimeError(
                    f"the kernel started {seen.copies} DMA{'s' if seen.copies > 1 else ''} of "
                    f"its own and pipelines no input with more than two buffers, so nothing "
                    f"here was checked. This adapter reads only emit_pipeline schedules, and "
                    f"cannot tell whether each of the kernel's own copies is waited for")
            named = len(seen.pipes) > 1
            found = []
            for i, grid in seen.grids:
                n = math.prod(grid)
                steps = seen.blocks.get(i, {})
                if sorted(steps) != list(range(n)):
                    raise RuntimeError(f"pipeline {i} reported block indices for {len(steps)} "
                                       f"of its {n} steps")
                for j, (pipe, b, lookahead) in enumerate(seen.pipes[i]):
                    found.append((f"pipeline{i}:{pipe}" if named else pipe, b, lookahead, n,
                                  [steps[s][j] for s in range(n)]))
            return found

        trip_counts_for([])
        tile_m = seen.cfgs[-1].tiles.tile_m
        for description, group_sizes in inputs(tile_m):
            out.append((description, trip_counts_for(group_sizes)))
    return out


def trip_counts(handle, name, params):
    found = pipelines_at(handle, params,
                         functools.partial(_runtime_inputs, params["size_m"]))
    return [{"pipe": pipe, "buffer_count": b, "trip_count": n, "lookahead": lookahead,
             "blocks": blocks, "input": description}
            for description, pipes in found for pipe, b, lookahead, n, blocks in pipes]
