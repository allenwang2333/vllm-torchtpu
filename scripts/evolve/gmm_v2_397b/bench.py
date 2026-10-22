"""Campaign scorer: megablox gmm_v2 at Qwen3.5-397B-A17B-FP8 TP8+EP8 serving shapes.

`perfkit scorer run` (on the cluster, `perfkit cluster score`) loads this file once
per arm, in the arm's own process. An ARM ROOT IS THE KERNEL DIRECTORY
`src/vllm_torchtpu/kernels/megablox` of a checkout, shipped alone, so the `tree`
perfkit reports for the arm is that directory's own git subtree hash. Run it as
the campaign's scorer command (README.md beside this file) says:

    perfkit cluster score --bench scripts/evolve/gmm_v2_397b/bench.py \
        --image <image@digest> --jax 0.10.2 --libtpu 0.0.47 \
        --arm-path . --source-dir . --kernel-dir . \
        --arm "golden=$GOLDEN/src/vllm_torchtpu/kernels/megablox:control" \
        --arm "cand=$WORKTREE/src/vllm_torchtpu/kernels/megablox" --candidate cand

One TPU v7x TensorCore, no mesh. `setup` refuses any other device.

The call
--------
`gmm_v2.gmm_v2` from the arm's `gmm_v2.py`, with the arguments
`layers/core/fused_moe_gmm.py:gmm_wrapper` passes on this recipe: bf16 lhs the
kernel quantizes to fp8 (`maybe_quantize_lhs=True`), fp8 e4m3 expert weights with
one fp32 scale per output column (`rhs_scale` [64,1,1,n]), no bias, 64 experts on
the rank (512 over EP=8), `group_offset` 0, `zero_initialize=False`, the kernel's
own tiler. Gate/up fuses silu and returns lhs's dtype; down returns bf16.
`size_m` is the padded token count times top-10, before the kernel's own sublane
padding (40 -> 48 at decode), so the compiled op name is the one in the served
trace (`CONFIGS[*]["trace_signature"]`).

Only the kernel files are imported, never the serving package. The serving
process's environment is reproduced instead: `LIBTPU_INIT_ARGS` is exactly the
string the served job ran with (`SERVED_LIBTPU_INIT_ARGS`), and the one import-time
patch golden's `env_override.py` applies to Pallas lowering is applied here from a
verbatim copy (`_patch_jax_pallas_fori_lowering`; `test_bench.py` checks it
against golden's source). Neither is in the campaign's search space.

Configs
-------
The shapes the served trace records, one per call site and regime (REPORT.md has
the evidence): the fused gate/up projection (k=4096, n=2048, silu) and the down
projection (k=1024, n=4096), at a decode step (4 tokens = 1 sequence x 4 MTP
positions, m=40 -> 48) and at a prefill chunk. `active` is the number of
non-empty expert groups, which sets how many weight matrices a call streams and
is most of the decode cost. `rows` is how many routed rows the rank owns.
The `_m80`, `_m160` and `_m10240` configs are the neighbouring decode buckets
(2 and 3-4 sequences) and a 1024-token prefill chunk, at the same rows-to-groups
proportion; `occ_*` is the occupancy sweep scoping used to choose `active`. The
spec freezes which of these the campaign measures and with what weight.

Timing
------
`latency_us` is the time per call inside one device program, which is how the
server runs the kernel: no host dispatch per call. A program runs a
`fori_loop` whose body makes `BLOCK` independent calls back to back (each with
its own `group_offset` operand, `0 + zeros[j]` from a vector passed in, so XLA
cannot merge or hoist them; each output's first element feeds a sum). The wall
clock around `block_until_ready`, best of `REPS`, times `ITERS_1` and `ITERS_2`
iterations; the slope is the per-call time and the program's fixed cost
cancels. Scoping chose this form on golden (REPORT.md): one call per loop
iteration added ~6 us per call over the served per-call time, fully unrolled
calls ~3 us but with a 2% A/A spread on a short program; the block keeps the
loop's cost amortized over 16 calls and times 224 of them. The loop body must
hold exactly `BLOCK` gmm_v2 calls and the time must scale with the iterations,
or the config raises.

Correctness
-----------
`worst_error`: the worst written row's relative L2 error against an fp32
reference built from the rows the kernel writes (rows past `sum(group_sizes)`
are never written). The kernel runs on the same inputs before and after the
timed programs and the larger error is reported. `checksum` hashes the written
rows. Weights and activations are seeded `jax.random` draws, identical in every
arm of a job. Experts with no rows hold zeros: the kernel never reads them.

What ran
--------
`dispatched` is `pallas-gmm:` plus a digest of the arm's kernel directory as
loaded, and is only set when the compiled program launches a Mosaic call whose
name starts with `gmm_v2`; otherwise the config is `declined`. A digest of the
compiled Mosaic payload would equal golden's on every config a candidate leaves
alone, which the harness voids. `binary_hash` is the sha256 of the compiled
program text, which carries the source paths it was traced from.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
import time
import types
from pathlib import Path

#: `device_kind` of the chip the campaign measures on.
REQUIRED_DEVICE_KIND = "TPU7x"

#: The kernel file, relative to the arm root (the kernel directory itself).
KERNEL_REL = "gmm_v2.py"

#: The kernel module's name, as the serving stack imports it.
KERNEL_MODULE = "vllm_torchtpu.kernels.megablox.gmm_v2"

#: LIBTPU_INIT_ARGS of the served job of record (j-e8281eaa-7238-47b0-be0c): the
#: recipe's string, after golden's env_override prepended the one flag the recipe
#: does not set. Collective flags are inert on one device; the SMEM flag is not.
SERVED_LIBTPU_INIT_ARGS = (
    "--xla_tpu_use_dynamic_smem_negotiation=true "
    " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
    " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
    " --xla_tpu_ars_combiner_threshold_in_bytes=0"
    " --xla_tpu_enable_async_collective_merger=false"
    " --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true"
    " --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=8388608"
    " --xla_enable_async_all_gather=true"
    " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
    " --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144")

#: Experts on one rank.
NUM_GROUPS = 64

#: The scorer's own bar on `worst_error`, equal to the campaign's frozen tolerance:
#: 1.5x golden's worst measured error (0.054, prefill gate/up), 6x under the wrong-kernel fault's.
TOLERANCE = 0.08

#: What `worst_error` reports for an output that is not finite.
NONFINITE_ERROR = 1.0e30

#: Every shape this benchmark can run. A literal: perfkit reads it with `ast`.
CONFIGS = {
    "decode_gate_up": {"size_m": 40, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 4, "rows": 5, "seed": 4001, "trace_signature": 'gmm_v2-g_64-m_48-k_4096-act_silu-n_2048-tm_48-tk_4096-tn_1024', "trace_us": 15.85},
    "decode_down": {"size_m": 40, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 4, "rows": 5, "seed": 4001, "trace_signature": 'gmm_v2-g_64-m_48-k_1024-act_None-n_4096-tm_48-tk_1024-tn_4096', "trace_us": 10.79},
    "prefill_gate_up": {"size_m": 20480, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 64, "rows": 2560, "seed": 6401},
    "prefill_down": {"size_m": 20480, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 64, "rows": 2560, "seed": 6401},
    "decode_gate_up_m80": {"size_m": 80, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 8, "rows": 10, "seed": 4002},
    "decode_down_m80": {"size_m": 80, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 8, "rows": 10, "seed": 4002},
    "decode_gate_up_m160": {"size_m": 160, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 15, "rows": 20, "seed": 4003},
    "decode_down_m160": {"size_m": 160, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 15, "rows": 20, "seed": 4003},
    "prefill_gate_up_m10240": {"size_m": 10240, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 64, "rows": 1280, "seed": 6402},
    "prefill_down_m10240": {"size_m": 10240, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 64, "rows": 1280, "seed": 6402},
    "occ_gate_up_g1": {"size_m": 40, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 1, "rows": 5, "seed": 5001},
    "occ_down_g1": {"size_m": 40, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 1, "rows": 5, "seed": 5001},
    "occ_gate_up_g2": {"size_m": 40, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 2, "rows": 5, "seed": 5002},
    "occ_down_g2": {"size_m": 40, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 2, "rows": 5, "seed": 5002},
    "occ_gate_up_g3": {"size_m": 40, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 3, "rows": 5, "seed": 5003},
    "occ_down_g3": {"size_m": 40, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 3, "rows": 5, "seed": 5003},
    "occ_gate_up_g4": {"size_m": 40, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 4, "rows": 5, "seed": 5004},
    "occ_down_g4": {"size_m": 40, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 4, "rows": 5, "seed": 5004},
    "occ_gate_up_g5": {"size_m": 40, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 5, "rows": 5, "seed": 5005},
    "occ_down_g5": {"size_m": 40, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 5, "rows": 5, "seed": 5005},
    "occ_gate_up_g6": {"size_m": 40, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 6, "rows": 6, "seed": 5006},
    "occ_down_g6": {"size_m": 40, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 6, "rows": 6, "seed": 5006},
    "occ_gate_up_g7": {"size_m": 40, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 7, "rows": 7, "seed": 5007},
    "occ_down_g7": {"size_m": 40, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 7, "rows": 7, "seed": 5007},
    "occ_gate_up_g8": {"size_m": 40, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 8, "rows": 8, "seed": 5008},
    "occ_down_g8": {"size_m": 40, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 8, "rows": 8, "seed": 5008},
    "occ_gate_up_g9": {"size_m": 40, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 9, "rows": 9, "seed": 5009},
    "occ_down_g9": {"size_m": 40, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 9, "rows": 9, "seed": 5009},
    "occ_gate_up_g10": {"size_m": 40, "size_k": 4096, "size_n": 2048, "fuse_act": 'silu', "out_dtype": None, "active": 10, "rows": 10, "seed": 5010},
    "occ_down_g10": {"size_m": 40, "size_k": 1024, "size_n": 4096, "fuse_act": None, "out_dtype": 'bfloat16', "active": 10, "rows": 10, "seed": 5010},
}

#: Calls unrolled in one loop iteration; loop iterations of the short and the long
#: timed program; and the timings each is the best of.
BLOCK, ITERS_1, ITERS_2, REPS = 16, 2, 16, 10
#: The long program must take at least this share of `ITERS_2 / ITERS_1` times the short one.
SLOPE_MIN_FRAC = 0.5


# ---------------------------------------------------------------------------
# the serving process's environment
# ---------------------------------------------------------------------------

# Verbatim from golden's src/vllm_torchtpu/env_override.py (test_bench.py checks it).
def _patch_jax_pallas_fori_lowering() -> None:
    """Elide the dead remainder ``scf.for`` from Pallas ``fori_loop`` lowering.

    JAX Pallas' Mosaic lowering unconditionally emits a remainder loop for
    dynamic-bound ``jax.lax.fori_loop`` even when ``unroll=1`` (the default),
    where the main loop already covers the full range and the remainder is a
    no-op. libtpu still lowers the empty remainder to device code, which
    materially grows the emitted Mosaic body and slows decode-heavy MoE
    workloads that rely on ``fori_loop`` in Pallas scheduler kernels.

    Rewrites ``_lower_jaxpr_to_for_loop`` in place to gate the remainder
    emission on ``unroll != 1``. Applied at import so it takes effect before
    any Pallas kernel is traced. No-op if the source no longer contains the
    exact needle (upstream fix landed, or the file was refactored), so it
    remains safe across JAX bumps.
    """
    try:
        from jax._src.pallas.mosaic import lowering as _lowering
    except Exception:
        return
    if getattr(_lowering, "_torchtpu_fori_patch_applied", False):
        return
    orig = getattr(_lowering, "_lower_jaxpr_to_for_loop", None)
    if orig is None:
        return

    import inspect
    import textwrap
    try:
        src = textwrap.dedent(inspect.getsource(orig))
    except Exception:
        return
    needle = "elif has_dynamic_remainder:"
    fix = "elif has_dynamic_remainder and unroll != 1:"
    if needle not in src or fix in src:
        return
    patched_src = src.replace(needle, fix, 1)
    ns: dict = {"__name__": _lowering.__name__}
    try:
        exec(compile(patched_src, _lowering.__file__, "exec"),
             _lowering.__dict__, ns)
    except Exception:
        return
    new_fn = ns.get("_lower_jaxpr_to_for_loop")
    if new_fn is None:
        return
    _lowering._lower_jaxpr_to_for_loop = new_fn
    _lowering._torchtpu_fori_patch_applied = True


# ---------------------------------------------------------------------------
# loading the arm's kernel
# ---------------------------------------------------------------------------

def _arm_root() -> Path:
    """The first PYTHONPATH entry holding the kernel file: the arm root perfkit set."""
    for entry in (os.environ.get("PYTHONPATH") or "").split(os.pathsep):
        if entry and (Path(entry) / KERNEL_REL).is_file():
            return Path(entry).resolve()
    raise RuntimeError(f"gmm_v2 397b scorer: no PYTHONPATH entry holds {KERNEL_REL}; the arm "
                       f"root is the kernel directory itself")


def load_kernel(root: Path):
    """Import the arm's gmm_v2 under its serving name, with the arm root as its package.

    Sibling modules the kernel may import resolve inside the arm root; the
    serving package's own `__init__` never runs.
    """
    for name, path in (("vllm_torchtpu", None), ("vllm_torchtpu.kernels", None),
                       ("vllm_torchtpu.kernels.megablox", root)):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)] if path else []
        sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location(KERNEL_MODULE, root / KERNEL_REL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"gmm_v2 397b scorer: cannot load {root / KERNEL_REL}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[KERNEL_MODULE] = mod
    spec.loader.exec_module(mod)
    return mod


def kernel_dir_digest(root: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        h.update(str(f.relative_to(root)).encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def setup(*, accept_device_kind: str | None = None, root: Path | None = None):
    """Set the served environment, load the arm's kernel, refuse any chip but a v7x.

    `accept_device_kind` and `root` are test hooks; a scorer job passes neither.
    """
    if "jax" in sys.modules:
        raise RuntimeError("gmm_v2 397b scorer: jax was imported before LIBTPU_INIT_ARGS was set")
    sys.dont_write_bytecode = True
    os.environ["LIBTPU_INIT_ARGS"] = SERVED_LIBTPU_INIT_ARGS
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x")
    _patch_jax_pallas_fori_lowering()
    from jax._src.pallas.mosaic import lowering as _lowering
    if not getattr(_lowering, "_torchtpu_fori_patch_applied", False):
        raise RuntimeError("gmm_v2 397b scorer: the served Pallas fori_loop lowering patch did "
                           "not apply to this jax, so the program would not be the server's")
    root = root or _arm_root()
    kernel = load_kernel(root)

    import jax
    import jax.numpy as jnp
    import numpy as np

    kind = jax.devices()[0].device_kind
    if kind != REQUIRED_DEVICE_KIND and kind != accept_device_kind:
        raise RuntimeError(f"gmm_v2 397b scorer: the first device is {kind!r}; this campaign "
                           f"measures on {REQUIRED_DEVICE_KIND!r} and prints nothing elsewhere")
    return types.SimpleNamespace(kernel=kernel, root=root, jax=jax, jnp=jnp, np=np,
                                 digest=kernel_dir_digest(root))


def origin(handle) -> str:
    return handle.kernel.__file__


# ---------------------------------------------------------------------------
# inputs and the fp32 reference
# ---------------------------------------------------------------------------

def group_sizes_for(cfg: dict):
    """int32[64] with exactly `active` non-empty groups summing to `rows`, fixed by seed."""
    import numpy as np

    active, rows = cfg["active"], cfg["rows"]
    if not 1 <= active <= min(rows, NUM_GROUPS):
        raise ValueError(f"{cfg}: need 1 <= active <= min(rows, {NUM_GROUPS})")
    rng = np.random.default_rng(cfg["seed"])
    which = np.sort(rng.choice(NUM_GROUPS, size=active, replace=False))
    extra = rng.multinomial(rows - active, np.full(active, 1.0 / active))
    g = np.zeros(NUM_GROUPS, dtype=np.int32)
    g[which] = 1 + extra
    return g


def make_inputs(handle, cfg: dict):
    """`(args, host)`: device operands of the call, and what the reference needs."""
    jax, jnp, np = handle.jax, handle.jnp, handle.np
    m, k, n = cfg["size_m"], cfg["size_k"], cfg["size_n"]
    gs = group_sizes_for(cfg)
    active = np.flatnonzero(gs)
    key = jax.random.key(cfg["seed"])
    k_lhs, k_w, k_s = jax.random.split(key, 3)
    lhs = jax.random.normal(k_lhs, (m, k), jnp.float32).astype(jnp.bfloat16)
    w_active = (jax.random.normal(k_w, (len(active), k, n), jnp.float32) * 0.05
                ).astype(jnp.float8_e4m3fn)
    rhs = jnp.zeros((NUM_GROUPS, k, n), jnp.float8_e4m3fn).at[active].set(w_active)
    scale = jax.random.uniform(k_s, (NUM_GROUPS, 1, 1, n), jnp.float32, 0.5, 1.5)
    args = (lhs, rhs, jnp.asarray(gs), scale, jnp.zeros((1,), jnp.int32))
    host = {"lhs": np.asarray(lhs.astype(jnp.float32)),
            "w": np.asarray(w_active.astype(jnp.float32)),
            "scale": np.asarray(scale)[active, 0, 0], "gs": gs, "active": active}
    return args, host


def fp32_reference(cfg: dict, host: dict):
    """fp32 over exactly the rows the kernel writes; gate and up are the halves of n."""
    import numpy as np

    lhs, gs = host["lhs"], host["gs"]
    n_out = cfg["size_n"] // (2 if cfg["fuse_act"] else 1)
    out = np.zeros((int(gs.sum()), n_out), dtype=np.float32)
    row = 0
    for j, g in enumerate(host["active"]):
        count = int(gs[g])
        acc = lhs[row:row + count] @ (host["w"][j] * host["scale"][j])
        if cfg["fuse_act"] == "silu":
            gate, up = acc[:, :n_out], acc[:, n_out:]
            acc = gate / (1.0 + np.exp(-gate)) * up
        out[row:row + count] = acc
        row += count
    return out


def worst_row_error(got, ref) -> float:
    import numpy as np

    g = np.asarray(got, dtype=np.float32)[:ref.shape[0]]
    if g.shape != ref.shape or not np.isfinite(g).all():
        return NONFINITE_ERROR
    num = np.linalg.norm(g - ref, axis=1)
    den = np.linalg.norm(ref, axis=1)
    err = float((num / np.where(den > 0, den, 1.0)).max())
    return err if np.isfinite(err) else NONFINITE_ERROR


def checksum(rows) -> str:
    import numpy as np

    a = np.ascontiguousarray(np.asarray(rows, dtype=np.float32))
    return "sha256:" + hashlib.sha256(a.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# what ran, from the compiled text
# ---------------------------------------------------------------------------

_CUSTOM_CALL = re.compile(r'^\s*(?:ROOT\s+)?%(\S+)\s+=\s.*?custom_call_target="tpu_custom_call"',
                          re.MULTILINE)
_SIGNATURE = re.compile(r"(gmm_v2-g_\d+-m_\d+-k_\d+-act_\w+?-n_\d+-tm_\d+-tk_\d+-tn_\d+)")


def gmm_calls(text: str) -> list[str]:
    return [n for n in _CUSTOM_CALL.findall(text) if n.startswith("gmm_v2")]


def signature(text: str) -> str:
    """The first gmm_v2 call's op name without XLA's numeric suffix, as the trace shows it."""
    for name in gmm_calls(text):
        m = _SIGNATURE.match(name)
        if m:
            return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def kernel_call(handle, cfg: dict):
    """gmm_v2 as gmm_wrapper calls it, as a function of its operands."""
    gmm = handle.kernel.gmm_v2
    out_dtype = handle.jnp.bfloat16 if cfg["out_dtype"] == "bfloat16" else None

    def call(lhs, rhs, group_sizes, rhs_scale, group_offset):
        return gmm(lhs=lhs, rhs=rhs, rhs_scale=rhs_scale, rhs_bias=None,
                   group_sizes=group_sizes, group_offset=group_offset[0],
                   zero_initialize=False, fuse_act=cfg["fuse_act"],
                   preferred_element_type=out_dtype, maybe_quantize_lhs=True,
                   rhs_quant_dtype=None)

    return call


def compile_call(handle, cfg: dict, args):
    return handle.jax.jit(kernel_call(handle, cfg)).lower(*args).compile()


def compile_calls(handle, cfg: dict, args, iterations: int):
    """One program making `iterations * BLOCK` independent calls.

    A `fori_loop` runs `iterations` times over a body of `BLOCK` unrolled calls;
    each call has its own `group_offset` operand, `group_offset + zeros[j]` for a
    zero vector passed in, so XLA can neither merge the calls nor hoist them, and
    each output's first element feeds a sum.
    """
    jax, jnp = handle.jax, handle.jnp
    call = kernel_call(handle, cfg)

    def program(lhs, rhs, group_sizes, rhs_scale, group_offset, zeros):
        def body(i, acc):
            for j in range(BLOCK):
                go = group_offset + jax.lax.dynamic_slice_in_dim(zeros, i * BLOCK + j, 1)
                acc = acc + call(lhs, rhs, group_sizes, rhs_scale, go)[0, 0].astype(jnp.float32)
            return acc

        return jax.lax.fori_loop(0, iterations, body, jnp.float32(0))

    zeros = jax.ShapeDtypeStruct((iterations * BLOCK,), jnp.int32)
    return jax.jit(program).lower(*args, zeros).compile()


def _wall_us(fn, args) -> float:
    t0 = time.perf_counter()
    fn(*args).block_until_ready()
    return (time.perf_counter() - t0) * 1e6


def calls_time(handle, cfg: dict, args, *, check: bool = True) -> tuple[float, dict]:
    """`(us per call, timing metrics)` from the slope between the two programs."""
    jnp = handle.jnp
    best = {}
    for it in (ITERS_1, ITERS_2):
        compiled = compile_calls(handle, cfg, args, it)
        found = len(gmm_calls(compiled.as_text()))
        if check and found != BLOCK:
            raise RuntimeError(f"the program's loop body launches {found} gmm_v2 calls, not "
                               f"{BLOCK}; XLA merged or removed some, so the slope would not be "
                               f"a per-call time")
        call_args = (*args, jnp.zeros((it * BLOCK,), jnp.int32))
        compiled(*call_args).block_until_ready()
        best[it] = min(_wall_us(compiled, call_args) for _ in range(REPS))
    calls = (ITERS_2 - ITERS_1) * BLOCK
    us = (best[ITERS_2] - best[ITERS_1]) / calls
    scaling = best[ITERS_2] / best[ITERS_1]
    if check and scaling < SLOPE_MIN_FRAC * ITERS_2 / ITERS_1:
        raise RuntimeError(f"the {ITERS_2 * BLOCK}-call program took {scaling:.2f}x the "
                           f"{ITERS_1 * BLOCK}-call one; it did not run every call")
    if not us > 0:
        raise RuntimeError(f"the per-call slope is {us} us; programs {best}")
    return float(us), {"calls_scaling": scaling, "calls_short_us": best[ITERS_1],
                       "calls_long_us": best[ITERS_2]}


def measure(handle, name: str, cfg: dict) -> dict:
    """One config on this arm: time, correctness, and what ran."""
    jax, np = handle.jax, handle.np
    args, host = make_inputs(handle, cfg)
    try:
        compiled = compile_call(handle, cfg, args)
    except Exception as exc:                                      # noqa: BLE001
        raise RuntimeError(f"{name}: gmm_v2 did not compile: {type(exc).__name__}: {exc}") from exc
    text = compiled.as_text()
    declined = not gmm_calls(text)
    sig = signature(text)
    written = int(host["gs"].sum())

    first = np.asarray(jax.device_get(compiled(*args)), dtype=np.float32)[:written]
    us, timing = calls_time(handle, cfg, args, check=not declined)
    again = np.asarray(jax.device_get(compiled(*args)), dtype=np.float32)[:written]

    ref = fp32_reference(cfg, host)
    err = max(worst_row_error(first, ref), worst_row_error(again, ref))
    digest = checksum(first)
    trace_sig = cfg.get("trace_signature")
    print(f"gmm_v2 {name}: {us:8.2f} us (trace {cfg.get('trace_us') or '-'}) err {err:.3e} "
          f"signature {sig or '-'}{' == trace' if sig == trace_sig else ''} "
          f"programs {timing['calls_short_us']:.0f}/{timing['calls_long_us']:.0f} us "
          f"{'DECLINED' if declined else ''}", file=sys.stderr)
    if digest != checksum(again):
        print(f"gmm_v2 {name}: two calls on the same inputs disagree; the larger error is "
              f"reported", file=sys.stderr)
    metrics = dict(timing, signature_matches_trace=float(bool(trace_sig) and sig == trace_sig))
    tiles = re.search(r"-tm_(\d+)-tk_(\d+)-tn_(\d+)", sig)
    if tiles:
        metrics.update(tile_m=int(tiles[1]), tile_k=int(tiles[2]), tile_n=int(tiles[3]))
    return {
        "latency_us": us,
        "worst_error": err,
        "checksum": digest,
        "dispatched": "" if declined else f"pallas-gmm:{handle.digest}",
        "declined": declined,
        "binary_hash": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
        "metrics": metrics,
    }
