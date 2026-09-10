"""AOT-compile ONE arm of v10 at the production shape, with no TPU attached.

    python aot.py base | sc2 | sc4 | sc16 | control | poscontrol

One arm per process, deliberately.  `jax.jit` caches the inner pjit jaxpr on
`(function object, avals)`, so two arms selected by rebinding a module attribute
inside one process can silently compare an arm against itself.  Every number
here is produced by a fresh interpreter, and the arm's identity is asserted from
the Pallas kernel NAME embedded in the emitted HLO -- `_sc_topk_direct` appends
`_sc{arity}` to it, so the scope name proves which body was traced.

`control` is the mandatory negative control: an 8 MiB VMEM scratch that the
SparseCore allocator must refuse with
`E3000: CompileTimeSparseCoreAllocationFailure ... memory space spmem`.
Any other failure message means the harness is not proving what it claims --
in particular `NotImplementedError: Swap does not support storing to HBM` is an
MLIR-layer refusal and is NOT a control.

`poscontrol` is its paired POSITIVE control: the byte-identical kernel with a
4 KiB scratch instead, which must COMPILE.  Without it, "8 MiB fails" is equally
consistent with this DMA form being broken for some unrelated reason.  The two
arms differ in exactly one integer -- `_ctl_words` -- so allocation size is the
only variable between compile and refusal.

Dump flags: `--xla_sc_dump_*` are LIBTPU_INIT_ARGS, read once when libtpu
loads, so they are set below BEFORE `import jax`.  XLA rejects them by name if
they are put in XLA_FLAGS.  `--xla_dump_to` is an XLA_FLAG and gives
buffer-assignment and memory-usage reports that `compiled.as_text()` does not
carry.
"""

import os
import pathlib
import random
import re
import sys
import time

HERE = pathlib.Path(__file__).parent
ARM = (sys.argv[1] if len(sys.argv) > 1 else "base")
OUT = HERE / "aot" / ARM
OUT.mkdir(parents=True, exist_ok=True)

os.environ["LIBTPU_INIT_ARGS"] = (f"--xla_sc_dump_bundles_to={OUT} "
                                  f"--xla_sc_dump_llvm_ir_to={OUT} "
                                  f"--xla_sc_dump_mlir_to={OUT}")
os.environ["XLA_FLAGS"] = f"--xla_dump_to={OUT}"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax._src.pallas.mosaic import tpu_info as _tpu_info  # noqa: E402

_TPU7X = _tpu_info.get_tpu_info_for_chip(_tpu_info.ChipVersion.TPU_7X, 1)
_tpu_info.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]
_tpu_info.is_tpu_device = lambda: True  # type: ignore[assignment]
_tpu_info.get_device_kind = lambda: "TPU7x"  # type: ignore[assignment]
_tpu_info.get_num_device_cores = lambda: 1  # type: ignore[assignment]

from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental import topologies  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.experimental.pallas import tpu_sc as plsc  # noqa: E402
from jax.sharding import SingleDeviceSharding  # noqa: E402

pltpu.get_tpu_info = lambda: _TPU7X  # type: ignore[assignment]

sys.path.insert(0, str(HERE))
import sparsecore_topk as mod  # noqa: E402

B, N, K = 1024, 10240, 2048  # the production prefill shape, report.md §4
LANES = mod.LANES


def _topology():
    """`/tmp/libtpu_lockfile` serialises host AOT compiles machine-wide and
    several agents hold it; a collision surfaces as ABORTED, not as a queue."""
    last = None
    for attempt in range(8):
        try:
            return topologies.get_topology_desc(topology_name="tpu7x:2x2x1",
                                                platform="tpu").devices[0]
        except Exception as exc:  # noqa: BLE001
            if "lockfile" not in str(exc) and "ABORTED" not in str(exc):
                raise
            last = exc
            time.sleep(2 ** attempt * 0.5 + random.random())
    raise SystemExit(f"libtpu lockfile contention, gave up: {last}")


DEV = _topology()
SH = SingleDeviceSharding(DEV)


# The control pair.  Everything below is shared; `words` is the ONLY thing that
# differs between the arm that must be refused and the arm that must compile.
_CTL_COPY = 1024  # int32 moved VMEM -> HBM, identical in both arms


def _control(words: int, name: str):
    def kernel(o_hbm, big, sems):
        cp = pltpu.make_async_copy(big.at[pl.ds(0, _CTL_COPY)], o_hbm.at[0],
                                   sems.at[0])
        cp.start()
        cp.wait()

    mesh = plsc.VectorSubcoreMesh(
        num_cores=_TPU7X.sparse_core.num_cores,
        num_subcores=_TPU7X.sparse_core.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore")
    return pl.kernel(kernel,
                     out_type=jax.ShapeDtypeStruct((B, _CTL_COPY), jnp.int32),
                     mesh=mesh,
                     scratch_types=(pltpu.VMEM((words, ), jnp.int32),
                                    pltpu.SemaphoreType.DMA((1, ))),
                     name=name)()


# 2 Mi int32 = 8 MiB, far past the 512 KiB SparseCore tile memory.
def negative_control():
    return _control(2 * 1024 * 1024, "aot_negctl_oversized_vmem")


# 1 Ki int32 = 4 KiB, comfortably inside it.
def positive_control():
    return _control(1024, "aot_posctl_small_vmem")


def main():
    if ARM == "control":
        try:
            jax.jit(negative_control, out_shardings=SH).lower().compile()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            (OUT / "control.txt").write_text(msg)
            ok = ("CompileTimeSparseCoreAllocationFailure" in msg
                  and "spmem" in msg)
            print(f"negative control raised {type(exc).__name__}")
            print(f"  E3000 + spmem : {ok}")
            print("  " + "\n  ".join(msg.splitlines()[:6]))
            raise SystemExit(0 if ok else 1)
        raise SystemExit("negative control COMPILED -- it is not a control")

    if ARM == "poscontrol":
        # Same kernel, same DMA, 4 KiB instead of 8 MiB.  Must compile, or the
        # negative control proved only that this DMA form does not work.
        c = jax.jit(positive_control, out_shardings=SH).lower().compile()
        txt = c.as_text()
        (OUT / "poscontrol.hlo.txt").write_text(txt)
        print("positive control COMPILED")
        print(f"  kernel name in HLO : "
              f"{'aot_posctl_small_vmem' in txt}")
        print(f"  copy words         : {_CTL_COPY} (identical to negctl)")
        raise SystemExit(0)

    arm = dict(base=dict(),
               sc2=dict(sample_and_count=True, count_arity=2),
               sc4=dict(sample_and_count=True, count_arity=4),
               sc16=dict(sample_and_count=True, count_arity=16))[ARM]

    scores = jax.ShapeDtypeStruct((B, N), jnp.float32, sharding=SH)
    lens = jax.ShapeDtypeStruct((B, ), jnp.int32, sharding=SH)
    fn = jax.jit(lambda s, l: mod._sc_topk_direct(s, K, l, **arm),
                 out_shardings=SH)
    hlo = fn.lower(scores, lens).compile().as_text()
    (OUT / "optimized.hlo.txt").write_text(hlo)

    # Arm activation, from the kernel name XLA carried through, not from the
    # flag we passed in.  `_sc{arity}` is appended only on the sample path.
    names = sorted(set(re.findall(r"sc_topk_direct_b\d+_n\d+_k\d+\w*", hlo)))
    want = "" if ARM == "base" else f"_sc{arm['count_arity']}"
    got = names[0] if names else "<none>"
    active = names and got.endswith(want) and (want != "" or "_sc" not in got)
    m = re.search(r"^ENTRY .*?^}", hlo, re.S | re.M)
    entry = m.group(0) if m else hlo

    print(f"--- arm {ARM}")
    print(f"  kernel name in HLO      : {got}   arm-active={bool(active)}")
    print(f"  ENTRY instructions      : {len(entry.splitlines()) - 2}")
    for key in ("used_scoped_memory_configs", "custom_call_target"):
        for ln in entry.splitlines():
            if key in ln:
                print(f"  [{key}] {ln.strip()[:200]}")
                break
    for f in sorted(OUT.iterdir()):
        if f.stat().st_size and f.suffix in (".txt", ".ll", ".mlir", ".pb"):
            print(f"  dump {f.name:<62} {f.stat().st_size:>9} B")
    if not active:
        raise SystemExit("ARM NOT ACTIVE -- the compiled body is not this arm")


main()
