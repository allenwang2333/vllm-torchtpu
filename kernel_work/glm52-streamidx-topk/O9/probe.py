"""What has to be true before an overlap is worth building. Four questions.

  1. WAVE STRUCTURE. The change the design sheet names is cross-wave: hide wave
     w+1's fetch behind wave w's compute. That needs two waves that both carry
     work. `num_waves = cdiv(b*p, 32)` and rows past `row_lengths` skip their
     DMA entirely, so the number of waves that fetch anything is arithmetic on
     the shape table -- printable without a device, and decisive.
  2. VMEM. A second resident buffer at (B) is 256 KiB on top of 256 KiB in a
     512 KiB vector subcore. The ledger says whether it fits; the compile says
     whether the ledger is right.
  3. `emit_pipeline` FROM A SparseCore KERNEL. Named as a route by
     `design/area3-pipelining.md` and never tried on this chip. Owed to
     `references/08-sparsecore.md` either way.
  4. SEMAPHORE PROTOCOL. Does a plsc kernel tolerate several outstanding
     `make_async_copy`s on distinct semaphores at all? A trace-and-compile of
     the chunked arms answers it before a benchmark is spent on them.

    python3 probe.py            # all four
"""
import importlib.util
import os
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arms  # noqa: E402
from pallas_mosaic_tpu import (Config, pick_partition, slice_geometry,  # noqa: E402
                               vmem_ledger)

INSTRUMENTS = os.environ.get(
    "INST_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "instruments"))

ROWS = ("A-D16-8k", "A-D64-8k", "B-D4-256k", "B-D1-1M", "B-P1024-256k")


def load_instrument(name):
    path = os.path.join(INSTRUMENTS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"instr_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def wave_structure():
    print("\n" + "=" * 74)
    print("1. WAVE STRUCTURE -- how many waves actually fetch anything")
    print("=" * 74, flush=True)
    shapes = load_instrument("shapes")
    print(f"{'row':14} {'b':>4} {'n':>9} {'P':>4} {'n_p':>7} {'slice_KiB':>10} "
          f"{'waves':>6} {'waves w/ work':>14} {'stage1 DMA MiB':>15}")
    for name in ROWS:
        r = shapes.ROWS[name]
        n = shapes.padded_n(r["psz"], r["pages_per_seq"])
        b = r["T"]
        k = shapes.K
        # Reproduce bench.py --decompose's row_lengths without building inputs.
        qlen = r.get("qlen", 1)
        lens = shapes._seq_lens_for(r)
        rl = np.zeros(b, dtype=np.int64)
        t = 0
        for i in range(r["R"]):
            for u in range(qlen):
                rl[t] = min(int(lens[i]) - qlen + u + 1, n)
                t += 1
        p = pick_partition(b, n, k)
        n_p = n // p if p > 1 else n
        b1 = b * p
        lengths_p = np.clip(rl[:, None] - np.arange(p) * n_p, 0,
                            n_p).reshape(-1)
        slice_len, waves = slice_geometry(b1, n_p, k)
        live = lengths_p > 0
        per_wave = [int(live[w * 32:(w + 1) * 32].sum()) for w in range(waves)]
        working = sum(1 for c in per_wave if c)
        dma_mib = live.sum() * slice_len * 4 / 2**20
        print(f"{name:14} {b:>4} {n:>9} {p:>4} {n_p:>7} "
              f"{slice_len*4/1024:>10.0f} {waves:>6} {working:>14} "
              f"{dma_mib:>15.1f}")
        print(f"{'':14} live rows per wave: {per_wave}", flush=True)
    print("\n  A cross-wave double buffer can hide at most (waves with work - 1)"
          "\n  of the fetches. Where that is 0, the change has nothing to hide.",
          flush=True)


def vmem():
    print("\n" + "=" * 74)
    print("2. VMEM -- can a second resident buffer exist?")
    print("=" * 74, flush=True)
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    print(f"  sparse_core: num_cores={sc.num_cores} "
          f"num_subcores={sc.num_subcores} num_lanes={sc.num_lanes} "
          f"vmem_capacity_bytes={sc.vmem_capacity_bytes} "
          f"({sc.vmem_capacity_bytes/1024:.0f} KiB per vector subcore)",
          flush=True)
    print(f"  TensorCore vmem_capacity_bytes={info.vmem_capacity_bytes} "
          f"({info.vmem_capacity_bytes/2**20:.1f} MiB) -- NOT the number that "
          f"binds here", flush=True)
    shapes = load_instrument("shapes")
    for name in ("A-D64-8k", "B-D4-256k"):
        r = shapes.ROWS[name]
        n = shapes.padded_n(r["psz"], r["pages_per_seq"])
        b, k = r["T"], shapes.K
        p = pick_partition(b, n, k)
        b1, n1 = b * p, (n // p if p > 1 else n)
        for label, cfg in (("1 buffer", Config()),
                           ("2 buffers", Config(second_buffer=True))):
            led = vmem_ledger(b1, n1, k, cfg)
            print(f"  {name:12} {label:10} used={led['used_bytes']/1024:8.1f} "
                  f"KiB of {led['capacity_bytes']/1024:.0f} KiB  "
                  f"fits={led['fits']}   {led['items']}", flush=True)


def compile_arms():
    print("\n" + "=" * 74)
    print("4. TRACE AND COMPILE -- semaphore protocol and allocation, for real")
    print("=" * 74, flush=True)
    shapes = load_instrument("shapes")
    for name in ("B-D4-256k", "A-D64-8k"):
        r = shapes.ROWS[name]
        n = shapes.padded_n(r["psz"], r["pages_per_seq"])
        b = r["T"]
        scores = jnp.zeros((b, n), jnp.float32)
        rl = jnp.full((b, ), n // 4, jnp.int32)
        for nm in ("base", "r2", "q2", "c2d2", "c4d2", "c8d2", "c8d4", "c16d4",
                   "buf2"):
            sel = arms.selector(nm)
            try:
                fn = jax.jit(lambda s, rr, _s=sel: _s(s, shapes.K, rr))
                jax.block_until_ready(fn(scores, rl))
                print(f"  {name:12} {nm:6} OK", flush=True)
            except Exception as e:  # noqa: BLE001
                msg = str(e).replace("\n", " ")[:400]
                print(f"  {name:12} {nm:6} {type(e).__name__}: {msg}",
                      flush=True)


def emit_pipeline_on_sc():
    print("\n" + "=" * 74)
    print("3. emit_pipeline FROM A SparseCore KERNEL -- is it usable at all?")
    print("=" * 74, flush=True)
    print(f"  jax {jax.__version__}", flush=True)
    print(f"  pltpu.emit_pipeline present: {hasattr(pltpu, 'emit_pipeline')}",
          flush=True)
    print(f"  plsc.emit_pipeline present:  {hasattr(plsc, 'emit_pipeline')}  "
          f"(tpu_sc exports: nothing pipeline-shaped)", flush=True)
    try:
        from jax._src import config as _cfg
        print("  config.use_emit_pipeline_primitive = "
              f"{_cfg.use_emit_pipeline_primitive.value}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  config.use_emit_pipeline_primitive unreadable: {e}",
              flush=True)

    n, blk = 8192, 1024
    sc = pltpu.get_tpu_info().sparse_core
    mesh = plsc.VectorSubcoreMesh(num_cores=sc.num_cores,
                                  num_subcores=sc.num_subcores,
                                  core_axis_name="core",
                                  subcore_axis_name="subcore")

    def kern(x_hbm, o_hbm):
        def body(x_ref, o_ref):
            o_ref[...] = x_ref[...] + jnp.int32(1)

        pltpu.emit_pipeline(
            body,
            grid=(n // blk, ),
            in_specs=[pl.BlockSpec((blk, ), lambda i: (i, ))],
            out_specs=[pl.BlockSpec((blk, ), lambda i: (i, ))],
        )(x_hbm, o_hbm)

    x = jnp.arange(n, dtype=jnp.int32)
    try:
        f = pl.kernel(kern,
                      out_type=jax.ShapeDtypeStruct((n, ), jnp.int32),
                      mesh=mesh,
                      name="o9_emit_pipeline_sc_probe")
        out = np.asarray(jax.block_until_ready(jax.jit(f)(x)))
        ok = bool((out == np.arange(n) + 1).all())
        print(f"  RESULT: emit_pipeline LOWERED on SparseCore. correct={ok}",
              flush=True)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()
        print("  RESULT: emit_pipeline FAILED from a plsc kernel. Tail:",
              flush=True)
        for line in tb[-14:]:
            print(f"    {line}", flush=True)


if __name__ == "__main__":
    wave_structure()
    vmem()
    emit_pipeline_on_sc()
    compile_arms()
    print("\nprobe done", flush=True)
