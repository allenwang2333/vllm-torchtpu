"""Step-1 probe: is SparseCore reachable on this stack, and does sparsecore_topk run?

Question: `references/hardware-facts.md` records that no SparseCore fact has been
measured on tpu7x here ("there is no SparseCore path on this stack to check it
against"), while `sparsecore_topk` raises NotImplementedError unless
`pltpu.get_tpu_info().sparse_core is not None`. Stage 2 of streamindex_topk is
that kernel, so whether it runs at all decides whether the incumbent can be
measured end to end, and whether a challenger may keep the top-k on SparseCore.

Reports UNSUPPORTED (the feature is absent) distinctly from FAIL (present and
misbehaving), per PROBING.md.
"""

import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu


def _p(tag, *rest):
    print(f"[probe] {tag}:", *rest, flush=True)


def report_tpu_info():
    info = pltpu.get_tpu_info()
    _p("device_kind", jax.devices()[0].device_kind)
    for f in ("mem_bw_bytes_per_second", "bf16_ops_per_second",
              "fp8_ops_per_second", "vmem_capacity_bytes",
              "smem_capacity_bytes", "hbm_capacity_bytes"):
        _p(f, getattr(info, f, "<absent>"))
    sc = getattr(info, "sparse_core", None)
    _p("sparse_core", sc)
    if sc is None:
        _p("VERDICT_sparse_core", "UNSUPPORTED - get_tpu_info().sparse_core is None")
        return None
    for f in ("num_cores", "num_subcores", "vmem_capacity_bytes"):
        _p(f"sparse_core.{f}", getattr(sc, f, "<absent>"))
    _p("VERDICT_sparse_core", "PRESENT")
    return sc


def time_call(fn, *args, reps=5):
    out = jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best, out


def try_topk(b, n, k):
    """Run sparsecore_topk at (b, n, k) and check it against a numpy oracle."""
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import \
        sparsecore_topk
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((b, n), dtype=np.float32)
    sj = jnp.asarray(scores)
    fn = jax.jit(lambda s: sparsecore_topk(s, k, write_empty_rows=True))
    dt, idx = time_call(fn, sj)
    idx = np.asarray(idx)
    # Oracle: the selected set must equal numpy's top-k set (order unspecified).
    want = np.argpartition(-scores, k - 1, axis=-1)[:, :k]
    ok = all(
        set(idx[r].tolist()) == set(want[r].tolist()) for r in range(b))
    gb = b * n * 4 / 1e9
    _p(f"topk b={b} n={n} k={k}", f"{dt*1e6:.1f}us",
       f"exact={ok}", f"scores={gb:.4f}GB", f"eff_BW={gb/dt/1e3:.2f}TB/s")
    return ok


def main():
    sc = report_tpu_info()
    if sc is None:
        _p("VERDICT_overall", "UNSUPPORTED - stage 2 cannot run on this stack")
        return
    # Small shape first: if this fails, the larger ones tell us nothing new.
    for (b, n, k) in [(32, 4096, 512), (32, 32768, 2048),
                      (64, 65536, 2048), (256, 65536, 2048)]:
        try:
            try_topk(b, n, k)
        except NotImplementedError as e:
            _p(f"topk b={b} n={n} k={k}", "UNSUPPORTED", repr(e))
        except Exception:
            _p(f"topk b={b} n={n} k={k}", "FAIL")
            traceback.print_exc()
    _p("VERDICT_overall", "see per-shape lines")


if __name__ == "__main__":
    main()
