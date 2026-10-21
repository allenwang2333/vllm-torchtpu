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
"""Is a TensorCore row gather bound by DMA issue or by the DMA engine?

One kernel issues M single-descriptor row copies (random rows of an HBM
table into VMEM) and, depending on the mode, a contiguous "weight" stream
of large chunks at the same time:

  g     gather alone
  w     weight stream alone
  gw    weight stream issued first, gather issued while it flows
  g2    gather split over two semaphores / two issue streams interleaved
  gs    gather with extra dependent scalar work per descriptor
  wg    gather issued first, weight stream after it
  gpage CSA-style: page-table lookup per row, two descriptors (NoPE, RoPE row)
  gN    (rows_per=N) each descriptor copies N consecutive rows

If the gather is issue-bound, `gw` ~ max(g, w) and `gs` > g; if it is
engine-bound, `gw` ~ g + w and `gs` ~ g. Prints RESULT lines.
"""

import argparse
import functools
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_attn_device import chain_time_pure  # noqa: E402

CH_ROWS = 2560  # 1.25 MiB weight chunk of int32 [rows, 128]


def probe(idx, table, weights, *, mode, rows_per, n_chunks, reps=1, interpret=False):
    M = idx.shape[0]

    def kernel(idx_ref, tab_hbm, w_hbm, out_ref, gbuf, wbuf, sems):
        def wstart():
            for c in range(n_chunks):
                pltpu.make_async_copy(w_hbm.at[c], wbuf, sems.at[2]).start()

        def wwait():
            for c in range(n_chunks):
                pltpu.make_async_copy(w_hbm.at[c], wbuf, sems.at[2]).wait()

        def gather(sem_of):
            def body(i, s):
                r = idx_ref[i]
                if mode == "gpage":  # idx = logical row; 256-row pages via a table
                    phys = idx_ref[(r >> 8) & 1023]  # stand-in page table (first 1024 entries)
                    row = (phys & 1023) * 256 + (r & 255)
                    pltpu.make_async_copy(tab_hbm.at[pl.ds(row, 1)], gbuf.at[pl.ds(i, 1)],
                                          sems.at[0]).start()
                    pltpu.make_async_copy(tab_hbm.at[pl.ds(row >> 2, 1)],
                                          gbuf.at[pl.ds(M + i, 1)], sems.at[1]).start()
                    return s
                if mode == "gs":  # dependent scalar busy-work, kept live through s
                    for _ in range(8):
                        s = (s * 1103515245 + r) & 0x7FFFFFFF
                    r = r + (s & 0)
                pltpu.make_async_copy(tab_hbm.at[pl.ds(r * rows_per, rows_per)],
                                      gbuf.at[pl.ds(i * rows_per, rows_per)],
                                      sems.at[sem_of(i)]).start()
                return s
            return lax.fori_loop(0, M, body, jnp.int32(1), unroll=8)

        def gwait(n_sems):
            per = M * rows_per // n_sems if mode != "gpage" else M
            for k in range(n_sems):
                blk = gbuf.at[pl.ds(k * per, per)]
                pltpu.make_async_copy(blk, blk, sems.at[k]).wait()

        def once(_, s):
            if mode in ("w", "gw"):
                wstart()
            if mode.startswith("g") or mode == "wg":
                n_sems = 2 if mode in ("g2", "gpage") else 1
                s = s + gather((lambda i: i % 2) if mode == "g2" else (lambda i: 0))
                if mode == "wg":
                    wstart()
                gwait(n_sems)
            if "w" in mode:
                wwait()
            return s

        s = lax.fori_loop(0, reps, once, jnp.int32(0))
        out_ref[...] = gbuf[0:8] + wbuf[0:8] + s

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.HBM),
                  pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[pltpu.VMEM((2 * M * rows_per, 128), jnp.int32),
                        pltpu.VMEM((CH_ROWS, 128), jnp.int32),
                        pltpu.SemaphoreType.DMA((3,))],
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             vmem_limit_bytes=60 * 1024 * 1024),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(idx, table, weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=7168)
    ap.add_argument("--chunks", type=int, default=16)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    n_tab = 65536 * 4
    table = jnp.asarray(rng.integers(0, 1 << 30, (n_tab, 128), dtype=np.int32))
    weights = jnp.asarray(rng.integers(0, 1 << 30, (a.chunks, CH_ROWS, 128), dtype=np.int32))
    print(jax.devices()[0], flush=True)
    for mode, rows_per in [("w", 1), ("g", 1), ("gw", 1), ("wg", 1), ("g2", 1), ("gs", 1),
                           ("gpage", 1), ("g", 4), ("gw", 4), ("g", 8)]:
        idx = jnp.asarray(rng.integers(0, n_tab // rows_per, a.m, dtype=np.int32))
        def timed(reps):
            jf = jax.jit(functools.partial(probe, mode=mode, rows_per=rows_per,
                                           n_chunks=a.chunks, reps=reps))
            jax.block_until_ready(jf(idx, table, weights))
            ts = []
            for _ in range(40):
                s0 = time.perf_counter()
                jax.block_until_ready(jf(idx, table, weights))
                ts.append(time.perf_counter() - s0)
            return np.array(ts)
        t1, t6 = timed(1), timed(21)
        per = (np.median(t6) - np.median(t1)) / 20
        spread = ((np.percentile(t6, 25) - np.percentile(t1, 75)) / 20,
                  (np.percentile(t6, 75) - np.percentile(t1, 25)) / 20)
        t = (per, *spread)
        wb = a.chunks * CH_ROWS * 512
        print(f"RESULT probe mode={mode} rows_per={rows_per} M={a.m} us={t[0]*1e6:.1f} "
              f"[{t[1]*1e6:.1f},{t[2]*1e6:.1f}] ns_per_desc={t[0]*1e9/a.m:.2f} "
              f"gather_MB={a.m*rows_per*512/1e6:.2f} weight_MB={wb/1e6 if 'w' in mode else 0:.1f}",
              flush=True)


if __name__ == "__main__":
    main()
