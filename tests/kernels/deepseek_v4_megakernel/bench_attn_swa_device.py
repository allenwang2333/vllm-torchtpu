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
"""On device: owner_attention_phase kind "swa" (DSpark draft layer) vs the
NumPy stock-path reference (caches bit-exact, attention error), and its time
per layer (slope between 1 and 11 in-kernel repetitions)."""

import argparse
import os
import sys
import time

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_swa_harness as H  # noqa: E402


def timed(f, ops, n=15):
    jax.block_until_ready(f(*ops))
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(f(*ops))
        ts.append(time.perf_counter() - t)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--req", default="1,4")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--attn_heads", type=int, default=128)
    a = ap.parse_args()
    print(jax.devices()[0], flush=True)
    for R in map(int, a.req.split(",")):
        rng = np.random.default_rng(R)
        bases = [a.ctx - 9 - 131 * r for r in range(R)]
        acc = [(b - 1 - (r % 6), 2 + r % 6) for r, b in enumerate(bases)]  # accepted 1..6
        ctx = [(c0, 7) for c0, _ in acc]
        bases = [c0 + n for c0, n in acc]
        ss = H.build(rng, 3, bases, ctx, 6, a.ctx + 1024)
        inp = H.make_inputs(rng, ss)
        hh = a.attn_heads
        fn, ops = H.build_call(ss, 1, inp, interpret=False, attn_heads=hh)
        res = jax.jit(fn)(*ops)
        caches = [np.asarray(x) for x in res[:-1]]
        out = np.asarray(res[-1], np.float32)
        exp_c, exp_o = H.reference(ss, 1, inp)
        bit = all(np.array_equal(g, e) for g, e in zip(caches, exp_c))
        o = out[:, :6 * hh].reshape(R, 6, hh, 512)
        e = exp_o[:, :, :hh]
        err = float(np.abs(o - e).max() / np.abs(e).max())
        f1, o1 = H.build_call(ss, 1, inp, interpret=False, reps=1, attn_heads=hh)
        f11, o11 = H.build_call(ss, 1, inp, interpret=False, reps=11, attn_heads=hh)
        t1, t11 = np.median(timed(jax.jit(f1), o1)), np.median(timed(jax.jit(f11), o11))
        print(f"RESULT kind=attn_swa heads={hh} R={R} ctx={a.ctx} us_per_layer={(t11 - t1) / 10 * 1e6:.1f} "
              f"err_max_rel={err:.2e} caches_bitexact={bit}", flush=True)


if __name__ == "__main__":
    main()
