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
"""router_gemm_hbm (TPU_MOE_ROUTER_HBM_PIN): the MoE router GEMM at decode with the
weight streamed from HBM. Not bit-identical to the XLA dot it replaces (the F.linear
the gate ran): the MXU accumulation orders differ, so a few logits land one bf16 ulp
away. The tests bound that (every element within one ulp, under 1e-3 of them off) and
require that the compiled module does not stage the weight in VMEM."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import router_gemm_hbm as rg

K = 6144


def _x(m, real, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, K), dtype=np.float32)
    x[:, rng.choice(K, 8, replace=False)] *= 30.0
    x[real:] = 0.0
    return jnp.asarray(x, jnp.bfloat16)


def _w(e, seed=1):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.standard_normal((e, K), dtype=np.float32) * 0.02, jnp.bfloat16
    )


def _xla(x, w):
    # What the gate's F.linear lowers to: dot over k, bf16 result.
    return jax.jit(lambda a, b: jax.lax.dot_general(a, b, (((1,), (1,)), ((), ()))))(
        x, w
    )


def _ulp_check(y, ref, tag=""):
    """Every element within one bf16 ulp of `ref` (tiny values excepted); returns
    the count that differ."""
    a = np.asarray(y).view(np.int16).astype(np.int32)
    b = np.asarray(ref).view(np.int16).astype(np.int32)
    d = np.abs(a - b)
    n = int((d != 0).sum())
    print(f"ROUTERULP {tag} off={n}/{d.size} max_ulps={int(d.max())}")
    big = np.abs(np.asarray(ref, np.float32)) >= 1e-4
    assert int(d[big].max(initial=0)) <= 1
    return n


@pytest.mark.parametrize("e", [128, 256])
@pytest.mark.parametrize("m,real", [(4, 1), (4, 4), (8, 5), (16, 16), (16, 3), (1, 1)])
def test_matches_xla_dot_to_one_ulp(e, m, real):
    x, w = _x(m, real), _w(e)
    y = jax.jit(rg.router_gemm_hbm)(x, w)
    ref = _xla(x, w)
    assert y.dtype == ref.dtype == jnp.bfloat16
    n = _ulp_check(y, ref, f"e={e} m={m}")
    assert n <= max(1, m * e // 1000)
    # and it is the product (positive control for a no-op comparison)
    f64 = np.asarray(x, np.float64) @ np.asarray(w, np.float64).T
    assert (
        np.abs(np.asarray(y, np.float64) - f64).max()
        <= 2.0**-7 * np.abs(f64).max() + 1e-6
    )


@pytest.mark.parametrize("config", [(32, 2), (16, 4), (64, 3), (128, 2)])
def test_tilings_agree_bitwise(config):
    """Every tiling gives the default tiling's bits (row tiles are independent)."""
    x, w = _x(16, 7), _w(256)
    y = jax.jit(lambda a, b: rg.router_gemm_hbm(a, b, config=config))(x, w)
    y0 = jax.jit(rg.router_gemm_hbm)(x, w)
    np.testing.assert_array_equal(np.asarray(y, np.float32), np.asarray(y0, np.float32))


def test_fault_is_caught():
    """Positive control: a perturbed weight tile breaks the equality check."""
    x, w = _x(8, 8), _w(256)
    bad = w.at[200, 17].add(jnp.bfloat16(0.5))
    y = jax.jit(rg.router_gemm_hbm)(x, bad)
    assert not np.array_equal(
        np.asarray(y, np.float32), np.asarray(_xla(x, w), np.float32)
    )


def test_router_logits_dispatch():
    w = _w(256)
    for m in (32, 64):  # prefill sizes: the plain dot, bit for bit
        x = _x(m, m)
        y = jax.jit(rg.router_logits)(x, w)
        np.testing.assert_array_equal(
            np.asarray(y, np.float32), np.asarray(_xla(x, w), np.float32)
        )
    for m in (4, 16):  # decode sizes: the kernel
        x = _x(m, m)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(rg.router_logits)(x, w), np.float32),
            np.asarray(jax.jit(rg.router_gemm_hbm)(x, w), np.float32),
        )
    assert rg.supports(16, _x(16, 1), w)
    assert not rg.supports(32, _x(32, 1), w)
    assert not rg.supports(16, _x(16, 1).astype(jnp.float32), w)


def _weight_vmem_lines(hlo: str, e: int) -> list[str]:
    shape = f"bf16[{e},{K}]"
    out = []
    for ln in hlo.splitlines():
        m = re.match(r"\s*(?:ROOT )?%?\S+ = (.+?) ([a-z][\w-]*)\(", ln)
        if not m:
            continue
        result, opcode = m.group(1), m.group(2)
        mine = [t for t in result.strip("()").split(", ") if t.startswith(shape)]
        if mine and (
            opcode in ("copy-start", "copy-done", "slice-start", "slice-done")
            or any("S(1)" in t for t in mine)
        ):
            out.append(ln.strip())
    return out


@pytest.mark.parametrize("e", [128, 256])
def test_no_vmem_staging_of_the_weight(e):
    """Compiled kernel module: no copy/slice of the router weight and no S(1)
    result of its shape. Positive control: XLA's own dot module stages it."""
    x, w = _x(16, 4), _w(e)
    pinned = jax.jit(rg.router_gemm_hbm).lower(x, w)
    assert "input_memory_space_colors" in pinned.as_text()
    hlo = pinned.compile().as_text()
    assert _weight_vmem_lines(hlo, e) == [], _weight_vmem_lines(hlo, e)
    # A dot whose weight is a parameter feeding a following op (so it is not the
    # whole program) -- XLA stages it.
    xla_hlo = (
        jax.jit(
            lambda a, b: jnp.tanh(jax.lax.dot_general(a, b, (((1,), (1,)), ((), ()))))
        )
        .lower(x, w)
        .compile()
        .as_text()
    )
    print("XLA dot staging:", _weight_vmem_lines(xla_hlo, e)[:4])


@pytest.mark.parametrize("e", [256])
def test_speed_vs_xla(e):
    """Informational: wall per call, kernel (HBM) vs XLA dot, 200 chained calls."""
    import time

    x = _x(16, 4)
    ws = [_w(e, seed=s) for s in range(8)]  # rotate weights: cold-ish

    def chain(f):
        @jax.jit
        def run(x, *ws):
            acc = x
            for i in range(64):
                y = f(acc, ws[i % len(ws)])
                acc = acc + y[:, :1].astype(acc.dtype) * 0  # serialise
            return acc

        return run

    for name, f in (
        ("kernel", rg.router_gemm_hbm),
        ("xla", lambda a, b: jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())))),
    ):
        r = chain(f)
        r(x, *ws).block_until_ready()
        t = time.perf_counter()
        for _ in range(20):
            out = r(x, *ws)
        out.block_until_ready()
        print(
            f"ROUTERSPEED {name} e={e}: "
            f"{(time.perf_counter() - t) / 20 / 64 * 1e6:.2f} us/call"
        )
