# SPDX-License-Identifier: Apache-2.0
"""fp8_oproj_all_reduce `weight_vmem` (TPU_FUSED_OPROJ_AR_VMEM_W) on the 8 cores of
one host: the weight as a whole-array VMEM operand XLA stages before the call is
bitwise the in-kernel HBM ring, against the float64 sum of the per-core products
within the kernel's rounding, and a perturbed weight on one core is caught."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce as fused

from .oracle import AXIS, N, custom_call_operands_in_vmem

pytestmark = pytest.mark.multichip

K, NOUT = 1024, 6144
F8 = jnp.float8_e4m3fn


def _inputs(m, real, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, m, K), dtype=np.float32)
    x[:, real:] = 0.0
    w = rng.standard_normal((N, K, NOUT), dtype=np.float32) * 0.02
    ws = np.abs(w).max(axis=1) / 448.0  # [N, NOUT]
    wq = np.clip(w / ws[:, None], -448, 448)
    return (
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(wq, F8),
        jnp.asarray(ws, jnp.float32),
    )


def _fn(mesh, *, waves, weight_vmem):
    def body(x, w, s, r):
        return fused.fp8_matmul_all_reduce(
            x[0], w[0], s[0], r, num_cores=N, axis=AXIS, waves=waves,
            weight_vmem=weight_vmem,
        )[None]

    return jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS),) * 4,
            out_specs=P(AXIS),
            check_vma=False,
        )
    )


def _run(mesh, x, wq, ws, *, waves, weight_vmem):
    spec = NamedSharding(mesh, P(AXIS))
    args = [jax.device_put(a, spec) for a in (x, wq, ws)]
    rank = jax.device_put(jnp.arange(N, dtype=jnp.int32), spec)
    fn = _fn(mesh, waves=waves, weight_vmem=weight_vmem)
    return np.asarray(fn(*args, rank), np.float32), fn, (*args, rank)


@pytest.mark.parametrize("m,real", [(4, 4), (16, 16), (16, 3)])
@pytest.mark.parametrize("waves", [False, True])
def test_weight_vmem_bitwise_the_ring(mesh, m, real, waves):
    x, wq, ws = _inputs(m, real, seed=m * 10 + real + waves)
    ring, _, _ = _run(mesh, x, wq, ws, waves=waves, weight_vmem=False)
    vmem, _, _ = _run(mesh, x, wq, ws, waves=waves, weight_vmem=True)
    np.testing.assert_array_equal(vmem, ring)
    # every core holds the same full sum
    for c in range(1, N):
        np.testing.assert_array_equal(vmem[c], vmem[0])
    # and it is the sum: float64 oracle of the per-core products (loose: the
    # activation is fp8-quantized in kernel; this checks the plumbing, the bitwise
    # equality above is the claim)
    xf = np.asarray(x, np.float64)
    want = np.einsum(
        "cmk,ckn->mn",
        xf,
        np.asarray(wq, np.float64) * np.asarray(ws, np.float64)[:, None, :],
    )
    got = vmem[0].astype(np.float64)
    assert np.count_nonzero(got[:real]) > 0
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert rel < 0.05, rel
    assert not np.any(got[real:]), "padding rows must stay zero"


def test_weight_vmem_positive_control(mesh):
    """A sign flip of one weight column on one core changes that output column."""
    x, wq, ws = _inputs(16, 16, seed=3)
    bad = wq.at[5, :, 777].set(-wq[5, :, 777])
    a, _, _ = _run(mesh, x, wq, ws, waves=True, weight_vmem=True)
    b, _, _ = _run(mesh, x, bad, ws, waves=True, weight_vmem=True)
    cols = np.nonzero((a[0] != b[0]).any(axis=0))[0]
    assert list(cols) == [777]


def test_weight_vmem_operand_staged_in_vmem(mesh):
    """Compiled: the fused call's weight operand (the 5th, f8e4m3fn[1,1024,6144])
    is in VMEM (S(1)) with weight_vmem and not with the ring."""
    x, wq, ws = _inputs(4, 4, seed=9)
    for weight_vmem in (False, True):
        _, fn, args = _run(mesh, x, wq, ws, waves=True, weight_vmem=weight_vmem)
        hlo = fn.lower(*args).compile().as_text()
        found = custom_call_operands_in_vmem(hlo, "fp8_oproj_all_reduce")
        assert found, "no fused call in the module"
        for name, ops in found.items():
            line = next(ln for ln in hlo.splitlines() if f"%{name} = " in ln)
            args_txt = line.split("custom-call(", 1)[1]
            w_name = [a for a in args_txt.split("%")[1:]][4].split(")")[0].split(",")[0]
            print(weight_vmem, name, ops, w_name)
            assert (w_name in ops) == weight_vmem, (weight_vmem, w_name, ops)
