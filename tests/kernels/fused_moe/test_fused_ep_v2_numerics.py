# SPDX-License-Identifier: Apache-2.0
"""Numerics gate for the v2 fused EP MoE kernel -- the real Pallas program,
its in-kernel all-to-all included, on CPU.

v1 could not be checked this way: jax's Pallas interpreter dies on the reshaped
and bitcast DMA refs v1 stages its weights through
(``'BitcastTransform' object has no attribute 'indices'``), so v1's numerics
question needed a TPU job, and the answer when it came was 14% rel-L2 off this
repo's fp32 reference with only 4.4% explained -- unresolved to this day.
v2's ref-level bitcast is on the four-bit weight stream only, so the fp8 path
runs under ``force_tpu_interpret_mode`` and the whole question is answerable
here for nothing.

Two assertions, in increasing strength:

* the kernel equals an op-for-op model of the arithmetic it documents -- per-row
  fp8 quantization applied through a bf16 reciprocal, f32 matmul accumulation,
  a bf16 intermediate requantized per row, fp8 rows on the wire, f32 combine.
  This is the one that would catch a real fault: it is tight to 1e-3, so it has
  no room to absorb one.
* the gap to an fp32 oracle *is* that model's gap, and nothing else.

The interpreter simulates the DMAs, so this gates arithmetic and transport
addressing, not timing or the on-device DMA engine; ``test_fused_ep_device.py``
remains the on-chip gate.
"""

import functools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0
EP = 4
HIDDEN = 256
INTER = 128
EXPERTS = 16


@pytest.fixture(scope="module")
def devices():
    """``EP`` addressable devices, on a host that has one CPU.

    ``jax_num_cpu_devices`` only takes effect before the backend initializes,
    so in a whole-suite run some earlier test has already claimed it and this
    would *skip* -- which is how the most valuable gate in this module quietly
    stops running. ``test_this_module_runs_even_in_a_full_suite`` catches that;
    read its docstring before weakening either.
    """
    try:
        jax.config.update("jax_num_cpu_devices", EP)
    except Exception as exc:  # already initialized, or the knob moved
        pytest.skip(f"cannot ask for {EP} CPU devices: {exc}")
    devs = jax.devices()
    if len(devs) < EP:
        pytest.skip(f"{len(devs)} devices, need {EP}")
    return tuple(devs[:EP])


def test_this_module_runs_even_in_a_full_suite():
    """Re-run this file in a subprocess that definitely has the devices.

    The gate above is the only check in the repository that executes the real
    Pallas program with its in-kernel all-to-all, so a silent skip is worse
    than a failure. Ordering decides whether it skips, and ordering is not
    ours: ``jax.config.update("jax_num_cpu_devices", ...)`` is a no-op once any
    earlier test has initialized the backend. A subprocess with the devices
    requested up front cannot be ordered out of existence.
    """
    import os
    import subprocess
    import sys

    if len(jax.devices()) >= EP:
        pytest.skip("already have the devices; the in-process tests ran")
    env = dict(os.environ,
               XLA_FLAGS=f"--xla_force_host_platform_device_count={EP} " +
               os.environ.get("XLA_FLAGS", ""))
    done = subprocess.run([
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        __file__, "-k", "not full_suite"
    ],
                          env=env,
                          capture_output=True,
                          text=True,
                          timeout=900)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
    assert " skipped" not in done.stdout.splitlines()[-1], (
        f"the subprocess skipped too: {done.stdout.splitlines()[-1]}")


@pytest.fixture
def tpu7x(monkeypatch):
    """Name the chip the kernel's VMEM accounting reads, on a host with none."""
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu

    info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(tpu_info, "get_tpu_info", lambda: info)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    return pltpu


def _bf16(a):
    return np.asarray(
        jnp.asarray(a, jnp.float32).astype(jnp.bfloat16), np.float32)


def _rowquant(rows_bf16):
    """``kernel.rowquant_fp8``: the reciprocal is taken in f32 and APPLIED in
    the row's own dtype, so the multiply rounds to bf16 before the fp8 cast."""
    amax = np.abs(rows_bf16).max(axis=-1, keepdims=True)
    scale = (amax / FP8_MAX).astype(np.float32)
    usable = np.isfinite(scale) & (scale > 0)
    scale = np.where(usable, scale, 0.0)
    sinv = _bf16(np.where(usable, 1.0 / np.where(usable, scale, 1.0), 0.0))
    q = np.asarray(
        jnp.asarray(_bf16(_bf16(rows_bf16) * sinv)).astype(FP8).astype(
            jnp.float32))
    return np.where(usable, q, 0.0), scale


def _silu(v):
    return np.asarray(jax.nn.silu(jnp.asarray(v)), np.float32)


def _moe(x, w1d, w2d, gating, topk, *, as_kernel):
    """The oracle (``as_kernel=False``) and the kernel's own arithmetic."""
    T, H = x.shape
    F = w1d.shape[2] // 2
    scores = np.asarray(
        jax.nn.softmax(jnp.asarray(gating, jnp.float32), axis=-1), np.float32)
    idx = np.argsort(-scores, axis=-1, kind="stable")[:, :topk]
    wts = np.take_along_axis(scores, idx, axis=-1).astype(np.float32)
    wts = wts / wts.sum(axis=-1, keepdims=True)
    xq, xs = (_rowquant(_bf16(x)) if as_kernel else
              (x.astype(np.float32), np.ones((T, 1), np.float32)))
    out = np.zeros((T, H), np.float32)
    for t in range(T):
        for k in range(topk):
            acc1 = xq[t] @ w1d[idx[t, k]]
            gate, up = acc1[:F] * xs[t, 0], acc1[F:] * xs[t, 0]
            mid = _silu(gate) * up
            if not as_kernel:
                out[t] += wts[t, k] * (mid @ w2d[idx[t, k]])
                continue
            mq, ms = _rowquant(_bf16(mid)[None])
            down = _bf16((mq[0] @ w2d[idx[t, k]]) * ms[0, 0])
            wq, ws = _rowquant(down[None])
            out[t] += wts[t, k] * (wq[0] * ws[0, 0])
    return _bf16(out) if as_kernel else out


def _rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _run(devices,
         tpu7x,
         topk,
         t_local,
         seed,
         sharded_plan=False,
         capacity_skew=None):
    from jax.sharding import AxisType, Mesh

    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME_1D
    from vllm_torchtpu.layers.common.fused_moe_ep_v2 import \
        fused_ep_moe_v2_func

    mesh = Mesh(np.asarray(devices).reshape(EP),
                axis_names=(EP_AXIS_NAME_1D, ),
                axis_types=(AxisType.Explicit, ))
    T = t_local * EP
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(T, HIDDEN)), jnp.bfloat16)
    w1 = jnp.asarray(rng.normal(size=(EXPERTS, HIDDEN, 2 * INTER)) *
                     0.1).astype(FP8)
    w2 = jnp.asarray(rng.normal(size=(EXPERTS, INTER, HIDDEN)) *
                     0.1).astype(FP8)
    # The loader's layout, singleton axes and all.
    w1s = jnp.asarray(rng.uniform(0.5, 1.5, size=(EXPERTS, 1, 1, 2 * INTER)),
                      jnp.float32)
    w2s = jnp.asarray(rng.uniform(0.5, 1.5, size=(EXPERTS, 1, 1, HIDDEN)),
                      jnp.float32)
    gating = jnp.asarray(rng.normal(size=(T, EXPERTS)), jnp.bfloat16)

    fn = functools.partial(fused_ep_moe_v2_func,
                           mesh=mesh,
                           top_k=topk,
                           renormalize=True,
                           activation="silu",
                           sharded_plan=sharded_plan,
                           capacity_skew=capacity_skew)
    with tpu7x.force_tpu_interpret_mode():
        got = np.asarray(
            jax.device_get(jax.jit(fn)(x, gating, w1, w2, w1s, w2s)),
            np.float32)

    w1d = np.asarray(w1, np.float32) * np.asarray(w1s).reshape(
        EXPERTS, 1, 2 * INTER)
    w2d = np.asarray(w2, np.float32) * np.asarray(w2s).reshape(
        EXPERTS, 1, HIDDEN)
    args = (np.asarray(x,
                       np.float32), w1d, w2d, np.asarray(gating,
                                                         jnp.float32), topk)
    return got, _moe(*args, as_kernel=True), _moe(*args, as_kernel=False)


@pytest.mark.parametrize("topk,t_local,seed", [(2, 16, 0), (8, 8, 5)])
def test_the_kernel_computes_exactly_the_arithmetic_it_documents(
        devices, tpu7x, topk, t_local, seed):
    got, model, _ = _run(devices, tpu7x, topk, t_local, seed)
    gap = _rel_l2(got, model)
    print(f"[v2 topk={topk} t_local={t_local}] kernel vs model {gap:.2e}")
    # Only f32 dot-product accumulation order is left; measured 0 to 2e-4.
    assert gap < 1e-3, (
        "the kernel does not compute what it documents; every rounding in "
        "_moe(as_kernel=True) is quoted from kernel.py, so a gap here is a "
        "fault in the kernel or a change in it, not slack in the model")


def test_the_gap_to_an_fp32_oracle_is_that_rounding_and_nothing_else(
        devices, tpu7x):
    got, model, exact = _run(devices, tpu7x, 8, 8, 5)
    kernel_gap, model_gap = _rel_l2(got, exact), _rel_l2(model, exact)
    print(
        f"[v2] vs fp32 oracle: kernel {kernel_gap:.4f} model {model_gap:.4f}")
    assert kernel_gap == pytest.approx(model_gap, rel=0.05)
    # The production GMM path scores 0.045 against the same kind of oracle at
    # the serving geometry, and v2 scores 0.053 there; this bound is the small
    # shape's headroom over that, not a tolerance anyone should widen.
    assert kernel_gap < 0.08


@pytest.mark.parametrize("topk,t_local,seed", [(2, 16, 0), (8, 8, 5)])
# capacity_skew=0 throughout: the decomposed plan does not build the drop mask
# a sized slab needs, and the skew is on by default, so the two are exclusive
# today. That is a real limitation of the decomposed plan and not of the tests
# -- see the refusal in layer.fused_ep_moe_v2.
def test_the_decomposed_routing_plan_delivers_the_same_rows(
        devices, tpu7x, topk, t_local, seed):
    """The 1/ep routing plan, through the real kernel and its all-to-all.

    ``test_fused_ep_v2_plan_decomp.py`` asserts the two plans hand the kernel
    the same operands. This asserts what that is FOR: that the rows still reach
    the rank that owns the token. It is also the only check that the one
    operand the two plans do not agree on bit for bit -- the scale slab's
    overhang row, which the whole plan fills with an accidental sum of
    off-shard pairs and the decomposed one leaves clean -- is in fact dead.
    """
    whole, _, _ = _run(devices, tpu7x, topk, t_local, seed, capacity_skew=0)
    sharded, _, _ = _run(devices,
                         tpu7x,
                         topk,
                         t_local,
                         seed,
                         sharded_plan=True,
                         capacity_skew=0)
    np.testing.assert_array_equal(whole, sharded)


def test_a_slab_at_the_drop_free_size_changes_nothing(devices, tpu7x):
    """A skew margin big enough to reach the drop-free bound must be a no-op on
    the answer -- the path that computes the drop mask included."""
    base, _, _ = _run(devices, tpu7x, 8, 8, 5)
    got, _, _ = _run(devices, tpu7x, 8, 8, 5, capacity_skew=16.0)
    np.testing.assert_array_equal(got, base)
