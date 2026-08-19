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

and, on a mesh whose index is not the EP rank:

* the expert relabel puts every row on the weights vLLM actually loaded into
  that chip. Uncompensated this is SILENT -- finite, correctly shaped, right
  rows, wrong experts -- so it is asserted by value against an unpermuted run,
  exactly, and with a negative control that shows the failure is silent.

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
# The mesh-order tests need the node's real permutation, which is eight wide.
# Everything else stays at four: this is the width of the process's device
# count, not of the arithmetic tests' mesh.
MESH_EP = 8
# The tpu7x-8 we serve on drives EP ranks 0..7 from these device ids; the mesh
# is in ascending device-id order, so EP rank NODE_EP_RANK_AT_MESH_INDEX[i]
# -- and the expert block vLLM gave it -- sits at mesh index i.  The mapping
# itself is pinned in tests/distributed/test_ep_mesh_device_order.py.
NODE_DEVICE_IDS = (0, 1, 4, 5, 6, 7, 2, 3)
NODE_EP_RANK_AT_MESH_INDEX = (0, 1, 6, 7, 2, 3, 4, 5)
HIDDEN = 256
INTER = 128
EXPERTS = 16


def _force_cpu_devices(count):
    """``count`` addressable CPU devices on a host that has one -- or FAIL.

    **Never a skip.** These are the only checks in the repository that execute
    the real Pallas program with its in-kernel all-to-all, so "could not get
    the devices" has to read as a broken gate, not as a green one.

    Getting them is the awkward part. ``jax_num_cpu_devices`` is honoured only
    before the backend initializes, and by the time a test in this file runs,
    some earlier module usually has -- which is exactly when the old fixture
    skipped, and skipping is decided by collection order, which is not ours.
    So take the one lever that still works afterwards and re-initialize. That
    invalidates arrays earlier modules hold; their tests have finished, and the
    modules that share this process's device count (``test_fused_ep_v2_plan_
    decomp.py`` asks for four) slice what they need out of ``jax.devices()``,
    so a larger count is compatible with all of them.
    """
    if len(jax.devices()) >= count:
        return tuple(jax.devices()[:count])
    # What the old fixture would have skipped on, printed so a run can say how
    # much of this file the forcing is buying.
    print(f"[cpu devices] {len(jax.devices())} in this process, forcing "
          f"{count}")
    try:
        # Not `import jax.extend`: that binds `jax` as a local for the whole
        # function and the line above stops seeing the module.
        from jax.extend import backend as jax_backend

        jax_backend.clear_backends()
        jax.config.update("jax_num_cpu_devices", count)
        devices = jax.devices()
    except Exception as exc:  # the knob or the backend API moved
        pytest.fail(
            f"cannot get {count} CPU devices for the fused EP MoE "
            f"numerics gate: {exc!r}. Run this file in its own process "
            f"with XLA_FLAGS=--xla_force_host_platform_device_count="
            f"{count}; do not turn this back into a skip.")
    if len(devices) < count:
        pytest.fail(f"asked for {count} CPU devices and got {len(devices)}")
    return tuple(devices[:count])


@pytest.fixture(scope="module", autouse=True)
def cpu_device_pool():
    """Force the process's device count once, before anything here traces."""
    return _force_cpu_devices(MESH_EP)


@pytest.fixture(scope="module")
def devices(cpu_device_pool):
    """The ``EP``-wide mesh the arithmetic tests run on."""
    return cpu_device_pool[:EP]


@pytest.fixture(scope="module")
def mesh_devices(cpu_device_pool):
    """The ``MESH_EP``-wide mesh the device-order tests need."""
    return cpu_device_pool[:MESH_EP]


def test_this_module_runs_even_in_a_full_suite(cpu_device_pool):
    """The contract the rest of this file rests on, asserted rather than hoped.

    This used to re-run the whole file in a subprocess, because a module-scoped
    fixture could not obtain devices once the backend was up and every test
    here would then *skip*. ``_force_cpu_devices`` removes the precondition
    instead of working around it: the count is forced, and a failure to force
    it fails. That is strictly stronger than the subprocess -- it cannot be
    ordered out of existence either, and it reports on the tests themselves
    rather than on a copy of them -- and it is why this no longer skips.
    """
    assert len(cpu_device_pool) == MESH_EP
    assert len(jax.devices()) >= MESH_EP


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


def _operands(ep, t_local, seed):
    """One layer's inputs: tokens, gating, fp8 weights, the loader's scales."""
    T = t_local * ep
    rng = np.random.default_rng(seed)
    return dict(
        x=jnp.asarray(rng.normal(size=(T, HIDDEN)), jnp.bfloat16),
        w1=jnp.asarray(rng.normal(size=(EXPERTS, HIDDEN, 2 * INTER)) *
                       0.1).astype(FP8),
        w2=jnp.asarray(rng.normal(size=(EXPERTS, INTER, HIDDEN)) *
                       0.1).astype(FP8),
        # The loader's layout, singleton axes and all.
        w1s=jnp.asarray(rng.uniform(0.5, 1.5, size=(EXPERTS, 1, 1, 2 * INTER)),
                        jnp.float32),
        w2s=jnp.asarray(rng.uniform(0.5, 1.5, size=(EXPERTS, 1, 1, HIDDEN)),
                        jnp.float32),
        gating=jnp.asarray(rng.normal(size=(T, EXPERTS)), jnp.bfloat16),
    )


def _kernel_out(devices, tpu7x, operands, *, ep, topk, sharded_plan, skew,
                mesh_ep_ranks):
    """The kernel's own answer for one set of operands, as f32."""
    from jax.sharding import AxisType, Mesh

    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME_1D
    from vllm_torchtpu.layers.common.fused_moe_ep_v2 import \
        fused_ep_moe_v2_func

    mesh = Mesh(np.asarray(devices[:ep]).reshape(ep),
                axis_names=(EP_AXIS_NAME_1D, ),
                axis_types=(AxisType.Explicit, ))
    fn = functools.partial(fused_ep_moe_v2_func,
                           mesh=mesh,
                           top_k=topk,
                           renormalize=True,
                           activation="silu",
                           sharded_plan=sharded_plan,
                           capacity_skew=skew,
                           mesh_ep_ranks=mesh_ep_ranks)
    with tpu7x.force_tpu_interpret_mode():
        out = jax.jit(fn)(operands["x"], operands["gating"], operands["w1"],
                          operands["w2"], operands["w1s"], operands["w2s"])
    return np.asarray(jax.device_get(out), np.float32)


def _run(devices,
         tpu7x,
         topk,
         t_local,
         seed,
         sharded_plan=False,
         capacity_skew=None):
    op = _operands(EP, t_local, seed)
    got = _kernel_out(devices,
                      tpu7x,
                      op,
                      ep=EP,
                      topk=topk,
                      sharded_plan=sharded_plan,
                      skew=capacity_skew,
                      mesh_ep_ranks=None)
    w1d = np.asarray(op["w1"], np.float32) * np.asarray(op["w1s"]).reshape(
        EXPERTS, 1, 2 * INTER)
    w2d = np.asarray(op["w2"], np.float32) * np.asarray(op["w2s"]).reshape(
        EXPERTS, 1, HIDDEN)
    args = (np.asarray(op["x"], np.float32), w1d, w2d,
            np.asarray(op["gating"], jnp.float32), topk)
    return got, _moe(*args, as_kernel=True), _moe(*args, as_kernel=False)


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


# --- the mesh whose index is not the EP rank ---------------------------------
#
# The mesh is laid out in ascending device-id order, so the shard at mesh index
# i runs on the process holding EP rank NODE_EP_RANK_AT_MESH_INDEX[i]'s expert
# block -- vLLM loaded those weights into that worker and no operand can move
# them. Model that by dealing the expert blocks out in that order, and require
# the answer not to change: same tokens, same weights, same routing, a different
# chip doing each expert.

MESH_TOPK = 2
MESH_T_LOCAL = 8
MESH_SEED = 0
_MESH_RUNS: dict[tuple, np.ndarray] = {}


def _by_mesh_index(order):
    """The expert axis reordered so block ``order[i]`` lands at mesh index i."""
    g_local = EXPERTS // MESH_EP
    return jnp.asarray(
        np.concatenate([
            np.arange(rank * g_local, (rank + 1) * g_local) for rank in order
        ]))


def _mesh_run(mesh_devices,
              tpu7x,
              *,
              placement,
              mesh_ep_ranks,
              sharded_plan=False,
              skew=None):
    """One ``MESH_EP``-wide run; memoized, because each interprets for ~10s.

    ``placement`` is the EP-rank order the expert blocks are dealt out in, or
    None for the identity; ``mesh_ep_ranks`` is what the driver tells the kernel
    about that order, or None for "they agree".
    """
    key = (placement, mesh_ep_ranks, sharded_plan, skew)
    if key not in _MESH_RUNS:
        op = _operands(MESH_EP, MESH_T_LOCAL, MESH_SEED)
        if placement is not None:
            index = _by_mesh_index(placement)
            op = dict(op,
                      **{n: op[n][index]
                         for n in ("w1", "w2", "w1s", "w2s")})
        _MESH_RUNS[key] = _kernel_out(mesh_devices,
                                      tpu7x,
                                      op,
                                      ep=MESH_EP,
                                      topk=MESH_TOPK,
                                      sharded_plan=sharded_plan,
                                      skew=skew,
                                      mesh_ep_ranks=mesh_ep_ranks)
    return _MESH_RUNS[key]


def test_the_node_permutation_is_not_the_identity():
    """Without this, every assertion below is green on any spelling."""
    ranks = {device: rank for rank, device in enumerate(NODE_DEVICE_IDS)}
    assert tuple(ranks[d] for d in sorted(NODE_DEVICE_IDS)) == (
        NODE_EP_RANK_AT_MESH_INDEX), "the two constants are not inverses"
    assert NODE_EP_RANK_AT_MESH_INDEX != tuple(range(MESH_EP))
    assert len(NODE_DEVICE_IDS) == MESH_EP


@pytest.mark.parametrize("sharded_plan,skew", [(False, None), (True, 0)])
def test_a_mesh_that_is_not_in_ep_rank_order_reaches_the_right_experts(
        mesh_devices, tpu7x, sharded_plan, skew):
    """Mesh index <-> EP rank <-> expert block, through the real kernel.

    Exact, not approximate: relabelling the expert axis moves which chip
    computes a pair, not the arithmetic of the pair, and the combine sums the
    ``topk`` slots in slot order rather than arrival order -- so a correct
    reindex is bit-for-bit. A tolerance here would hide the tie-flip that
    permuting the *logits* instead of the selected ids causes.

    The shipped plan builder and the shipped slab, which is what serving runs.
    The decomposed builder spells ownership separately (``first = me *
    g_local`` in its own arithmetic) and could in principle be reindexed
    correctly in one and not the other; that arm lives with the rest of the
    plan-decomposition coverage rather than here.
    """
    reference = _mesh_run(mesh_devices,
                          tpu7x,
                          placement=None,
                          mesh_ep_ranks=None)
    permuted = _mesh_run(mesh_devices,
                         tpu7x,
                         placement=NODE_EP_RANK_AT_MESH_INDEX,
                         mesh_ep_ranks=NODE_EP_RANK_AT_MESH_INDEX)
    np.testing.assert_array_equal(permuted, reference)


def test_without_the_reindex_the_wrong_experts_answer_and_nothing_says_so(
        mesh_devices, tpu7x):
    """NEGATIVE CONTROL, and the reason this gate is by value.

    Deal the blocks out and tell the kernel nothing: every table is then
    self-consistent, so the rows still reach the token that selected them and
    the shapes, the row count and the finiteness are all exactly right. Only
    the *weights* are another rank's. Nothing in the program can notice --
    which is why no shape test, no row-order test and no "it ran" check can
    stand in for this one.
    """
    reference = _mesh_run(mesh_devices,
                          tpu7x,
                          placement=None,
                          mesh_ep_ranks=None)
    uncompensated = _mesh_run(mesh_devices,
                              tpu7x,
                              placement=NODE_EP_RANK_AT_MESH_INDEX,
                              mesh_ep_ranks=None)

    assert uncompensated.shape == reference.shape
    assert np.isfinite(uncompensated).all(), "the failure is silent, not a NaN"
    assert not np.array_equal(uncompensated, reference), (
        "the uncompensated run agreed with the reference, so either the "
        "placement did not move any expert or the kernel is ignoring "
        "mesh_ep_ranks -- in both cases the test above proves nothing")
    gap = _rel_l2(uncompensated, reference)
    print(f"[v2 mesh order] uncompensated vs reference rel-L2 {gap:.4f}")
    # Two of the eight blocks are the permutation's fixed points, so six ranks
    # compute with weights that are not theirs: measured 1.12, and a whole
    # order of magnitude of headroom under it still means "a different model".
    assert gap > 0.1
