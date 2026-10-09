# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Accuracy instruments for the fused decode MoE layer kernel.

Three arms per cell, all from the same inputs:

* float64 oracle: dequantized FP8 weights, unquantized activations, exact
  routing weights, routed sum (x routed scale) + shared expert;
* native reference: today's path -- fused_moe_func's EP remap/mask and BF16
  weight cast, the incumbent ``dense_expert_moe``, then the XLA shared expert
  (``xla_quantized_matmul`` gate/up as its own op; silu*up fused with the down
  activation quantization, as XLA fuses it) and the BF16 shared + routed add
  (the runner's routed scale, when not 1, applied to the BF16 routed output);
* candidate: ``dense_expert_moe_fused`` with global ids, the EP offset and the
  shared expert inside.

Pass: finite, relative L2 vs native < 0.008, relative L2 vs float64 within
1.5x the native's own float64 error, padded rows bit-equal to the native's
(exact zeros routed-only; exactly the shared expert's output with it).
Fault controls run the candidate wrongly and the same checker must reject it.

Runs on the CPU Pallas interpreter (JAX_PLATFORMS=cpu) and on TPU.
"""

import functools
import zlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import moe_dense_expert as incumbent
from vllm_torchtpu.kernels.experimental import moe_dense_expert_fused as fused
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

M = 16
REL_L2_NATIVE = 0.008
ORACLE_SLACK = 1.5


@pytest.fixture(autouse=True)
def _interpret_on_cpu(monkeypatch):
    if jax.default_backend() == "cpu":
        monkeypatch.setattr(
            fused.pl,
            "pallas_call",
            functools.partial(fused.pl.pallas_call, interpret=True),
        )
        monkeypatch.setattr(
            fused.pltpu,
            "get_tpu_info",
            lambda: SimpleNamespace(vmem_capacity_bytes=64 * 1024**2),
        )
    incumbent.dense_expert_moe.clear_cache()
    fused.dense_expert_moe_fused.clear_cache()
    yield
    incumbent.dense_expert_moe.clear_cache()
    fused.dense_expert_moe_fused.clear_cache()


# ---------------------------------------------------------------- inputs


@functools.lru_cache(maxsize=2)
def _weights(e, h, i, shared_i, active_key):
    """Random FP8 weights for the experts that can be touched, zeros elsewhere.

    ``active_key`` lists local experts that get random weights (the active
    set and its +-1 neighbours, which the offset fault maps onto). Inactive
    experts are never DMA'd, so zeros there cost nothing and keep the real
    64-expert cell's host memory small.
    """
    rng = np.random.default_rng(1000 + e + h + shared_i)
    w1 = np.zeros((e, h, 2 * i), np.float32)
    w2 = np.zeros((e, i, h), np.float32)
    for expert in active_key:
        w1[expert] = rng.normal(size=(h, 2 * i))
        w2[expert] = rng.normal(size=(i, h))
    w1 = jnp.asarray(w1, jnp.float8_e4m3fn)
    w2 = jnp.asarray(w2, jnp.float8_e4m3fn)
    # Scales keep activations O(1): gate/up ~ sqrt(H) * s1, down ~ sqrt(I) * s2.
    s1 = jnp.asarray(
        rng.uniform(0.5, 1.5, size=(e, 1, 1, 2 * i)) / np.sqrt(h), jnp.float32
    )
    s2 = jnp.asarray(rng.uniform(0.5, 1.5, size=(e, 1, 1, h)) / np.sqrt(i), jnp.float32)
    ws13 = jnp.asarray(rng.normal(size=(h, 2 * shared_i)), jnp.float8_e4m3fn)
    ss13 = jnp.asarray(
        rng.uniform(0.5, 1.5, size=(2 * shared_i,)) / np.sqrt(h), jnp.float32
    )
    ws2 = jnp.asarray(rng.normal(size=(shared_i, h)), jnp.float8_e4m3fn)
    ss2 = jnp.asarray(rng.uniform(0.5, 1.5, size=(h,)) / np.sqrt(shared_i), jnp.float32)
    return (w1, w2, s1, s2), (ws13, ss13, ws2, ss2)


def _routing(e, k, live, active, offset, dup, seed):
    """Global top-k ids and renormalized+scaled weights, padded rows zeroed.

    Every one of the ``active`` local experts appears in some live row; the
    remaining routes go to other EP groups, including the ids adjacent to
    this rank's range so a +-1 offset error changes the result. Padded rows
    still carry a local id: only their zero weight drops them.
    """
    rng = np.random.default_rng(seed)
    assert active <= live * k
    act = sorted(rng.choice(e, size=active, replace=False).tolist()) if active else []
    foreign = [offset - 1, offset + e, offset - 2, offset + e + 1]
    foreign += [
        int(g)
        for g in rng.choice(
            [g for g in range(4 * e) if not offset <= g < offset + e], size=8
        )
    ]
    ids = rng.choice(foreign, size=(M, k)).astype(np.int64)
    slots = rng.permutation(live * k)[: len(act)]
    for a, slot in zip(act, slots):
        ids[slot // k, slot % k] = offset + a
    if act:
        # Extra traffic onto the active set (shared experts between rows).
        extra = rng.random((live, k)) < 0.3
        taken = np.zeros((live, k), bool)
        taken.reshape(-1)[slots] = True
        fill = rng.choice(act, size=(live, k)) + offset
        ids[:live] = np.where(extra & ~taken, fill, ids[:live])
    if dup and live and act:
        # Repeated ids on one row: the kernel must sum their coefficients.
        ids[0, : min(3, k)] = offset + act[0]
    if live < M and act:
        ids[live:, 0] = offset + act[-1]
    raw = rng.uniform(0.05, 1.0, size=(M, k))  # sigmoid scores
    weights = raw / raw.sum(axis=1, keepdims=True) * 2.5
    weights[live:] = 0.0
    raw[live:] = 0.0
    # Every active expert keeps at least one live route after the dup rewrite.
    present = {int(v) - offset for v in ids[:live].reshape(-1)}
    act = [a for a in act if a in present]
    return (
        jnp.asarray(ids, jnp.int32),
        jnp.asarray(weights, jnp.float32),
        jnp.asarray(raw, jnp.float32),
        act,
    )


# ------------------------------------------------------------------ arms


def _xla_shared(x, ws13, ss13, ws2, ss2):
    shared_i = ws2.shape[0]
    gate_up = jax.jit(xla_quantized_matmul)(x, ws13, ss13)

    on_tpu = jax.default_backend() == "tpu"

    @jax.jit
    def down(gu):
        if on_tpu:
            # Today's form: torch SiluAndMul on BF16, fused by XLA into the
            # down quantization.
            act = jax.nn.silu(gu[:, :shared_i]) * gu[:, shared_i:]
        else:
            # XLA:TPU computes the form above as FP32 silu*up rounded once to
            # BF16 (bit-exact, probe j-1999b8dc-ecff-439d-b68e); XLA:CPU keeps
            # excess precision through the fusion instead. State the TPU
            # semantics explicitly so the CPU run checks the same numerics.
            g = gu[:, :shared_i].astype(jnp.float32)
            u = gu[:, shared_i:].astype(jnp.float32)
            act = (jax.nn.silu(g) * u).astype(jnp.bfloat16)
        return xla_quantized_matmul(act, ws2, ss2)

    return down(gate_up)


def _native(x, routed, weights, ids, offset, shared, rsf):
    w1, w2, s1, s2 = routed
    e = w1.shape[0]
    tw = weights.astype(x.dtype)  # fused_moe_func: weights pinned to BF16
    local = ids - offset
    valid = (local >= 0) & (local < e)
    out = incumbent.dense_expert_moe(
        x, w1, w2, s1, s2, jnp.where(valid, tw, 0), jnp.where(valid, local, -1)
    )
    if rsf != 1.0:
        out = (out * rsf).astype(jnp.bfloat16)  # runner: fused_output *= rsf
    if shared is not None:
        out = (_xla_shared(x, *shared) + out).astype(jnp.bfloat16)
    return np.asarray(out, np.float32)


def _candidate(x, routed, weights, ids, offset, shared, rsf):
    w1, w2, s1, s2 = routed
    out = fused.dense_expert_moe_fused(
        x,
        w1,
        w2,
        s1,
        s2,
        weights,
        ids,
        jnp.int32(offset),
        shared,
        routed_scale=rsf,  # the runner skips its own scaling on this path
    )
    return np.asarray(out, np.float32)


def _oracle(x, routed, weights, ids, offset, shared, rsf):
    w1, w2, s1, s2 = routed
    e, _, two_i = w1.shape
    i = two_i // 2
    f64 = lambda a: np.asarray(a.astype(jnp.float32), np.float64)  # noqa: E731
    xf = f64(x)
    ids_np, w_np = np.asarray(ids), np.asarray(weights, np.float64)

    def mlp(a, wa, sa, wb, sb, inter):
        gu = a @ (wa * sa)
        g, u = gu[:, :inter], gu[:, inter:]
        return (g / (1 + np.exp(-g)) * u) @ (wb * sb)

    out = np.zeros((M, xf.shape[1]))
    for expert in sorted({int(v) - offset for v in ids_np.reshape(-1)}):
        if not 0 <= expert < e:
            continue
        coeff = np.where(ids_np - offset == expert, w_np, 0).sum(axis=1)
        if not coeff.any():
            continue
        y = mlp(
            xf,
            f64(w1[expert]),
            f64(s1[expert, 0]),
            f64(w2[expert]),
            f64(s2[expert, 0]),
            i,
        )
        out += coeff[:, None] * y
    out *= rsf
    if shared is not None:
        ws13, ss13, ws2, ss2 = shared
        out += mlp(
            xf, f64(ws13), f64(ss13)[None], f64(ws2), f64(ss2)[None], ws2.shape[0]
        )
    return out


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _check(cand, native, oracle, live):
    """Failure reasons; empty means the candidate passes the bar."""
    fails = []
    if not np.isfinite(cand).all():
        return ["non-finite"]
    if np.linalg.norm(native) == 0:
        if np.any(cand != 0):
            fails.append("expected exact zeros")
        return fails
    rel_native = _rel(cand, native)
    if rel_native >= REL_L2_NATIVE:
        fails.append(f"rel L2 vs native {rel_native:.4g}")
    native_err, cand_err = _rel(native, oracle), _rel(cand, oracle)
    if cand_err > ORACLE_SLACK * native_err:
        fails.append(f"oracle err {cand_err:.4g} > 1.5 x native {native_err:.4g}")
    if live < M and not np.array_equal(cand[live:], native[live:]):
        fails.append("padded rows differ from native")
    return fails


# ----------------------------------------------------------------- cells

# (class, e, h, i, live, k, active, shared_i, dup, rsf)
FAST = [
    ("fast", 4, 512, 512, 4, 4, 1, 128, False, 1.0),
    ("fast", 4, 512, 512, 16, 8, 4, 256, True, 1.0),
    ("fast", 4, 512, 512, 1, 8, 0, 128, False, 1.0),
    ("fast", 4, 512, 512, 4, 8, 4, 128, False, 2.5),
]
E64 = [
    ("e64", 64, 512, 512, 1, 8, 0, 128, False, 1.0),
    ("e64", 64, 512, 512, 1, 8, 1, 256, False, 1.0),
    ("e64", 64, 512, 512, 1, 8, 8, 128, True, 1.0),
    ("e64", 64, 512, 512, 4, 4, 1, 128, False, 1.0),
    ("e64", 64, 512, 512, 4, 8, 8, 256, True, 1.0),
    ("e64", 64, 512, 512, 4, 8, 32, 128, False, 1.0),
    ("e64", 64, 512, 512, 16, 4, 8, 256, False, 1.0),
    ("e64", 64, 512, 512, 16, 8, 32, 128, True, 1.0),
]
REAL = [
    ("real", 64, 6144, 512, 4, 8, 8, 128, False, 1.0),
    ("real", 64, 6144, 512, 16, 8, 32, 256, True, 1.0),
]
OFFSET = 64  # this rank owns global experts [64, 64 + E)


def _cell_id(cell):
    cls, e, h, i, live, k, a, si, dup, rsf = cell
    return f"{cls}-e{e}-h{h}-live{live}-k{k}-a{a}-s{si}{'-dup' if dup else ''}" + (
        f"-rsf{rsf}" if rsf != 1.0 else ""
    )


def _setup(cell, seed):
    cls, e, h, i, live, k, active, shared_i, dup, rsf = cell
    ids, weights, raw, act = _routing(e, k, live, active, OFFSET, dup, seed)
    touched = tuple(sorted({a + d for a in act for d in (-1, 0, 1) if 0 <= a + d < e}))
    routed, shared = _weights(e, h, i, shared_i, touched)
    rng = np.random.default_rng(seed + 7)
    x = jnp.asarray(rng.normal(size=(M, h)), jnp.bfloat16)
    return x, routed, shared, ids, weights, raw, act, live, rsf


def _faults(x, routed, shared, ids, weights, raw, act, live, rsf):
    """name -> faulty candidate thunk (None when the cell cannot express it)."""
    faults = {}
    if act:
        for d in (1, -1):
            faults[f"offset{d:+d}"] = lambda d=d: _candidate(
                x, routed, weights, ids, OFFSET + d, shared, rsf
            )
        faults["not_renormalized"] = lambda: _candidate(
            x, routed, raw, ids, OFFSET, shared, rsf
        )
        last = OFFSET + act[-1]
        dropped = jnp.where(ids == last, OFFSET - 3, ids)
        faults["last_active_dropped"] = lambda: _candidate(
            x, routed, weights, dropped, OFFSET, shared, rsf
        )
    if live < M and act:
        unmasked = weights.at[live:].set(0.3)
        faults["pad_not_masked"] = lambda: _candidate(
            x, routed, unmasked, ids, OFFSET, shared, rsf
        )
    faults["shared_omitted"] = lambda: _candidate(
        x, routed, weights, ids, OFFSET, None, rsf
    )
    return faults


def _run_cell(cell, seed, with_faults=True):
    x, routed, shared, ids, weights, raw, act, live, rsf = _setup(cell, seed)
    native = _native(x, routed, weights, ids, OFFSET, shared, rsf)
    oracle = _oracle(x, routed, weights, ids, OFFSET, shared, rsf)
    cand = _candidate(x, routed, weights, ids, OFFSET, shared, rsf)
    fails = _check(cand, native, oracle, live)
    assert not fails, (fails, _rel(native, oracle))
    # Padded rows carry exactly the shared expert (zero routed coefficient).
    if live < M:
        np.testing.assert_array_equal(
            cand[live:], np.asarray(_xla_shared(x, *shared), np.float32)[live:]
        )
    caught = {}
    if with_faults:
        for name, thunk in _faults(
            x, routed, shared, ids, weights, raw, act, live, rsf
        ).items():
            caught[name] = bool(_check(thunk(), native, oracle, live))
        print(f"\n{_cell_id(cell)} faults caught: {caught}")
        assert all(caught.values()), caught
    return caught


@pytest.mark.parametrize("cell", FAST + E64, ids=_cell_id)
def test_fused_layer_matches_native_and_oracle(cell):
    _run_cell(cell, seed=zlib.crc32(_cell_id(cell).encode()))


@pytest.mark.parametrize("cell", REAL, ids=_cell_id)
def test_real_shape_cell(cell):
    # H=6144, I=512, E=64: served per-rank shape. Faults on the first cell.
    _run_cell(cell, seed=20261008, with_faults=cell is REAL[0])


@pytest.mark.parametrize("cell", [FAST[1], E64[4], E64[7]], ids=_cell_id)
def test_routed_only_is_bit_identical_to_incumbent(cell):
    """Without a shared operand the routed arithmetic is the incumbent's."""
    x, routed, _, ids, weights, _, _, live, _ = _setup(cell, seed=5)
    native = _native(x, routed, weights, ids, OFFSET, None, 1.0)
    cand = _candidate(x, routed, weights, ids, OFFSET, None, 1.0)
    np.testing.assert_array_equal(cand, native)
    np.testing.assert_array_equal(cand[live:], 0)


def test_no_ep_offset_none_matches_offset_zero():
    cell = FAST[0]
    x, routed, shared, ids, weights, *_ = _setup(cell, seed=9)
    local_ids = ids - OFFSET
    w1, w2, s1, s2 = routed
    a = fused.dense_expert_moe_fused(
        x, w1, w2, s1, s2, weights, local_ids, None, shared
    )
    b = fused.dense_expert_moe_fused(
        x, w1, w2, s1, s2, weights, local_ids, jnp.int32(0), shared
    )
    np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))


# ------------------------------------------------------------ admission


def _sds(shape, dtype):
    return jax.ShapeDtypeStruct(shape, dtype)


def _shared_sds(h, si, scale_shape=None):
    return (
        _sds((h, 2 * si), jnp.float8_e4m3fn),
        _sds(scale_shape or (2 * si,), jnp.float32),
        _sds((si, h), jnp.float8_e4m3fn),
        _sds((h,), jnp.float32),
    )


@pytest.mark.parametrize(
    "si, ok", [(128, True), (256, True), (384, True), (64, False), (200, False)]
)
def test_shared_admission_needs_only_lane_alignment(si, ok):
    x = _sds((16, 6144), jnp.bfloat16)
    assert fused.can_use_fused_shared(x, *_shared_sds(6144, si)) == ok


def test_shared_admission_rejects_blockwise_scales_and_wrong_layout():
    x = _sds((16, 6144), jnp.bfloat16)
    assert not fused.can_use_fused_shared(x, *_shared_sds(6144, 128, (1, 12, 1, 256)))
    ws13, ss13, ws2, ss2 = _shared_sds(6144, 128)
    assert not fused.can_use_fused_shared(
        x, _sds((256, 6144), jnp.float8_e4m3fn), ss13, ws2, ss2
    )
    assert not fused.can_use_fused_shared(
        x, ws13, ss13, _sds((128, 6144), jnp.bfloat16), ss2
    )


def test_buffers_at_served_shape_stay_three():
    for si in (128, 256):
        assert fused._select_expert_buffers(16, 6144, 512, 64, 8, si, 64 * 1024**2) == 3


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("1", True)])
def test_fused_decode_layer_environment(monkeypatch, value, expected):
    from vllm_torchtpu import envs

    name = "TPU_MOE_FUSED_DECODE_LAYER"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert envs.environment_variables[name]() == expected


# --------------------------------------------------------------- wiring


class _Recorder:
    def __init__(self):
        self.calls = []


def test_runner_patch_skips_shared_only_when_admitted(monkeypatch):
    """Flag on: an admitted batch skips the shared MLP, the routed layer sees
    the shared operands and routed scale during its call, the runner sees no
    shared output and does not scale again. A rejected batch takes upstream.
    Needs vLLM (runs in the serving image)."""
    pytest.importorskip("vllm")
    pytest.importorskip("torch_tpu")
    import torch
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExpertsOrder,
    )

    import vllm_torchtpu

    cls = mr.MoERunner
    for name in (
        "_apply_quant_method",
        "_maybe_combine",
        "_maybe_apply_routed_scale_to_output",
        "_tpu_fused_decode_layer_patch",
    ):
        monkeypatch.setattr(cls, name, getattr(cls, name, False), raising=False)
    monkeypatch.setattr(cls, "_tpu_fused_decode_layer_patch", False)
    rec = _Recorder()
    monkeypatch.setattr(
        cls,
        "_apply_quant_method",
        lambda self, *a, **k: rec.calls.append("upstream_apply") or ("S", "R"),
    )
    monkeypatch.setattr(
        cls,
        "_maybe_apply_routed_scale_to_output",
        lambda self, s, f: rec.calls.append("upstream_scale") or (s, f * 2),
    )
    monkeypatch.setattr(
        cls,
        "_maybe_combine",
        lambda self, s, h: (
            rec.calls.append(("upstream_combine", s is not None)),
            (s, h),
        )[1],
    )
    monkeypatch.setenv("TPU_MOE_FUSED_DECODE_LAYER", "1")
    vllm_torchtpu._patch_moe_runner_fused_decode_layer()
    assert cls._tpu_fused_decode_layer_patch

    hs = torch.zeros(16, 8)
    seen = {}

    class Routed:
        quant_method = SimpleNamespace(is_monolithic=True)
        _tpu_fused_decode_layer = None

        def forward_monolithic(self, x, router_logits, input_ids=None):
            seen["during"] = self._tpu_fused_decode_layer
            return x + 1

    class Shared:
        _layer = object()

        def _determine_shared_experts_order(self, x):
            return SharedExpertsOrder.NO_OVERLAP

    admit = {"value": ("w13", "s13", "w2", "s2")}
    routed = Routed()
    fake = SimpleNamespace(
        _shared_experts=Shared(),
        routed_input_transform=None,
        routed_output_transform=None,
        shared_expert_gate=None,
        moe_config=SimpleNamespace(is_sequence_parallel=False),
        routed_experts=routed,
        routed_scaling_factor=2.5,
        _quant_method=SimpleNamespace(
            tpu_fused_decode_shared_operands=lambda layer, x, mlp, rsf: (
                admit["value"] if rsf == 2.5 else None
            )
        ),
    )
    shared_out, fused_out = cls._apply_quant_method(fake, hs, hs, hs)
    assert shared_out is None and torch.equal(fused_out, hs + 1)
    assert seen["during"] == admit["value"]
    assert routed._tpu_fused_decode_layer is None
    assert "upstream_apply" not in rec.calls
    assert cls._maybe_combine(fake, None, fused_out)[0] is None
    s, f = cls._maybe_apply_routed_scale_to_output(fake, None, fused_out)
    assert torch.equal(f, fused_out) and "upstream_scale" not in rec.calls
    # Rejected batch: the upstream methods, scaling included.
    admit["value"] = None
    assert cls._apply_quant_method(fake, hs, hs, hs) == ("S", "R")
    assert cls._maybe_apply_routed_scale_to_output(fake, "S", 1) == ("S", 2)
    assert rec.calls.count("upstream_apply") == 1
    assert rec.calls.count("upstream_scale") == 1


def test_runner_patch_not_installed_when_flag_off(monkeypatch):
    pytest.importorskip("vllm")
    pytest.importorskip("torch_tpu")
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    import vllm_torchtpu

    cls = mr.MoERunner
    before = cls.__dict__.get("_apply_quant_method")
    monkeypatch.delenv("TPU_MOE_FUSED_DECODE_LAYER", raising=False)
    vllm_torchtpu._patch_moe_runner_fused_decode_layer()
    assert cls.__dict__.get("_apply_quant_method") is before


def test_decode_layer_op_is_never_built_in_the_forward():
    """The forward only looks the op up; an unprebuilt key fails loudly
    instead of building pallas.jax_op under Dynamo tracing."""
    pytest.importorskip("vllm")
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.adapter import fused_moe as adapter

    config = dict(
        topk=7,
        activation="silu",
        use_ep=True,
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
        routed_scale=2.5,
    )
    with pytest.raises(RuntimeError, match="not prebuilt"):
        adapter._get_fused_decode_layer_op(**config)
    adapter.prebuild_fused_moe_decode_layer(**config)
    op = adapter._get_fused_decode_layer_op(**config)
    adapter.prebuild_fused_moe_decode_layer(**config)
    assert adapter._get_fused_decode_layer_op(**config) is op
