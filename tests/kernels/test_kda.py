# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the ported KDA Pallas TPU forward kernel.

Reference: ``naive_recurrent_kda`` from vLLM's
``tests/models/kimi_k3/test_kda.py`` @ fa2a2589 (itself ported from FLA's
``naive.py``), reproduced verbatim below and run in float32 **on CPU**. It is a
literal transcription of the KDA recurrence, so it is an independent check on the
chunked Pallas implementation rather than a rearrangement of the same algorithm.

Layouts differ and the tests transpose explicitly:

* naive reference: ``q/k/v/g`` are ``[B, T, H, K|V]``, ``beta`` is ``[B, T, H]``
* KDA kernel:      ``q/k/v/g`` are ``[H, B, T, K|V]``, ``beta`` is ``[H, B, T]``

Comparison uses the reference's own RMSE-relative metric (``assert_close``)
rather than elementwise ``allclose``: the kernel accumulates chunk-wise in
bfloat16/float32 while the reference is a sequential float32 loop, so individual
elements legitimately differ while the aggregate error stays small.
"""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax", reason="KDA kernel requires JAX")
jnp = pytest.importorskip("jax.numpy", reason="KDA kernel requires JAX")


def _has_tpu() -> bool:
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


requires_tpu = pytest.mark.skipif(
    not _has_tpu(), reason="KDA Pallas kernel requires a TPU device")


# ---------------------------------------------------------------------------
# Reference implementation (verbatim from vLLM tests/models/kimi_k3/test_kda.py)
# ---------------------------------------------------------------------------
def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive recurrent KDA reference, ported from FLA's naive.py."""
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5

    q, k, v, g, beta = (x.to(torch.float) for x in [q, k, v, g, beta])
    q = q * scale

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "bhk,bhv->bhkv",
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),
        )
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def assert_close(
    name: str,
    ref: torch.Tensor,
    tri: torch.Tensor,
    ratio: float,
    err_atol: float = 1e-6,
):
    """RMSE-based relative error comparison (from the same vLLM test file)."""
    abs_err = (ref.detach() - tri.detach()).flatten().abs().max().item()
    rmse_diff = (ref.detach() -
                 tri.detach()).flatten().square().mean().sqrt().item()
    rmse_base = ref.detach().flatten().square().mean().sqrt().item()
    rel_err = rmse_diff / (rmse_base + 1e-8)
    print(f"{name:>6} | abs={abs_err:.6f} | rmse={rel_err:.6f} | thr={ratio}")
    if abs_err <= err_atol:
        return
    assert not torch.isnan(ref).any(), f"{name}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{name}: NaN detected in tri"
    assert rel_err < ratio, (
        f"{name}: max abs err {abs_err:.6f}, rmse ratio {rel_err:.6f} "
        f">= {ratio}")


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------
def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize along the last dim, matching FLA's ``l2norm_fwd``."""
    return torch.nn.functional.normalize(x.to(torch.float32), p=2.0, dim=-1)


def _make_inputs(B: int, T: int, H: int, K: int, V: int, seed: int = 0):
    """Build a batch-major ``[B, T, H, *]`` input set in float32 on CPU.

    Deliberately mirrors the distributions in vLLM's own ``test_chunk_kda``:

    * ``q``/``k``/``v`` are **uniform** ``rand``, not ``randn``
    * ``g = logsigmoid(randn)``, i.e. already log-space and strictly negative,
      which is the contract when ``use_gate_in_kernel=False``
    * ``beta = rand().sigmoid()``

    This matters: with unnormalized ``randn`` q/k at K=128 the delta-rule update
    diverges and the *reference itself* produces NaN over a few hundred steps.
    vLLM's test avoids that by L2-normalizing q/k before the reference (see
    ``_ref_kda``), which is also what the kernel's ``use_qk_l2norm_in_kernel``
    does internally.
    """
    gen = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.rand(*shape, generator=gen, dtype=torch.float32)
    randn = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32)

    q = rand(B, T, H, K)
    k = rand(B, T, H, K)
    v = rand(B, T, H, V)
    g = torch.nn.functional.logsigmoid(randn(B, T, H, K))
    beta = rand(B, T, H).sigmoid()
    return q, k, v, g, beta


def _ref_kda(q, k, v, g, beta, **kwargs):
    """Naive reference on L2-normalized q/k, as vLLM's test does.

    The kernel is given raw q/k with ``use_qk_l2norm_in_kernel=True``, so the
    normalization happens on both sides and the comparison stays apples-to-apples.
    """
    return naive_recurrent_kda(_l2norm(q), _l2norm(k), v, g, beta, **kwargs)


# The chunked kernel is scoped to Kimi-K3's bfloat16 path: float32 inputs now
# raise, because the fp32 fallback (unfused intra solve + exact forward
# substitution) was removed as unreachable in production. Inputs are therefore
# cast to bfloat16 here while the reference stays float32 on CPU -- which is the
# comparison that matters, and is what vLLM's own test does. Note numpy cannot
# represent bfloat16, so the cast happens on the jax side in both directions.
KERNEL_DTYPE = jnp.bfloat16


def _to_kernel_layout(x: torch.Tensor) -> jnp.ndarray:
    """``[B, T, H, D]`` (torch, CPU) -> ``[H, B, T, D]`` (jax), bfloat16."""
    return jnp.asarray(
        x.permute(2, 0, 1, 3).contiguous().numpy()).astype(KERNEL_DTYPE)


def _beta_to_kernel_layout(beta: torch.Tensor) -> jnp.ndarray:
    """``[B, T, H]`` (torch, CPU) -> ``[H, B, T]`` (jax), bfloat16."""
    return jnp.asarray(
        beta.permute(2, 0, 1).contiguous().numpy()).astype(KERNEL_DTYPE)


def _from_kernel_layout(o: jnp.ndarray) -> torch.Tensor:
    """``[H, B, T, V]`` (jax) -> ``[B, T, H, V]`` (torch, float32)."""
    return torch.from_numpy(
        np.asarray(o.astype(jnp.float32))).permute(1, 2, 0, 3).contiguous()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_naive_reference_is_self_consistent():
    """Guard the reference itself: state carried across a split must match.

    Runs T steps in one call, then the same inputs as two halves chaining
    ``initial_state``. This runs on CPU only and needs no kernel, so it fails
    loudly if the reference or the harness is wrong before any kernel result is
    interpreted.
    """
    B, T, H, K, V = 1, 16, 2, 32, 32
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, seed=1)

    whole, final = naive_recurrent_kda(q, k, v, g, beta,
                                       output_final_state=True)

    half = T // 2
    first, mid = naive_recurrent_kda(q[:, :half], k[:, :half], v[:, :half],
                                     g[:, :half], beta[:, :half],
                                     output_final_state=True)
    second, final2 = naive_recurrent_kda(q[:, half:], k[:, half:], v[:, half:],
                                        g[:, half:], beta[:, half:],
                                        initial_state=mid,
                                        output_final_state=True)

    assert_close("split-o", whole, torch.cat([first, second], dim=1), 1e-5)
    assert_close("split-S", final, final2, 1e-5)


@requires_tpu
@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 64, 1, 64, 64),
    (1, 128, 2, 64, 64),
    (2, 256, 4, 128, 128),
])
def test_kda_forward_matches_naive_reference(B, T, H, K, V):
    """Kernel output vs the naive CPU reference, no initial state."""
    from vllm_torchtpu.kernels.kda import kda_forward

    q, k, v, g, beta = _make_inputs(B, T, H, K, V, seed=0)
    ref_o, _ = _ref_kda(q, k, v, g, beta)

    o, final_state = kda_forward(
        _to_kernel_layout(q),
        _to_kernel_layout(k),
        _to_kernel_layout(v),
        _to_kernel_layout(g),
        _beta_to_kernel_layout(beta),
        use_qk_l2norm_in_kernel=True,
        output_final_state=False,
    )
    assert final_state is None, "final_state must be None when not requested"

    got_o = _from_kernel_layout(o)
    assert got_o.shape == ref_o.shape, (f"shape {got_o.shape} != {ref_o.shape}")
    assert_close(f"o[{B}x{T}x{H}]", ref_o, got_o, 5e-3)


@requires_tpu
@pytest.mark.parametrize("B,T,H,K,V", [(1, 128, 2, 64, 64)])
def test_kda_forward_final_state_matches_naive_reference(B, T, H, K, V):
    """Also check the returned recurrent state, which decode depends on."""
    from vllm_torchtpu.kernels.kda import kda_forward

    q, k, v, g, beta = _make_inputs(B, T, H, K, V, seed=2)
    ref_o, ref_state = _ref_kda(q, k, v, g, beta, output_final_state=True)

    o, final_state = kda_forward(
        _to_kernel_layout(q),
        _to_kernel_layout(k),
        _to_kernel_layout(v),
        _to_kernel_layout(g),
        _beta_to_kernel_layout(beta),
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
    )
    assert final_state is not None, "final_state requested but not returned"

    assert_close("o", ref_o, _from_kernel_layout(o), 5e-3)

    # Reference state is [B, H, K, V]; the kernel may add a leading segment
    # dim ([B, N, H, K, V]) for the varlen path. Squeeze it when N == 1.
    got_state = torch.from_numpy(
        np.asarray(final_state.astype(jnp.float32)))
    if got_state.ndim == 5:
        assert got_state.shape[1] == 1, (
            f"unexpected segment dim {got_state.shape[1]} for a non-varlen call")
        got_state = got_state[:, 0]
    assert got_state.shape == ref_state.shape, (
        f"state shape {got_state.shape} != {ref_state.shape}")
    assert_close("S", ref_state, got_state, 5e-3)


@requires_tpu
@pytest.mark.parametrize("B,T,H,K,V", [(1, 128, 2, 64, 64)])
def test_kda_forward_honours_initial_state(B, T, H, K, V):
    """Chaining state across a split must match one contiguous run.

    Exercises the same property as the reference self-consistency test, but
    through the kernel: run the first half, feed its final state into the second
    half, and compare the concatenation against a single full-length call.
    """
    from vllm_torchtpu.kernels.kda import kda_forward

    q, k, v, g, beta = _make_inputs(B, T, H, K, V, seed=3)
    ref_o, _ = _ref_kda(q, k, v, g, beta)

    half = T // 2
    kern = lambda sl, init: kda_forward(
        _to_kernel_layout(q[:, sl]),
        _to_kernel_layout(k[:, sl]),
        _to_kernel_layout(v[:, sl]),
        _to_kernel_layout(g[:, sl]),
        _beta_to_kernel_layout(beta[:, sl]),
        initial_state=init,
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
    )

    o1, s1 = kern(slice(0, half), None)
    o2, _ = kern(slice(half, T), s1)

    got = torch.cat([_from_kernel_layout(o1), _from_kernel_layout(o2)], dim=1)
    assert got.shape == ref_o.shape
    assert_close("chained-o", ref_o, got, 5e-3)


# ---------------------------------------------------------------------------
# The flag combination Kimi-K3 actually uses
#
# KimiK3DeltaAttention's prefill call (see the ported layer) is:
#
#     chunk_kda_with_fused_gate(
#         q, k, v,
#         raw_g=..., raw_beta=...,        # -> use_gate_in_kernel=True
#         A_log=..., g_bias=dt_bias,
#         lower_bound=self.gate_lower_bound,      # -5.0 in the real config
#         initial_state=..., output_final_state=True,
#         use_qk_l2norm_in_kernel=True,
#         cu_seqlens=non_spec_query_start_loc,    # varlen
#     )
#
# so the tests above (log-space gate, no lower bound, dense) exercise a
# configuration K3 never runs. These cover the real one.
# ---------------------------------------------------------------------------
def _activate_gate_torch(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor:
    """Batch-major torch port of tokamax's ``base._activate_gate``.

    Upstream operates on head-major ``[H, B, T, K]`` and reshapes ``A_log`` to
    ``(H,1,1,1)`` and ``dt_bias`` to ``(H,1,1,K)``. Here ``g`` is
    ``[B, T, H, K]``, so the broadcast targets become ``(1,1,H,1)`` and
    ``(1,1,H,K)``; the underlying ``(H, K)`` ordering of ``dt_bias`` is the same.
    """
    _, _, H, K = g.shape
    g_f = g.to(torch.float32)
    if dt_bias is not None:
        g_f = g_f + dt_bias.to(torch.float32).reshape(1, 1, H, K)
    A = torch.exp(A_log.to(torch.float32)).reshape(1, 1, H, 1)
    if lower_bound is None:
        return -A * torch.nn.functional.softplus(g_f)
    return lower_bound * torch.sigmoid(A * g_f)


def _make_gate_params(H: int, K: int, seed: int = 7):
    """``A_log`` of shape ``[H]`` and ``dt_bias`` of shape ``[H*K]``.

    Shapes are enforced by ``base._validate_gate_args``; ``A_log`` is kept small
    and negative so ``A = exp(A_log)`` stays near 1 and the gate does not
    saturate.
    """
    gen = torch.Generator().manual_seed(seed)
    A_log = torch.randn(H, generator=gen, dtype=torch.float32) * 0.5 - 1.0
    dt_bias = torch.randn(H * K, generator=gen, dtype=torch.float32) * 0.1
    return A_log, dt_bias


@requires_tpu
@pytest.mark.parametrize("lower_bound", [-5.0, -1.0, None])
def test_kda_gate_in_kernel_matches_reference(lower_bound):
    """``use_gate_in_kernel=True``: kernel activates the raw gate itself.

    ``lower_bound=-5.0`` is the real config's ``gate_lower_bound``. ``None``
    selects the unbounded ``-A * softplus(g)`` branch, which requires
    ``safe_gate=False`` (``_validate_gate_args`` rejects an unbounded gate while
    ``safe_gate`` is on).
    """
    from vllm_torchtpu.kernels.kda import kda_forward

    B, T, H, K, V = 1, 128, 2, 64, 64
    q, k, v, raw_g, beta = _make_inputs(B, T, H, K, V, seed=11)
    A_log, dt_bias = _make_gate_params(H, K)

    # Reference: activate the gate outside, then run the log-space recurrence.
    log_g = _activate_gate_torch(raw_g, A_log, dt_bias, lower_bound)
    ref_o, _ = _ref_kda(q, k, v, log_g, beta)

    o, _ = kda_forward(
        _to_kernel_layout(q),
        _to_kernel_layout(k),
        _to_kernel_layout(v),
        _to_kernel_layout(raw_g),
        _beta_to_kernel_layout(beta),
        A_log=jnp.asarray(A_log.numpy()),
        dt_bias=jnp.asarray(dt_bias.numpy()),
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        safe_gate=lower_bound is not None,
        output_final_state=False,
    )
    assert_close(f"gate(lb={lower_bound})", ref_o, _from_kernel_layout(o), 5e-3)


def _segment_ids_from_row_boundaries(
    T: int,
    row_bounds: list[list[int]],
) -> torch.Tensor:
    """1-indexed ``[B, T]`` segment IDs, one boundary list per batch row.

    Each row's ``bounds`` is cu_seqlens-style (``[0, n1, n1+n2, ...]``) and is
    numbered independently from 1, because segments are tracked per row. Positions
    beyond a row's last boundary stay 0, which the kernel treats as padding.
    """
    seg = torch.zeros(len(row_bounds), T, dtype=torch.int32)
    for b, bounds in enumerate(row_bounds):
        for i in range(len(bounds) - 1):
            seg[b, bounds[i]:bounds[i + 1]] = i + 1
    return seg


def _segment_ids_from_boundaries(T: int, bounds: list[int]) -> torch.Tensor:
    """``[1, T]`` segment IDs: the single-row case of the above."""
    return _segment_ids_from_row_boundaries(T, [bounds])


@requires_tpu
@pytest.mark.parametrize("bounds", [
    [0, 64, 128],            # aligned to chunk_size
    [0, 64, 192, 256],       # aligned, uneven segment lengths
    [0, 15, 100, 256],       # ragged, not chunk-aligned
])
def test_kda_varlen_matches_per_segment_reference(bounds):
    """Varlen via ``segment_ids``, against a per-segment reference loop.

    Mirrors how vLLM's own ``test_chunk_kda`` builds its reference: each segment
    is an independent sequence, so the reference runs the naive recurrence once
    per segment (each starting from zero state) and concatenates along T. Ragged
    boundaries are included because the kernel pads to ``chunk_size`` internally
    and that alignment is exactly where varlen bugs hide.
    """
    from vllm_torchtpu.kernels.kda import kda_forward

    T = bounds[-1]
    B, H, K, V = 1, 2, 64, 64
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, seed=13)

    ref_parts = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        o_i, _ = _ref_kda(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e],
                          beta[:, s:e])
        ref_parts.append(o_i)
    ref_o = torch.cat(ref_parts, dim=1)

    segment_ids = _segment_ids_from_boundaries(T, bounds)
    o, _ = kda_forward(
        _to_kernel_layout(q),
        _to_kernel_layout(k),
        _to_kernel_layout(v),
        _to_kernel_layout(g),
        _beta_to_kernel_layout(beta),
        segment_ids=jnp.asarray(segment_ids.numpy()),
        N_max=len(bounds) - 1,
        use_qk_l2norm_in_kernel=True,
        output_final_state=False,
    )
    got = _from_kernel_layout(o)
    assert got.shape == ref_o.shape, f"{got.shape} != {ref_o.shape}"
    assert_close(f"varlen{bounds}", ref_o, got, 5e-3)


@requires_tpu
def test_kda_kimi_k3_flag_combination():
    """All six flags the real K3 prefill call sets, together.

    varlen + raw gate (A_log/dt_bias) + lower_bound=-5.0 + in-kernel L2 norm +
    per-segment initial_state + output_final_state. This is the configuration a
    Kimi-K3 prefill actually runs; the individual tests above isolate the pieces.
    """
    from vllm_torchtpu.kernels.kda import kda_forward

    bounds = [0, 64, 192, 256]
    T = bounds[-1]
    B, H, K, V = 1, 2, 64, 64
    N = len(bounds) - 1
    q, k, v, raw_g, beta = _make_inputs(B, T, H, K, V, seed=17)
    A_log, dt_bias = _make_gate_params(H, K)
    lower_bound = -5.0

    gen = torch.Generator().manual_seed(19)
    # One recurrent state per segment: [B, N, H, K, V].
    init = torch.randn(B, N, H, K, V, generator=gen, dtype=torch.float32) * 0.1

    log_g = _activate_gate_torch(raw_g, A_log, dt_bias, lower_bound)

    ref_parts, ref_states = [], []
    for i in range(N):
        s, e = bounds[i], bounds[i + 1]
        o_i, s_i = _ref_kda(q[:, s:e], k[:, s:e], v[:, s:e], log_g[:, s:e],
                            beta[:, s:e],
                            initial_state=init[:, i],
                            output_final_state=True)
        ref_parts.append(o_i)
        ref_states.append(s_i)
    ref_o = torch.cat(ref_parts, dim=1)
    ref_state = torch.stack(ref_states, dim=1)  # [B, N, H, K, V]

    o, final_state = kda_forward(
        _to_kernel_layout(q),
        _to_kernel_layout(k),
        _to_kernel_layout(v),
        _to_kernel_layout(raw_g),
        _beta_to_kernel_layout(beta),
        A_log=jnp.asarray(A_log.numpy()),
        dt_bias=jnp.asarray(dt_bias.numpy()),
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        safe_gate=True,
        segment_ids=jnp.asarray(_segment_ids_from_boundaries(T, bounds).numpy()),
        N_max=N,
        initial_state=jnp.asarray(init.numpy()),
        output_final_state=True,
    )

    assert_close("k3-o", ref_o, _from_kernel_layout(o), 5e-3)

    assert final_state is not None
    got_state = torch.from_numpy(
        np.asarray(final_state.astype(jnp.float32)))
    assert got_state.shape == ref_state.shape, (
        f"state {got_state.shape} != {ref_state.shape}")
    assert_close("k3-S", ref_state, got_state, 5e-3)


# ---------------------------------------------------------------------------
# B > 1 *and* varlen together
#
# The tests above cover the two batching mechanisms separately: the ``B`` axis of
# ``[H, B, T, K]`` (equal-length sequences, one state each) and ``segment_ids``
# (variable-length sequences packed along ``T``, ``N`` states). Nothing exercised
# them at once, so it was unverified whether segments are numbered per row or
# collapse across rows -- and in particular whether per-``(row, segment)``
# ``final_state`` slots land in the right places when rows disagree on how many
# segments they hold.
#
# vLLM serving itself uses B=1 with a flat packed token dim, so this is defensive
# coverage of the kernel's contract rather than of a path K3 currently takes.
# ---------------------------------------------------------------------------
def _batched_varlen_reference(q, k, v, g, beta, row_bounds, init, N_max):
    """Naive reference run independently per ``(row, segment)``.

    Returns ``(ref_o, ref_state, valid, live)``:

    * ``valid`` is a ``[B, T]`` bool mask of non-padding positions. Output at
      padding positions is unconstrained, so the comparison is masked.
    * ``live`` lists the ``(row, segment)`` pairs that carry a real sequence.
      State slots outside ``live`` are likewise unconstrained -- a row with fewer
      segments than ``N_max`` leaves the trailing slots unused, and nothing reads
      them -- so only live slots are asserted on.
    """
    B, T, H, _ = q.shape
    V = v.shape[-1]
    ref_o = torch.zeros_like(v)
    ref_state = torch.zeros(B, N_max, H, q.shape[-1], V, dtype=torch.float32)
    valid = torch.zeros(B, T, dtype=torch.bool)
    live: list[tuple[int, int]] = []
    for b, bounds in enumerate(row_bounds):
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            o_i, s_i = _ref_kda(q[b:b + 1, s:e], k[b:b + 1, s:e],
                                v[b:b + 1, s:e], g[b:b + 1, s:e],
                                beta[b:b + 1, s:e],
                                initial_state=init[b:b + 1, i],
                                output_final_state=True)
            ref_o[b, s:e] = o_i[0]
            ref_state[b, i] = s_i[0]
            valid[b, s:e] = True
            live.append((b, i))
    return ref_o, ref_state, valid, live


def _assert_batched_varlen(name, ref_o, got_o, ref_state, got_state, valid,
                           live):
    assert got_o.shape == ref_o.shape, f"{got_o.shape} != {ref_o.shape}"
    assert_close(f"{name}-o", ref_o[valid], got_o[valid], 5e-3)

    assert got_state is not None, "final_state requested but not returned"
    assert got_state.shape == ref_state.shape, (
        f"state {got_state.shape} != {ref_state.shape}")
    # Compare only the slots that hold a real segment; see _batched_varlen_reference.
    assert_close(f"{name}-S",
                 torch.stack([ref_state[b, i] for b, i in live]),
                 torch.stack([got_state[b, i] for b, i in live]), 5e-3)


@requires_tpu
@pytest.mark.parametrize("row_bounds", [
    [[0, 64, 192, 256], [0, 64, 192, 256]],  # identical segmentation
    [[0, 64, 192, 256], [0, 128, 256]],      # different boundaries and counts
    [[0, 15, 100, 256], [0, 200, 256]],      # ragged, different per row
    [[0, 64, 192, 256], [0, 256]],           # 3 segments vs 1 (N < N_max)
    [[0, 64, 192, 256], [0, 100]],           # row 1 full, row 2 trailing padding
    [[0, 64, 128], [0, 32, 96], [0, 128]],   # B=3, all three shapes differing
])
def test_kda_varlen_batched_matches_per_segment_reference(row_bounds):
    """B>1 with per-row independent segmentation, including per-segment state.

    Log-space gate, so a failure here isolates batching rather than the gate
    (covered separately by ``test_kda_gate_in_kernel_matches_reference``). The
    per-segment ``initial_state`` is nonzero, which is what makes this a real test
    of state routing: with a zero state the output would be insensitive to
    mis-indexed slots.
    """
    from vllm_torchtpu.kernels.kda import kda_forward

    B = len(row_bounds)
    T = max(bounds[-1] for bounds in row_bounds)
    N_max = max(len(bounds) - 1 for bounds in row_bounds)
    H, K, V = 2, 64, 64
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, seed=23)

    gen = torch.Generator().manual_seed(29)
    init = torch.randn(B, N_max, H, K, V, generator=gen,
                       dtype=torch.float32) * 0.1

    ref_o, ref_state, valid, live = _batched_varlen_reference(
        q, k, v, g, beta, row_bounds, init, N_max)

    o, final_state = kda_forward(
        _to_kernel_layout(q),
        _to_kernel_layout(k),
        _to_kernel_layout(v),
        _to_kernel_layout(g),
        _beta_to_kernel_layout(beta),
        segment_ids=jnp.asarray(
            _segment_ids_from_row_boundaries(T, row_bounds).numpy()),
        N_max=N_max,
        initial_state=jnp.asarray(init.numpy()),
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
    )

    _assert_batched_varlen(f"batched{row_bounds}", ref_o,
                           _from_kernel_layout(o), ref_state,
                           torch.from_numpy(
                               np.asarray(final_state.astype(jnp.float32))),
                           valid,
                           live)


@requires_tpu
def test_kda_kimi_k3_flag_combination_batched():
    """The full K3 flag set at B>1 with rows holding different segment counts.

    ``test_kda_kimi_k3_flag_combination`` pins the real flag combination at B=1;
    this is the same six flags with the batch axis also in play.
    """
    from vllm_torchtpu.kernels.kda import kda_forward

    row_bounds = [[0, 64, 192, 256], [0, 128, 256], [0, 256]]
    B = len(row_bounds)
    T = 256
    N_max = max(len(bounds) - 1 for bounds in row_bounds)
    H, K, V = 2, 64, 64
    lower_bound = -5.0

    q, k, v, raw_g, beta = _make_inputs(B, T, H, K, V, seed=31)
    A_log, dt_bias = _make_gate_params(H, K)
    gen = torch.Generator().manual_seed(37)
    init = torch.randn(B, N_max, H, K, V, generator=gen,
                       dtype=torch.float32) * 0.1

    # Reference activates the gate outside the kernel, as the B=1 test does.
    log_g = _activate_gate_torch(raw_g, A_log, dt_bias, lower_bound)
    ref_o, ref_state, valid, live = _batched_varlen_reference(
        q, k, v, log_g, beta, row_bounds, init, N_max)

    o, final_state = kda_forward(
        _to_kernel_layout(q),
        _to_kernel_layout(k),
        _to_kernel_layout(v),
        _to_kernel_layout(raw_g),
        _beta_to_kernel_layout(beta),
        A_log=jnp.asarray(A_log.numpy()),
        dt_bias=jnp.asarray(dt_bias.numpy()),
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        safe_gate=True,
        segment_ids=jnp.asarray(
            _segment_ids_from_row_boundaries(T, row_bounds).numpy()),
        N_max=N_max,
        initial_state=jnp.asarray(init.numpy()),
        output_final_state=True,
    )

    _assert_batched_varlen("k3-batched", ref_o, _from_kernel_layout(o),
                           ref_state,
                           torch.from_numpy(
                               np.asarray(final_state.astype(jnp.float32))),
                           valid,
                           live)


# ---------------------------------------------------------------------------
# The decode kernel (kda/decode.py)
#
# A separate Pallas kernel: GDN v3's single-token recurrent path with KDA's
# per-channel gate, and the per-sequence state gather expressed as a paged DMA
# instead of an XLA gather. Its layout and conventions differ from
# ``kda_forward`` deliberately, so these tests use their own input builder:
#
#   kda_forward : q/k/v/g ``[H, B, T, D]``, dt_bias ``[H*K]``, varlen segments
#   kda_decode  : q/k/v/g ``[num_seqs, n_v, D]``, dt_bias ``[n_v, d_k]``,
#                 one token per sequence, state in a paged pool
#
# Tolerances are looser here than for the chunked kernel (1e-7). Both
# `jax.lax.dot` calls run at the MXU's default precision, as GDN v3's do, so
# float32 inputs still multiply at reduced precision. That is the intended
# trade; the tests assert against the same 5e-3 RMSE ratio used above.
# ---------------------------------------------------------------------------
def _make_decode_inputs(N: int, n_v: int, d_k: int, d_v: int, seed: int = 0):
    """Inputs in the decode kernel's ``[num_seqs, n_v, D]`` layout.

    Same distributions as ``_make_inputs``: uniform q/k/v, ``logsigmoid`` raw
    gate, sigmoid beta (the decode kernel takes beta already activated, matching
    ``kda_forward``).
    """
    gen = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.rand(*s, generator=gen, dtype=torch.float32)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32)

    q = rand(N, n_v, d_k)
    k = rand(N, n_v, d_k)
    v = rand(N, n_v, d_v)
    raw_g = torch.nn.functional.logsigmoid(randn(N, n_v, d_k))
    beta = rand(N, n_v).sigmoid()
    a_log = randn(n_v) * 0.5 - 1.0
    dt_bias = randn(n_v, d_k) * 0.1
    return q, k, v, raw_g, beta, a_log, dt_bias


def _decode_reference(q, k, v, raw_g, beta, a_log, dt_bias, state,
                      lower_bound):
    """One KDA token step per sequence, float32 on CPU.

    Reuses the same activation and recurrence as the chunked tests: activate the
    per-channel gate, L2-normalize q/k, then one step of
    ``naive_recurrent_kda``'s loop body.
    """
    n_v, d_k = q.shape[1], q.shape[2]
    A = torch.exp(a_log).reshape(1, n_v, 1)
    if lower_bound is None:
        log_g = -A * torch.nn.functional.softplus(q.new_tensor(0.0) + raw_g +
                                                  dt_bias.reshape(1, n_v, d_k))
    else:
        log_g = lower_bound * torch.sigmoid(
            A * (raw_g + dt_bias.reshape(1, n_v, d_k)))

    qn = _l2norm(q) * (d_k**-0.5)
    kn = _l2norm(k)
    S = state * log_g[..., None].exp()
    kv = (kn[..., None] * S).sum(-2)
    S = S + (beta[..., None] * kn)[..., None] * (v - kv)[..., None, :]
    out = (qn[..., None] * S).sum(-2)
    return out, S


def _run_decode(q, k, v, raw_g, beta, a_log, dt_bias, pool, indices,
                has_init, lower_bound, cast=None):
    """Run the decode kernel. ``cast`` optionally lowers q/k/v/g/beta precision.

    The decode kernel accepts float32 (it computes in float32 regardless), so
    these tests feed float32 for the tightest possible comparison against the
    reference. ``cast`` exists for the cross-check against the chunked kernel,
    which is bfloat16-only.
    """
    from vllm_torchtpu.kernels.kda.decode import kda_decode

    j = lambda x: (jnp.asarray(x.numpy()) if cast is None else
                   jnp.asarray(x.numpy()).astype(cast))
    f32 = lambda x: jnp.asarray(x.numpy())
    out, new_pool = kda_decode(j(q), j(k), j(v), j(raw_g), j(beta), f32(pool),
                               f32(a_log), f32(dt_bias),
                               jnp.asarray(indices.numpy()),
                               jnp.asarray(has_init.numpy()),
                               lower_bound=lower_bound)
    # numpy has no bfloat16, so go through float32 when `cast` lowered dtype.
    to_t = lambda x: torch.from_numpy(np.asarray(x.astype(jnp.float32))).clone()
    return to_t(out), to_t(new_pool)


def _pool_from_states(states, indices, num_slots):
    """Scatter per-sequence states into a pool; slot 0 stays reserved."""
    n_v, d_k, d_v = states.shape[1], states.shape[2], states.shape[3]
    pool = torch.zeros(num_slots, n_v, d_k, d_v, dtype=torch.float32)
    for s, slot in enumerate(indices.tolist()):
        if slot > 0:
            pool[slot] = states[s]
    return pool


@requires_tpu
@pytest.mark.parametrize("N,n_v,d_k,d_v", [
    (1, 2, 128, 128),
    (8, 2, 128, 128),
    (32, 6, 128, 128),
])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_kda_decode_matches_naive_reference(N, n_v, d_k, d_v, lower_bound):
    """One decode step, nonzero carried state, against the naive reference."""
    q, k, v, raw_g, beta, a_log, dt_bias = _make_decode_inputs(
        N, n_v, d_k, d_v, seed=41)
    gen = torch.Generator().manual_seed(43)
    state0 = torch.randn(N, n_v, d_k, d_v, generator=gen,
                         dtype=torch.float32) * 0.1

    ref_out, ref_state = _decode_reference(q, k, v, raw_g, beta, a_log, dt_bias,
                                           state0, lower_bound)

    indices = torch.arange(1, N + 1, dtype=torch.int32)
    pool = _pool_from_states(state0, indices, N + 1)
    got_out, got_pool = _run_decode(q, k, v, raw_g, beta, a_log, dt_bias, pool,
                                    indices, torch.ones(N, dtype=torch.int32),
                                    lower_bound)

    assert got_out.shape == ref_out.shape
    assert_close(f"decode-o[N{N}h{n_v}lb{lower_bound}]", ref_out, got_out, 5e-3)
    got_state = torch.stack([got_pool[i] for i in indices.tolist()])
    assert_close(f"decode-S[N{N}h{n_v}lb{lower_bound}]", ref_state, got_state,
                 5e-3)


@requires_tpu
def test_kda_decode_paged_state_routing():
    """Shuffled, non-contiguous slots: each sequence must hit its own slot.

    The whole point of the kernel is that the state gather is a paged DMA driven
    by ``state_indices``. A routing bug would still produce plausible numbers, so
    this uses a permuted, sparse slot assignment in an oversized pool and also
    asserts that untouched slots are left alone.
    """
    N, n_v, d_k, d_v = 8, 2, 128, 128
    lower_bound = -5.0
    q, k, v, raw_g, beta, a_log, dt_bias = _make_decode_inputs(
        N, n_v, d_k, d_v, seed=47)
    gen = torch.Generator().manual_seed(53)
    state0 = torch.randn(N, n_v, d_k, d_v, generator=gen,
                         dtype=torch.float32) * 0.1

    # Sparse and out of order, in a pool with room to spare.
    indices = torch.tensor([9, 3, 14, 1, 7, 12, 5, 11], dtype=torch.int32)
    num_slots = 16
    pool = _pool_from_states(state0, indices, num_slots)
    # Poison every unused slot: if the kernel mis-routes, it reads poison.
    used = set(indices.tolist())
    poison = torch.full((n_v, d_k, d_v), 7.0, dtype=torch.float32)
    for slot in range(1, num_slots):
        if slot not in used:
            pool[slot] = poison

    ref_out, ref_state = _decode_reference(q, k, v, raw_g, beta, a_log, dt_bias,
                                           state0, lower_bound)
    got_out, got_pool = _run_decode(q, k, v, raw_g, beta, a_log, dt_bias, pool,
                                    indices, torch.ones(N, dtype=torch.int32),
                                    lower_bound)

    assert_close("paged-o", ref_out, got_out, 5e-3)
    got_state = torch.stack([got_pool[i] for i in indices.tolist()])
    assert_close("paged-S", ref_state, got_state, 5e-3)

    for slot in range(1, num_slots):
        if slot not in used:
            assert torch.equal(got_pool[slot], poison), (
                f"kernel wrote to unused slot {slot}")


@requires_tpu
def test_kda_decode_null_slot_and_first_step():
    """``state_indices <= 0`` and ``has_initial_state == 0`` must both be safe.

    Slot 0 is the reserved null block and may hold garbage. A null sequence must
    emit zeros and leave the pool alone; a sequence on its first step must ignore
    whatever its slot currently holds and start from zero state.
    """
    N, n_v, d_k, d_v = 4, 2, 128, 128
    lower_bound = -5.0
    q, k, v, raw_g, beta, a_log, dt_bias = _make_decode_inputs(
        N, n_v, d_k, d_v, seed=59)
    gen = torch.Generator().manual_seed(61)
    state0 = torch.randn(N, n_v, d_k, d_v, generator=gen,
                         dtype=torch.float32) * 0.1

    indices = torch.arange(1, N + 1, dtype=torch.int32)
    pool = _pool_from_states(state0, indices, N + 1)
    pool[0] = 13.0  # garbage in the reserved null slot

    # Sequence 2 is null; sequence 3 is on its first step.
    idx_null = indices.clone()
    idx_null[2] = 0
    has_init = torch.ones(N, dtype=torch.int32)
    has_init[3] = 0

    zero_state = state0.clone()
    zero_state[3] = 0.0  # what sequence 3's reference should start from
    ref_out, ref_state = _decode_reference(q, k, v, raw_g, beta, a_log, dt_bias,
                                           zero_state, lower_bound)

    got_out, got_pool = _run_decode(q, k, v, raw_g, beta, a_log, dt_bias, pool,
                                    idx_null, has_init, lower_bound)

    assert torch.equal(got_out[2], torch.zeros_like(got_out[2])), (
        "null sequence must emit zeros")
    for s in (0, 1, 3):
        assert_close(f"null-o[seq{s}]", ref_out[s], got_out[s], 5e-3)
    # Sequence 3 started from zero state despite its slot holding state0[3].
    assert_close("first-step-S", ref_state[3], got_pool[indices[3].item()],
                 5e-3)
    # The live sequences' slots are correct and slot 0 was not adopted as state.
    for s in (0, 1):
        assert_close(f"null-S[seq{s}]", ref_state[s],
                     got_pool[indices[s].item()], 5e-3)


@requires_tpu
def test_kda_decode_multistep_does_not_drift():
    """Chain 8 steps through the pool and compare against 8 reference steps.

    Single-step agreement does not prove the state write-back is right, and MXU
    default precision could compound across steps. This walks the same inputs
    through both and checks the error does not grow.
    """
    N, n_v, d_k, d_v, steps = 8, 2, 128, 128, 8
    lower_bound = -5.0
    indices = torch.arange(1, N + 1, dtype=torch.int32)
    pool = torch.zeros(N + 1, n_v, d_k, d_v, dtype=torch.float32)
    ref_state = torch.zeros(N, n_v, d_k, d_v, dtype=torch.float32)
    has_init = torch.zeros(N, dtype=torch.int32)  # first step: no carried state

    rel_errs = []
    for t in range(steps):
        q, k, v, raw_g, beta, a_log, dt_bias = _make_decode_inputs(
            N, n_v, d_k, d_v, seed=100 + t)
        ref_out, ref_state = _decode_reference(q, k, v, raw_g, beta, a_log,
                                               dt_bias, ref_state, lower_bound)
        got_out, pool = _run_decode(q, k, v, raw_g, beta, a_log, dt_bias, pool,
                                    indices, has_init, lower_bound)
        has_init = torch.ones(N, dtype=torch.int32)  # state carried from now on

        got_state = torch.stack([pool[i] for i in indices.tolist()])
        rel = (ref_state - got_state).square().mean().sqrt().item() / (
            ref_state.square().mean().sqrt().item() + 1e-8)
        rel_errs.append(rel)
        assert_close(f"multistep-o[t{t}]", ref_out, got_out, 5e-3)
        assert_close(f"multistep-S[t{t}]", ref_state, got_state, 5e-3)

    print(f"state rmse by step: {['%.2e' % e for e in rel_errs]}")
    assert rel_errs[-1] < 5e-3, f"state drifted to {rel_errs[-1]:.3e}"


@requires_tpu
def test_kda_decode_agrees_with_chunked_kernel():
    """Cross-check the two kernels against each other, not just the reference.

    ``kda_forward`` fed one token per varlen segment computes exactly what the
    decode kernel computes, by a completely different route (chunked, padded to
    64, triangular solve). Agreement between them is a much stronger signal than
    either matching the reference alone, and it is the property that lets decode
    be swapped in without changing results.
    """
    from vllm_torchtpu.kernels.kda import kda_forward

    N, n_v, d_k, d_v = 8, 2, 128, 128
    lower_bound = -5.0
    q, k, v, raw_g, beta, a_log, dt_bias = _make_decode_inputs(
        N, n_v, d_k, d_v, seed=67)
    gen = torch.Generator().manual_seed(71)
    state0 = torch.randn(N, n_v, d_k, d_v, generator=gen,
                         dtype=torch.float32) * 0.1

    # Chunked path: N segments of 1 token, one initial state each.
    # [num_seqs, n_v, D] -> [H, B, T, D] with B=1, T=N.
    # Both kernels get bfloat16 inputs, so the comparison isolates the two
    # algorithms rather than a dtype difference (the chunked path is bf16-only).
    to_hbtd = lambda x: jnp.asarray(
        x.permute(1, 0, 2).unsqueeze(1).contiguous().numpy()).astype(
            KERNEL_DTYPE)
    o_chunked, s_chunked = kda_forward(
        to_hbtd(q),
        to_hbtd(k),
        to_hbtd(v),
        to_hbtd(raw_g),
        jnp.asarray(beta.permute(1, 0).unsqueeze(1).contiguous().numpy()).astype(
            KERNEL_DTYPE),
        A_log=jnp.asarray(a_log.numpy()),
        dt_bias=jnp.asarray(dt_bias.reshape(-1).numpy()),
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        safe_gate=True,
        segment_ids=jnp.asarray(
            (torch.arange(N, dtype=torch.int32) + 1).reshape(1, N).numpy()),
        N_max=N,
        initial_state=jnp.asarray(
            state0.unsqueeze(0).permute(0, 1, 2, 3, 4).numpy()),
        output_final_state=True,
    )
    # [H, B, T, V] -> [num_seqs, n_v, d_v]
    chunked_out = torch.from_numpy(
        np.asarray(o_chunked.astype(jnp.float32)))[:, 0].permute(
        1, 0, 2).contiguous()
    chunked_state = torch.from_numpy(
        np.asarray(s_chunked.astype(jnp.float32)))[0]  # [N,H,K,V]

    indices = torch.arange(1, N + 1, dtype=torch.int32)
    pool = _pool_from_states(state0, indices, N + 1)
    decode_out, decode_pool = _run_decode(q, k, v, raw_g, beta, a_log, dt_bias,
                                          pool, indices,
                                          torch.ones(N, dtype=torch.int32),
                                          lower_bound, cast=KERNEL_DTYPE)
    decode_state = torch.stack([decode_pool[i] for i in indices.tolist()])

    assert_close("chunked-vs-decode-o", chunked_out, decode_out, 5e-3)
    assert_close("chunked-vs-decode-S", chunked_state, decode_state, 5e-3)


@requires_tpu
def test_kda_decode_state_transposed_layout():
    """``state_transposed=True`` reads/writes a ``[d_v, d_k]`` pool.

    vLLM's KDA recurrent cache stores each slot as the transpose of what the
    kernels work in, so the decode kernel converts around its paged load/store
    (see ``_VLLM_STATE_IS_VK`` in models/kimi_k3/kda.py). Feeding a transposed
    pool with the flag on must reproduce the untransposed run exactly, transposed
    -- otherwise the conversion is lopsided (applied on load but not on store, or
    vice versa), which would silently corrupt state across decode steps.
    """
    from vllm_torchtpu.kernels.kda.decode import kda_decode

    N, n_v, d_k, d_v = 8, 2, 128, 128
    lower_bound = -5.0
    q, k, v, raw_g, beta, a_log, dt_bias = _make_decode_inputs(
        N, n_v, d_k, d_v, seed=73)
    gen = torch.Generator().manual_seed(79)
    state0 = torch.randn(N, n_v, d_k, d_v, generator=gen,
                         dtype=torch.float32) * 0.1
    indices = torch.arange(1, N + 1, dtype=torch.int32)
    pool = _pool_from_states(state0, indices, N + 1)

    j = lambda x: jnp.asarray(x.numpy())
    common = dict(lower_bound=lower_bound)
    out_kv, pool_kv = kda_decode(
        j(q), j(k), j(v), j(raw_g), j(beta), j(pool), j(a_log), j(dt_bias),
        jnp.asarray(indices.numpy()), jnp.ones((N, ), dtype=jnp.int32),
        state_transposed=False, **common)
    # Same data, pool stored transposed, flag on.
    out_vk, pool_vk = kda_decode(
        j(q), j(k), j(v), j(raw_g), j(beta),
        j(pool.transpose(-1, -2).contiguous()), j(a_log), j(dt_bias),
        jnp.asarray(indices.numpy()), jnp.ones((N, ), dtype=jnp.int32),
        state_transposed=True, **common)

    o_kv = torch.from_numpy(np.asarray(out_kv))
    o_vk = torch.from_numpy(np.asarray(out_vk))
    assert_close("transposed-o", o_kv, o_vk, 5e-3)

    s_kv = torch.from_numpy(np.asarray(pool_kv))
    s_vk = torch.from_numpy(np.asarray(pool_vk)).transpose(-1, -2)
    live = [i for i in indices.tolist()]
    assert_close("transposed-S", torch.stack([s_kv[i] for i in live]),
                 torch.stack([s_vk[i] for i in live]), 5e-3)
