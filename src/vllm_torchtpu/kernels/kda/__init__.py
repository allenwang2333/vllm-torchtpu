# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention (KDA) Pallas TPU kernel.

Derived from tokamax's experimental KDA op; see ``README.md`` for provenance.
**Forward path only** — inference needs no
gradients, so tokamax's backward kernel, its ``op.Op`` wrapper and its custom VJP
are not ported. Everything here is plain JAX/Pallas, matching the style of the
plugin's other kernels.

KDA is the per-channel-gated variant of the gated delta rule used by Kimi models.
It differs from the Qwen-style GDN kernel in ``kernels/gdn/``: the decay ``g`` is
per channel (shape ``[H, B, T, K]``) rather than per head.

Layout note — the kernel is **head-major** (``[H, B, T, K]``), whereas vLLM's
model code and its naive reference are batch-major (``[B, T, H, K]``). Callers
must transpose; ``kda_forward`` does not, to keep it a faithful passthrough.

Usage::

    from vllm_torchtpu.kernels.kda import kda_forward

    o, final_state = kda_forward(q, k, v, g, beta, scale=..., chunk_size=64)

See ``tests/kernels/test_kda.py`` for correctness against vLLM's naive recurrent
reference, and ``docs/ops/kda/chunk_kda_fwd_design_doc.md`` in the tokamax
checkout for the algorithm.
"""

from typing import Any

from vllm_torchtpu.kernels.kda.decode import kda_decode
from vllm_torchtpu.kernels.kda.pallas_tpu_fwd import chunk_kda_fwd_custom
from vllm_torchtpu.kernels.kda.preprocess import (_preprocess_inputs,
                                                  check_inputs_support)

# `kda_forward` is the chunked path (prefill); `kda_decode` is the
# single-token recurrent path (decode). See decode.py for why they are separate
# kernels rather than one.
__all__ = ["chunk_kda_fwd_custom", "kda_decode", "kda_forward"]


def kda_forward(
    q,
    k,
    v,
    g,
    beta,
    *,
    A_log=None,
    dt_bias=None,
    scale: float | None = None,
    initial_state=None,
    output_final_state: bool = False,
    use_gate_in_kernel: bool = False,
    segment_ids=None,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    chunk_size: int = 64,
    **kwargs: Any,
):
    """Run the KDA forward kernel and return ``(output, final_state)``.

    Thin wrapper over :func:`chunk_kda_fwd_custom` that drops the residuals
    (needed only for a backward pass) so callers get the tuple they actually
    want. Argument meanings are unchanged from tokamax:

    Args:
      q, k: ``[H, B, T, K]`` queries and keys.
      v: ``[H, B, T, V]`` values.
      g: ``[H, B, T, K]`` per-channel gate. In log space unless
        ``use_gate_in_kernel`` is set, in which case it is raw and is activated
        in-kernel using ``A_log`` and ``dt_bias``.
      beta: ``[H, B, T]`` per-token delta-rule learning rate.
      A_log: ``[H]`` gate parameter; required when ``use_gate_in_kernel``.
      dt_bias: ``[H*K]`` optional gate bias.
      scale: query scale, defaults to ``K ** -0.5``.
      initial_state: ``[B, H, K, V]`` or ``[B, N, H, K, V]`` recurrent state.
      output_final_state: whether to return the final state (else ``None``).
      use_gate_in_kernel: treat ``g`` as raw gate input.
      segment_ids: ``[B, T]`` 1-indexed varlen segment IDs, 0 for padding.
      safe_gate: match the kernel's gate validation.
      lower_bound: optional sigmoid-gate lower bound.
      chunk_size: chunk length used by the kernel.
      **kwargs: forwarded verbatim (varlen metadata such as ``cu_seqlens``,
        ``chunk_indices`` and ``aligned_segment_ids``).

    Returns:
      ``(output, final_state)`` where output is ``[H, B, T, V]`` and
      ``final_state`` is ``[B, N, H, K, V]`` (segment dim always present, as in
      tokamax) or ``None`` unless ``output_final_state`` is set.

    Note:
      ``chunk_kda_fwd_custom`` expects inputs already canonicalized by
      tokamax's TPU adapter (padding to chunk multiples, optional q/k L2 norm,
      varlen metadata, resolved ``scale``). This function reproduces that
      adapter: the default-``scale`` resolution from ``base.KimiDeltaAttention``
      and ``check_inputs_support`` / ``_preprocess_inputs`` from
      ``PallasTpuKimiDeltaAttention``. Calling the kernel directly with raw
      arguments fails (e.g. ``scale=None`` gives "unsupported operand type(s)
      for *: 'NoneType' and ...").
    """
    use_qk_l2norm_in_kernel = kwargs.pop("use_qk_l2norm_in_kernel", False)
    N_max = kwargs.pop("N_max", None)
    disable_recompute = kwargs.pop("disable_recompute", True)

    # base.KimiDeltaAttention.bind resolves the default scale from the key dim.
    if scale is None:
        scale = q.shape[-1]**-0.5

    # Cross-argument shape agreement. jaxtyping used to enforce this via shared
    # dimension names ("H B T K" across q/k/g), but the annotations are now plain
    # comments per the repo convention, so check it explicitly here --
    # check_inputs_support only validates q and v.
    if not (q.shape == k.shape == g.shape):
        raise ValueError(
            f"q, k and g must share [H, B, T, K]; got q={q.shape} "
            f"k={k.shape} g={g.shape}")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError(
            f"v must share [H, B, T] with q; got v={v.shape} q={q.shape}")
    if beta.shape != q.shape[:3]:
        raise ValueError(
            f"beta must be [H, B, T] = {q.shape[:3]}; got {beta.shape}")

    # The kernel's intra-chunk decay relies on the activated gate being <= 0, so
    # that the cumulative gate decreases along the chunk and every causal
    # exponent g_r - g_t is non-positive (see the Aqk/L section of
    # pallas_tpu_fwd.py). Both activations satisfy this only for a negative
    # bound: -exp(A_log) * softplus(.) is always <= 0, while
    # lower_bound * sigmoid(.) needs lower_bound < 0. A positive bound would
    # make the decay grow along the chunk and silently return wrong values.
    if lower_bound is not None and lower_bound >= 0:
        raise ValueError(
            "lower_bound must be negative (it bounds a log-decay); got "
            f"{lower_bound}. A non-negative bound makes the gate increase "
            "along the chunk, which this kernel does not support.")

    check_inputs_support(
        q,
        v,
        initial_state=initial_state,
        output_final_state=output_final_state,
        segment_ids=segment_ids,
        chunk_size=chunk_size,
        N_max=N_max,
    )

    prepared = _preprocess_inputs(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        chunk_size=chunk_size,
        N_max=N_max,
    )

    (value, final_state), _residuals = chunk_kda_fwd_custom(
        prepared.q,
        prepared.k,
        prepared.v,
        prepared.g,
        prepared.beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=prepared.initial_state,
        output_final_state=output_final_state,
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        disable_recompute=disable_recompute,
        chunk_size=chunk_size,
        return_residuals=False,
        cu_seqlens=prepared.cu_seqlens,
        aligned_cu_seqlens=prepared.aligned_cu_seqlens,
        chunk_indices=prepared.chunk_indices,
        aligned_segment_ids=prepared.aligned_segment_ids,
        q_rstd=prepared.q_rstd,
        k_rstd=prepared.k_rstd,
        cp_metadata=prepared.cp_metadata,
        **kwargs,
    )
    # Match tokamax: always expose the segment dim on the returned state.
    if final_state is not None and final_state.ndim == 4:
        final_state = final_state[:, None]
    return value, final_state
