"""Input canonicalization for the KDA Pallas TPU forward kernel."""

import dataclasses
from typing import Any
import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from vllm_torchtpu.kernels.kda.pallas_tpu_fwd import (
    chunk_kda_fwd_custom,
)
from vllm_torchtpu.kernels.kda.pallas_tpu_types import (
    CpMetadata,
    KdaResiduals,
)
from vllm_torchtpu.kernels.kda.utils import (
    _align_seqs,
    align_segment_ids,
    l2norm_fwd,
    prepare_chunk_indices,
    segment_ids_to_cu_seqlens,
)
from typing_extensions import override

@dataclasses.dataclass(frozen=True)
class _PreparedKdaInputs:
  q: jax.Array
  k: jax.Array
  v: jax.Array
  g: jax.Array
  beta: jax.Array
  initial_state: jax.Array | None
  cu_seqlens: jax.Array | None
  aligned_cu_seqlens: jax.Array | None
  chunk_indices: jax.Array | None
  aligned_segment_ids: jax.Array | None
  q_rstd: jax.Array | None
  k_rstd: jax.Array | None
  cp_metadata: CpMetadata

def check_inputs_support(
    q: jax.Array,
    v: jax.Array,
    *,
    initial_state: jax.Array | None,
    output_final_state: bool,
    segment_ids: jax.Array | None,
    chunk_size: int,
    N_max: int | None,
) -> None:
  """Checks whether the Pallas TPU backend supports the static inputs."""
  if q.dtype not in (jnp.bfloat16, jnp.float32):
    raise NotImplementedError(
        "`pallas_tpu` currently supports bfloat16 and float32 inputs only."
    )
  heads, batch, seq_len, key_dim = q.shape
  value_dim = v.shape[-1]
  if heads < 1 or batch < 1 or seq_len < 1:
    raise NotImplementedError(
        "`pallas_tpu` requires positive head, batch, and sequence "
        f"dimensions; got H={heads}, B={batch}, T={seq_len}."
    )
  if key_dim < 1 or value_dim < 1:
    raise NotImplementedError(
        "`pallas_tpu` requires positive key and value dimensions; got "
        f"K={key_dim}, V={value_dim}."
    )
  if key_dim > 256:
    raise NotImplementedError(
        "`pallas_tpu` currently supports key dimensions up to 256; got "
        f"K={key_dim}."
    )
  if initial_state is not None and segment_ids is None:
    if initial_state.shape[1] != 1:
      raise NotImplementedError(
          "`pallas_tpu` fixed-length execution requires exactly one "
          "recurrent state per batch item; got "
          f"N={initial_state.shape[1]}."
      )
  if chunk_size != 64:
    raise NotImplementedError("`pallas_tpu` currently supports chunk_size=64.")
  if segment_ids is None and seq_len % chunk_size != 0:
    raise NotImplementedError(
        "`pallas_tpu` requires the sequence length to be divisible by "
        f"`chunk_size`; got T={seq_len}, chunk_size={chunk_size}."
    )

def _preprocess_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    *,
    initial_state: jax.Array | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    segment_ids: jax.Array | None,
    chunk_size: int,
    N_max: int | None,
) -> _PreparedKdaInputs:
  """Canonicalizes inputs shared by the forward and backward kernels."""
  # Context parallelism is not supported (Kimi-K3 does not use it), so the
  # varlen metadata always comes from segment_ids.
  cu_seqlens = None
  if cu_seqlens is None:
    cu_seqlens, N_max = segment_ids_to_cu_seqlens(
        segment_ids,
        initial_state=initial_state,
        N_max=N_max,
    )

  aligned_cu_seqlens = None
  chunk_indices = None
  if cu_seqlens is None:
    q_aligned, k_aligned, v_aligned = q, k, v
    g_aligned, beta_aligned = g, beta
  else:
    (
        [q_aligned, k_aligned, v_aligned, g_aligned],
        [beta_aligned],
        aligned_cu_seqlens,
        _,
    ) = _align_seqs(
        [q, k, v, g],
        [beta],
        cu_seqlens,
        align=chunk_size,
    )
    chunk_indices = prepare_chunk_indices(
        aligned_cu_seqlens,
        chunk_size,
        max_T=q_aligned.shape[2],
    )

    if use_gate_in_kernel:
      aligned_seq_len = g_aligned.shape[2]
      original_lengths = jnp.diff(cu_seqlens, axis=-1)
      aligned_starts = aligned_cu_seqlens[..., :-1]
      positions = jnp.arange(aligned_seq_len)
      for batch_index in range(cu_seqlens.shape[0]):
        in_range = (
            positions[None, :]
            >= aligned_starts[batch_index, :, None]
        ) & (
            positions[None, :]
            < (
                aligned_starts[batch_index]
                + original_lengths[batch_index]
            )[:, None]
        )
        valid_mask = in_range.any(axis=0)
        g_aligned = g_aligned.at[:, batch_index].set(
            jnp.where(
                valid_mask[None, :, None],
                g_aligned[:, batch_index],
                -1e4,
            )
        )

  initial_state_prepared = initial_state
  if initial_state is not None:
    if aligned_cu_seqlens is None:
      # Fixed-length execution has one recurrent state per batch item.
      if initial_state.ndim == 5:
        initial_state_prepared = initial_state[:, 0]
    else:
      state_count = aligned_cu_seqlens.shape[-1] - 1
      if initial_state.shape[1] < state_count:
        initial_state_prepared = jnp.pad(
            initial_state,
            (
                (0, 0),
                (0, state_count - initial_state.shape[1]),
                (0, 0),
                (0, 0),
                (0, 0),
            ),
        )
      if initial_state_prepared.shape[1] != state_count:
        raise ValueError(
            "`initial_state` state count must match aligned segment "
            f"count {state_count}; got {initial_state.shape[1]}."
        )

  aligned_segment_ids = None
  if aligned_cu_seqlens is not None and segment_ids is not None:
    effective_n_max = (
        N_max
        if N_max is not None
        else aligned_cu_seqlens.shape[-1] - 1
    )
    aligned_segment_ids = jnp.stack(
        [
            align_segment_ids(
                segment_ids[batch_index], effective_n_max, chunk_size
            )
            for batch_index in range(segment_ids.shape[0])
        ]
    )

  if use_qk_l2norm_in_kernel:
    q_prepared, q_rstd = l2norm_fwd(q_aligned)
    k_prepared, k_rstd = l2norm_fwd(k_aligned)
  else:
    q_prepared, k_prepared = q_aligned, k_aligned
    q_rstd = k_rstd = None

  cp_metadata = None

  return _PreparedKdaInputs(
      q=q_prepared,
      k=k_prepared,
      v=v_aligned,
      g=g_aligned,
      beta=beta_aligned,
      initial_state=initial_state_prepared,
      cu_seqlens=cu_seqlens,
      aligned_cu_seqlens=aligned_cu_seqlens,
      chunk_indices=chunk_indices,
      aligned_segment_ids=aligned_segment_ids,
      q_rstd=q_rstd,
      k_rstd=k_rstd,
      cp_metadata=cp_metadata,
  )
