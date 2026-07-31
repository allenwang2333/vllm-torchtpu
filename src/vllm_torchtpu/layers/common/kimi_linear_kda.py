# SPDX-License-Identifier: Apache-2.0
"""JAX core for Kimi-Linear KDA attention on TPU.

Structured after ``layers/common/gdn_attention.py``: one JAX function that takes
the TPU platform's attention metadata verbatim -- ``query_start_loc``,
``mamba_state_indices``, ``request_distribution``, ``seq_lens`` -- and does conv1d
plus KDA end to end, so nothing branches on the host and nothing needs a
device->host sync.

That contract is why this file exists at all. An earlier version of the layer
overrode upstream's ``_forward`` and reused its control flow, which reads
``num_prefills`` / ``has_initial_state`` off vLLM's ``GDNAttentionMetadata``. On
this platform the metadata is ``layers/common/attention_metadata.py``'s
``AttentionMetadata``, whose fields are **jax.Array** and which describes the
batch with a ``distribution`` triple instead of prefill/decode counts. The layer
died with ``AttributeError: 'AttentionMetadata' object has no attribute
'num_prefills'`` in the first engine forward.

``distribution`` is ``(decode_end, prefill_end, mixed_end)``: sequences
``[:decode_end]`` are decodes, ``[decode_end:prefill_end]`` are prefills. Both
``ragged_causal_conv1d`` and the chunked KDA kernel consume the whole padded batch
in one call and handle the segments internally, so the split never surfaces here.

Known gap: KDA runs the **chunked** kernel over every sequence, including
1-token decode rows, which pad to ``chunk_size=64`` and so cost ~64x their useful
work. It is correct, and it is what makes a single graph-friendly call possible.
Routing ``[:decode_end]`` through ``kernels.kda.kda_decode`` (13.9x faster at 128
sequences) is the obvious follow-up and needs the decode rows split out of the
packed layout first.
"""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.causal_conv1d.causal_conv1d import \
    ragged_causal_conv1d
from vllm_torchtpu.kernels.kda import kda_forward
from vllm_torchtpu.utils import get_mesh_shape_product

__all__ = ["kimi_linear_kda_core"]

# vLLM's KDA recurrent cache stores each slot as [num_heads, d_v, d_k], the
# transpose of the [d_k, d_v] the kernels use. Established by validating the
# Kimi-K3 prototype against GPU dumps (rmse_rel 1.41 raw vs 0.0038 transposed).
_VLLM_STATE_IS_VK = True


def _derive_has_initial_state(query_start_loc: jax.Array,
                             seq_lens: jax.Array) -> jax.Array:
    """A sequence carries state in iff it has previously computed context.

    ``seq_lens`` is the total length (computed + scheduled) while
    ``query_start_loc`` gives what is scheduled now, so the difference is the
    prefix already processed. This is the same derivation vLLM does; the TPU
    metadata does not carry ``has_initial_state`` directly.
    """
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    return seq_lens > query_lens


def _segment_ids_from_cu_seqlens(query_start_loc: jax.Array, num_tokens: int,
                                 num_seqs: jax.Array) -> jax.Array:
    """1-indexed ``[1, T]`` segment ids; 0 marks padding. Device-side.

    The chunked kernel wants ``segment_ids`` where the runner supplies
    cu_seqlens. Doing this with ``searchsorted`` keeps it in the graph -- the
    previous host-side version called ``.tolist()``, which forces a sync per layer
    per step and cannot be traced.
    """
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seg = jnp.searchsorted(query_start_loc, tok, side="right").astype(jnp.int32)
    # Tokens past the last real sequence are padding.
    valid = tok < query_start_loc[num_seqs]
    return jnp.where(valid, seg, 0).reshape(1, num_tokens)


def kimi_linear_kda_core(
    q_proj_states: jax.Array,  # [num_tokens, proj]
    *args,
    mesh: jax.sharding.Mesh,
    num_heads: int,
    head_dim: int,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Conv1d + KDA for one Kimi-Linear layer, bound to the worker's mesh.

    The body runs under ``jax.shard_map`` on the mesh the runner built for this
    worker, mirroring every ``gdn_attention.py`` path, so the computation is
    placed on the worker's assigned device rather than JAX's default one. (Each
    worker's mesh is single-device, so this has not been shown to change
    behaviour on its own -- it is here to match GDN, not as a fix for a
    diagnosed bug.)

    Head-parallel axes are sharded on the mesh's ``"model"`` axis. Every
    current TPU config gives each worker a single-device mesh (vLLM does TP
    across processes, so each worker already owns its head shard) making these
    specs a no-op, but they keep the op correct if a multi-device mesh is ever
    used -- and a replicated spec would silently compute the wrong thing there.
    """
    tp_size = get_mesh_shape_product(mesh, "model")

    in_specs = (
        P(None, "model"),  # q_proj_states
        P(None, "model"),  # k_proj_states
        P(None, "model"),  # v_proj_states
        P(None, "model"),  # g1
        P(None, "model"),  # beta
        P(None, None, "model"),  # conv_state
        P(None, "model", None, None),  # recurrent_state
        P("model", None, None),  # q_conv_weight
        P("model", None, None),  # k_conv_weight
        P("model", None, None),  # v_conv_weight
        P("model"),  # A_log
        P("model"),  # dt_bias
        P(None),  # state_indices
        P(None),  # query_start_loc
        P(None),  # distribution
        P(None),  # seq_lens
    )
    out_specs = (
        P(None, None, "model"),  # new_conv_state
        P(None, "model", None, None),  # new_recurrent_state
        P(None, "model", None),  # out
    )

    body = functools.partial(
        _kimi_linear_kda_local,
        num_heads=num_heads // tp_size,
        head_dim=head_dim,
        kernel_size=kernel_size,
    )
    mapped = jax.shard_map(body,
                           mesh=mesh,
                           in_specs=in_specs,
                           out_specs=out_specs,
                           check_vma=False)
    return mapped(q_proj_states, *args)


def _kimi_linear_kda_local(
    q_proj_states: jax.Array,  # [num_tokens, proj]
    k_proj_states: jax.Array,  # [num_tokens, proj]
    v_proj_states: jax.Array,  # [num_tokens, proj]
    g1: jax.Array,  # [num_tokens, proj]  raw gate
    beta: jax.Array,  # [num_tokens, num_heads]  already activated
    conv_state: jax.Array,  # [max_num_seqs, kernel - 1, 3 * proj]
    recurrent_state: jax.Array,  # [num_slots, num_heads, d_v, d_k]
    q_conv_weight: jax.Array,  # [proj, 1, kernel]
    k_conv_weight: jax.Array,
    v_conv_weight: jax.Array,
    A_log: jax.Array,  # [num_heads]
    dt_bias: jax.Array,  # [proj]
    state_indices: jax.Array,  # [max_num_seqs]
    query_start_loc: jax.Array,  # [max_num_seqs + 1]
    distribution: jax.Array,  # [3] = (decode_end, prefill_end, mixed_end)
    seq_lens: jax.Array,  # [max_num_seqs]
    *,
    num_heads: int,
    head_dim: int,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Conv1d + KDA for one Kimi-Linear layer.

    Returns ``(new_conv_state, new_recurrent_state, out)`` where ``out`` is
    ``[num_tokens, num_heads, head_dim]``. Both states come back whole so the
    caller can write them into the donated cache buffers.
    """
    num_tokens = q_proj_states.shape[0]
    n_max = state_indices.shape[0]
    has_initial_state = _derive_has_initial_state(query_start_loc, seq_lens)

    # --- conv1d ------------------------------------------------------------
    # The cache packs q|k|v along the channel axis; split, convolve each, and
    # reassemble so the caller sees one buffer.
    proj = q_proj_states.shape[-1]
    cs_q, cs_k, cs_v = (conv_state[..., 0:proj], conv_state[..., proj:2 * proj],
                        conv_state[..., 2 * proj:3 * proj])

    def conv(x, cs, w):
        """Ragged causal conv1d followed by SiLU, as the reference does.

        Upstream calls ``causal_conv1d_fn(..., activation="silu")`` for all three
        of q, k and v (see ``kimi_gdn_linear_attn.py``), and the GDN wrapper does
        the same with ``jax.nn.silu`` after its fused conv. Omitting it is silent:
        magnitudes stay comparable so the model still produces fluent text, but
        every q/k/v is systematically wrong. It cost a gsm8k score of 0.16 while
        the kernel tests -- which never touch conv1d -- stayed green.

        The activation applies to the conv *output* only. ``new_conv_state`` is a
        rolling window of raw input tokens and must stay pre-activation, which is
        also what ``causal_conv1d_fn`` does with its ``conv_states`` argument.
        """
        out, new_cs = ragged_causal_conv1d(x, cs, w, None, query_start_loc,
                                           state_indices, distribution,
                                           has_initial_state,
                                           kernel_size=kernel_size)
        return jax.nn.silu(out), new_cs

    q, new_cs_q = conv(q_proj_states, cs_q, q_conv_weight)
    k, new_cs_k = conv(k_proj_states, cs_k, k_conv_weight)
    v, new_cs_v = conv(v_proj_states, cs_v, v_conv_weight)
    new_conv_state = jnp.concatenate([new_cs_q, new_cs_k, new_cs_v], axis=-1)

    # --- KDA ---------------------------------------------------------------
    # [T, H*d] -> head-major [H, 1, T, d] for the kernel.
    hm = lambda x: jnp.transpose(
        x.reshape(num_tokens, num_heads, head_dim), (1, 0, 2))[:, None]

    segment_ids = _segment_ids_from_cu_seqlens(query_start_loc, num_tokens,
                                               distribution[2])

    init = jnp.take(recurrent_state, state_indices, axis=0)  # [N, H, V, K]
    if _VLLM_STATE_IS_VK:
        init = jnp.swapaxes(init, -1, -2)  # -> [N, H, K, V]
    init = jnp.where(has_initial_state[:, None, None, None], init, 0.0)

    out, final_state = kda_forward(
        hm(q),
        hm(k),
        hm(v),
        hm(g1),
        # beta is [T, H] -> [H, 1, T]
        jnp.transpose(beta, (1, 0))[:, None],
        A_log=A_log,
        dt_bias=dt_bias,
        # Kimi-Linear has no gate_lower_bound, so the gate is the unbounded
        # -exp(A_log) * softplus(g + dt_bias) branch, which the kernel requires
        # be paired with safe_gate=False.
        lower_bound=None,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        safe_gate=False,
        segment_ids=segment_ids,
        N_max=n_max,
        initial_state=init[None],
        output_final_state=True,
    )

    scatter = final_state[0]
    if _VLLM_STATE_IS_VK:
        scatter = jnp.swapaxes(scatter, -1, -2)
    # Only sequences that ran should update the pool. Padded entries point at
    # slot 0, which the runner reserves.
    new_recurrent_state = recurrent_state.at[state_indices].set(
        scatter.astype(recurrent_state.dtype))

    # [H, 1, T, V] -> [T, H, V]
    out = jnp.transpose(out[:, 0], (1, 0, 2))
    return new_conv_state, new_recurrent_state, out
