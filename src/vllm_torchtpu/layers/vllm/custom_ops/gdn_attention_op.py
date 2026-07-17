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

import dataclasses
import functools

import jax
import torch
from einops import rearrange
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import \
    QwenGatedDeltaNetAttention
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.pcp import (get_pcp_group, get_pcp_rank,
                                           get_pcp_world_size)
from vllm_torchtpu.layers.common.gdn_attention import (
    GdnAttentionConfig, run_jax_gdn_attention,
    run_jax_gdn_attention_pcp_local_prefill,
    run_jax_gdn_attention_pcp_tp_prefill)
from vllm_torchtpu.layers.common.ragged_gated_delta_rule_wrapper import \
    RaggedGatedDeltaRuleImpl
from vllm_torchtpu.layers.common.sequence_layout import \
    is_pcp_streaming_attention_metadata
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from vllm_torchtpu.utils import get_dp_size


def _normalize_dim(dim: int, ndim: int) -> int:
    if not -ndim <= dim < ndim:
        raise RuntimeError(f"axis={dim} is invalid for a {ndim}D tensor")
    return dim + ndim if dim < 0 else dim


def _validate_pcp_rank(pcp_rank: int, pcp_size: int) -> None:
    if pcp_size <= 0:
        raise RuntimeError(f"pcp_size must be positive, got {pcp_size}")
    if not 0 <= pcp_rank < pcp_size:
        raise RuntimeError(f"pcp_rank={pcp_rank} is outside [0, {pcp_size})")


def _pcp_slice_dim_torch(tensor: torch.Tensor, pcp_rank: torch.Tensor,
                         pcp_size: int, *, axis: int) -> torch.Tensor:
    """Select this rank's shard with the rank supplied as a tensor.

    The rank stays runtime data, so the traced program is identical on
    every PCP rank.
    """
    axis = _normalize_dim(axis, tensor.dim())
    if tensor.shape[axis] % pcp_size != 0:
        raise RuntimeError(
            "PCP shard axis is not divisible by pcp_size: "
            f"shape={tuple(tensor.shape)} axis={axis} pcp_size={pcp_size}")
    shard_size = tensor.shape[axis] // pcp_size
    index = (pcp_rank * shard_size +
             torch.arange(shard_size, device=tensor.device))
    return tensor.index_select(axis, index)


def _pcp_slice_reordered_concatenated_torch(
    tensor: torch.Tensor,
    split_sizes: list[int],
    pcp_rank: torch.Tensor,
    pcp_size: int,
    *,
    axis: int,
) -> torch.Tensor:
    """Select this rank's shard of each split with the rank as a tensor.

    The rank stays runtime data, so the traced program is identical on
    every PCP rank.
    """
    axis = _normalize_dim(axis, tensor.dim())
    if sum(split_sizes) != tensor.shape[axis]:
        raise RuntimeError(
            "PCP concatenated splits do not cover the shard axis: "
            f"shape={tuple(tensor.shape)} axis={axis} "
            f"split_sizes={split_sizes}")
    parts = []
    offset = 0
    for split_size in split_sizes:
        if split_size % pcp_size != 0:
            raise RuntimeError("PCP split is not divisible by pcp_size: "
                               f"split_size={split_size} pcp_size={pcp_size}")
        shard_size = split_size // pcp_size
        index = (offset + pcp_rank * shard_size +
                 torch.arange(shard_size, device=tensor.device))
        parts.append(tensor.index_select(axis, index))
        offset += split_size
    return torch.cat(parts, dim=axis).contiguous()


def _pcp_reorder_concatenated_torch(
    tensor: torch.Tensor,
    split_sizes: list[int],
    pcp_size: int,
    *,
    axis: int,
) -> torch.Tensor:
    """Group concatenated Q/K/V slices by destination PCP rank."""
    axis = _normalize_dim(axis, tensor.dim())
    if sum(split_sizes) != tensor.shape[axis]:
        raise RuntimeError(
            "PCP concatenated splits do not cover the reorder axis: "
            f"shape={tuple(tensor.shape)} axis={axis} "
            f"split_sizes={split_sizes}")
    split_tensors = torch.split(tensor, split_sizes, dim=axis)
    rank_parts: list[torch.Tensor] = []
    for pcp_rank in range(pcp_size):
        for split_tensor, split_size in zip(split_tensors, split_sizes):
            if split_size % pcp_size != 0:
                raise RuntimeError(
                    "PCP split is not divisible by pcp_size: "
                    f"split_size={split_size} pcp_size={pcp_size}")
            shard_size = split_size // pcp_size
            rank_parts.append(
                split_tensor.narrow(axis, pcp_rank * shard_size, shard_size))
    return torch.cat(rank_parts, dim=axis).contiguous()


def _pcp_all_to_all_single_torch(send: torch.Tensor) -> torch.Tensor:
    """Run an equal-split Torch all-to-all over the native PCP group."""
    group = get_pcp_group()
    if group is None or int(group.world_size) == 1:
        return send
    if not send.is_contiguous():
        send = send.contiguous()
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(
        recv.view(-1),
        send.view(-1),
        group=group.device_group,
        async_op=False,
    )
    return recv


def _pcp_exchange_token_shards_for_head_shards_torch(
    tensor: torch.Tensor,
    pcp_size: int,
) -> torch.Tensor:
    """Exchange local-token/all-head input for all-token/local-head input."""
    if tensor.dim() != 2:
        raise RuntimeError(
            "Torch PCP token/head exchange expects a 2D tensor, got "
            f"shape={tuple(tensor.shape)}")
    local_tokens, width = tensor.shape
    if width % pcp_size != 0:
        raise RuntimeError("PCP head width is not divisible by pcp_size: "
                           f"width={width} pcp_size={pcp_size}")
    width_per_rank = width // pcp_size
    send = tensor.reshape(local_tokens, pcp_size, width_per_rank).permute(
        1, 0, 2).contiguous()
    recv = _pcp_all_to_all_single_torch(send)
    return recv.reshape(pcp_size * local_tokens, width_per_rank)


def _pcp_exchange_head_shards_for_token_shards_torch(
    tensor: torch.Tensor,
    pcp_size: int,
) -> torch.Tensor:
    """Inverse exchange restoring local-token/full-head output."""
    if tensor.dim() != 2:
        raise RuntimeError(
            "Torch PCP head/token exchange expects a 2D tensor, got "
            f"shape={tuple(tensor.shape)}")
    packed_tokens, width_per_rank = tensor.shape
    if packed_tokens % pcp_size != 0:
        raise RuntimeError(
            "Packed PCP token count is not divisible by pcp_size: "
            f"tokens={packed_tokens} pcp_size={pcp_size}")
    local_tokens = packed_tokens // pcp_size
    send = tensor.reshape(pcp_size, local_tokens, width_per_rank).contiguous()
    recv = _pcp_all_to_all_single_torch(send)
    return recv.permute(1, 0, 2).reshape(local_tokens,
                                         pcp_size * width_per_rank)


def gdn_attention_core_tpu(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    A_log: jax.Array,
    dt_bias: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    config: GdnAttentionConfig,
    dp_enabled: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    (new_conv_state, new_recurrent_state), output = run_jax_gdn_attention(
        mixed_qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        mesh=mesh,
        dp_enabled=dp_enabled,
        config=config,
    )

    return new_conv_state, new_recurrent_state, output


def gdn_attention_core_tpu_pcp_prefill(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    A_log: jax.Array,
    dt_bias: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pcp_size: int,
    interleave_size: int,
    config: GdnAttentionConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    (new_conv_state,
     new_recurrent_state), output = run_jax_gdn_attention_pcp_tp_prefill(
         mixed_qkv,
         b,
         a,
         conv_state,
         recurrent_state,
         conv_weight,
         conv_bias,
         A_log,
         dt_bias,
         state_indices,
         query_start_loc,
         distribution,
         seq_lens,
         n_kq=n_kq,
         n_v=n_v,
         d_k=d_k,
         d_v=d_v,
         kernel_size=kernel_size,
         pcp_size=pcp_size,
         interleave_size=interleave_size,
         mesh=mesh,
         config=config,
     )
    return new_conv_state, new_recurrent_state, output


def gdn_attention_core_tpu_pcp_torch_a2a_prefill(
    packed_mixed_qkv: jax.Array,
    packed_b: jax.Array,
    packed_a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    A_log: jax.Array,
    dt_bias: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pcp_size: int,
    interleave_size: int,
    config: GdnAttentionConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Worker-local GDN entry used between two Torch PCP all-to-alls."""
    (new_conv_state,
     new_recurrent_state), packed_output = (
         run_jax_gdn_attention_pcp_local_prefill(
             packed_mixed_qkv,
             packed_b,
             packed_a,
             conv_state,
             recurrent_state,
             conv_weight,
             conv_bias,
             A_log,
             dt_bias,
             state_indices,
             query_start_loc,
             distribution,
             seq_lens,
             n_kq=n_kq,
             n_v=n_v,
             d_k=d_k,
             d_v=d_v,
             kernel_size=kernel_size,
             pcp_size=pcp_size,
             interleave_size=interleave_size,
             config=config,
         ))
    return new_conv_state, new_recurrent_state, packed_output


def _get_pcp_size(vllm_config: VllmConfig) -> int:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
    return pcp_size if isinstance(pcp_size, int) and pcp_size > 1 else 1


def _gdn_state_donate_argnums(vllm_config: VllmConfig,
                              op_name: str) -> tuple[int, int] | None:
    """Donate GDN state only while the op stays inside a compiled graph.

    vLLM executes an op named in ``splitting_ops`` between the compiled graph
    pieces. Donating state in that mode invalidates the Torch buffers that
    ``gdn_impl`` refills immediately after the custom-op call. Full-graph
    compilation rewires the aliases, so keep the existing donation there.
    """
    compilation_config = getattr(vllm_config, "compilation_config", None)
    splitting_ops = getattr(compilation_config, "splitting_ops", None) or ()
    is_split = (op_name in splitting_ops
                or f"{op_name}.default" in splitting_ops)
    return None if is_split else (3, 4)


def _localize_gdn_mamba_spec_for_pcp(
    spec: MambaSpec,
    pcp_size: int,
) -> MambaSpec:
    if pcp_size <= 1 or len(spec.shapes) != 2:
        return spec

    conv_shape = tuple(spec.shapes[0])
    recurrent_shape = tuple(spec.shapes[1])
    if not conv_shape or not recurrent_shape:
        return spec
    if conv_shape[-1] % pcp_size != 0:
        raise ValueError("GDN conv state width must be divisible by PCP size: "
                         f"shape={conv_shape}, pcp_size={pcp_size}")
    if recurrent_shape[0] % pcp_size != 0:
        raise ValueError(
            "GDN recurrent state head count must be divisible by PCP size: "
            f"shape={recurrent_shape}, pcp_size={pcp_size}")

    unpadded_page_size = dataclasses.replace(
        spec, page_size_padded=None).page_size_bytes
    page_size_padded = spec.page_size_padded
    if page_size_padded == unpadded_page_size:
        page_size_padded = None

    return dataclasses.replace(
        spec,
        shapes=(
            (*conv_shape[:-1], conv_shape[-1] // pcp_size),
            (recurrent_shape[0] // pcp_size, *recurrent_shape[1:]),
        ),
        page_size_padded=page_size_padded,
    )


@QwenGatedDeltaNetAttention.register_oot
class VllmGatedDeltaNetAttention(QwenGatedDeltaNetAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gdn_op = self._build_gdn_op()
        if self._pcp_streaming_enabled():
            # The PCP rank must reach the compiled graph as runtime data.
            # Read as a Python int inside the traced op, it is baked in as
            # a compile-time constant, so every rank compiles a
            # structurally different program around the in-graph PCP
            # collectives and their collective schedules desynchronize.
            _validate_pcp_rank(get_pcp_rank(), get_pcp_world_size())
            self.register_buffer(
                "_pcp_rank",
                torch.tensor(get_pcp_rank(), dtype=torch.int64),
                persistent=False)
        self.gdn_pcp_op = (self._build_gdn_op(
            pcp_streaming=True) if self._pcp_streaming_enabled() else None)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None:
            return None
        assert isinstance(spec, MambaSpec)
        return _localize_gdn_mamba_spec_for_pcp(spec,
                                                _get_pcp_size(vllm_config))

    @staticmethod
    def _pcp_streaming_enabled() -> bool:
        vllm_context = get_vllm_model_wrapper_context()
        parallel_config = getattr(vllm_context.vllm_config, "parallel_config",
                                  None)
        pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
        return isinstance(pcp_size, int) and pcp_size > 1

    def _build_gdn_op(self, *, pcp_streaming: bool = False):
        ragged_gated_delta_rule_impl = RaggedGatedDeltaRuleImpl(
            envs.RAGGED_GATED_DELTA_RULE_IMPL)
        config = GdnAttentionConfig(
            ragged_gated_delta_rule_impl=ragged_gated_delta_rule_impl)
        local_num_v_heads = self.num_v_heads // self.tp_size
        local_num_kq_heads = self.num_k_heads // self.tp_size
        vllm_context = get_vllm_model_wrapper_context()
        parallel_config = getattr(vllm_context.vllm_config, "parallel_config",
                                  None)
        dp_enabled = (parallel_config is not None
                      and get_dp_size(parallel_config) > 1)
        if pcp_streaming:
            interleave_size = getattr(parallel_config,
                                      "cp_kv_cache_interleave_size", 0)
            if not isinstance(interleave_size, int) or interleave_size <= 0:
                raise ValueError("GDN PCP streaming requires "
                                 "cp_kv_cache_interleave_size > 0.")
            pcp_size = get_pcp_world_size()
            if local_num_kq_heads % pcp_size != 0:
                raise ValueError("local GDN K/Q heads "
                                 f"{local_num_kq_heads} must be divisible by "
                                 f"pcp_size={pcp_size}.")
            if local_num_v_heads % pcp_size != 0:
                raise ValueError(
                    f"local GDN V heads {local_num_v_heads} must be divisible "
                    f"by pcp_size={pcp_size}.")
            op_name = ("pallas::gdn_attention_pcp_"
                       f"{self.prefix.replace('.', '_')}")
            wrapped_fn = functools.partial(
                gdn_attention_core_tpu_pcp_torch_a2a_prefill,
                n_kq=local_num_kq_heads // pcp_size,
                n_v=local_num_v_heads // pcp_size,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                kernel_size=self.conv_kernel_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                config=config,
            )
            donate_argnums = _gdn_state_donate_argnums(
                vllm_context.vllm_config, op_name)
            gdn_jax_op = pallas.jax_op(op_name,
                                       wrapped_fn,
                                       donate_argnums=donate_argnums)

            def _fake_gdn(mixed_qkv, _b, _a, conv_state, recurrent_state,
                          *args, **kwargs):
                num_tokens = mixed_qkv.size(0)
                out_shape = (num_tokens, local_num_v_heads // pcp_size *
                             self.head_v_dim)
                return torch.empty_like(conv_state), torch.empty_like(
                    recurrent_state), torch.empty(out_shape,
                                                  dtype=mixed_qkv.dtype,
                                                  device=mixed_qkv.device)

            gdn_jax_op.register_fake(_fake_gdn)
        else:
            op_name = f"pallas::gdn_attention_{self.prefix.replace('.', '_')}"
            wrapped_fn = functools.partial(
                gdn_attention_core_tpu,
                mesh=vllm_context.mesh,
                n_kq=local_num_kq_heads,
                n_v=local_num_v_heads,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                kernel_size=self.conv_kernel_size,
                config=config,
                dp_enabled=dp_enabled,
            )
            donate_argnums = _gdn_state_donate_argnums(
                vllm_context.vllm_config, op_name)
            gdn_jax_op = pallas.jax_op(op_name,
                                       wrapped_fn,
                                       donate_argnums=donate_argnums)

            def _fake_gdn(mixed_qkv, _b, _a, conv_state, recurrent_state,
                          *args, **kwargs):
                num_tokens = mixed_qkv.size(0)
                out_shape = (num_tokens, local_num_v_heads, self.head_v_dim)
                return torch.empty_like(conv_state), torch.empty_like(
                    recurrent_state), torch.empty(out_shape,
                                                  dtype=mixed_qkv.dtype,
                                                  device=mixed_qkv.device)

            gdn_jax_op.register_fake(_fake_gdn)

        def gdn_impl(
            mixed_qkv: torch.Tensor,
            b: torch.Tensor,
            a: torch.Tensor,
            conv_state: torch.Tensor,
            recurrent_state: torch.Tensor,
            conv_weight: torch.Tensor,
            conv_bias: torch.Tensor | None,
            A_log: torch.Tensor,
            dt_bias: torch.Tensor,
            state_indices: torch.Tensor,
            query_start_loc: torch.Tensor,
            request_distribution: torch.Tensor,
            seq_lens: torch.Tensor,
        ) -> torch.Tensor:
            if pcp_streaming:
                pcp_rank = self._pcp_rank.to(mixed_qkv.device)
                local_key_dim = local_num_kq_heads * self.head_k_dim
                local_value_dim = local_num_v_heads * self.head_v_dim
                split_sizes = [local_key_dim, local_key_dim, local_value_dim]
                # The KV cache spec is already physically PCP-local. Only
                # model weights still need to be sliced at the torch boundary.
                conv_state_arg = conv_state
                recurrent_state_arg = recurrent_state
                conv_weight_arg = _pcp_slice_reordered_concatenated_torch(
                    conv_weight,
                    split_sizes,
                    pcp_rank,
                    pcp_size,
                    axis=0,
                )
                conv_bias_arg = (None if conv_bias is None else
                                 _pcp_slice_reordered_concatenated_torch(
                                     conv_bias,
                                     split_sizes,
                                     pcp_rank,
                                     pcp_size,
                                     axis=0,
                                 ))
                A_log_arg = _pcp_slice_dim_torch(
                    A_log,
                    pcp_rank,
                    pcp_size,
                    axis=0,
                ).contiguous()
                dt_bias_arg = _pcp_slice_dim_torch(
                    dt_bias,
                    pcp_rank,
                    pcp_size,
                    axis=0,
                ).contiguous()
                mixed_qkv_arg = _pcp_reorder_concatenated_torch(
                    mixed_qkv,
                    split_sizes,
                    pcp_size,
                    axis=-1,
                )
                mixed_qkv_arg = (
                    _pcp_exchange_token_shards_for_head_shards_torch(
                        mixed_qkv_arg, pcp_size))
                b_arg = _pcp_exchange_token_shards_for_head_shards_torch(
                    b, pcp_size)
                a_arg = _pcp_exchange_token_shards_for_head_shards_torch(
                    a, pcp_size)
            else:
                conv_state_arg = conv_state
                recurrent_state_arg = recurrent_state
                conv_weight_arg = conv_weight
                conv_bias_arg = conv_bias
                A_log_arg = A_log
                dt_bias_arg = dt_bias
                mixed_qkv_arg = mixed_qkv
                b_arg = b
                a_arg = a

            new_conv, new_rec, outputs = gdn_jax_op(
                mixed_qkv_arg, b_arg, a_arg, conv_state_arg,
                recurrent_state_arg,
                conv_weight_arg, conv_bias_arg, A_log_arg, dt_bias_arg,
                state_indices, query_start_loc, request_distribution, seq_lens)
            if pcp_streaming:
                outputs = _pcp_exchange_head_shards_for_token_shards_torch(
                    outputs, pcp_size)
                outputs = outputs.reshape(-1,
                                          local_num_v_heads * self.head_v_dim)
            # Both PCP and non-PCP caches are complete from this worker's
            # perspective, so refill the whole physical local state.
            conv_state.copy_(new_conv)
            recurrent_state.copy_(new_rec)

            return outputs

        return gdn_impl

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if hasattr(self, "in_proj_qkv"):
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba)
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"),
                    (query, key, value))
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                # ``z`` crosses the GDN splitting-op boundary while still
                # being a tail view of ``mixed_qkvz``.  Materialize it so the
                # next compiled piece does not have to reconstruct a
                # non-zero-offset, non-contiguous view.
                z = z.reshape(z.size(0), -1, self.head_v_dim).clone(
                    memory_format=torch.contiguous_format)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        kv_cache = getattr(self, "kv_cache", None)
        local_num_v_heads = self.num_v_heads // self.tp_size

        # During warmup or memory profiling, the kv_cache might not be allocated yet
        if kv_cache is None or kv_cache[0].numel() == 0:
            core_attn_out = torch.zeros(
                (num_tokens, local_num_v_heads, self.head_v_dim),
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )
            conv_state, recurrent_state = None, None
        else:
            fc = get_forward_context()
            attn_metadata = fc.attn_metadata[self.prefix]

            conv_state, recurrent_state = kv_cache
            # Recurrent-state slot id per persistent-batch position.
            # Compact-mamba: the mamba pool has only `_mamba_num_blocks` slots
            # (< attention `num_blocks`), so the slot id is carried explicitly
            # in `mamba_state_indices` (in [0, _mamba_num_blocks)) rather than
            # derived from the attention `block_tables[:, 0]`. Fall back to
            # `block_tables[:, 0]` only when compact sizing was skipped and
            # mamba shares the attention block pool (uniform layout).
            if attn_metadata.mamba_state_indices is not None:
                state_indices = attn_metadata.mamba_state_indices.to(
                    torch.int32)
            else:
                max_reqs = attn_metadata.seq_lens.shape[0]
                max_blocks_per_req = (attn_metadata.block_tables.shape[0] //
                                      max_reqs)
                block_tables_2d = torch.reshape(
                    attn_metadata.block_tables,
                    (max_reqs, max_blocks_per_req),
                )
                state_indices = block_tables_2d[:, 0].to(torch.int32)

            # Execute the TorchTPU custom op
            use_pcp_streaming = is_pcp_streaming_attention_metadata(
                attn_metadata)
            if not use_pcp_streaming:
                core_attn_out = self.gdn_op(
                    mixed_qkv, b, a, conv_state, recurrent_state,
                    self.conv1d.weight, self.conv1d.bias, self.A_log,
                    self.dt_bias, state_indices, attn_metadata.query_start_loc,
                    attn_metadata.request_distribution, attn_metadata.seq_lens)
            else:
                gdn_pcp_op = getattr(self, "gdn_pcp_op", None)
                if gdn_pcp_op is None:
                    raise RuntimeError(
                        "GDN PCP prefill op was not initialized during model "
                        "loading.")
                core_attn_out = gdn_pcp_op(
                    mixed_qkv, b, a, conv_state, recurrent_state,
                    self.conv1d.weight, self.conv1d.bias, self.A_log,
                    self.dt_bias, state_indices, attn_metadata.query_start_loc,
                    attn_metadata.request_distribution, attn_metadata.seq_lens)
            if core_attn_out.shape[0] != num_tokens:
                raise RuntimeError("GDN op returned an incompatible output "
                                   f"shape: {tuple(core_attn_out.shape)}.")

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        core_attn_out = core_attn_out.view(num_tokens, local_num_v_heads,
                                           self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)
