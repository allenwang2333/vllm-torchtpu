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

"""TPU Custom Op implementation for Kimi Delta Attention (KDA)."""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from einops import rearrange

from vllm.config import VllmConfig
from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


# NOTE: Currently @KimiK3DeltaAttention.register_oot and inheriting from
# upstream KimiK3DeltaAttention are used for Out-Of-Tree layer substitution
# when running against upstream vLLM models.
#
# Once clean model code in `vllm-torchtpu` is available:
# 1. Remove the `@KimiK3DeltaAttention.register_oot` decorator and import.
# 2. Change base class to `nn.Module` and initialize sub-layers in `__init__`.
# 3. The clean model in `vllm_torchtpu/models/kimi_k3.py` will import directly.
@KimiK3DeltaAttention.register_oot
class VllmKimiK3DeltaAttention(KimiK3DeltaAttention):
    """TPU Out-Of-Tree implementation of Kimi-K3 Delta Attention (KDA).

    Integrates KDA layers into vLLM's TPU execution engine using MambaSpec
    for stateful KV cache allocation (conv_state + recurrent_state).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Bound by TPU runner during KV cache initialization
        self.kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Returns the dual-state MambaSpec layout for TPU memory allocation.

        conv_state shape: (3 * local_projection_size, conv_width - 1)
        recurrent_state shape: (local_num_heads, head_dim, head_dim)
        """
        conv_width = self.conv1d.weight.size(2)
        conv_state_shape = (
            3 * self.local_projection_size,
            conv_width - 1,
        )
        recurrent_state_shape = (
            self.local_num_heads,
            self.head_dim,
            self.head_dim,
        )
        cache_config = getattr(vllm_config, "cache_config", None)
        block_size = (
            cache_config.block_size
            if vllm_config and cache_config is not None
            else 16
        )
        return MambaSpec(
            block_size=block_size,
            shapes=(conv_state_shape, recurrent_state_shape),
            dtypes=(self.conv1d.weight.dtype, self.conv1d.weight.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,  # Required by vLLM layer interface
    ) -> torch.Tensor:
        """Forward pass executing bring-up PyTorch KDA state recurrence."""
        num_tokens = hidden_states.size(0)
        projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
        split_sizes = [
            3 * self.local_projection_size,
            self.local_projection_size,
            self.head_dim,
            self.local_num_heads,
        ]
        if self.in_proj_padding:
            split_sizes.append(self.in_proj_padding)

        projected = projected_qkvgfab.split(split_sizes, dim=-1)
        mixed_qkv, g_proj_states, f_a, beta = projected[:4]

        g1 = self.f_b_proj(f_a)[0]
        beta = beta.unsqueeze(0)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        self._forward_tpu(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            core_attn_out=core_attn_out,
        )

        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        return self.o_proj(core_attn_out)[0]

    def _forward_tpu(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        """Pure PyTorch/JAX bring-up implementation of KDA recurrence."""
        if self.kv_cache is None:
            # Fallback if kv_cache uninitialized during mock/eager testing
            core_attn_out.zero_()
            return

        conv_state, recurrent_state = self.kv_cache

        # 1. Update Conv State & Causal Conv1D
        groups = (
            self.conv1d.groups
            if hasattr(self.conv1d, "groups")
            else self.conv1d.weight.shape[0]
        )
        if conv_state.ndim == 3:
            # conv_state: (batch, 3 * local_proj_size, conv_width - 1)
            qkv_3d = mixed_qkv.t().unsqueeze(0)
            conv_state_slice = conv_state[0:1]
            conv_input = torch.cat([conv_state_slice, qkv_3d], dim=-1)
            conv_out = F.silu(
                F.conv1d(
                    conv_input,
                    self.conv1d.weight,
                    bias=self.conv1d.bias,
                    groups=groups,
                )
            )[0]
            conv_state[0:1].copy_(conv_input[:, :, -conv_state.size(-1):])
        else:
            qkv_2d = mixed_qkv.t()
            conv_input = torch.cat([conv_state, qkv_2d], dim=-1)
            conv_out = F.silu(
                F.conv1d(
                    conv_input.unsqueeze(0),
                    self.conv1d.weight,
                    bias=self.conv1d.bias,
                    groups=groups,
                )
            ).squeeze(0)
            conv_state.copy_(conv_input[:, -conv_state.size(-1):])

        q_conv, k_conv, v_conv = conv_out.t().split(
            self.local_projection_size, dim=-1
        )

        q = rearrange(q_conv, "n (h d) -> n h d", d=self.head_dim)
        k = rearrange(k_conv, "n (h d) -> n h d", d=self.head_dim)
        v = rearrange(v_conv, "n (h d) -> n h d", d=self.head_dim)

        # 2. Recurrent KDA Loop
        dt_bias = self.dt_bias.view(1, self.local_num_heads, -1)
        A_log = self.A_log.view(1, self.local_num_heads, -1)

        num_tokens = q.size(0)
        out_tokens = []

        rec_state = (
            recurrent_state[0]
            if recurrent_state.ndim == 4
            else recurrent_state
        )

        for t in range(num_tokens):
            qt = q[t:t + 1]  # (1, h, d)
            kt = k[t:t + 1]  # (1, h, d)
            vt = v[t:t + 1]  # (1, h, d)
            g1t = g1[:, t:t + 1]  # (1, 1, h, d) or (1, h, d)
            betat = torch.sigmoid(beta[..., t, :]).squeeze(0)  # (h,)

            # Compute gate decay
            gate = torch.sigmoid(A_log * dt_bias + g1t.squeeze(0))

            # Update state S_t = S_{t-1} * gate + beta * (V - S_{t-1} K) K^T
            decayed_state = rec_state * gate.squeeze(0).unsqueeze(-1)
            raw_delta_v = vt.squeeze(0) - torch.bmm(
                decayed_state, kt.squeeze(0).unsqueeze(-1)
            ).squeeze(-1)
            delta_v = betat.unsqueeze(-1) * raw_delta_v

            delta_state = torch.bmm(
                delta_v.unsqueeze(-1), kt.squeeze(0).unsqueeze(1)
            )
            rec_state.copy_(decayed_state + delta_state)

            yt = torch.bmm(
                rec_state, qt.squeeze(0).unsqueeze(-1)
            ).squeeze(-1)
            out_tokens.append(yt)

        output_tensor = torch.stack(out_tokens, dim=0)  # (num_tokens, h, d)

        # 3. Apply Head-wise RMSNorm and Output Gating g2
        variance = output_tensor.pow(2).mean(-1, keepdim=True)
        output_tensor = output_tensor * torch.rsqrt(variance + 1e-6)

        has_o_norm = hasattr(self, "o_norm")
        norm_weight = (
            getattr(self.o_norm, "weight", None) if has_o_norm else None
        )
        if norm_weight is not None:
            output_tensor = output_tensor * norm_weight

        if g2 is not None:
            output_tensor = output_tensor * torch.sigmoid(g2)

        core_attn_out[0].copy_(output_tensor)
