# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi MLA and KDA layers for TPU."""

from __future__ import annotations

# `jax` must be a *module* global, not a local of _build_kda_op: this file uses
# deferred annotations, so `pallas.jax_op` resolving the wrapper's `jax.Array`
# hints via inspect.signature(eval_str=True) evaluates them against module
# globals and otherwise dies with "NameError: name 'jax' is not defined".
import jax  # noqa: F401
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import (get_forward_context,
                                  is_forward_context_available)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator,
    is_conv_state_dim_first)
from vllm.model_executor.layers.mla import (MLAModules,
                                            MultiHeadLatentAttentionWrapper)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, sharded_weight_loader)
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_torchtpu.layers.vllm.custom_ops.kda_attention_op import (
    build_kimi_kda_op, build_kimi_sconv_op)


def _load_a_log(parameter: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Load either the old ``[1, 1, H, 1]`` or current ``[H]`` layout."""

    if loaded_weight.ndim == 4:
        loaded_weight = loaded_weight.flatten()
    rank = get_tensor_model_parallel_rank()
    shard_size = parameter.shape[0]
    loaded_weight = loaded_weight.narrow(0, rank * shard_size, shard_size)
    default_weight_loader(parameter, loaded_weight)


class MultiHeadLatentAttention(nn.Module):
    """NoPE MLA dispatched through the TPU Pallas MLA custom op."""

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        if not config.mla_use_nope:
            raise ValueError("The TPU Kimi model supports NoPE MLA only")
        required = {
            "kv_lora_rank": config.kv_lora_rank,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "v_head_dim": config.v_head_dim,
        }
        if any(value is None for value in required.values()):
            raise ValueError(f"Incomplete Kimi K3 MLA config: {required}")
        assert config.kv_lora_rank is not None
        assert config.qk_nope_head_dim is not None
        assert config.qk_rope_head_dim is not None
        assert config.v_head_dim is not None

        tp_size = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp_size:
            raise ValueError(
                "num_attention_heads must be divisible by TP size")
        num_heads = config.num_attention_heads // tp_size
        qk_head_dim = int(config.qk_nope_head_dim) + int(
            config.qk_rope_head_dim)
        quant_config = vllm_config.quant_config
        super().__init__()

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = qk_head_dim

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                config.hidden_size,
                config.num_attention_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.fused_qkv_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
        else:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                config.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                config.num_attention_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.q_proj = None
            self.kv_a_proj_with_mqa = None
        if self.q_lora_rank is None:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                config.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            config.num_attention_heads *
            (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.g_proj = (ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.v_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        ) if config.mla_use_output_gate else None)
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=None,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
            g_proj=self.g_proj,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            config.hidden_size,
            num_heads,
            qk_head_dim**-0.5,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            vllm_config.cache_config,
            quant_config,
            prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)


class CausalDepthwiseConv1d(nn.Module):
    """Checkpoint-compatible depthwise-convolution weight container."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if channels % tp_size:
            raise ValueError(
                "Convolution channels must be divisible by TP size")
        self.channels = channels // tp_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(self.channels, 1, kernel_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(
        self,
        parameter: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        if loaded_weight.ndim == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        rank = get_tensor_model_parallel_rank()
        loaded_weight = loaded_weight.chunk(
            get_tensor_model_parallel_world_size(), dim=0)[rank]
        parameter.data.copy_(loaded_weight)


class KimiDeltaAttention(nn.Module, MambaBase):
    """KDA registered with vLLM and executed by TPU Pallas custom ops."""

    supports_dcp = False

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "TPU KDA does not support speculative decode")
        kda_config = config.linear_attn_config
        if kda_config is None:
            raise ValueError("KDA requires linear_attn_config")

        self.prefix = prefix
        self.vllm_config = vllm_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = int(kda_config["head_dim"])
        self.total_heads = int(kda_config["num_heads"])
        if self.total_heads % self.tp_size:
            raise ValueError("KDA num_heads must be divisible by TP size")
        self.num_heads = self.total_heads // self.tp_size
        self.projection_size = self.total_heads * self.head_dim
        self.local_projection_size = self.num_heads * self.head_dim
        self.conv_size = int(kda_config["short_conv_kernel_size"])
        self.gate_lower_bound = kda_config.get("gate_lower_bound")
        self.use_full_rank_gate = kda_config.get("use_full_rank_gate", False)
        quant_config = vllm_config.quant_config

        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.q_conv1d = CausalDepthwiseConv1d(self.projection_size,
                                              self.conv_size)
        self.k_conv1d = CausalDepthwiseConv1d(self.projection_size,
                                              self.conv_size)
        self.v_conv1d = CausalDepthwiseConv1d(self.projection_size,
                                              self.conv_size)
        self.f_a_proj = ReplicatedLinear(
            config.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_a_proj",
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.b_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
        )
        if self.use_full_rank_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                self.projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        else:
            self.g_a_proj = ReplicatedLinear(
                config.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_a_proj",
            )
            self.g_b_proj = ColumnParallelLinear(
                self.head_dim,
                self.projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_b_proj",
            )

        self.A_log = nn.Parameter(
            torch.empty(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32))
        set_weight_attrs(self.A_log, {"weight_loader": _load_a_log})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})
        self.o_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            self.projection_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.kv_cache = None
        # One fused op (conv1d + KDA) over this repo's validated chunked/decode
        # KDA kernels, replacing PR 220's `build_kimi_sconv_op` +
        # `build_kimi_kda_op` pair. Those wrap a token-at-a-time
        # `jax.lax.fori_loop` recurrence, which is correct but sequential in
        # sequence length -- hopeless for prefill -- and its own unit test misses
        # an independent torch reference by 4.7e-4 against a 2e-5 tolerance.
        # `kimi_linear_kda_core` is the implementation validated end to end on
        # Kimi-Linear (gsm8k 0.93), so K3 and Kimi-Linear now share one kernel
        # path and a K3 discrepancy can be bisected against a working model.
        #
        # Built here, not lazily: `pallas.jax_op` / `register_fake` are Python
        # construction and would otherwise run during Dynamo tracing of the
        # compiled model graph.
        self.kda_op = self._build_kda_op()

        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.LINEAR

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.vllm_config.model_config.dtype,
            self.vllm_config.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.total_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=0,
        )

    def _build_kda_op(self):
        """Wrap ``kimi_linear_kda_core`` as this layer's torch<->jax op."""
        import functools

        from torch_tpu._internal import pallas

        from vllm_torchtpu.layers.common.kimi_linear_kda import \
            kimi_linear_kda_core
        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
            get_vllm_model_wrapper_context

        fn = functools.partial(
            kimi_linear_kda_core,
            # The mesh binds the computation to this worker's device; every GDN
            # op path threads it for the same reason.
            mesh=get_vllm_model_wrapper_context().mesh,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            kernel_size=self.conv_size,
            # K3 sets gate_lower_bound (-5.0): the bounded
            # lower_bound * sigmoid(exp(A_log) * (g + dt_bias)) gate. Kimi-Linear
            # leaves it None for the unbounded softplus form.
            lower_bound=self.gate_lower_bound,
        )

        def wrapped(q_proj_states: jax.Array, k_proj_states: jax.Array,
                    v_proj_states: jax.Array, g1: jax.Array, beta: jax.Array,
                    conv_state: jax.Array, recurrent_state: jax.Array,
                    q_conv_weight: jax.Array, k_conv_weight: jax.Array,
                    v_conv_weight: jax.Array, A_log: jax.Array,
                    dt_bias: jax.Array, state_indices: jax.Array,
                    query_start_loc: jax.Array, distribution: jax.Array,
                    seq_lens: jax.Array
                    ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return fn(q_proj_states, k_proj_states, v_proj_states, g1, beta,
                      conv_state, recurrent_state, q_conv_weight,
                      k_conv_weight, v_conv_weight, A_log, dt_bias,
                      state_indices, query_start_loc, distribution, seq_lens)

        name = f"pallas::kimik3_kda_{self.prefix.replace('.', '_')}"
        # conv_state (5) and recurrent_state (6) are donated and also written
        # back with copy_(): XLA aliases the donated input to the program output.
        # Without donation torch_tpu's compiled mode aborts in
        # TraverseAndCompile with "identified an argument that was not
        # provided", naming those buffers.
        op = pallas.jax_op(name, wrapped, donate_argnums=(5, 6))

        def _fake(q, _k, _v, _g, _beta, conv_state, recurrent_state, *a, **kw):
            out = torch.empty((q.size(0), self.num_heads, self.head_dim),
                              dtype=q.dtype, device=q.device)
            return (torch.empty_like(conv_state),
                    torch.empty_like(recurrent_state), out)

        op.register_fake(_fake)
        return op

    def _gated_norm(self, out: torch.Tensor,
                    output_gate: torch.Tensor) -> torch.Tensor:
        """``RMSNorm(out) * sigmoid(output_gate)``, per head.

        PR 220 folds this into its kernel (it passes ``o_norm.weight`` and
        ``output_gate`` in). Our kernel returns the pre-norm core output, so the
        layer applies it -- the same split the working Kimi-Linear layer uses.
        Formula and float32 accumulation match ``GatedRMSNormSigmoid`` in
        ``models/kimi_k3/kda.py``, which was validated against real GPU dumps
        (rmse_rel 0.006).
        """
        x = out.float()
        x = x * torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.o_norm.variance_epsilon)
        x = x * self.o_norm.weight.float()
        gate = output_gate.view(-1, self.num_heads, self.head_dim).float()
        return (x * torch.sigmoid(gate)).to(out.dtype)

    def _metadata(self) -> LinearAttentionMetadata | None:
        if not is_forward_context_available():
            return None
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        layer_metadata = metadata[self.prefix]
        if not isinstance(layer_metadata, LinearAttentionMetadata):
            raise TypeError("KDA received incompatible attention metadata")
        return layer_metadata

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del positions
        query, _ = self.q_proj(hidden_states)
        key, _ = self.k_proj(hidden_states)
        value, _ = self.v_proj(hidden_states)
        mixed_qkv = torch.cat((query, key, value), dim=-1)
        f_a, _ = self.f_a_proj(hidden_states)
        raw_gate, _ = self.f_b_proj(f_a)
        beta, _ = self.b_proj(hidden_states)
        if self.use_full_rank_gate:
            output_gate, _ = self.g_proj(hidden_states)
        else:
            g_a, _ = self.g_a_proj(hidden_states)
            output_gate, _ = self.g_b_proj(g_a)

        # Metadata comes from the TPU runner (layers/common/attention_metadata.py),
        # whose fields are jax.Arrays and which describes the batch with a
        # `request_distribution` triple. PR 220's `_metadata()` instead requires
        # vLLM's `LinearAttentionMetadata` (host-side num_prefills/num_decodes)
        # and raises TypeError on anything else, so it cannot run on this runner
        # -- the same contract mismatch that broke the first Kimi-Linear attempt.
        # Read it the way the working Kimi-Linear layer does.
        from vllm.forward_context import get_forward_context

        del mixed_qkv  # our core takes q/k/v separately and packs internally

        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            metadata = metadata.get(self.prefix)
        cache_missing = (self.kv_cache is None or len(self.kv_cache) < 2
                         or self.kv_cache[0].numel() == 0)
        if (metadata is None or cache_missing
                or getattr(metadata, "mamba_state_indices", None) is None):
            # Warmup and the memory-profiling run precede KV-cache allocation.
            output = torch.zeros(
                (query.shape[0], self.num_heads * self.head_dim),
                dtype=query.dtype,
                device=query.device)
        else:
            conv_cache, recurrent_cache = self.kv_cache
            # K3's b_proj is raw; the kernels take beta already activated.
            # (Kimi-Linear's own forward sigmoids before calling the core.)
            beta_act = torch.sigmoid(beta.float()).to(beta.dtype)
            new_conv, new_rec, core_out = self.kda_op(
                query,
                key,
                value,
                raw_gate,
                beta_act,
                conv_cache,
                recurrent_cache,
                self.q_conv1d.weight,
                self.k_conv1d.weight,
                self.v_conv1d.weight,
                self.A_log,
                self.dt_bias,
                metadata.mamba_state_indices.to(torch.int32),
                metadata.query_start_loc,
                metadata.request_distribution,
                metadata.seq_lens,
            )
            conv_cache.copy_(new_conv)
            recurrent_cache.copy_(new_rec)
            output = self._gated_norm(core_out, output_gate).flatten(1)

        output, _ = self.o_proj(output)
        return output
