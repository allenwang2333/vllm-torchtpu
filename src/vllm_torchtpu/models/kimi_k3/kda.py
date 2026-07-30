# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools

import jax
import jax.numpy as jnp
import torch
from einops import rearrange
from torch import nn

from vllm_torchtpu.kernels.kda import kda_decode, kda_forward

__all__ = ["KdaMetadata", "KimiK3DeltaAttention", "GatedRMSNormSigmoid"]

_KDA_GATE_LOGBOUND_MIN = -5.0

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class KdaMetadata:
    """The subset of upstream ``KimiK3KDAMetadata`` this prototype consumes.

    Kept as a standalone dataclass so accuracy tests can drive the layer
    directly without standing up a vLLM engine and forward context. In a real
    integration this would be built by the TPU attention-metadata builder.

    Attributes:
        num_prefills / num_decodes: batch composition. A batch is treated as
            prefill if ``num_prefills > 0``; mixed batches are rejected, since
            resolving them needs the token-index split upstream gets from
            ``spec_token_indx`` / ``non_spec_token_indx``.
        query_start_loc: ``[num_seqs + 1]`` cu_seqlens over the token axis.
        state_indices: ``[num_seqs]`` slot per sequence in the state pool.
        has_initial_state: ``[num_seqs]`` bool -- whether the sequence carries
            recurrent/conv state in from a previous step.
        num_actual_tokens: tokens before padding.
    """
    num_prefills: int
    num_decodes: int
    query_start_loc: torch.Tensor
    state_indices: torch.Tensor
    has_initial_state: torch.Tensor
    num_actual_tokens: int
    num_spec_decodes: int = 0


# ---------------------------------------------------------------------------
# Gated RMSNorm
# ---------------------------------------------------------------------------
class GatedRMSNormSigmoid(nn.Module):
    """Plain-torch equivalent of upstream ``FusedRMSNormGated``."""

    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_f * torch.rsqrt(variance + self.eps)
        x_normed = x_normed * self.weight.float()
        g_f = g.float()
        out = x_normed * torch.sigmoid(g_f)
        return out.to(x.dtype)


# ---------------------------------------------------------------------------
# causal conv1d, torch reference
# ---------------------------------------------------------------------------
def _causal_conv1d_varlen(
    x: torch.Tensor,  # [num_tokens, dim]
    conv_state: torch.Tensor,  # [num_slots, state_len, dim]
    weight: torch.Tensor,  # [dim, kernel_size]
    bias: torch.Tensor | None,
    query_start_loc: torch.Tensor,  # [num_seqs + 1]
    state_indices: torch.Tensor,  # [num_seqs]
    has_initial_state: torch.Tensor,  # [num_seqs]
) -> torch.Tensor:
    """Prefill conv1d + SiLU, per sequence, writing back the tail state.

    Reference implementation: loops over sequences on the host. Correct and easy
    to check against upstream, but it materialises one conv per sequence and
    cannot be traced into a single graph --- replace with
    ``kernels.causal_conv1d.ragged_causal_conv1d`` before this is used for
    anything but validation.
    """
    dim, kernel_size = weight.shape
    state_len = conv_state.shape[1]
    out = torch.empty_like(x)
    starts = query_start_loc.tolist()
    slots = state_indices.tolist()
    has_init = has_initial_state.tolist()

    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        if e <= s:
            continue
        seq = x[s:e]  # [t, dim]
        # Left-pad with carried state (or zeros on the first chunk).
        if has_init[i]:
            prev = conv_state[slots[i], -(kernel_size - 1):, :]
        else:
            prev = torch.zeros(kernel_size - 1,
                               dim,
                               dtype=x.dtype,
                               device=x.device)
        padded = torch.cat([prev.to(x.dtype), seq], dim=0)  # [t + w - 1, dim]
        # Depthwise causal conv: out[t] = sum_j padded[t + j] * weight[:, j]
        windows = padded.unfold(0, kernel_size, 1)  # [t, dim, w]
        y = (windows * weight.unsqueeze(0)).sum(-1)
        if bias is not None:
            y = y + bias
        out[s:e] = torch.nn.functional.silu(y).to(out.dtype)

        # Tail becomes the next step's state, right-aligned as vLLM stores it.
        tail = padded[-(state_len):] if padded.shape[0] >= state_len else \
            torch.cat([
                torch.zeros(state_len - padded.shape[0], dim,
                            dtype=padded.dtype, device=padded.device), padded
            ], dim=0)
        conv_state[slots[i]] = tail.to(conv_state.dtype)
    return out


def _causal_conv1d_update(
    x: torch.Tensor,  # [num_seqs, dim]  one token per sequence
    conv_state: torch.Tensor,  # [num_slots, state_len, dim]
    weight: torch.Tensor,  # [dim, kernel_size]
    bias: torch.Tensor | None,
    state_indices: torch.Tensor,  # [num_seqs]
) -> torch.Tensor:
    """Decode conv1d + SiLU: roll the window, apply, store back."""
    _dim, kernel_size = weight.shape
    slots = state_indices.to(torch.long)
    # [num_seqs, w - 1, dim] -> window of the last w tokens including this one.
    prev = conv_state[slots][:, -(kernel_size - 1):, :].to(x.dtype)
    window = torch.cat([prev, x.unsqueeze(1)], dim=1)  # [n, w, dim]
    y = (window.transpose(1, 2) * weight.unsqueeze(0)).sum(-1)  # [n, dim]
    if bias is not None:
        y = y + bias
    # Store the rolled window back into the pool.
    new_state = conv_state[slots].clone()
    new_state[:, :-1, :] = new_state[:, 1:, :].clone()
    new_state[:, -1, :] = x.to(conv_state.dtype)
    conv_state[slots] = new_state
    return torch.nn.functional.silu(y).to(x.dtype)


# ---------------------------------------------------------------------------
# JAX side: the two kernel call sites
# ---------------------------------------------------------------------------
def _kda_prefill_jax(
    q: jax.Array,  # [1, T, H, D]
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,  # raw per-channel gate, [1, T, H, D]
    beta: jax.Array,  # [1, T, H]
    pool: jax.Array,  # [num_slots, H, K, V]
    a_log: jax.Array,  # [H]
    dt_bias: jax.Array,  # [H * K]
    segment_ids: jax.Array,  # [1, T]
    state_indices: jax.Array,  # [num_seqs]
    has_initial_state: jax.Array,  # [num_seqs]
    *,
    lower_bound: float | None,
    n_max: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunked KDA over a packed prefill batch; returns (out, updated pool).

    Layout bridge: the layer works in vLLM's ``[1, T, H, D]`` while the chunked
    kernel wants head-major ``[H, B, T, D]``.
    """
    hm = lambda x: jnp.transpose(x, (2, 0, 1, 3))  # [1,T,H,D] -> [H,1,T,D]

    init = jnp.take(pool, state_indices, axis=0)  # [N, H, V, K]
    init = jnp.swapaxes(init, -1, -2)  # -> [N, H, K, V]
    init = jnp.where(has_initial_state[:, None, None, None], init, 0.0)
    init = init[None]  # [1, N, H, K, V]

    out, final_state = kda_forward(
        hm(q),
        hm(k),
        hm(v),
        hm(g),
        jnp.transpose(beta, (2, 0, 1)),  # [1,T,H] -> [H,1,T]
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=True,
        safe_gate=True,
        segment_ids=segment_ids,
        N_max=n_max,
        initial_state=init,
        output_final_state=True,
    )
    # [H, 1, T, V] -> [1, T, H, V]
    out = jnp.transpose(out, (1, 2, 0, 3))
    # Scatter the per-segment final states back, transposed into vLLM's layout.
    pool = pool.at[state_indices].set(
        jnp.swapaxes(final_state[0], -1, -2).astype(pool.dtype))
    return out, pool


def _kda_decode_jax(
    q: jax.Array,  # [1, N, H, D]
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,  # [1, N, H]
    pool: jax.Array,  # [num_slots, H, K, V]
    a_log: jax.Array,  # [H]
    dt_bias: jax.Array,  # [H * K]
    state_indices: jax.Array,
    has_initial_state: jax.Array,
    *,
    lower_bound: float | None,
) -> tuple[jax.Array, jax.Array]:
    """Single-token recurrent KDA; returns (out, updated pool)."""
    n_v, d_k = q.shape[2], q.shape[3]
    out, pool = kda_decode(
        q[0],
        k[0],
        v[0],
        g[0],
        beta[0],
        pool,
        a_log,
        dt_bias.reshape(n_v, d_k),
        state_indices,
        has_initial_state,
        lower_bound=lower_bound,
        # TODO(gnovack) - Determine correct state layout ([d_v, d_k] vs. [d_k, d_v]); ported kernel and vllm reference disagree)
        state_transposed=True, 
    )
    return out[None], pool  # [1, N, H, V]


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------
class KimiK3DeltaAttention(nn.Module):
    """KDA layer wired to the TPU kernels. See module docstring for scope."""

    def __init__(self, config, vllm_config, prefix: str = "") -> None:
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear)
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.model_executor.utils import set_weight_attrs
        from vllm.utils.math_utils import divide

        del extract_layer_index  # kept for parity with upstream's imports

        self.prefix = prefix
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = getattr(vllm_config, "quant_config", None)
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()

        kda_config = config.linear_attn_config
        assert kda_config is not None, "linear_attn_config must be set"
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)
        self.projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(self.projection_size, self.tp_size)
        self.conv_size = kda_config["short_conv_kernel_size"]
        assert kda_config.get("use_full_rank_gate", False), (
            "KimiK3DeltaAttention requires a full-rank gate")

        # Fused projection: q, k, v, output-gate, f_a, beta (+ alignment pad).
        # Same output-size list and 16-element alignment as upstream, because the
        # split sizes in forward() depend on it.
        qkvg_output_sizes = [self.projection_size] * 4
        in_proj_output_sizes = qkvg_output_sizes + [
            self.head_dim,
            self.num_heads,
        ]
        local_output_size = (4 * self.local_projection_size + self.head_dim +
                             self.local_num_heads)
        self.in_proj_padding = -local_output_size % 16
        if self.in_proj_padding:
            in_proj_output_sizes.append(self.in_proj_padding * self.tp_size)
        # NOTE: upstream uses _KimiGDNMergedColumnParallelLinear so that shard 4
        # (f_a, of size head_dim) is REPLICATED rather than split across ranks.
        # This plain merged linear is only correct at tp_size == 1.
        self.in_proj_qkvgfab = MergedColumnParallelLinear(
            self.hidden_size,
            in_proj_output_sizes,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvgfab",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )

        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32))
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32))
        try:
            from vllm.model_executor.model_loader.weight_utils import \
                sharded_weight_loader
            set_weight_attrs(self.dt_bias,
                             {"weight_loader": sharded_weight_loader(0)})
            set_weight_attrs(self.A_log,
                             {"weight_loader": sharded_weight_loader(0)})
        except ImportError:
            pass

        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=3 * self.projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.gate_lower_bound: float | None = kda_config.get(
            "gate_lower_bound", None)
        if self.gate_lower_bound is not None:
            assert _KDA_GATE_LOGBOUND_MIN <= self.gate_lower_bound < 0, (
                "KDA gate lower bound must be in "
                f"[{_KDA_GATE_LOGBOUND_MIN}, 0). Got {self.gate_lower_bound}.")

        self.o_norm = GatedRMSNormSigmoid(self.head_dim)
        self.o_proj = RowParallelLinear(
            self.projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # (conv_state, recurrent_state); assigned by the runner, or by a test.
        self.kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._prefill_op = None
        self._decode_op = None

    # -- state shapes, for whoever allocates the cache -----------------------
    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        from vllm.model_executor.layers.mamba.mamba_utils import \
            MambaStateDtypeCalculator
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        from vllm.model_executor.layers.mamba.mamba_utils import \
            MambaStateShapeCalculator
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=0,
        )

    # -- torch <-> jax op construction --------------------------------------
    def _get_prefill_op(self, n_max: int):
        """Build (and cache per n_max) the prefill jax_op.

        ``N_max`` is static to the kernel, so one op is compiled per distinct
        sequence count. A real integration would bucket this the way the runner
        buckets token counts.
        """
        from torch_tpu._internal import pallas

        key = ("prefill", n_max)
        if self._prefill_op is not None and self._prefill_op[0] == key:
            return self._prefill_op[1]

        fn = functools.partial(_kda_prefill_jax,
                               lower_bound=self.gate_lower_bound,
                               n_max=n_max)

        def wrapped(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array,
                    beta: jax.Array, pool: jax.Array, a_log: jax.Array,
                    dt_bias: jax.Array, segment_ids: jax.Array,
                    state_indices: jax.Array,
                    has_initial_state: jax.Array) -> tuple[jax.Array,
                                                           jax.Array]:
            return fn(q, k, v, g, beta, pool, a_log, dt_bias, segment_ids,
                      state_indices, has_initial_state)

        name = (f"pallas::kda_prefill_{self.prefix.replace('.', '_')}"
                f"_n{n_max}")
        # NOT donated. Donating the pool (as gdn_attention_op.py does) consumes
        # the caller's buffer, but `_forward` also writes the result back with
        # `recurrent_state.copy_(new_pool)` and the same cache tensor is reused
        # on the next call -- torch_tpu then rejects the second execution with
        # "Donation requested for invalid buffer", and the failure is only
        # visible as a MaterializationWorker error while the caller sees stale
        # results. Correctness first for a prototype; the cost is one pool-sized
        # copy per layer per step, which is what to revisit when this is wired
        # into the runner and the cache lifetime is owned there.
        op = pallas.jax_op(name, wrapped)

        def _fake(q, _k, _v, _g, _beta, pool, *args, **kwargs):
            return (torch.empty_like(q), torch.empty_like(pool))

        op.register_fake(_fake)
        self._prefill_op = (key, op)
        return op

    def _get_decode_op(self):
        from torch_tpu._internal import pallas

        if self._decode_op is not None:
            return self._decode_op

        fn = functools.partial(_kda_decode_jax,
                               lower_bound=self.gate_lower_bound)

        def wrapped(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array,
                    beta: jax.Array, pool: jax.Array, a_log: jax.Array,
                    dt_bias: jax.Array, state_indices: jax.Array,
                    has_initial_state: jax.Array) -> tuple[jax.Array,
                                                           jax.Array]:
            return fn(q, k, v, g, beta, pool, a_log, dt_bias, state_indices,
                      has_initial_state)

        name = f"pallas::kda_decode_{self.prefix.replace('.', '_')}"
        op = pallas.jax_op(name, wrapped)  # not donated; see _get_prefill_op

        def _fake(q, _k, _v, _g, _beta, pool, *args, **kwargs):
            return (torch.empty_like(q), torch.empty_like(pool))

        op.register_fake(_fake)
        self._decode_op = op
        return op

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: KdaMetadata | None = None,
    ) -> torch.Tensor:
        """Step-for-step the same as upstream's ``forward``.

        ``metadata`` is an addition: upstream reads it from the global forward
        context. Passing it explicitly is what lets an accuracy test drive this
        layer without a vLLM engine; when omitted it is read from the forward
        context exactly as upstream does.
        """
        del positions  # KDA is position-free; kept for interface parity.
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
        self._forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            core_attn_out=core_attn_out,
            metadata=metadata,
        )
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        return self.o_proj(core_attn_out)[0]

    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        metadata: KdaMetadata | None = None,
    ) -> None:
        if metadata is None:
            from vllm.forward_context import get_forward_context
            raw = get_forward_context().attn_metadata
            if raw is None:
                return
            assert isinstance(raw, dict), (
                "expected per-layer attn_metadata dict")
            metadata = raw[self.prefix]
        m = metadata

        if m.num_spec_decodes:
            raise NotImplementedError(
                "Speculative decoding is not supported by the TPU KDA decode "
                "kernel (window_size == 1). Upstream routes these through "
                "fused_recurrent_kda with num_accepted_tokens rollback.")
        if m.num_prefills > 0 and m.num_decodes > 0:
            raise NotImplementedError(
                "Mixed prefill+decode batches are not split yet; upstream uses "
                "spec_token_indx / non_spec_token_indx to separate them.")

        assert self.kv_cache is not None, "kv_cache must be assigned"
        conv_state, recurrent_state = self.kv_cache
        # vLLM stores the conv state in either orientation (see
        # MambaStateShapeCalculator._orient_conv_shape). Upstream normalizes to
        # dim-at-(-2); the torch reference conv here wants
        # [num_slots, state_len, dim], so normalize the other way. transpose()
        # returns a view, so in-place writes still land in the real cache.
        try:
            from vllm.model_executor.layers.mamba.mamba_utils import \
                is_conv_state_dim_first
            if is_conv_state_dim_first():
                conv_state = conv_state.transpose(-1, -2)
        except ImportError:
            pass

        n = m.num_actual_tokens
        mixed_qkv = mixed_qkv[:n]
        g1 = g1[:, :n]
        beta = beta[:, :n]

        # beta arrives RAW from the projection. Upstream hands it to the kernel as
        # `raw_beta=beta` and the kernel applies the sigmoid itself; the ported
        # kernels do not (they take beta already activated, like the reference in
        # tests/kernels/test_kda.py), so it has to happen here.
        #
        # Getting this wrong is nearly silent: raw beta is ~N(0, 0.17) where
        # sigmoid(beta) ~ 0.5, so the delta-rule update is scaled ~0 and the
        # module's output collapses to ~1/3 of its correct magnitude while the
        # recurrent state still looks approximately right. Against the GPU dump
        # that showed up as output rmse_rel 1.03 with final_conv_state and
        # final_recurrent_state both passing.
        beta = torch.sigmoid(beta.float()).to(beta.dtype)

        # conv1d weight is stored as [3 * local_proj, 1, conv_size].
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))
        conv_bias = self.conv1d.bias

        is_prefill = m.num_prefills > 0
        if is_prefill:
            conv_out = _causal_conv1d_varlen(
                mixed_qkv,
                conv_state,
                conv_weights,
                conv_bias,
                m.query_start_loc,
                m.state_indices,
                m.has_initial_state,
            )
        else:
            conv_out = _causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                conv_bias,
                m.state_indices,
            )

        q, k, v = (rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                   for x in conv_out.split(self.local_projection_size, dim=-1))

        num_seqs = m.state_indices.numel()
        has_init = m.has_initial_state.to(torch.int32)
        state_indices = m.state_indices.to(torch.int32)

        if is_prefill:
            segment_ids = self._segment_ids(m.query_start_loc, n,
                                            mixed_qkv.device)
            op = self._get_prefill_op(num_seqs)
            out, new_pool = op(q, k, v, g1, beta, recurrent_state, self.A_log,
                               self.dt_bias, segment_ids, state_indices,
                               has_init)
        else:
            op = self._get_decode_op()
            out, new_pool = op(q, k, v, g1, beta, recurrent_state, self.A_log,
                               self.dt_bias, state_indices, has_init)

        recurrent_state.copy_(new_pool)
        core_attn_out[:, :n] = out[:, :n].to(core_attn_out.dtype)
        core_attn_out.copy_(self.o_norm(core_attn_out, g2))

    @staticmethod
    def _segment_ids(query_start_loc: torch.Tensor, num_tokens: int,
                     device) -> torch.Tensor:
        """``[1, T]`` 1-indexed segment ids from cu_seqlens; 0 marks padding.

        The chunked kernel takes ``segment_ids`` where upstream's
        ``chunk_kda_with_fused_gate`` takes ``cu_seqlens``; this is the
        conversion. Built on host from ``query_start_loc``, so like the reference
        conv1d it is not graph-friendly.
        """
        seg = torch.zeros(1, num_tokens, dtype=torch.int32, device=device)
        starts = query_start_loc.tolist()
        for i in range(len(starts) - 1):
            s, e = starts[i], min(starts[i + 1], num_tokens)
            if e > s:
                seg[0, s:e] = i + 1
        return seg
