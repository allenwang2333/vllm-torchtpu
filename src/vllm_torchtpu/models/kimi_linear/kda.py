# SPDX-License-Identifier: Apache-2.0
"""TPU KDA attention for Kimi-Linear, as an out-of-tree pluggable layer.

Subclasses vLLM's in-tree ``KimiGatedDeltaNetAttention`` and overrides **only**
``_forward`` -- the method that runs conv1d and the KDA kernels. Everything else
is inherited: the projections and their weight loaders, ``o_norm``/``o_proj``,
``get_state_shape``/``get_state_dtype``, and the ``forward`` that wraps
``_forward`` with the gated norm and output projection.

Registration goes through vLLM's ``PluggableLayer`` OOT mechanism rather than a
monkeypatch. ``PluggableLayer.__new__`` keys on the *in-tree* class name, so
registering under ``"KimiGatedDeltaNetAttention"`` means
``vllm/model_executor/models/kimi_linear.py`` builds this class instead, with no
changes to the model definition. ``__init__`` only extends the parent's: it
chains to ``super()`` and then builds the jax op and initialises ``kv_cache``,
both of which must happen outside the compiled forward.

``_forward`` here is only an adapter: it reads the TPU platform's attention
metadata and hands it to ``layers/common/kimi_linear_kda.py``, which does conv1d
and KDA in one JAX call. All the substance -- and the reasoning about why the
metadata contract differs from upstream's -- lives there.

Conventions that differ from the Kimi-K3 variant of this layer, and which cost
real debugging time on the K3 prototype:

* **beta arrives already activated.** Kimi-Linear's ``forward`` does
  ``self.b_proj(...).float().sigmoid()`` before calling ``_forward``, which
  matches what the ported kernels expect. (K3 passes raw beta and lets its kernel
  sigmoid internally.) So there is deliberately no sigmoid here.
* **The gate is unbounded.** Kimi-Linear's config has no ``gate_lower_bound``, so
  the activation is ``-exp(A_log) * softplus(g + dt_bias)``. That is
  ``lower_bound=None``, which the kernels require be paired with
  ``safe_gate=False``.
* **``A_log`` is ``[1, 1, num_heads, 1]``** (sharded on axis 2 upstream), not
  ``[num_heads]``; it is reshaped for the kernels.
* **The recurrent-state cache is ``[d_v, d_k]``** per slot, the transpose of what
  the kernels use; converted at the cache boundary in the JAX core.

Not yet implemented: decode rows go through the chunked kernel rather than
``kda_decode`` (see the core's module docstring), and speculative decoding is
unsupported.
"""

import functools

import jax
import torch

from vllm_torchtpu.layers.common.kimi_linear_kda import kimi_linear_kda_core

__all__ = ["KimiLinearKDAAttention"]


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------
def _register():
    """Build the class and register it, deferring the vLLM import."""
    from vllm.model_executor.custom_op import PluggableLayer
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import \
        KimiGatedDeltaNetAttention

    # Keyed on the in-tree class name: PluggableLayer.__new__ looks up
    # `cls.__name__`, so this substitutes for every construction site.
    @PluggableLayer.register_oot(name="KimiGatedDeltaNetAttention")
    class KimiLinearKDAAttention(KimiGatedDeltaNetAttention):
        """Kimi-Linear KDA on TPU. Only ``_forward`` differs from upstream."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Both of these follow VllmGatedDeltaNetAttention in
            # layers/vllm/custom_ops/gdn_attention_op.py, and both matter for
            # the compiled path:
            #
            # * The op is built here, not lazily in `_forward`. `pallas.jax_op`
            #   and `register_fake` are Python-level construction, so building
            #   on first call would run them *during* Dynamo tracing of the
            #   model graph.
            # * `kv_cache` is bound by the runner during KV-cache init and must
            #   exist before then, since warmup and the memory-profiling run
            #   call `_forward` first. Setting it here means the guard below is
            #   a plain None check rather than a getattr for a missing
            #   attribute.
            self.kv_cache = None
            self.kda_op = self._build_op()

        def _build_op(self):
            """Build this layer's torch<->jax op."""
            from torch_tpu._internal import pallas

            from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
                get_vllm_model_wrapper_context

            # The mesh binds the computation to *this worker's* device. Every
            # GDN op path threads it for the same reason; omitting it places
            # the op on JAX's default device, which only shows up at TP>1.
            mesh = get_vllm_model_wrapper_context().mesh

            fn = functools.partial(
                kimi_linear_kda_core,
                mesh=mesh,
                num_heads=self.local_num_heads,
                head_dim=self.head_dim,
                kernel_size=self.conv_size,
            )

            def wrapped(q_proj_states: jax.Array, k_proj_states: jax.Array,
                        v_proj_states: jax.Array, g1: jax.Array,
                        beta: jax.Array, conv_state: jax.Array,
                        recurrent_state: jax.Array, q_conv_weight: jax.Array,
                        k_conv_weight: jax.Array, v_conv_weight: jax.Array,
                        A_log: jax.Array, dt_bias: jax.Array,
                        state_indices: jax.Array, query_start_loc: jax.Array,
                        distribution: jax.Array, seq_lens: jax.Array
                        ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return fn(q_proj_states, k_proj_states, v_proj_states, g1, beta,
                          conv_state, recurrent_state, q_conv_weight,
                          k_conv_weight, v_conv_weight, A_log, dt_bias,
                          state_indices, query_start_loc, distribution,
                          seq_lens)

            name = f"pallas::kimilinear_kda_{self.prefix.replace('.', '_')}"
            # conv_state (5) and recurrent_state (6) are donated, and are ALSO
            # written back with copy_() below -- XLA aliases the donated input
            # to the program output, so the copy is in-place rather than a
            # second full-sized write. This is the contract the GDN op uses
            # (donate_argnums=(3, 4) plus copy_ writeback); without donation,
            # torch_tpu's compiled mode aborts with "compiled_mode.cc: Check
            # failed: ValidateAndReorderArguments ... identified an argument
            # that was not provided", naming these two buffers. Eager mode
            # tolerates the undonated form, which is why this only appeared
            # once the model was compiled.
            op = pallas.jax_op(name, wrapped, donate_argnums=(5, 6))

            def _fake(q, _k, _v, _g, _beta, conv_state, recurrent_state, *a,
                      **kw):
                out = torch.empty(
                    (q.size(0), self.local_num_heads, self.head_dim),
                    dtype=q.dtype, device=q.device)
                return (torch.empty_like(conv_state),
                        torch.empty_like(recurrent_state), out)

            op.register_fake(_fake)
            return op

        def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            output: torch.Tensor,
        ) -> None:
            """Full layer forward, overriding upstream's.

            Upstream's ``forward`` routes the core through
            ``torch.ops.vllm.kda_attention``, a ``direct_register_custom_op``
            wrapper around ``_forward``. Dynamo does not trace into that op, so
            its only graph inputs are the five tensors handed to it -- while
            ``_forward`` goes on to read ``self.kv_cache``, the conv weights,
            ``A_log`` and the attention metadata at runtime. Those buffers are
            then referenced by the program without being arguments to it, and
            torch_tpu aborts in ``TraverseAndCompile`` with "identified an
            argument that was not provided". Materializing any one operand only
            promotes the next: the boundary is the problem, not the tensor.

            Overriding ``forward`` (as ``VllmGatedDeltaNetAttention`` does for
            GDN) puts the jax op directly in the traced graph, so every operand
            is a real graph input. Eager mode never showed this because there
            is no graph to capture.

            The body mirrors upstream's exactly apart from the core call, and
            skips its ``[1, n, h, d]`` round-trip for ``g1``/``beta`` since this
            core is token-major.
            """
            from einops import rearrange

            num_tokens = hidden_states.size(0)
            q = self.q_proj(hidden_states)[0]
            k = self.k_proj(hidden_states)[0]
            v = self.v_proj(hidden_states)[0]

            # Already-activated beta, matching what the ported kernels expect.
            beta = self.b_proj(hidden_states)[0].float().sigmoid()
            g1 = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]

            g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
            g2 = rearrange(g_proj_states,
                           "... (h d) -> ... h d",
                           d=self.head_dim)

            core_attn_out = self._core(q, k, v, g1, beta, num_tokens,
                                       hidden_states)

            core_attn_out = self.o_norm(core_attn_out, g2)
            core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
            output[:] = self.o_proj(core_attn_out)[0]

        def _core(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g1: torch.Tensor,
            beta: torch.Tensor,
            num_tokens: int,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            """conv1d + KDA, returning ``[1, num_tokens, heads, head_dim]``.

            Does no host-side work -- no ``.tolist()``, no branching on
            prefill-vs-decode. The metadata arrays go in as-is and the kernels
            resolve the batch composition from ``request_distribution``.
            """
            from vllm.forward_context import get_forward_context

            zeros = lambda: torch.zeros(
                (1, num_tokens, self.local_num_heads, self.head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device)

            # Warmup and the memory-profiling run happen before the KV cache
            # exists, so emit correctly shaped zeros and touch no state. Gate on
            # the cache rather than the metadata: the TPU runner supplies
            # metadata (with its mamba fields unset) even on those runs, unlike
            # CUDA where upstream's `attn_metadata is None` check suffices.
            kv_cache = self.kv_cache
            if (kv_cache is None or len(kv_cache) < 2
                    or kv_cache[0].numel() == 0):
                return zeros()

            md = get_forward_context().attn_metadata
            if md is None:
                return zeros()
            if isinstance(md, dict):
                md = md[self.prefix]
            if md.mamba_state_indices is None:
                raise RuntimeError(
                    "Kimi-Linear KDA has a KV cache but the metadata carries "
                    "no mamba_state_indices, so there is no slot mapping to "
                    "use. The runner did not treat this layer as a mamba/SSM "
                    "layer.")

            conv_state, recurrent_state = kv_cache

            # NOTE: no logging in here -- this is inside the compiled graph and
            # Dynamo cannot trace a logger call under fullgraph=True.
            new_conv, new_rec, out = self.kda_op(
                q, k, v, g1, beta, conv_state, recurrent_state,
                self.q_conv1d.weight, self.k_conv1d.weight,
                self.v_conv1d.weight, self.A_log.reshape(-1), self.dt_bias,
                md.mamba_state_indices.to(torch.int32), md.query_start_loc,
                md.request_distribution, md.seq_lens)

            conv_state.copy_(new_conv)
            recurrent_state.copy_(new_rec)

            # Eager-only diagnostic (KDA_DEBUG=1). Never enable under
            # torch.compile: the logging call and the .item() syncs are both
            # untraceable. This is the only way to confirm the op actually ran
            # and produced finite values -- with random weights a silently
            # zeroed layer is indistinguishable from a working one.
            import os as _os
            if _os.environ.get("KDA_DEBUG") == "1":
                from vllm.logger import init_logger as _il
                _il(__name__).info(
                    "[kda-debug] %s ran: out finite=%s absmax=%.4g "
                    "state finite=%s", self.prefix,
                    bool(torch.isfinite(out).all().item()),
                    out.float().abs().max().item(),
                    bool(torch.isfinite(new_rec).all().item()))

            return out.unsqueeze(0)

    return KimiLinearKDAAttention
