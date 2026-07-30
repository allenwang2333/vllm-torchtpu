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
changes to the model definition. Because we do not define ``__init__``, the
parent's runs unmodified.

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

        def _get_op(self):
            """Build (and cache) this layer's torch<->jax op."""
            op = getattr(self, "_tpu_kda_op", None)
            if op is not None:
                return op

            from torch_tpu._internal import pallas

            fn = functools.partial(
                kimi_linear_kda_core,
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
            # Not donated: the states are also written back with copy_(), and the
            # same cache tensors are reused next call -- donating them makes
            # torch_tpu reject the second execution with "Donation requested for
            # invalid buffer", visible only as a worker error while the caller
            # silently gets stale results.
            op = pallas.jax_op(name, wrapped)

            def _fake(q, _k, _v, _g, _beta, conv_state, recurrent_state, *a,
                      **kw):
                out = torch.empty(
                    (q.size(0), self.local_num_heads, self.head_dim),
                    dtype=q.dtype, device=q.device)
                return (torch.empty_like(conv_state),
                        torch.empty_like(recurrent_state), out)

            op.register_fake(_fake)
            self._tpu_kda_op = op
            return op

        def _forward(
            self,
            q_proj_states: torch.Tensor,
            k_proj_states: torch.Tensor,
            v_proj_states: torch.Tensor,
            g1: torch.Tensor,
            beta: torch.Tensor,
            core_attn_out: torch.Tensor,
        ) -> None:
            """Adapter: hand the TPU metadata straight to the JAX core.

            Deliberately does no host-side work -- no ``.tolist()``, no branching
            on prefill-vs-decode. The metadata arrays go in as-is and the kernels
            resolve the batch composition from ``request_distribution``.
            """
            from vllm.forward_context import get_forward_context

            md = get_forward_context().attn_metadata
            if md is None:
                return  # profile run before metadata exists
            if isinstance(md, dict):
                md = md[self.prefix]

            assert md.mamba_state_indices is not None, (
                "Kimi-Linear KDA needs mamba_state_indices; the runner did not "
                "provide a mamba state mapping for this layer.")

            conv_state, recurrent_state = self.kv_cache

            # The parent hands g1/beta with a leading batch axis ([1, T, H, d]
            # and [1, T, H]); the core is token-major.
            g1 = g1[0].reshape(g1.shape[1], -1)
            beta = beta[0]

            new_conv, new_rec, out = self._get_op()(
                q_proj_states, k_proj_states, v_proj_states, g1, beta,
                conv_state, recurrent_state, self.q_conv1d.weight,
                self.k_conv1d.weight, self.v_conv1d.weight,
                self.A_log.reshape(-1), self.dt_bias,
                md.mamba_state_indices.to(torch.int32), md.query_start_loc,
                md.request_distribution, md.seq_lens)

            conv_state.copy_(new_conv)
            recurrent_state.copy_(new_rec)
            core_attn_out[0, :out.shape[0]] = out.to(core_attn_out.dtype)

    return KimiLinearKDAAttention
