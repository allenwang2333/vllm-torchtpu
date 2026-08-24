# SPDX-License-Identifier: Apache-2.0
"""MoE runner that reduces K3's latent-MoE pair as one collective."""

import torch
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (MoERunner,
                                                                    _unpack)

from vllm_torchtpu.distributed import low_latency_all_reduce as _llar
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class KimiK3LatentMoERunner(MoERunner):
    """Reduce the latent-MoE pair with one low-latency collective, not two.

    ``MoERunner.forward`` reduces twice for a latent MoE, on two independent
    locals in consecutive statements: the routed experts' output at the latent
    width, and the shared experts' at the hidden width. Upstream says so itself
    -- "there are two all-reduce points below, they are mutually exclusive,
    controlled by _fused_output_is_reduced". On K3 with TP the first fires and
    flips that flag, the second fires to match, and the late reduce is skipped.

    Being independent is why XLA's CRS combiner merges them into a single 42 KiB
    collective, and that amortisation is what defeated every attempt to put a
    kernel on one half: splitting the pair costs 18 us per layer, more than the
    faster collective saves (+5.5% device time on K3). So this does the
    combiner's job explicitly -- concatenate, reduce once with the kernel, split
    -- measured at 15.07 us against the combiner's 21.14, worth -10.5 us/layer
    of collective time reproduced across three runs.

    Ordering is preserved exactly. Both reductions are plain sums over the same
    TP group and an all-reduce is elementwise, so slicing the concatenation
    apart gives each half bit-identical values. The latent half is still reduced
    *before* ``apply_routed_output_transform``, which is required rather than
    incidental -- that transform contains an RMS norm, and TP partials have to be
    summed before a non-linearity. That constraint is also why the two reductions
    cannot simply be deferred and merged later: the concatenation has to happen
    while both are still unreduced, which is exactly where they are adjacent.

    Selected by passing ``runner_cls`` to ``FusedMoEFactory``, the hook upstream
    added for exactly this and uses itself for ``LatentMoERunner``. An earlier
    revision monkeypatched ``MoERunner.forward`` instead, which cost two
    failures worth recording:

    * A patch on the base class does not apply to a subclass that overrides
      ``forward``, silently. A run logged the pair as wired, built the kernel,
      and never executed the concat op once; only a trace post-mortem found it.
      Choosing the class removes the question -- nothing can be built but what
      was asked for.
    * The patched ``forward`` was a closure in ``vllm_torchtpu/__init__.py``, so
      installing it put that file in the *traced* path of every model, and
      vLLM's ``source_info.add_code`` calls ``inspect.getsourcelines`` on each
      traced code object. If the file changed on disk after a worker imported it
      (a lint pass, a branch switch mid-run) compilation died with "IndexError:
      list index out of range" from ``inspect.findsource``, 25 minutes in and
      nowhere near the real cause. A module-level class has no such hazard.
    """

    def _use_concat_path(self) -> bool:
        """Whether this layer's two reductions are ours to merge.

        Every term is structural and known at construction, so the decision is
        made before any work happens and the fallback below can be a plain
        ``super().forward()``. Anything else -- no shared experts, no output
        transform, a combine kernel that already reduced, sequence parallelism
        -- keeps upstream's own sequencing untouched.
        """
        return (self._shared_experts is not None
                and self.routed_output_transform is not None
                and not self._fused_output_is_reduced
                and not self.moe_config.is_sequence_parallel and
                (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1))

    def forward(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            input_ids: torch.Tensor | None = None,
            shared_experts_input: torch.Tensor | None = None) -> torch.Tensor:
        if not self._use_concat_path():
            return super().forward(hidden_states, router_logits, input_ids,
                                   shared_experts_input)

        # A bucket above the configured batch threshold is expected and must
        # fall back silently: warmup wires only the buckets it was asked for,
        # and the rest have to keep upstream's pair so XLA's combiner can fuse
        # it. Concatenating anyway is what regressed job 139's c4/c8 arms by
        # 5.2% and 4.0% -- the cat and its two slices cost ~300 us per decode
        # step and bought a kernel that never ran, because at K3's bucket 16 the
        # pair is 336 KiB and two_shot is 0.75x XLA there. No threshold change
        # rescues that: two_shot is bandwidth-bound past ~112 KiB and only gets
        # worse, which is why _GATE has the final say over the threshold.
        #
        # Asked here, at the top, rather than after the expert compute: this
        # indexes a dict with a Python int from the forward context and dynamo
        # folds it to a constant, so the whole branch resolves at trace time.
        entries = _llar._MOE_CONCAT_ENTRIES
        if not entries:
            # Loud, not a silent fallback. This class is only ever built when
            # the feature was explicitly requested, and the failure mode it
            # exists to prevent is precisely a run that looks wired and quietly
            # does the old thing.
            raise RuntimeError(
                "KimiK3LatentMoERunner is in use, but warmup wired no "
                "concatenated plan for any bucket. Check that "
                "TPU_LOW_LATENCY_ALLREDUCE=1, that a compile bucket is at or "
                "below TPU_LOW_LATENCY_ALLREDUCE_MAX_BATCH_SIZE, and that the "
                "pair's payload is inside the two_shot band for this device "
                "count.")

        entry = _llar.select_entry(entries)
        if entry is None:
            return super().forward(hidden_states, router_logits, input_ids,
                                   shared_experts_input)

        return self._concat_forward(entry, hidden_states, router_logits,
                                    input_ids, shared_experts_input)

    def _concat_forward(
            self, entry: tuple, hidden_states: torch.Tensor,
            router_logits: torch.Tensor, input_ids: torch.Tensor | None,
            shared_experts_input: torch.Tensor | None) -> torch.Tensor:
        """``MoERunner.forward`` with its two reductions merged into one.

        The body up to ``_forward_entry`` and after the reduction is upstream's,
        copied because the merge happens in the middle of it -- the same reason
        upstream's own ``LatentMoERunner._fused_forward`` copies it.
        """
        if shared_experts_input is None:
            hidden_states, shared_experts_input = (
                self.apply_routed_input_transform(hidden_states))
        (hidden_states, og_pre,
         og_post) = self._maybe_pad_hidden_states(shared_experts_input,
                                                  hidden_states)

        result = self._forward_entry(
            hidden_states, router_logits, shared_experts_input, input_ids,
            self._encode_layer_name(), self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output else 0)

        shared_output, fused_output = _unpack(result)
        assert shared_output is not None, (
            "_use_concat_path requires shared experts")
        if og_pre is not None:
            fused_output = fused_output[..., :og_pre]

        n_latent = fused_output.numel()
        pair = torch.cat([fused_output.reshape(-1), shared_output.reshape(-1)])
        op, buffers = entry
        reduced = op(
            pair.reshape(shared_output.shape[0], -1).contiguous(), *buffers)
        flat = reduced.reshape(-1)
        fused_output = flat[:n_latent].reshape(fused_output.shape)
        shared_output = flat[n_latent:].reshape(shared_output.shape)

        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
            shared_output, fused_output)
        fused_output = self.apply_routed_output_transform(fused_output)
        out = shared_output + fused_output
        # Both halves are reduced, which is what upstream's flag means, so the
        # late all-reduce is skipped and this only strips padding.
        out = self._maybe_reduce_final_output(out,
                                              og_post,
                                              output_is_reduced=True)
        return self._maybe_add_zero_expert_output(out)
