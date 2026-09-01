# SPDX-License-Identifier: Apache-2.0
"""RowParallelLinear whose TP all-reduce runs through the two-shot kernel op.

This is the explicit call site for the low-latency all-reduce: the layer
invokes ``vllm_torchtpu::llar_out_proj_reduce_*`` unconditionally, and the op
itself chooses between the two-shot Pallas kernel and the stock XLA ring
reduction based on the concrete token count it sees at execution time. The
choice cannot live here -- the forward is traced once with the token dim
dynamic, and reading it in Python raises ConstraintViolationError -- so this
class adds nothing to the traced graph but one opaque op call with a fixed
signature. See ``distributed/low_latency_all_reduce.py``.
"""

import torch
from vllm.distributed import (split_tensor_along_last_dim,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.linear import (RowParallelLinear,
                                               UnquantizedLinearMethod)


class LowLatencyRowParallelLinear(RowParallelLinear):
    """A RowParallelLinear that routes its TP all-reduce through the LLAR op.

    Constructed in place of RowParallelLinear by the model when
    ``TPU_LOW_LATENCY_ALLREDUCE`` is set; identical to it otherwise. The op
    and its workspaces are built by ``initialize_llar``, which the runner
    calls after weights load and before torch.compile -- registration and
    workspace allocation cannot happen inside the traced forward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # None means "stock all-reduce": the flag is off, the gate rejected
        # every payload at this width, or the mesh could not be built.
        self._llar_entry = None
        # Position among the armed layers in execution order, assigned by the
        # runner after initialize_llar. The kernel's workspace slot rotation
        # is driven by it: consecutive calls alternate slots, and the first
        # call of every execution barriers before sending. The default keeps
        # an unassigned layer fully safe (slot 0, barrier first).
        self._llar_ordinal = 0

    def initialize_llar(self) -> bool:
        from vllm_torchtpu.distributed.low_latency_all_reduce import \
            get_out_proj_reduce_entry

        if not (self.reduce_results and self.tp_size > 1):
            return False
        # The workspace dtype comes from the weight, which is only the
        # activation dtype when nothing quantizes the layer.
        if not isinstance(self.quant_method, UnquantizedLinearMethod):
            return False
        self._llar_entry = get_out_proj_reduce_entry(self.output_size,
                                                     self.weight.dtype)
        return self._llar_entry is not None

    def forward(self, input_: torch.Tensor):
        # Mirrors the TPU RowParallelLinear forward (the deferred-bias variant
        # installed by vllm_torchtpu._patch_rowparallel_defer_bias), with the
        # all-reduce line swapped for the opaque op when it is armed.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        do_all_reduce = self.reduce_results and self.tp_size > 1
        # Defer the (replicated) bias past the all-reduce so every rank runs
        # an identical GEMM graph -> matching XLA all_reduce channel IDs.
        defer_tpu_bias = (do_all_reduce and not self.skip_bias_add
                          and self.bias is not None)
        if defer_tpu_bias:
            bias_ = None
        else:
            bias_ = (None if
                     (self.tp_rank > 0 or self.skip_bias_add) else self.bias)
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        entry = self._llar_entry
        if do_all_reduce and entry is not None:
            op, rs_list, ag_list = entry
            output = op(output_parallel, rs_list, ag_list,
                        self._llar_ordinal % 2,
                        1 if self._llar_ordinal == 0 else 0)
        elif do_all_reduce:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if defer_tpu_bias:
            output = output + self.bias

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
