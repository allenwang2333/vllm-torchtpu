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

import types
from collections.abc import Callable

import torch
from torch.nn import Parameter
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import RowParallelLinear

_TPU_OP_CACHE: dict[tuple, Callable] = {}


def _get_tpu_op(key: tuple, fn: Callable) -> Callable:
    op = _TPU_OP_CACHE.get(key)
    if op is not None:
        return op
    name = f"tpu_row_linear_op_{len(_TPU_OP_CACHE)}"
    specialized = types.FunctionType(
        fn.__code__.replace(co_name=name),
        fn.__globals__,
        name,
        fn.__defaults__,
        fn.__closure__,
    )
    op = torch.compile(specialized, backend="tpu", dynamic=False)
    _TPU_OP_CACHE[key] = op
    return op


def _reduce_add_bias_impl(
    output_parallel: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    output = tensor_model_parallel_all_reduce(output_parallel)
    return output + bias


@RowParallelLinear.register_oot
class TpuRowParallelLinear(RowParallelLinear):
    """RowParallelLinear that adds the bias after the TP all-reduce.

    Upstream fuses the bias into the GEMM on rank 0 only, so the per-rank FX
    graphs differ and the torch_tpu XLA backend hands the all_reduce mismatched
    channel IDs, breaking the collective. The bias is replicated on every rank,
    so adding it afterwards sums the same terms in a different order and keeps
    the graphs identical.

    Only the deferred case is overridden; everything else delegates to
    ``super().forward()``.
    """

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias
        defer_bias = (
            self.reduce_results
            and self.tp_size > 1
            and not self.skip_bias_add
            and bias is not None
        )
        if not defer_bias:
            return super().forward(input_)

        assert bias is not None
        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        output_parallel = self.quant_method.apply(self, input_parallel, None)
        if output_parallel.device.type == "tpu" and not torch.compiler.is_compiling():
            post_op = _get_tpu_op(
                (
                    "row_parallel_post",
                    id(tensor_model_parallel_all_reduce),
                    output_parallel.shape,
                    output_parallel.dtype,
                    bias.shape,
                    bias.dtype,
                ),
                _reduce_add_bias_impl,
            )
            output = post_op(output_parallel, bias)
        else:
            output = _reduce_add_bias_impl(output_parallel, bias)

        # skip_bias_add is False here, so there is no bias left to return.
        if not self.return_bias:
            return output
        return output, None
