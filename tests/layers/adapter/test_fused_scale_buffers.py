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
"""The fused EP scale buffers are real, materialized copies.

A view of the 4-D scale parameter has no device layout for the compiler to
pin, so the compiled forward re-materialized it with an eager `as_strided`
program on every step. These tests pin the properties that prevent that.
"""

import torch
from torch_tpu._internal.compile.compiler import tpu_torch_compile

from vllm_torchtpu.layers.adapter.fused_moe_ep import register_fused_scale_buffers


def _layer(experts=4, hidden=256, inter=128, block=64):
    layer = torch.nn.Module()
    for name, k, n in (
        ("w13_weight_scale", hidden, 2 * inter),
        ("w2_weight_scale", inter, hidden),
    ):
        scale = torch.rand(experts, k // block, 1, n).to("tpu")
        layer.register_parameter(name, torch.nn.Parameter(scale, requires_grad=False))
    return layer


def test_scale_buffers_are_materialized_copies_not_views():
    layer = _layer()
    # What `squeeze(2).contiguous()` hands back: a view, with no layout.
    view = layer.w13_weight_scale.squeeze(2).contiguous()
    assert view._is_view()
    assert tpu_torch_compile.get_device_layout_if_materialized(view) is None

    expected = {
        name: getattr(layer, name).squeeze(2).cpu()
        for name in ("w13_weight_scale", "w2_weight_scale")
    }
    register_fused_scale_buffers(layer)

    for param_name, buffer_name in (
        ("w13_weight_scale", "_tpu_fused_w13_scale"),
        ("w2_weight_scale", "_tpu_fused_w2_scale"),
    ):
        buf = getattr(layer, buffer_name)
        assert not buf._is_view()
        assert tpu_torch_compile.get_device_layout_if_materialized(buf) is not None
        torch.testing.assert_close(buf.cpu(), expected[param_name])
        # The 4-D parameter now aliases the copy instead of keeping the old
        # storage alive: the scales are held once.
        param = getattr(layer, param_name)
        assert isinstance(param, torch.nn.Parameter)
        assert param.shape == (buf.shape[0], buf.shape[1], 1, buf.shape[2])
        assert param.untyped_storage().data_ptr() == buf.untyped_storage().data_ptr()
        assert buffer_name in layer._non_persistent_buffers_set
