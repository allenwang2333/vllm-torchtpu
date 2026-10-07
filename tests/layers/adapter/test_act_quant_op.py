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
"""TPU_ACT_QUANT against the unfused path on TPU: SiluAndMul / the attention
output gate, then the fp8 linear's own activation quant and matmul. Run with
--use-tpu."""

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul

from vllm_torchtpu.layers.adapter import act_quant_op as aqo
from vllm_torchtpu.layers.adapter import linear_common


@pytest.fixture(autouse=True)
def _vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def _same(a, b):
    return torch.equal(a.cpu().view(torch.int16), b.cpu().view(torch.int16))


def _w(k, n, g, device):
    w = (torch.randn(k, n, generator=g) * 100).clamp(-448, 448)
    return w.to(torch.float8_e4m3fn).to(device), (torch.rand(n, generator=g) * 1e-3).to(device)


@pytest.mark.parametrize("rows", [4, 8, 48, 1024])
def test_silu_mul_matches(device, rows):
    if device.type != "tpu":
        pytest.skip("Pallas kernel on TPU.")
    g = torch.Generator().manual_seed(rows)
    x = (torch.randn(rows, 256, generator=g) * 2).to(torch.bfloat16).to(device)
    x[1] = 0
    w, ws = _w(128, 4096, g, device)
    ref = linear_common.quantized_matmul(SiluAndMul()(x), w, ws)
    q, s = aqo._build("silu_mul")(x)
    assert _same(linear_common.quantized_matmul_prequant(q, s, w, ws), ref)


@pytest.mark.parametrize("rows", [4, 8, 48, 1024])
def test_sigmoid_gate_matches(device, rows):
    if device.type != "tpu":
        pytest.skip("Pallas kernel on TPU.")
    g = torch.Generator().manual_seed(rows + 1)
    a = torch.randn(rows, 1024, generator=g).to(torch.bfloat16).to(device)
    gate = (torch.randn(rows, 1024, generator=g) * 2).to(torch.bfloat16).to(device)
    w, ws = _w(1024, 4096, g, device)
    ref = linear_common.quantized_matmul(a * torch.sigmoid(gate), w, ws)
    q, s = aqo._build("sigmoid_gate")(a, gate)
    assert _same(linear_common.quantized_matmul_prequant(q, s, w, ws), ref)


def test_positive_control(device, monkeypatch):
    """A SiLU rounded differently (round_mid=1) is caught by the comparison."""
    if device.type != "tpu":
        pytest.skip("Pallas kernel on TPU.")
    monkeypatch.setenv("TPU_ACT_QUANT_ROUND_MID", "1")
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(8, 256, generator=g) * 2).to(torch.bfloat16).to(device)
    w, ws = _w(128, 4096, g, device)
    ref = linear_common.quantized_matmul(SiluAndMul()(x), w, ws)
    q, s = aqo._build("silu_mul")(x)
    assert not _same(linear_common.quantized_matmul_prequant(q, s, w, ws), ref)


def test_gate_positive_control(device, monkeypatch):
    """An f32 (unrounded) gate sigmoid (round_mid=4) is caught."""
    if device.type != "tpu":
        pytest.skip("Pallas kernel on TPU.")
    monkeypatch.setenv("TPU_ACT_QUANT_ROUND_MID", "4")
    g = torch.Generator().manual_seed(9)
    a = torch.randn(8, 1024, generator=g).to(torch.bfloat16).to(device)
    gate = (torch.randn(8, 1024, generator=g) * 2).to(torch.bfloat16).to(device)
    w, ws = _w(1024, 4096, g, device)
    ref = linear_common.quantized_matmul(a * torch.sigmoid(gate), w, ws)
    q, s = aqo._build("sigmoid_gate")(a, gate)
    assert not _same(linear_common.quantized_matmul_prequant(q, s, w, ws), ref)
