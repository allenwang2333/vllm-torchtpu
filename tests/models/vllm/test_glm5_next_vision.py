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
"""GLM-5.3-Flash's vision tower differs from GLM-OCR's only by a SwiGLU clamp.

That makes the clamp the whole correctness surface of the port, and it is
asymmetric in a way that is easy to get wrong: the gate is bounded **above
only** (``silu`` already handles the negative tail) while the up-projection is
bounded on both sides. Clamping the gate below as well would look reasonable
and quietly change the activation.

The tower classes themselves are imported lazily, because pulling them in
drags vLLM's ``glm4_1v`` image processor and therefore ``torchvision``, whose
compiled ops do not load against the CPU-only torch in the dev venv. The clamp
function has no such dependency, so the part that actually needs pinning is
always tested.
"""

import ast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

_VISION_SRC = "src/vllm_torchtpu/models/vllm/glm5_next/vision.py"
_LIMIT = 10.0  # GLM-5.3-Flash's vision_config.swiglu_limit


def _load_clamped_swiglu():
    """Pull `_clamped_swiglu` out of the module without importing the module."""
    tree = ast.parse(open(_VISION_SRC).read())
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_clamped_swiglu")
    ns = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<vision>", "exec"),
         ns)
    return ns["_clamped_swiglu"]


@pytest.fixture(scope="module")
def clamped_swiglu():
    return _load_clamped_swiglu()


def _hf_reference(gate, up, limit):
    """transformers' Glm5NextVisionMLP.forward, modular_glm5_next.py:1379."""
    return F.silu(gate.clamp(min=None, max=limit)) * up.clamp(min=-limit,
                                                              max=limit)


def test_matches_the_hf_activation(clamped_swiglu):
    torch.manual_seed(0)
    # x12 so the clamp genuinely engages on both halves.
    gate, up = torch.randn(64, 32) * 12, torch.randn(64, 32) * 12
    assert (gate > _LIMIT).any() and (up.abs() > _LIMIT).any(), \
        "inputs never reach the clamp; the test would be vacuous"

    got = clamped_swiglu(torch.cat([gate, up], dim=-1), _LIMIT)
    torch.testing.assert_close(got, _hf_reference(gate, up, _LIMIT))


def test_is_not_plain_swiglu(clamped_swiglu):
    """Guard against someone 'simplifying' this back to SiluAndMul."""
    torch.manual_seed(0)
    gate, up = torch.randn(64, 32) * 12, torch.randn(64, 32) * 12
    got = clamped_swiglu(torch.cat([gate, up], dim=-1), _LIMIT)
    assert (got - F.silu(gate) * up).abs().max() > 1.0


def test_gate_is_bounded_above_only(clamped_swiglu):
    """The asymmetry. A symmetric clamp would zero this instead."""
    gate = torch.full((4, 4), -50.0)
    up = torch.ones(4, 4)
    got = clamped_swiglu(torch.cat([gate, up], dim=-1), _LIMIT)
    # silu(-50) ~ 0, and crucially not silu(-10) ~ -4.5e-4.
    assert got.abs().max() < 1e-6
    symmetric = F.silu(gate.clamp(-_LIMIT, _LIMIT)) * up
    assert (got - symmetric).abs().max() > 1e-5


def test_split_order_is_gate_then_up(clamped_swiglu):
    """`SiluAndMul` puts gate first; the split must agree or the halves swap."""
    gate = torch.zeros(2, 3)  # silu(0) == 0 -> output must be 0
    up = torch.full((2, 3), 5.0)
    got = clamped_swiglu(torch.cat([gate, up], dim=-1), _LIMIT)
    assert got.abs().max() == 0.0
    # Swapped would give silu(5)*0 == 0 too, so also check the reverse case.
    gate2 = torch.full((2, 3), 5.0)
    up2 = torch.zeros(2, 3)
    assert clamped_swiglu(torch.cat([gate2, up2], dim=-1),
                          _LIMIT).abs().max() == 0.0
    # ...and an asymmetric case that distinguishes them.
    g3 = torch.full((2, 3), 1.0)
    u3 = torch.full((2, 3), 3.0)
    got3 = clamped_swiglu(torch.cat([g3, u3], dim=-1), _LIMIT)
    torch.testing.assert_close(got3, F.silu(g3) * u3)


def test_tower_config_sources():
    """The two values GlmOcr reads from the wrong place.

    Skipped where torchvision cannot load; the assertions are about
    construction, which needs the real classes.
    """
    # `importorskip` is not enough: torchvision imports fine while its
    # compiled ops fail to register against a CPU-only torch, and the failure
    # surfaces much later as `operator torchvision::nms does not exist`.
    try:
        from vllm_torchtpu.models.vllm.glm5_next.vision import \
            Glm5NextVisionTransformer  # noqa: F401
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"vision tower not importable here: {exc}")

    import inspect
    src = inspect.getsource(Glm5NextVisionTransformer)
    # context_dim must come from the vision config, not text_config
    # (12288 vs the checkpoint's 10240).
    assert "projection_intermediate_size" in src
    assert "text_config.intermediate_size" not in src
    # norm_eps must be read explicitly off the vision config.
    assert "vision_config" in src and "rms_norm_eps" in src
