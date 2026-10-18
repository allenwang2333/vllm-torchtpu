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
"""TPU_MLA_ABSORB_TOKEN_MAJOR torch bridge vs the bmm code it replaces.

The incumbent is the flag-off code of `TPUMLAAttentionImpl.forward` /
`project_uv_heads`, verbatim, compiled by the TPU backend -- the graph the
served model runs. GLM-5.2 per-rank shapes, decode bucket 128 and a 4096-token
prefill chunk, bf16 and fp8 (QUANTIZE_ON_LOAD_PREFIXES=attn) weights.
"""

import pytest

torch = pytest.importorskip("torch")

from vllm_torchtpu.layers.adapter import mla_head_matmul  # noqa: E402

pytestmark = pytest.mark.tpu_test

H, NOPE, ROPE, LORA, V = 64, 192, 64, 512, 256


def _fp8_weight(h, k, n, seed):
    from vllm_torchtpu.layers.core.quantization import quantize_tensor

    g = torch.Generator().manual_seed(seed)
    w = torch.randn(h, k, n, generator=g) * 0.05
    wq, ws = quantize_tensor(w, torch.float8_e4m3fn, axis=1)
    return wq.to("tpu"), ws.to(torch.bfloat16).to("tpu")


def _bf16_weight(h, k, n, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(h, k, n, generator=g) * 0.05).to(torch.bfloat16).to("tpu"), None


def _w_uk_bmm(q_nope, w_uk_t_src, w_uk_t_scale):
    # attention.py, flag off, verbatim.
    input_dtype = q_nope.dtype
    q_nope_t = q_nope.transpose(0, 1)
    w_uk_t = (
        w_uk_t_src.to(q_nope_t.dtype) if w_uk_t_src.dtype != q_nope_t.dtype else w_uk_t_src
    )
    ql_nope = torch.bmm(q_nope_t, w_uk_t)
    if w_uk_t_scale is not None:
        ql_nope = ql_nope * w_uk_t_scale
    return ql_nope.transpose(0, 1).to(input_dtype)


def _project_uv_bmm(partial, w_uv, w_uv_scale, out_dtype):
    # cp_mla_attention.project_uv_heads, flag off, verbatim.
    if out_dtype != torch.float32 and w_uv_scale is None:
        out = torch.bmm(partial.transpose(0, 1), w_uv.to(partial.dtype))
        return out.transpose(0, 1).to(out_dtype)
    out = torch.bmm(partial.transpose(0, 1).to(torch.float32), w_uv.to(torch.float32))
    if w_uv_scale is not None:
        out = out * w_uv_scale.to(torch.float32)
    return out.transpose(0, 1).to(out_dtype)


def _diff(got, ref):
    got, ref = got.cpu(), ref.cpu()
    assert got.shape == ref.shape and got.dtype == ref.dtype
    if torch.equal(got, ref):
        return 0, 0.0
    gi = got.view(torch.int16 if got.element_size() == 2 else torch.int32).long()
    ri = ref.view(torch.int16 if ref.element_size() == 2 else torch.int32).long()
    d = (gi - ri).abs()
    return int(d.max()), float((d > 0).float().mean())


@pytest.fixture(scope="module", autouse=True)
def _ops():
    mla_head_matmul.build_ops()


@pytest.mark.parametrize("tokens", [128, 4096])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("whole_q", [True, False])
def test_w_uk_absorb_matches_compiled_bmm(tokens, fp8, whole_q):
    torch._dynamo.reset()
    g = torch.Generator().manual_seed(tokens)
    q = torch.randn(tokens, H, NOPE + ROPE, generator=g).to(torch.bfloat16).to("tpu")
    w, s = (_fp8_weight if fp8 else _bf16_weight)(H, NOPE, LORA, 1)

    def flag_off(q, w, s):
        q_nope, _ = q.split([NOPE, ROPE], dim=-1)
        return _w_uk_bmm(q_nope, w, s)

    def flag_on(q, w, s):
        q_nope, _ = q.split([NOPE, ROPE], dim=-1)
        return mla_head_matmul.head_matmul(q if whole_q else q_nope, w, s)

    ref = torch.compile(flag_off, fullgraph=True, dynamic=False)(q, w, s)
    got = torch.compile(flag_on, fullgraph=True, dynamic=False)(q, w, s)
    eager = flag_on(q, w, s)
    print(f"W_UK T={tokens} fp8={fp8} whole_q={whole_q}: compiled {_diff(got, ref)} "
          f"eager {_diff(eager, ref)}")
    assert _diff(got, ref) == (0, 0.0)
    assert _diff(eager, ref) == (0, 0.0)


@pytest.mark.parametrize("tokens", [128, 4096])
@pytest.mark.parametrize(
    "fp8,out_dtype",
    [(False, torch.bfloat16), (True, torch.bfloat16), (True, torch.float32)],
)
def test_w_uv_projection_matches_compiled_bmm(tokens, fp8, out_dtype):
    torch._dynamo.reset()
    g = torch.Generator().manual_seed(tokens + 1)
    partial = torch.randn(tokens, H, LORA, generator=g).to(torch.bfloat16).to("tpu")
    w, s = (_fp8_weight if fp8 else _bf16_weight)(H, LORA, V, 2)
    narrow = out_dtype != torch.float32 and s is None

    def flag_off(p, w, s):
        return _project_uv_bmm(p, w, s, out_dtype)

    def flag_on(p, w, s):
        return mla_head_matmul.head_matmul(
            p, w, s, round_before_scale=narrow, out_dtype=out_dtype
        )

    ref = torch.compile(flag_off, fullgraph=True, dynamic=False)(partial, w, s)
    got = torch.compile(flag_on, fullgraph=True, dynamic=False)(partial, w, s)
    d = _diff(got, ref)
    print(f"W_UV T={tokens} fp8={fp8} out={out_dtype}: {d}")
    # The f32 path's reference is an f32 x f32 dot of bf16-exact operands; it
    # may sum in another order. Bound it at 1 ulp on a small fraction.
    if narrow:
        assert d == (0, 0.0)
    else:
        assert d[0] <= 1 and d[1] <= 1e-3, d
