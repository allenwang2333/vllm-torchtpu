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
"""DeepSeek-V4 CSA smoke test -- the half of it that runs without a TPU.

Before this file the only DSv4 coverage in the repo was
``tests/kernels/test_dsv4_streamindex_topk.py``, which needs a device (every
test in it fails on CPU with "Unsupported TPU device kind"). So any change to
the shared ``streamindex_topk`` kernel was unreviewable off-device, for the
model with the most to lose from it.

Two things are pinned here:

* **The ratio-4 visible-row arithmetic DSv4 CSA runs today.** This kernel
  counts a compressed row visible as soon as the query *enters* it
  (``pos // 4 + 1``), while upstream vLLM uses ``(pos + 1) // 4`` for DSv4 as
  well. That difference is deliberately preserved -- ``causal_row_offset``
  defaults to 0 -- so the numbers below are the *pre-change baseline*, not an
  endorsement. See ``docs/dsv4-streamindex-causal-off-by-one.md``.

* **The page geometry ``num_kv_pages_per_block`` has to satisfy**, read off
  DSv4's own call site rather than copied into the test. DSv4 gets away with a
  small tuple only because its ``block_size`` is 1024; the same ``(3, 2, 2)``
  is a trace-time ``ValueError`` for a hybrid model that keeps vLLM's default
  ``block_size = 16``, so a second caller has to derive the value rather than
  inherit it.

The generation half needs real weights and a TPU. ``DeepSeek-V4-Flash`` runs
TP=1 x DP=8 on v7x-8 (``scripts/vllm/benchmarking/configs/deepseek-v4-flash-dp8-ep.sh``),
so it is not a single-chip job either.
"""

import json
import os
import pathlib
import re

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: DeepSeek-V4 CSA's compressed-row ratio. Read off
#: ``models/vllm/deepseek_v4/attention.py:244`` /
#: ``layers/vllm/custom_ops/deepseek_v4/deepseek_v4_indexer.py:113``, and the
#: only ratio that reaches ``streamindex_topk`` (128 is the HCA compressor's,
#: and the compressor never calls this kernel).
DSV4_CSA_COMPRESS_RATIO = 4

#: DSv4 does not take vLLM's default block size: ``tpu_platform.py:1150-1157``
#: overrides it with the MLA page size because its pages hold compressed rows.
DSV4_BLOCK_SIZE = 1024

#: GLM-5.3 is hybrid, so ``tpu_platform.py:1159-1163`` leaves ``block_size`` at
#: vLLM's default. Kept here as the contrast that makes the geometry assertion
#: below mean something.
GLM5_3_BLOCK_SIZE = 16

_MODEL = os.environ.get("DSV4_SMOKE_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
_BASELINE = pathlib.Path(
    os.environ.get(
        "DSV4_SMOKE_BASELINE",
        pathlib.Path(__file__).with_name("dsv4_csa_smoke_baseline.json")))

_PROMPTS = (
    "The capital of France is",
    "Write one sentence explaining why the sky appears blue.",
)


def _visible_row_count(seq_len: int, q_len: int, i: int, ratio: int) -> int:
    """Independently written reference for the shipped convention.

    ``|{ j >= 0 : j * ratio <= pos }|``, clamped by the number of compressed
    rows that exist. Written as a set comprehension rather than as the closed
    form so it does not restate the implementation's arithmetic.
    """
    pos = seq_len - q_len + i
    return min(seq_len // ratio,
               len([j for j in range(pos + 1) if j * ratio <= pos]))


def _batch():
    import jax.numpy as jnp

    # seq 0: seq_len 40, q_len 1 (decode).
    # seq 1: seq_len 23, q_len 7 (chunked-prefill / spec row -- the only shape
    # where the off-by-one is a live future-token leak rather than clamped).
    seq_lens = jnp.array([40, 23, 0, 0], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 8, 8, 8], dtype=jnp.int32)
    distribution = jnp.array([1, 1, 2], dtype=jnp.int32)
    rows = [(40, 1, 0)] + [(23, 7, i) for i in range(7)]
    return seq_lens, cu_q_lens, distribution, rows


def test_ratio4_row_lengths_are_the_shipped_convention():
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        _effective_row_lengths

    seq_lens, cu_q_lens, distribution, rows = _batch()
    num_tokens = 10  # 8 real rows + 2 padding rows

    got = np.asarray(
        _effective_row_lengths(seq_lens, cu_q_lens, distribution, num_tokens,
                               4096, DSV4_CSA_COMPRESS_RATIO))
    want = [
        _visible_row_count(sl, ql, i, DSV4_CSA_COMPRESS_RATIO)
        for sl, ql, i in rows
    ] + [0, 0]

    assert got.tolist() == want, (
        "DeepSeek-V4 CSA visible-row counts moved. `causal_row_offset` must "
        "default to 0, which reproduces today's `pos // 4 + 1` bit for bit.")
    # Spelled out so the pre-change baseline is readable without running it.
    # The decode row at position 39 is clamped by kv_len = 40 // 4 = 10.
    # The 7-token row covers positions 16..22 against kv_len = 23 // 4 = 5,
    # and the clamp is what hides the off-by-one from position 20 onward.
    assert want == [10, 5, 5, 5, 5, 5, 5, 5, 0, 0]


def test_ratio4_offset_three_is_the_glm_rule_and_is_strictly_fewer_rows():
    """``causal_row_offset = ratio - 1`` is GLM-5.3 kpool's pool rule.

    Same kernel, same call, one static int: a pool is a candidate only once its
    *last* token is visible. Asserted here rather than only in the GLM tests
    because this is the DSv4-side statement that the offset does something --
    if it silently did nothing, a ``ratio == 1`` clamp (which is forbidden,
    because at ratio 1 the offset is emphatically not a no-op) could have
    crept in unnoticed.
    """
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        _effective_row_lengths

    seq_lens, cu_q_lens, distribution, rows = _batch()
    num_tokens = 10
    off = DSV4_CSA_COMPRESS_RATIO - 1

    got = np.asarray(
        _effective_row_lengths(seq_lens, cu_q_lens, distribution, num_tokens,
                               4096, DSV4_CSA_COMPRESS_RATIO, off))
    want = []
    for sl, ql, i in rows:
        pos = sl - ql + i
        # |{ j : 4j + 3 <= pos }| == floor((pos + 1) / 4), clamped.
        want.append(min(sl // 4, (pos + 1) // 4))
    want += [0, 0]

    assert got.tolist() == want
    baseline = [
        _visible_row_count(sl, ql, i, DSV4_CSA_COMPRESS_RATIO)
        for sl, ql, i in rows
    ]
    assert all(a <= b for a, b in zip(want, baseline))
    assert any(a < b for a, b in zip(want, baseline)), (
        "offset 3 selected the same rows as offset 0 on this batch, so this "
        "test proves nothing -- pick positions that are not all clamped")


def _num_kv_pages_per_block_at_the_dsv4_call_site() -> tuple[int, ...]:
    """Read DSv4's shipped tuple out of its source, not out of this test."""
    src = (_REPO_ROOT / "src/vllm_torchtpu/layers/vllm/custom_ops/"
           "deepseek_v4/deepseek_v4_indexer.py").read_text()
    m = re.search(r"num_kv_pages_per_block=\((\d+),\s*(\d+),\s*(\d+)\)", src)
    assert m, ("could not find `num_kv_pages_per_block=(a, b, c)` in "
               "deepseek_v4_indexer.py -- update this test with the call site")
    return tuple(int(g) for g in m.groups())


def test_dsv4_page_geometry_is_legal_and_glms_would_not_be():
    """``page_size * bkv_p % 128 == 0`` (``streamindex_topk.py:748-753``).

    ``page_size`` here is *compressed records per page*, i.e.
    ``spec.storage_block_size == block_size // compress_ratio``. DSv4 gets away
    with small ``bkv_p`` only because its ``block_size`` is 1024. The second
    half of this test is the load-bearing one for any second caller: the same
    tuple cannot build for a hybrid model at ``block_size = 16``, so the value
    has to be derived rather than inherited.
    """
    bkv = _num_kv_pages_per_block_at_the_dsv4_call_site()

    dsv4_page = DSV4_BLOCK_SIZE // DSV4_CSA_COMPRESS_RATIO
    assert dsv4_page == 256
    for p in bkv:
        assert dsv4_page * p % 128 == 0, (
            f"DSv4's own num_kv_pages_per_block={bkv} no longer satisfies the "
            f"kernel's page constraint at page_size={dsv4_page}")

    glm_page = GLM5_3_BLOCK_SIZE // DSV4_CSA_COMPRESS_RATIO
    assert glm_page == 4
    assert [glm_page * p for p in bkv] == [12, 8, 8]
    assert all(glm_page * p % 128 != 0 for p in bkv), (
        "DSv4's num_kv_pages_per_block became legal at GLM's page size of 4. "
        "If that is real, the derivation needs revisiting; if it is not, "
        "someone changed the tuple.")
    # What GLM has to use instead: at page_size 4 the constraint forces every
    # element to be a multiple of 32.
    assert min(p for p in range(1, 129) if glm_page * p % 128 == 0) == 32


def _tpu_is_real() -> bool:
    try:
        import torch_tpu
    except ImportError:
        return False
    return not getattr(torch_tpu, "_is_cpu_stub", False)


@pytest.mark.skipif(
    not _tpu_is_real() or os.environ.get("RUN_GENERATION_SMOKE") != "1",
    reason=("needs a real TPU and RUN_GENERATION_SMOKE=1. DeepSeek-V4-Flash "
            "runs TP=1 x DP=8 on v7x-8 per "
            "scripts/vllm/benchmarking/configs/deepseek-v4-flash-dp8-ep.sh."),
)
def test_dsv4_csa_greedy_generation_is_stable():
    """Greedy decode through the DSv4 CSA indexer, diffed against a record.

    Same three-part bar as the GLM-5.2 smoke: non-empty, factually right on the
    one prompt with a single correct continuation, reproducible, and equal to
    the recorded baseline if one exists (the first run writes it).
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=_MODEL,
        max_model_len=int(os.environ.get("DSV4_SMOKE_MAX_LEN", 9216)),
        tensor_parallel_size=int(os.environ.get("DSV4_SMOKE_TP", 1)),
        quantization="deepseek_v4_fp8",
        kv_cache_dtype="fp8_e4m3",
        trust_remote_code=True,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=32)
    first = [o.outputs[0].text for o in llm.generate(list(_PROMPTS), params)]
    second = [o.outputs[0].text for o in llm.generate(list(_PROMPTS), params)]

    for prompt, text in zip(_PROMPTS, first):
        print(f"[dsv4-csa] {prompt!r} -> {text!r}")
        assert text.strip(), f"empty completion for {prompt!r}"
    assert "Paris" in first[0], (
        f"DeepSeek-V4 did not answer the capital of France: {first[0]!r}")
    assert first == second, "greedy decode is not reproducible run to run"

    if _BASELINE.exists():
        recorded = json.loads(_BASELINE.read_text())
        assert recorded["model"] == _MODEL
        assert recorded["completions"] == first, (
            "DSv4 CSA greedy output changed against the recorded step-1 "
            f"baseline in {_BASELINE}")
    else:
        _BASELINE.write_text(
            json.dumps(
                {
                    "model": _MODEL,
                    "prompts": list(_PROMPTS),
                    "completions": first,
                },
                indent=2))
        print(f"[dsv4-csa] wrote a new baseline to {_BASELINE}")
