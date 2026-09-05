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
"""GLM-5.2 DSA smoke test -- the half of it that runs without a TPU.

Before this file there was nothing to compare a shared-kernel change against:
the only GLM-5.2 coverage was kernel/op level
(``tests/layers/vllm/test_tpu_indexer.py``, which cannot run off-device).

GLM-5.2 DSA runs ``streamindex_topk`` at ``compression_ratio=1`` with the
early-exit fast path **on** in production
(``layers/vllm/custom_ops/mla_attention_op.py:57``, passed at ``:121``,
ratio at ``:118``). That makes it the model most exposed to the
``causal_row_offset`` parameter, and in the nastiest way: at ratio 1 the offset
is *not* a no-op for any value but 0 -- the visible-row count is
``p + 1 - off`` -- so an offset that leaks in here gives every query one token
of its own future, with no assert, no NaN, and fluent output.

So the CPU half below pins the **visible-row arithmetic at ratio 1** against an
independently written model of the rule, in both of the forms the kernel needs
(a count for the top-k row lengths, an inclusive max index for the early-exit
path), plus the structural guard that makes ratio 1 admit only offset 0. These
are pure ``jnp`` and run on the JAX CPU backend in milliseconds.

The generation half needs real weights and a TPU; see ``_MODEL``. Note the
checkpoint is 761 GB at TP=1 x DP=16
(``scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh``), i.e. a v7x-16
job -- it does not fit the single-chip rung.
"""

import json
import os
import pathlib

import numpy as np
import pytest

# GLM-5.2 DSA's production indexer configuration. Both values are read off the
# only call site: layers/vllm/custom_ops/mla_attention_op.py:118 and :121.
GLM5_2_COMPRESSION_RATIO = 1
GLM5_2_ENABLE_EARLY_EXIT = True

_MODEL = os.environ.get("GLM5_2_SMOKE_MODEL", "zai-org/GLM-5.2-FP8")
_BASELINE = pathlib.Path(
    os.environ.get(
        "GLM5_2_SMOKE_BASELINE",
        pathlib.Path(__file__).with_name("glm5_2_dsa_smoke_baseline.json")))

#: Fixed greedy prompts. The first has a single correct one-word continuation,
#: which is what makes "the indexer selected the wrong keys" visible at all --
#: a broken key set still produces fluent text.
_PROMPTS = (
    "The capital of France is",
    "Write one sentence explaining why the sky appears blue.",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
)


def _visible_row_count(seq_len: int, q_len: int, i: int, ratio: int) -> int:
    """Reference for the row-length form, written independently of the kernel.

    Number of compressed rows visible to the ``i``-th query token of a
    ``(seq_len, q_len)`` row under the convention this kernel ships today: a
    row is visible as soon as the query *enters* it, i.e.
    ``|{ j >= 0 : j * ratio <= pos }|``, clamped by the number of rows that
    exist (``seq_len // ratio``).
    """
    pos = seq_len - q_len + i
    return min(seq_len // ratio,
               len([j for j in range(pos + 1) if j * ratio <= pos]))


def _batch():
    """A mixed batch: one decode row and one 5-token prefill/spec row."""
    import jax.numpy as jnp

    # seq 0: seq_len 9, q_len 1 (decode). seq 1: seq_len 20, q_len 5.
    seq_lens = jnp.array([9, 20, 0, 0], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 1, 6, 6, 6], dtype=jnp.int32)
    distribution = jnp.array([1, 1, 2], dtype=jnp.int32)
    rows = [(9, 1, 0)] + [(20, 5, i) for i in range(5)]
    return seq_lens, cu_q_lens, distribution, rows


def test_ratio1_row_lengths_match_the_documented_rule():
    """Row-length COUNT at GLM-5.2's ratio, incl. padding rows."""
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        _effective_row_lengths

    seq_lens, cu_q_lens, distribution, rows = _batch()
    num_tokens = 8  # 6 real rows + 2 padding rows

    got = np.asarray(
        _effective_row_lengths(seq_lens, cu_q_lens, distribution, num_tokens,
                               4096, GLM5_2_COMPRESSION_RATIO))

    want = [
        _visible_row_count(sl, ql, i, GLM5_2_COMPRESSION_RATIO)
        for sl, ql, i in rows
    ]
    # Padding rows must contribute no candidates at all.
    want += [0, 0]
    assert got.tolist() == want, (
        "GLM-5.2 DSA visible-row counts moved. At ratio 1 this is exactly "
        "`min(seq_len, pos + 1)`; any `causal_row_offset` other than 0 shifts "
        "every row by that amount, which is a future-token leak with no "
        "exception and fluent output.")
    # Explicit, so a future reader does not have to run it: the decode row at
    # position 8 sees 9 keys; the 5-token row at positions 15..19 sees 16..20.
    assert want == [9, 16, 17, 18, 19, 20, 0, 0]


def test_ratio1_fast_path_index_is_exactly_one_less_than_the_row_length():
    """The early-exit path is an inclusive max INDEX, not a count.

    GLM-5.2 is the model that runs the fast path in production, and a draft of
    the ``causal_row_offset`` change returned one row too many from it at the
    default offset -- one real, in-range, *future* key id per query. The
    invariant is index + 1 == count for every non-padding row, at every clamp.
    """
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        _effective_row_lengths, _fast_path_max_valid_idx)

    seq_lens, cu_q_lens, distribution, rows = _batch()
    num_tokens = 8

    counts = np.asarray(
        _effective_row_lengths(seq_lens, cu_q_lens, distribution, num_tokens,
                               4096, GLM5_2_COMPRESSION_RATIO))
    idx = np.asarray(
        _fast_path_max_valid_idx(seq_lens, cu_q_lens, num_tokens,
                                 GLM5_2_COMPRESSION_RATIO))

    n_real = len(rows)
    assert (idx[:n_real] + 1 == counts[:n_real]).all(), (
        f"fast-path index {idx[:n_real].tolist()} disagrees with row lengths "
        f"{counts[:n_real].tolist()} by something other than one")
    # Padding rows: -1 ("select nothing") in the index form, 0 in the count
    # form. Both mean empty; they are deliberately different spellings.
    assert idx[n_real:].tolist() == [-1, -1]
    assert counts[n_real:].tolist() == [0, 0]


def test_ratio1_admits_only_offset_zero_structurally():
    """`compression_ratio == 1` must admit only `causal_row_offset == 0`.

    Structural, not algebraic: the claim "the offset is a no-op at ratio 1" is
    false (the count is ``p + 1 - off``), so the protection has to be a guard
    that refuses every other value rather than a test that pretends they agree.
    Equally forbidden is a ``ratio == 1 -> offset = 0`` clamp, which would bury
    a silent ratio-dependent special case in a kernel shared by three shipping
    models.

    The guard raises ``ValueError`` and not ``AssertionError`` because
    ``python -O`` deletes ``assert``, and a deleted guard here is a
    future-token leak with fluent output.
    """
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        _effective_row_lengths

    seq_lens, cu_q_lens, distribution, _ = _batch()
    with pytest.raises(ValueError, match="causal_row_offset"):
        _effective_row_lengths(seq_lens, cu_q_lens, distribution, 8, 4096,
                               GLM5_2_COMPRESSION_RATIO, 1)
    # ... and 0 is accepted, so the guard is not simply refusing everything.
    assert _effective_row_lengths(seq_lens, cu_q_lens, distribution, 8, 4096,
                                  GLM5_2_COMPRESSION_RATIO, 0) is not None


def _tpu_is_real() -> bool:
    try:
        import torch_tpu
    except ImportError:
        return False
    return not getattr(torch_tpu, "_is_cpu_stub", False)


@pytest.mark.skipif(
    not _tpu_is_real() or os.environ.get("RUN_GENERATION_SMOKE") != "1",
    reason=(
        "needs a real TPU and RUN_GENERATION_SMOKE=1. GLM-5.2-FP8 is 761 GB: "
        "the recorded production topology is TP=1 x DP=16 on v7x-16 "
        "(scripts/vllm/benchmarking/configs/glm-5.2-fp8-dp16-ep.sh), so this "
        "is not a single-chip job."),
)
def test_glm5_2_dsa_greedy_generation_is_stable():
    """Greedy decode through the DSA indexer, diffed against a recorded run.

    Pass criteria, in order of what they catch:

    1. every prompt produces non-empty output (the run happened);
    2. the factual prompt continues with ``Paris`` -- a wrong key set still
       reads fluently, so a content assertion is the only cheap signal;
    3. two identical greedy requests agree token-for-token (nondeterminism
       here means the indexer is reading state it should not);
    4. if a baseline file exists, the completions match it exactly. The first
       run writes it, so a later run diffs against recorded numbers rather
       than against "it passed".
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=_MODEL,
        max_model_len=int(os.environ.get("GLM5_2_SMOKE_MAX_LEN", 4096)),
        tensor_parallel_size=int(os.environ.get("GLM5_2_SMOKE_TP", 1)),
        kv_cache_dtype="fp8",
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=32)
    first = [o.outputs[0].text for o in llm.generate(list(_PROMPTS), params)]
    second = [o.outputs[0].text for o in llm.generate(list(_PROMPTS), params)]

    for prompt, text in zip(_PROMPTS, first):
        print(f"[glm5.2-dsa] {prompt!r} -> {text!r}")
        assert text.strip(), f"empty completion for {prompt!r}"
    assert "Paris" in first[0], (
        f"GLM-5.2 did not answer the capital of France: {first[0]!r}")
    assert first == second, "greedy decode is not reproducible run to run"

    if _BASELINE.exists():
        recorded = json.loads(_BASELINE.read_text())
        assert recorded["model"] == _MODEL
        assert recorded["completions"] == first, (
            "GLM-5.2 DSA greedy output changed against the recorded step-1 "
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
        print(f"[glm5.2-dsa] wrote a new baseline to {_BASELINE}")
