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
"""The indexer query path: projection, fp8 quantization, scale folding.

The reference scores ``relu(q @ pool_keys^T * softmax_scale)`` and only then
applies the per-head weights (``modeling_glm5_next.py:824-830``).
``streamindex_topk`` applies the per-head weight **after** its own ReLU
(``streamindex_topk.py:333-335``), so the producer has to fold
``q_scale * softmax_scale * n_head**-0.5`` into the weights. The rearrangement
is exact because ``relu(a*x) == a*relu(x)`` for ``a > 0`` and all three factors
are strictly positive.

Why this needs a real test rather than a comment: dropping ``softmax_scale`` or
``n_head**-0.5`` changes **nothing** observable -- they are global positive
constants and top-k is invariant to a positive scalar -- so every small
selection test stays green. ``q_scale`` is per ``(token, head)`` and is not
invariant: dropping it reweights heads against one another per token, the
head-weighted sum ranks pools by a different functional, and the only symptom
is degraded long-context retrieval, i.e. the exact number this project exists
to measure.

**A test that looks right and is backwards.** "Multiply ``q`` by a per-head
positive constant vector and assert the selected set is unchanged; this holds
only if ``q_scale`` is folded" is the obvious check, and it is inverted. Scaling head ``h`` of ``q`` by ``c_h`` leaves ``q_fp8``
bit-identical (fp8 quantization is scale-invariant) and scales ``q_scale`` by
``c_h``; so a *folded* producer's scores change by ``c_h`` per head -- exactly
as the reference's do -- while an *unfolded* producer's scores do not change at
all. "The selected set is unchanged" is therefore the signature of the **bug**,
not of the fix. What is true, and what
:func:`test_per_head_scaling_is_tracked_only_when_q_scale_is_folded` asserts,
is that the device selection *tracks the reference* under per-head scaling, and
that an unfolded producer is provably blind to it.
"""

import types

import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.glm5_next.indexer import Glm5NextKpoolIndexer

from .kpool_oracle import pool_scores, pooled_keys, select_pool_ids

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

HIDDEN = 64
Q_LORA = 32
N_HEAD = 4
HEAD_DIM = 128
INDEX_TOPK = 32
INDEX_KPOOL = 4
SELECT_K = INDEX_TOPK // INDEX_KPOOL  # 8 pools out of 16 -- top-k must choose
SEQ_LEN = 64

#: Powers of two, so scaling ``wq_b``'s rows is exact in binary floating point
#: and ``q_fp8`` comes out bit-identical. A non-power-of-two would leave a
#: handful of last-bit differences and turn a sharp assertion into a fuzzy one.
PER_HEAD_SCALE = (0.25, 1.0, 4.0, 64.0)


@pytest.fixture(autouse=True)
def _fake_tp_group(monkeypatch):
    from vllm.distributed import parallel_state
    monkeypatch.setattr(parallel_state,
                        "_TP",
                        types.SimpleNamespace(rank_in_group=0, world_size=1),
                        raising=False)


def _config():
    return types.SimpleNamespace(
        hidden_size=HIDDEN,
        q_lora_rank=Q_LORA,
        index_n_heads=N_HEAD,
        index_head_dim=HEAD_DIM,
        index_topk=INDEX_TOPK,
        index_kpool=INDEX_KPOOL,
        index_kpool_always_select_tail=True,
        qk_rope_head_dim=0,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        indexer_types=["full"] * 4,
    )


def _build_indexer(per_head_scale=None, seed=7):
    """A float64 indexer with reproducible weights.

    float64 throughout so the only lossy step in the pipeline is the fp8
    quantization under test; any disagreement is then attributable to the
    algebra rather than to accumulated bf16 rounding.
    """
    indexer = Glm5NextKpoolIndexer(_config(),
                                   None,
                                   prefix="model.layers.3.self_attn.indexer",
                                   layer_idx=3,
                                   params_dtype=torch.float64).eval()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in indexer.parameters():
            param.copy_(
                torch.randn(param.shape, generator=gen, dtype=torch.float64) *
                HIDDEN**-0.5)
        # LayerNorm ships weight=1; keep it near 1 rather than near 0.
        indexer.k_norm.weight.add_(1.0)
        if per_head_scale is not None:
            for head, scale in enumerate(per_head_scale):
                rows = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
                indexer.wq_b.weight[rows].mul_(scale)
    return indexer


def _inputs(seed=11):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(SEQ_LEN, HIDDEN, generator=gen, dtype=torch.float64)
    q_c = torch.randn(SEQ_LEN, Q_LORA, generator=gen, dtype=torch.float64)
    return hidden, q_c


def _run(indexer, hidden, q_c):
    """Both arrangements of the same score, from one set of projections.

    Returns the device arrangement (folded weights over the fp8 queries), the
    reference arrangement (``pool_scores`` over the **dequantized** queries),
    and a deliberately unfolded mutant.

    ``q_scale`` is recovered from ``q`` rather than from the returned weights
    on purpose. Recovering it from the weights would make the comparison blind
    to a dropped *global* factor -- the error would cancel on both sides --
    which is precisely the class of omission this test has to catch.
    ``abs_max / dtype_max`` is ``quantize_tensor``'s documented contract
    (``layers/common/quantization.py:257-263``).
    """
    with torch.no_grad():
        k, gate = indexer.project_keys(hidden)
        q_fp8, weights = indexer.project_query(hidden, q_c)
        q_exact = indexer.wq_b(q_c)[0].view(SEQ_LEN, N_HEAD, HEAD_DIM)
        weights_raw = indexer.weights_proj(hidden)[0]

    q_scale = (q_exact.abs().amax(-1, keepdim=True) / FP8_MAX).numpy()
    q_fp8_np = q_fp8.to(torch.float64).numpy()
    q_dequant = q_fp8_np * q_scale

    pool_key = pooled_keys(k.numpy(),
                           gate.numpy(),
                           indexer.index_kpool_compress_ape.detach().numpy(),
                           index_kpool=INDEX_KPOOL)

    reference = pool_scores(q_dequant,
                            pool_key,
                            weights_raw.numpy(),
                            softmax_scale=indexer.softmax_scale,
                            n_heads=N_HEAD)

    per_head = np.maximum(np.einsum("thd,pd->thp", q_fp8_np, pool_key), 0.0)
    device = np.einsum("th,thp->tp", weights.numpy(), per_head)
    # The mutant: everything except the per-(token, head) q_scale.
    unfolded_weights = weights_raw.numpy() * (indexer.softmax_scale *
                                              N_HEAD**-0.5)
    unfolded = np.einsum("th,thp->tp", unfolded_weights, per_head)

    return {
        "device": device,
        "reference": reference,
        "unfolded": unfolded,
        "q_exact": q_exact.numpy(),
        "q_dequant": q_dequant,
        "q_fp8": q_fp8_np,
        "weights": weights.numpy(),
        "weights_raw": weights_raw.numpy(),
        "q_scale": q_scale,
    }


def _select(scores):
    return select_pool_ids(scores,
                           seq_len=SEQ_LEN,
                           index_kpool=INDEX_KPOOL,
                           select_k=SELECT_K)


def _rows_differing(a, b):
    return sum(set(x) != set(y) for x, y in zip(a, b))


# --- shape and dtype contract -------------------------------------------


def test_query_is_fp8_per_head_and_weights_keep_the_activation_dtype():
    """The op takes weights in the activation dtype, not fp32.

    ``q_scale`` is fp32, so the fold promotes; ``mla_attention_op.py:276-280``
    casts back for exactly this reason. Handing the kernel fp32 weights against
    a bf16 activation dtype is a dtype mismatch at the pallas boundary, not a
    numerical one -- but it is the kind of thing that only shows up on device.
    """
    indexer = Glm5NextKpoolIndexer(_config(),
                                   None,
                                   prefix="p",
                                   layer_idx=3,
                                   params_dtype=torch.bfloat16).eval()
    hidden, q_c = _inputs()
    with torch.no_grad():
        q_fp8, weights = indexer.project_query(hidden.to(torch.bfloat16),
                                               q_c.to(torch.bfloat16))
    assert q_fp8.shape == (SEQ_LEN, N_HEAD, HEAD_DIM)
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert weights.shape == (SEQ_LEN, N_HEAD)
    assert weights.dtype == torch.bfloat16


def test_no_rope_is_applied_and_none_is_needed():
    """GLM-5.3's indexer takes no ``position_embeddings`` at all (§2.3).

    ``project_query`` has no ``positions``/``rotary_emb`` parameter, which is
    the structural version of "we skip RoPE": there is nothing to forget to
    pass. Pinned so nobody reintroduces the DeepSeek split.
    """
    import inspect

    params = inspect.signature(Glm5NextKpoolIndexer.project_query).parameters
    assert set(params) == {"self", "hidden_states", "q_c"}


# --- the algebra --------------------------------------------------------


def test_the_fold_reproduces_the_reference_score_exactly():
    """The two arrangements agree to 1e-6 on the same queries.

    Measured max relative disagreement: 5.2e-08, which is the ULP difference
    between this test's ``abs_max / dtype_max`` and ``quantize_tensor``'s
    reciprocal-multiply, not an error in the rearrangement.
    """
    indexer = _build_indexer()
    got = _run(indexer, *_inputs())

    # The recovered fp8 scale really is the one the producer used.
    dequant_error = (np.abs(got["q_dequant"] - got["q_exact"]).max() /
                     np.abs(got["q_exact"]).max())
    assert dequant_error < 0.1, dequant_error  # e4m3 has 3 mantissa bits

    rel = (np.abs(got["device"] - got["reference"]).max() /
           np.abs(got["reference"]).max())
    assert rel < 1e-6, f"folded score disagrees with the reference by {rel}"
    assert _select(got["device"]) == _select(got["reference"])


@pytest.mark.parametrize("dropped,factor", [
    ("softmax_scale", HEAD_DIM**-0.5),
    ("n_head**-0.5", N_HEAD**-0.5),
])
def test_dropping_a_global_factor_is_visible_in_the_score(dropped, factor):
    """...and invisible in the selection. Both halves matter.

    The score comparison is what catches a missing global constant; the
    *selection* cannot, because top-k is invariant to a positive scalar. This
    pins that asymmetry so nobody "simplifies"
    ``test_the_fold_reproduces_the_reference_score_exactly`` down to a set
    comparison and loses the only assertion that sees these two factors.
    """
    del dropped
    indexer = _build_indexer()
    got = _run(indexer, *_inputs())
    mutant = got["device"] / factor

    rel = (np.abs(mutant - got["reference"]).max() /
           np.abs(got["reference"]).max())
    assert rel > 0.1, rel
    assert _select(mutant) == _select(got["reference"])


def test_q_scale_is_actually_folded_into_the_weights():
    """The folded weight equals ``weights_proj(h) * q_scale * sm * n^-0.5``.

    The folded expression, checked against an independently
    recovered ``q_scale``.
    """
    indexer = _build_indexer()
    got = _run(indexer, *_inputs())
    expected = (got["weights_raw"] * got["q_scale"][..., 0] *
                indexer.softmax_scale * N_HEAD**-0.5)
    np.testing.assert_allclose(got["weights"], expected, rtol=1e-6)
    # And q_scale is not a constant across heads, which is what makes it
    # impossible to absorb into a global factor.
    spread = got["q_scale"].max(axis=1) / got["q_scale"].min(axis=1)
    assert spread.max() > 1.05, spread.max()


# --- the cheap discriminating test --------------------------------------


def test_per_head_scaling_is_tracked_only_when_q_scale_is_folded():
    """Scale head ``h`` of ``q`` by ``c_h``; the selection must follow.

    The four things asserted, in the order they make the argument:

    1. ``q_fp8`` is bit-identical across the two runs -- fp8 quantization
       normalizes each ``(token, head)`` row, so *all* of the per-head scale
       information lives in ``q_scale`` and nowhere else.
    2. The reference's own selection **does** change, on 24 of 64 rows here.
       Without this the rest is vacuous.
    3. The folded device selection tracks the reference at both scalings.
    4. The unfolded mutant's scores are *byte-identical* across the two
       scalings -- it cannot see the change at all -- and it therefore
       disagrees with the reference on all 24 rows.
    """
    hidden, q_c = _inputs()
    flat = _run(_build_indexer(), hidden, q_c)
    scaled = _run(_build_indexer(PER_HEAD_SCALE), hidden, q_c)

    assert np.array_equal(flat["q_fp8"], scaled["q_fp8"])

    ref_flat, ref_scaled = _select(flat["reference"]), _select(
        scaled["reference"])
    changed = _rows_differing(ref_flat, ref_scaled)
    assert changed > 0, (
        "per-head scaling did not change the reference selection at all, so "
        "this test proves nothing; pick a stronger scale vector")

    assert _select(flat["device"]) == ref_flat
    assert _select(scaled["device"]) == ref_scaled

    # The mutant is blind: identical scores, hence identical selection.
    np.testing.assert_array_equal(flat["unfolded"], scaled["unfolded"])
    blind = _rows_differing(_select(scaled["unfolded"]), ref_scaled)
    assert blind == changed, (
        f"the unfolded mutant disagreed with the reference on {blind} rows "
        f"but the scaling moved {changed}; the test is not discriminating "
        "what it claims to")
