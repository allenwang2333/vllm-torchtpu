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
"""CPU ground truth for the GLM-5.3-Flash k-pool DSA indexer.

This is a numpy transcription of one DSA layer's *entire* lightning indexer,
written the way the **TPU** computes it rather than the way transformers does:

* one pooled row per **complete** pool, built from that pool's own
  ``index_kpool`` ``(k, gate)`` pairs plus the ``[index_kpool, index_head_dim]``
  absolute-position embedding, with a per-channel softmax over the pool's slots;
* pool-space scoring with the pool-granular causal predicate
  ``index_kpool * j + index_kpool - 1 <= p``;
* top-``select_k`` over the candidate prefix, ``-1`` **suffix** padded (the
  contract ``sparsecore_topk`` documents at ``sparsecore_topk.py:25-28``);
* expansion ``pool_id * index_kpool + [0 .. index_kpool - 1]`` written as a
  contiguous prefix, then the always-selected tail at column
  ``index_kpool * n_selected``, then ``-1`` to ``out_width``.

Everything here is deliberately **pure numpy in float64** so it can serve as the
reference the jnp ops in ``src/vllm_torchtpu/kernels/glm5_kpool/`` are diffed
against, and so no part of it can accidentally inherit a bug from the code under
test.

Verified against the real transformers implementation
(``Glm5NextTextIndexer`` in
``transformers/models/glm5_next/modeling_glm5_next.py``) by
``test_kpool_oracle.py``. Line citations below are into that file.

Semantics confirmed by reading the reference (do not "fix" these into a
difference):

* ``k = LayerNorm(wk(h))`` with ``eps=1e-6`` **and bias** (``:763``, ``:798``);
  ``gate = F.linear(h, index_kpool_compress_gate)`` (``:800``).
* pooling weights are ``softmax_over_slots(gate + ape)`` **per channel**
  (``:962-967``); the reference downcasts the fp32 probabilities to the key
  dtype before the ``index_kpool``-term sum (``:964-966``) -- this oracle
  accumulates in float64 and is therefore *more* accurate, so an exact
  elementwise diff against a bf16/fp32 reference will never close.
* a pool is a candidate iff it is **complete** (``grouped_valid_keys.all(-1)``
  at ``:958``) **and** its **last** token is visible (``pool_end =
  pool_indices[..., -1]`` at ``:833-839``). Those two together give exactly
  ``min(seq_len // index_kpool, (p + 1) // index_kpool)`` candidates for query
  position ``p`` -- the ``floor((p+1)/kpool)`` convention, **not**
  ``floor(p/kpool) + 1``. ``test_kpool_oracle.py::test_candidate_count_matches_floor_p_plus_1``
  pins this against the reference rather than assuming it.
* the tail is ``tail_count = (p + 1) % index_kpool`` raw ids starting at
  ``tail_start = (p + 1) - tail_count`` (``:998-1015``, with ``first_key == 0``
  under vLLM's unpadded ragged layout).
* scoring is ``relu(q @ pool_key.T * softmax_scale)`` per head, then a per-head
  weighted sum with ``weights = weights_proj(h) * n_heads**-0.5`` (``:825-830``).

The reference is **not** prefix-contiguous: it writes the tail at the fixed
column ``index_kpool * select_k`` regardless of how many pools were actually
valid, so a query with fewer than ``select_k`` candidates emits ``-1`` holes
*before* the tail. ``kernels/mla/sparse/kernel.py:209`` masks by column
**position**, so those holes would gather decoy rows at full weight. That is the
entire reason ``expand_pools_with_tail`` exists, and
``test_kpool_oracle.py`` asserts the reference fails the prefix property
positively rather than leaving it as a comment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# The real zai-org/GLM-5.3-Flash text config, recorded so a test can pin them.
REAL_INDEX_KPOOL = 4
REAL_INDEX_TOPK = 2048
REAL_INDEX_HEAD_DIM = 128
REAL_INDEX_N_HEADS = 32
REAL_INDEX_KPOOL_ALWAYS_SELECT_TAIL = True

# ``index_topk + index_kpool - 2``: the largest query position at which the
# selection is provably the identity (every causal token, nothing more). See
# ``identity_bound``.
REAL_IDENTITY_BOUND = REAL_INDEX_TOPK + REAL_INDEX_KPOOL - 2  # 2050


@dataclass(frozen=True)
class KpoolParams:
    """The indexer's static shape/geometry contract."""

    index_kpool: int = REAL_INDEX_KPOOL
    index_topk: int = REAL_INDEX_TOPK
    index_head_dim: int = REAL_INDEX_HEAD_DIM
    index_n_heads: int = REAL_INDEX_N_HEADS
    always_select_tail: bool = REAL_INDEX_KPOOL_ALWAYS_SELECT_TAIL

    def __post_init__(self):
        assert self.index_kpool >= 1
        assert self.index_topk % self.index_kpool == 0, (
            "index_topk must be a whole number of pools; "
            f"got {self.index_topk} % {self.index_kpool} != 0")

    @property
    def select_k(self) -> int:
        """The pool-space budget -- ``streamindex_topk``'s ``k``."""
        return self.index_topk // self.index_kpool

    @property
    def softmax_scale(self) -> float:
        """``modeling_glm5_next.py:765``."""
        return self.index_head_dim**-0.5

    @property
    def reference_out_width(self) -> int:
        """What ``Glm5NextTextIndexer.forward`` returns (``:866-874``)."""
        if not self.always_select_tail:
            return self.index_topk
        return self.index_topk + self.index_kpool - 1

    @property
    def identity_bound(self) -> int:
        """Largest ``p`` for which the selection is exactly ``range(p + 1)``.

        At query position ``p`` the reference selects
        ``index_kpool * min(select_k, (p+1) // index_kpool)`` pooled tokens plus
        ``(p+1) % index_kpool`` tail tokens. While
        ``(p+1) // index_kpool <= select_k`` that union is ``{0, ..., p}``
        exactly, which holds iff ``p <= index_topk + index_kpool - 2``.
        """
        return self.index_topk + self.index_kpool - 2


def num_complete_pools(seq_len: int, index_kpool: int) -> int:
    """Pools all of whose ``index_kpool`` tokens exist.

    ``streamindex_topk``'s ``kv_len = seq_len // compression_ratio``
    (``streamindex_topk.py:125``) is already exactly this.
    """
    return seq_len // index_kpool


def num_candidate_pools(p: int, seq_len: int, index_kpool: int) -> int:
    """Candidate pools visible to raw query position ``p``.

    ``floor((p + 1) / index_kpool)``, clamped by the complete-pool count. This
    is the value ``_effective_row_lengths`` must produce at
    ``causal_row_offset = index_kpool - 1``.
    """
    return min(num_complete_pools(seq_len, index_kpool),
               (p + 1) // index_kpool)


def pooled_keys(k: np.ndarray,
                gate: np.ndarray,
                ape: np.ndarray,
                *,
                index_kpool: int,
                softmax_dtype: np.dtype | None = None,
                prob_dtype: np.dtype | None = None) -> np.ndarray:
    """``[T, D], [T, D], [kpool, D] -> [T // kpool, D]``.

    ``pool_key[j, c] = sum_m softmax_m(gate[kpool*j + m, c] + ape[m, c])
                             * k[kpool*j + m, c]``

    The softmax runs over the ``kpool`` slot axis **per channel** -- not over
    channels, and not over heads (there are no heads here; ``wk`` is MQA-width).

    Two precision knobs, both defaulting to "more accurate than the reference"
    because the TPU path *is* more accurate and an exact elementwise diff will
    therefore never close:

    Args:
        softmax_dtype: precision of the ``gate + ape`` add and the softmax. The
            reference hard-codes **float32** here via ``.float()``
            (``modeling_glm5_next.py:962``) regardless of the module's dtype, so
            pass ``np.float32`` to reproduce it bit-for-bit-ish. ``None``
            (default) computes in float64.
        prob_dtype: if given, round the probabilities to this dtype before the
            ``index_kpool``-term sum, reproducing the reference's downcast to the
            *key* dtype at ``:964-966`` -- bf16 in the shipped checkpoint, which
            is a far coarser rounding than the float32 softmax above it.
    """
    seq_len, head_dim = k.shape
    assert gate.shape == (seq_len, head_dim)
    assert ape.shape == (index_kpool, head_dim)
    num_pools = num_complete_pools(seq_len, index_kpool)
    if num_pools == 0:
        return np.zeros((0, head_dim), dtype=np.float64)

    used = num_pools * index_kpool
    kg = k[:used].astype(np.float64).reshape(num_pools, index_kpool, head_dim)
    acc = np.float64 if softmax_dtype is None else softmax_dtype
    gg = gate[:used].astype(acc).reshape(num_pools, index_kpool, head_dim)
    logits = gg + ape.astype(acc)[None]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    if prob_dtype is not None:
        probs = probs.astype(prob_dtype)
    return (probs.astype(np.float64) * kg).sum(axis=1)


def pool_scores(q: np.ndarray, pool_key: np.ndarray, weights_raw: np.ndarray,
                *, softmax_scale: float, n_heads: int) -> np.ndarray:
    """``[T, H, D], [P, D], [T, H] -> [T, P]``.

    Mirrors ``modeling_glm5_next.py:825-830`` exactly: ReLU **after** the
    ``softmax_scale`` multiply, then the per-head weighted sum with the global
    ``n_heads ** -0.5``.

    ``weights_raw`` is the bare ``weights_proj(h)`` output. The device path
    folds ``q_scale * softmax_scale * n_heads**-0.5`` into it instead and lets
    ``streamindex_topk`` apply the weights *after* its own ReLU
    (``streamindex_topk.py:333-335``); that rearrangement is exact because
    ``relu(a * x) == a * relu(x)`` for ``a > 0`` and all three factors are
    strictly positive. This oracle keeps the reference's arrangement so the
    folding is something the device path has to *prove*, not something it
    inherits.
    """
    seq_len, n_head, head_dim = q.shape
    assert weights_raw.shape == (seq_len, n_head)
    assert n_head == n_heads
    if pool_key.shape[0] == 0:
        return np.zeros((seq_len, 0), dtype=np.float64)
    assert pool_key.shape[1] == head_dim

    # [T, H, P]
    per_head = np.einsum("thd,pd->thp", q.astype(np.float64),
                         pool_key.astype(np.float64))
    per_head = np.maximum(per_head * softmax_scale, 0.0)
    w = weights_raw.astype(np.float64) * (n_heads**-0.5)
    return np.einsum("th,thp->tp", w, per_head)


def candidate_mask(seq_len: int, num_tokens: int, *,
                   index_kpool: int) -> np.ndarray:
    """``bool[num_tokens, P]``: pool ``j`` is a candidate for query ``p``.

    ``index_kpool * j + index_kpool - 1 <= p``. Because valid pools are the
    prefix ``0 .. num_complete_pools - 1`` and the predicate is monotone in
    ``j``, the candidate set is always a *prefix* of length
    ``num_candidate_pools(p, ...)`` -- which is why ``sparsecore_topk``'s
    ``row_lengths`` argument is a faithful encoding of it.
    """
    num_pools = num_complete_pools(seq_len, index_kpool)
    j = np.arange(num_pools)[None, :]
    p = np.arange(num_tokens)[:, None]
    return index_kpool * j + index_kpool - 1 <= p


def select_pool_ids(scores: np.ndarray, *, seq_len: int, index_kpool: int,
                    select_k: int) -> list[list[int]]:
    """Exact top-``select_k`` over the candidate prefix of every row.

    Returns one ascending list of pool ids per query row (length
    ``min(select_k, num_candidate_pools)``). Order carries no meaning: top-k tie
    breaking is unspecified and differs between torch, XLA and SparseCore, so
    every consumer must compare **sets**. The list is sorted only to make
    failures readable.
    """
    num_tokens = scores.shape[0]
    out: list[list[int]] = []
    for p in range(num_tokens):
        n_cand = num_candidate_pools(p, seq_len, index_kpool)
        take = min(select_k, n_cand)
        if take == 0:
            out.append([])
            continue
        row = scores[p, :n_cand]
        # ``-row`` + stable argsort == descending top-k with a deterministic
        # tie break; callers must not depend on which tie wins.
        chosen = np.argsort(-row, kind="stable")[:take]
        out.append(sorted(int(c) for c in chosen))
    return out


def tail_span(p: int, index_kpool: int) -> tuple[int, int]:
    """``(tail_start, tail_count)`` for query position ``p``.

    ``modeling_glm5_next.py:998-1015`` with ``first_key == 0``:
    ``visible_count = p + 1``, ``tail_count = visible_count % index_kpool``,
    ``tail_start = visible_count - tail_count``.
    """
    visible = p + 1
    tail_count = visible % index_kpool
    return visible - tail_count, tail_count


def expand_with_tail(pool_ids_rows: list[list[int]],
                     *,
                     index_kpool: int,
                     out_width: int,
                     valid_token: np.ndarray | None = None) -> np.ndarray:
    """Pool ids -> raw ids, prefix-contiguous, ``-1`` suffix padded.

    This is the ground truth for ``expand_pools_with_tail``. Query
    positions are assumed to be ``0, 1, ..., T-1``; the ``positions`` argument
    the op takes exists for chunked prefill and decode, and is exercised by the
    op's own tests.
    """
    num_tokens = len(pool_ids_rows)
    out = np.full((num_tokens, out_width), -1, dtype=np.int32)
    for p, pool_ids in enumerate(pool_ids_rows):
        if valid_token is not None and not bool(valid_token[p]):
            continue
        col = 0
        for j in pool_ids:
            for m in range(index_kpool):
                out[p, col] = index_kpool * j + m
                col += 1
        tail_start, tail_count = tail_span(p, index_kpool)
        for m in range(tail_count):
            out[p, col] = tail_start + m
            col += 1
        assert col <= out_width, (
            f"row {p} needs {col} columns but out_width is {out_width}; "
            "out_width must be >= index_topk + index_kpool - 1")
    return out


def kpool_oracle_rows(k: np.ndarray,
                      gate: np.ndarray,
                      ape: np.ndarray,
                      q: np.ndarray,
                      weights_raw: np.ndarray,
                      *,
                      params: KpoolParams,
                      out_width: int | None = None,
                      softmax_dtype: np.dtype | None = None,
                      prob_dtype: np.dtype | None = None) -> np.ndarray:
    """The whole indexer, end to end, in the TPU's data flow.

    Args:
        k: ``[T, D]`` -- already ``LayerNorm(wk(h))``.
        gate: ``[T, D]`` -- ``F.linear(h, index_kpool_compress_gate)``.
        ape: ``[index_kpool, D]`` -- ``index_kpool_compress_ape``.
        q: ``[T, H, D]`` -- ``wq_b(q_resid)`` viewed per head. No RoPE: GLM-5.3
            is NoPE and ``Glm5NextTextIndexer.forward`` does not even accept
            ``position_embeddings`` (``modeling_glm5_next.py:774-779``).
        weights_raw: ``[T, H]`` -- bare ``weights_proj(h)``.

    Returns:
        ``int32[T, out_width]`` raw token ids, ``-1`` suffix padded.
    """
    seq_len = k.shape[0]
    if out_width is None:
        out_width = params.reference_out_width
    pk = pooled_keys(k,
                     gate,
                     ape,
                     index_kpool=params.index_kpool,
                     softmax_dtype=softmax_dtype,
                     prob_dtype=prob_dtype)
    scores = pool_scores(q,
                         pk,
                         weights_raw,
                         softmax_scale=params.softmax_scale,
                         n_heads=params.index_n_heads)
    rows = select_pool_ids(scores,
                           seq_len=seq_len,
                           index_kpool=params.index_kpool,
                           select_k=params.select_k)
    if not params.always_select_tail:
        # Same expansion with the tail suppressed.
        out = np.full((seq_len, out_width), -1, dtype=np.int32)
        for p, pool_ids in enumerate(rows):
            flat = [
                params.index_kpool * j + m for j in pool_ids
                for m in range(params.index_kpool)
            ]
            out[p, :len(flat)] = flat
        return out
    return expand_with_tail(rows,
                            index_kpool=params.index_kpool,
                            out_width=out_width)


def row_id_set(row: np.ndarray) -> set[int]:
    """The non-``-1`` ids of one emitted row, as a set."""
    return {int(v) for v in np.asarray(row).reshape(-1) if int(v) != -1}


def is_prefix_contiguous(row: np.ndarray) -> bool:
    """True iff every non-``-1`` entry precedes every ``-1`` entry.

    ``kernels/mla/sparse/kernel.py:209`` and ``:339`` mask by column
    **position** against ``kv_lens = sum(topk_indices != -1)`` (``:622``), so a
    row with an interior ``-1`` makes the kernel read past its own valid prefix
    and gather a decoy row at full weight.
    """
    flat = np.asarray(row).reshape(-1)
    n_valid = int((flat != -1).sum())
    return bool((flat[:n_valid] != -1).all())


# ---------------------------------------------------------------------------
# Ground truth from the real transformers implementation.
#
# Run in a **subprocess**, always, and this is not defensiveness -- it is the
# only arrangement that works when ``glm5_next`` lives in a transformers
# *overlay* rather than in the installed transformers, which is the usual
# situation while a model is new. vLLM imports transformers at collection
# time, so in any pytest session that touches vLLM first an in-process
# ``sys.path.insert`` is too late: ``import transformers`` has already bound
# the installed version and ``transformers.models.glm5_next`` raises
# ``ModuleNotFoundError``. A test that skips on that reads green while proving
# nothing -- the first version of this file did exactly that and silently
# skipped all 67 reference assertions whenever it was collected alongside
# ``tests/models``. Injecting the overlay into ``transformers.models.__path__``
# would "work" but would run the overlay's model code against a different
# core.
#
# The subprocess gets ``PYTHONPATH`` with the overlay first, so it sees a
# coherent transformers, and returns raw arrays. One process serves a whole
# test module (~12 s, dominated by the transformers import).
# ---------------------------------------------------------------------------

#: Directory holding a ``transformers/models/glm5_next/`` source
#: overlay. Unset means "use whatever transformers is installed",
#: which works once glm5_next ships in a release.
TRANSFORMERS_SRC_ENV = "GLM5_NEXT_TRANSFORMERS_SRC"
GLM5_NEXT_TRANSFORMERS_SRC = os.environ.get(TRANSFORMERS_SRC_ENV, "")
_HF_MODELING = (os.path.join(GLM5_NEXT_TRANSFORMERS_SRC, "transformers",
                             "models", "glm5_next", "modeling_glm5_next.py")
                if GLM5_NEXT_TRANSFORMERS_SRC else "")

# Every field of a case is a plain int/bool so a case list is hashable and can
# cross the process boundary as data rather than as code.
CASE_FIELDS = ("index_kpool", "index_topk", "index_head_dim", "index_n_heads",
               "hidden_size", "q_lora_rank", "weight_seed", "seq_len", "n_pad",
               "input_seed")


def glm5_next_source_path() -> str:
    return _HF_MODELING


def glm5_next_source_available() -> bool:
    """Whether a subprocess would be able to import ``glm5_next``."""
    if _HF_MODELING:
        return os.path.exists(_HF_MODELING)
    import importlib.util
    return importlib.util.find_spec(
        "transformers.models.glm5_next") is not None


def make_case(params: KpoolParams,
              *,
              hidden_size: int,
              q_lora_rank: int,
              seq_len: int,
              weight_seed: int = 0,
              input_seed: int = 0,
              n_pad: int = 0) -> dict:
    assert params.always_select_tail, "the reference cases all ship the tail"
    return {
        "index_kpool": params.index_kpool,
        "index_topk": params.index_topk,
        "index_head_dim": params.index_head_dim,
        "index_n_heads": params.index_n_heads,
        "hidden_size": hidden_size,
        "q_lora_rank": q_lora_rank,
        "weight_seed": weight_seed,
        "seq_len": seq_len,
        "n_pad": n_pad,
        "input_seed": input_seed,
    }


def case_key(case: dict) -> tuple:
    return tuple(case[f] for f in CASE_FIELDS)


def run_reference_cases(cases: list[dict]) -> dict[tuple, dict]:
    """Evaluate ``Glm5NextTextIndexer`` on every case, out of process.

    Returns a dict keyed by :func:`case_key`. Each value holds, as numpy
    arrays:

    ``k``, ``gate``, ``q``, ``weights_raw``, ``ape``
        the reference's own projections for the ``seq_len`` **real** tokens, in
        float64, so the oracle can be driven from bit-identical inputs and any
        disagreement is attributable to indexer logic rather than projections.
    ``ref``
        ``int32[n_pad + seq_len, index_topk + index_kpool - 1]`` --
        ``forward``'s output for the left-padded batch.
    ``pooled_ref``, ``pool_indices_ref``, ``pool_valid_ref``
        ``get_pooled_states`` on the unpadded packed states.

    Raises ``RuntimeError`` if the overlay is missing or the worker fails;
    callers should let that fail the test rather than skip, unless the overlay
    genuinely is not installed.
    """
    import os
    import pickle
    import subprocess
    import sys
    import tempfile

    if not glm5_next_source_available():
        raise FileNotFoundError(
            "transformers has no glm5_next; set "
            f"${TRANSFORMERS_SRC_ENV} to a source overlay that "
            "provides transformers/models/glm5_next/")
    if not cases:
        return {}

    env = dict(os.environ)
    here = os.path.dirname(os.path.abspath(__file__))
    # Overlay first: it must win over any installed transformers.
    overlay = ([GLM5_NEXT_TRANSFORMERS_SRC]
               if GLM5_NEXT_TRANSFORMERS_SRC else [])
    env["PYTHONPATH"] = os.pathsep.join([*overlay, here])
    env.pop("VLLM_TORCHTPU_CPU_STUB", None)

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "cases.pkl")
        out_path = os.path.join(tmp, "out.pkl")
        with open(in_path, "wb") as fh:
            pickle.dump(cases, fh)
        proc = subprocess.run(
            [sys.executable,
             os.path.abspath(__file__), in_path, out_path],
            env=env,
            capture_output=True,
            text=True)
        if proc.returncode != 0:
            raise RuntimeError("glm5_next reference worker failed:\n"
                               f"{proc.stdout}\n{proc.stderr}")
        with open(out_path, "rb") as fh:
            results = pickle.load(fh)
    return {case_key(c): r for c, r in zip(cases, results)}


# --- worker side (runs only in the subprocess) -----------------------------


def _worker_inputs(seq_len, hidden_size, q_lora_rank, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(1, seq_len, hidden_size, generator=g, dtype=torch.float64)
    qr = torch.randn(1, seq_len, q_lora_rank, generator=g, dtype=torch.float64)
    return h, qr


def _worker_build_indexer(case):
    """Construct a real ``Glm5NextTextIndexer`` with randomized weights.

    ``__init__`` reads only plain attributes off ``config``, so a
    ``SimpleNamespace`` is enough and no HF config class has to be built.

    The weights are re-randomized from an explicit generator rather than left
    at ``nn.Module`` defaults on purpose: a derived expectation vector taken
    from default-initialised weights is not reproducible, because the values
    depend on exact RNG consumption during module construction. Nothing in
    this file hard-codes a selected-pool vector; every expectation is derived
    from the tensors actually used.
    """
    import types

    import torch
    from transformers.models.glm5_next.modeling_glm5_next import \
        Glm5NextTextIndexer

    hidden_size = case["hidden_size"]
    cfg = types.SimpleNamespace(
        hidden_size=hidden_size,
        index_n_heads=case["index_n_heads"],
        index_head_dim=case["index_head_dim"],
        qk_rope_head_dim=0,  # GLM-5.3 is NoPE.
        index_topk=case["index_topk"],
        q_lora_rank=case["q_lora_rank"],
        index_kpool=case["index_kpool"],
        index_kpool_always_select_tail=True,
    )
    indexer = Glm5NextTextIndexer(cfg, layer_idx=0)
    gen = torch.Generator().manual_seed(case["weight_seed"])

    def fill(t, scale):
        with torch.no_grad():
            t.copy_(
                torch.randn(t.shape, generator=gen, dtype=torch.float64).to(
                    t.dtype) * scale)

    fill(indexer.wq_b.weight, hidden_size**-0.5)
    fill(indexer.wk.weight, hidden_size**-0.5)
    fill(indexer.weights_proj.weight, hidden_size**-0.5)
    # LayerNorm ships weight=1 / bias=0; a bias that is exactly zero cannot
    # distinguish "bias applied" from "bias dropped", so perturb both.
    fill(indexer.k_norm.weight, 0.1)
    with torch.no_grad():
        indexer.k_norm.weight.add_(1.0)
    fill(indexer.k_norm.bias, 0.1)
    fill(indexer.index_kpool_compress_ape, 1.0)
    fill(indexer.index_kpool_compress_gate, hidden_size**-0.5)

    assert isinstance(indexer.k_norm, torch.nn.LayerNorm)
    assert indexer.k_norm.bias is not None
    assert indexer.k_norm.eps == 1e-6
    return indexer.double().eval()


def _worker_one(indexer, case):
    import torch
    import torch.nn.functional as F

    seq_len, n_pad = case["seq_len"], case["n_pad"]
    hidden_size, q_lora_rank = case["hidden_size"], case["q_lora_rank"]
    head_dim = case["index_head_dim"]

    h_real, qr_real = _worker_inputs(seq_len, hidden_size, q_lora_rank,
                                     case["input_seed"])
    pad_h, pad_qr = _worker_inputs(n_pad, hidden_size, q_lora_rank,
                                   case["input_seed"] + 104729)
    h = torch.cat([pad_h, h_real], dim=1)
    qr = torch.cat([pad_qr, qr_real], dim=1)
    mask = torch.cat([
        torch.zeros(1, n_pad, dtype=torch.bool),
        torch.ones(1, seq_len, dtype=torch.bool),
    ],
                     dim=1)

    with torch.no_grad():
        ref = indexer(h, qr, mask, past_key_values=None).cpu().numpy()[0]

        head_shape = (1, seq_len, -1, head_dim)
        q = indexer.wq_b(qr_real).view(head_shape)
        k = indexer.k_norm(indexer.wk(h_real)).view(head_shape).squeeze(2)
        gate = F.linear(h_real, indexer.index_kpool_compress_gate)
        weights_raw = indexer.weights_proj(h_real)

        packed = torch.cat(
            [k, gate, torch.ones(1, seq_len, 1, dtype=torch.float64)], dim=-1)
        pooled, pool_indices, pool_valid = indexer.get_pooled_states(packed)

    def np64(t):
        return t[0].detach().to(torch.float64).numpy()

    return {
        "k": np64(k),
        "gate": np64(gate),
        "q": np64(q),
        "weights_raw": np64(weights_raw),
        "ape":
        indexer.index_kpool_compress_ape.detach().to(torch.float64).numpy(),
        "ref": ref.astype("int32"),
        "pooled_ref": np64(pooled),
        "pool_indices_ref": pool_indices[0].detach().cpu().numpy(),
        "pool_valid_ref": pool_valid[0].detach().cpu().numpy(),
    }


def _worker_main(in_path, out_path):
    import pickle

    with open(in_path, "rb") as fh:
        cases = pickle.load(fh)

    built: dict = {}
    results = []
    for case in cases:
        key = (case["index_kpool"], case["index_topk"], case["index_head_dim"],
               case["index_n_heads"], case["hidden_size"], case["q_lora_rank"],
               case["weight_seed"])
        if key not in built:
            built[key] = _worker_build_indexer(case)
        results.append(_worker_one(built[key], case))

    with open(out_path, "wb") as fh:
        pickle.dump(results, fh)


if __name__ == "__main__":
    import sys as _sys

    _worker_main(_sys.argv[1], _sys.argv[2])
