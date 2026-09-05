# Bug report: `streamindex_topk` uses the wrong causal convention for DeepSeek-V4

**Component:** `src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py`
(the TPU lightning-indexer top-k kernel)
**Affects:** DeepSeek-V4 CSA (`compress_ratio == 4`) — **prefill, chunked-prefill
and speculative-decode rows only** (`q_len > 1`).
**Does not affect:** pure decode rows at any ratio; DeepSeek-V3.2 / GLM-5.2 DSA
(`compress_ratio == 1`); GLM-5.3 kpool (opts into the correct convention).
**Found:** while porting the GLM-5.3-Flash kpool indexer, which needs the
opposite convention and so had to establish which one the kernel implements.
**Status:** written up here only. Nothing has been filed externally — no PR, no
issue, no upstream comment. Route this to whoever owns DeepSeek-V4/CSA on TPU.

---

## Summary

`streamindex_topk` decides which compressed KV rows a query may score. It counts
a compressed row as visible **as soon as the query enters it**:

```
visible_rows(p) = p // compress_ratio + 1        # the TPU kernel, today
```

Upstream vLLM — including the exact tag the TPU pods import, `v0.27.0` — counts a
compressed row as visible only once **its last token** is visible:

```
visible_rows(p) = (p + 1) // compress_ratio      # upstream vLLM, everywhere
```

At `compress_ratio == 1` the two are the same expression, which is why this has
never been noticed: the only production caller before DeepSeek-V4 ran at ratio 1.
At `compress_ratio == 4` they differ for three query positions in every four, and
the extra row a query is handed spans raw tokens **up to and beyond the query's
own position**.

This was originally written up as a *semantic difference between DeepSeek-V4
and GLM-5.3*. It is not. Both models want `floor((p+1)/ratio)`. The TPU kernel
is off by one against its own reference.

---

## 1. What the TPU kernel does

Three sites carry the predicate. They now all derive it from two shared helpers,
so the convention lives in one place — but the *default* is still the old one:

| site | what it produces | line (today) |
| --- | --- | --- |
| Pallas causal mask (`compute_scores`, fed from the pipeline body) | inclusive max visible row index | `streamindex_topk.py:464-466`, consumed at `:422-423` |
| `_effective_row_lengths` — the row length handed to `sparsecore_topk` | count of visible rows | `:576`, `:604-608` |
| `_fast_path_max_valid_idx` — the early-exit path | inclusive max visible row index | `:613`, `:643-647` |

All three call `_visible_compressed_rows(p, r, causal_row_offset)`
(`:89-102`), which is `(p + r - off) // r`. At the default `causal_row_offset = 0`
that is exactly `p // r + 1` — the historical behaviour, bit for bit.

The count is then clamped by `kv_len = seq_len // compress_ratio`
(`:200`, `:604`). **That clamp is the entire reason this bug has never produced a
visible symptom**, and it is also the reason it is not benign everywhere. See §4.

## 2. What upstream vLLM does

Checked against `v0.27.0` — the tag the TPU pods actually import
(`docs/vllm-kv-cache-api-pin.md`), read out of the local checkout with
`git show v0.27.0:<path>`, not out of `main`.

Every site in upstream's DeepSeek-V4 path uses `(pos + 1) // compress_ratio`:

| what | v0.27.0 citation | expression |
| --- | --- | --- |
| prefill indexer metadata (per-token context length) | `vllm/v1/attention/backends/mla/indexer.py:275-276` | `global_ctx = start_pos + 1 + offset; len_per_token = global_ctx // COMPRESS_RATIO` |
| short-context top-k shortcut | `vllm/models/deepseek_v4/attention.py:82` | `num_compressed = (positions[row] + 1) // COMPRESS_RATIO` |
| sparse-MLA reference | `vllm/models/deepseek_v4/sparse_mla.py:376` | `num_compressed = (position + 1) // compress_ratio` |
| indexer cache utils (two sites) | `vllm/models/deepseek_v4/common/ops/cache_utils.py:692`, `:1103` | `topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)` |
| ROCm variant | `vllm/models/deepseek_v4/amd/rocm.py:85` | `topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)` |

In `indexer.py:275`, `start_pos = uncompressed_seq_len - query_len` and `offset` is
the intra-query index, so `start_pos + offset` is the token's 0-based logical
position `p` and `global_ctx` is `p + 1`. It is `floor((p+1)/r)`, unambiguously.

There is exactly one `pos // COMPRESS_RATIO` in upstream's compressed path —
`vllm/v1/attention/backends/mla/compressor_utils.py:39` — and it is **not** a
counterexample: it is the write-side slot index, evaluated only under
`is_valid = (pos + 1) % COMPRESS_RATIO == 0`, where `pos // r == (pos+1)/r - 1`
identically.

Corroboration on `main` (`60857baa53`, 2026-09-03), for anyone reading a newer
tree: the same two DeepSeek-V4 sites are at
`vllm/v1/attention/backends/mla/indexer.py:360-361` and
`vllm/models/deepseek_v4/attention.py:82`, and Qwen4-exp QSA — a module that does
not exist at v0.27.0 — independently computes
`min((logical_position + 1) // compress_ratio, seq_len // compress_ratio)` at
`vllm/models/qwen4_exp/common/qsa_cache.py:274` and `:502`.

Note also that upstream's short-context shortcut
(`vllm/models/deepseek_v4/attention.py:845`, guarded by
`max_seq_len // compress_ratio <= topk_tokens`) is the direct analogue of this
kernel's `enable_early_exit` fast path — and upstream's version of that shortcut
uses the correct convention while ours does not.

## 3. This repo already disagrees with itself

The strongest evidence that `floor((p+1)/r)` is the intended convention is that
**the TPU port already uses it on the sibling code path**. In
`src/vllm_torchtpu/models/vllm/deepseek_v4/attention.py:614`, the non-CSA (HCA,
`compress_ratio == 128`) branch computes the per-token compressed KV length as

```python
extra = (positions + 1).to(torch.int32) // self.compress_ratio
```

and hands it to the core attention op as `kv_lens_to_attend`
(`layers/vllm/custom_ops/deepseek_v4/deepseek_v4_attention_op.py:95`, `:143`).
So at ratio 128 the port matches upstream; at ratio 4 the indexer's top-k does
not. Same model, same file, two conventions.

## 4. Severity, stated precisely

### Unaffected

- **`compress_ratio == 1`.** `p // 1 + 1 == (p + 1) // 1` for every `p`. This
  covers DeepSeek-V3.2 and GLM-5.2 DSA, i.e. the `VllmTPUIndexer` caller at
  `layers/vllm/custom_ops/mla_attention_op.py:107` (`compression_ratio=1` at
  `:116`). Verified over `p = 0..499`; identical.
- **Pure decode rows (`q_len == 1`), at any ratio.** Here `p = seq_len - 1`, and
  the clamp `min(kv_len, ...)` with `kv_len = seq_len // r` pulls the count back
  to the correct value for every residue of `seq_len mod r`. Verified
  exhaustively for `r = 4`, `L = 1..199`: **zero** rows differ.
- **GLM-5.3 kpool.** Its indexer will call the kernel with
  `causal_row_offset = index_kpool - 1 = 3`, so it gets the correct rule by
  construction, and `_assert_causal_row_offset` refuses anything else at ratio 4.

### Affected

- **DeepSeek-V4 CSA at `compress_ratio == 4`, on any row with `q_len > 1`** —
  prefill, chunked prefill, and speculative decode. The caller is
  `layers/vllm/custom_ops/deepseek_v4/deepseek_v4_indexer.py:83`
  (`compression_ratio=self.compress_ratio`; upstream only constructs
  `DeepseekV4Indexer` under `if self.compress_ratio == 4`,
  `v0.27.0:vllm/models/deepseek_v4/attention.py:284`, so ratio 4 is the only
  value that reaches this kernel from DSv4).
- Three query positions in every four are affected —
  `p mod 4 ∈ {0, 1, 2}` — except where the `kv_len` clamp bites, which is only
  the last few tokens of the sequence.
- **Whether the leaked row is actually *selected* is not left to chance in the
  common case.** Whenever the number of visible rows is `<= k`, top-k returns
  everything visible, so the extra row is selected with certainty. For DSv4,
  `k = config.index_topk`, so every prefill up to roughly `4k` raw tokens is in
  that regime. Above it the extra row merely competes on score — and it is a
  *nearby* row, so it competes well.

### Worked example

`compress_ratio = 4`, a single 32-token prefill, so `kv_len = 8`. Query at
`p = 8`:

| | visible rows | max row index |
| --- | --- | --- |
| TPU today (`causal_row_offset = 0`) | `min(8, (8+4-0)//4) = 3` → rows {0, 1, 2} | 2 |
| upstream / correct (`offset = 3`) | `min(8, (8+1)//4) = 2` → rows {0, 1} | 1 |

Row 2's boundary (last) token is raw token 11. The query at position 8 is handed a
row whose content includes raw tokens 9, 10 and 11 — **three tokens strictly in its
own future**. Under CSA's overlapping window (`overlap=True`,
`width = 2 * compress_ratio = 8`;
`kernels/deepseek_v4/compress_and_store/config.py:76-83`) row 2 actually pools raw
tokens 4..11, so the leak is a pooled mixture that includes the future tokens, not
a single future key.

Measured over the whole 32-token prefill: **24 of 32 query positions** get one
extra row. Reproduction is in §6.

### Why it is silent

Every symptom-generating mechanism is absent:

- the extra index is a **real, in-range** compressed row id, so there is no assert,
  no NaN, no shape error and no out-of-bounds gather;
- the `kv_len` clamp hides it at the final position of every sequence, which is
  the position a decode-only smoke test looks at;
- the sequence lengths in the existing kernel tests are short enough that
  everything visible is selected under either convention, so the *set* difference
  is one row that a set-equality test at ratio 1 cannot see. Before this change no
  test in `tests/kernels/test_dsv4_streamindex_topk.py` exercised the early-exit
  path at any ratio other than 1 (`_ee_both_paths` hard-codes
  `compression_ratio=1`).

The observable consequence is a model that is very slightly *better* than it should
be at prefill time — logits computed with a peek at the next up-to-3 tokens — and
that is inconsistent between prefill and decode for the same token. That is the
failure shape that inflates an eval and cannot be attributed afterwards.

## 5. The fix

Already implemented, already tested, **not yet enabled for DeepSeek-V4**.

`streamindex_topk` now takes a static integer:

> `causal_row_offset: int = 0` — compressed row `j` is visible to raw query
> position `p` iff `j * compression_ratio + causal_row_offset <= p`.

- `causal_row_offset = 0` reproduces today's arithmetic bit for bit (that is the
  regression contract, and it is tested by array-equality against the pre-change
  implementation over randomized `(seq_lens, cu_q_lens, distribution)` at ratios
  1, 4 and 128).
- `causal_row_offset = compression_ratio - 1` is the upstream / GLM rule.
- `_assert_causal_row_offset` (`streamindex_topk.py:65-86`) enforces
  `0 <= causal_row_offset < compression_ratio`, which makes "ratio 1 admits only
  offset 0" structural. Do **not** "fix" a failing ratio-1 test by clamping the
  offset to 0 when `compression_ratio == 1`: at ratio 1 the count is `p + 1 - off`,
  which moves with `off` at every value, and a clamp would bury a silent
  ratio-dependent special case inside a kernel shared by three production models.

**What the DSv4 owner has to do is one line**: pass
`causal_row_offset=compression_ratio - 1` at
`layers/vllm/custom_ops/deepseek_v4/deepseek_v4_indexer.py:83`. Nothing else in
the kernel changes.

It is deliberately *not* done in the change that added the parameter. Flipping it
changes DeepSeek-V4's selected key sets on every prefill row, which is a numerics
change to a shipping model that must be measured against DSv4's own baseline — and
doing it alongside an unrelated model port would make any DSv4 regression
un-attributable. The default stays 0, so adding the parameter is a no-op for DSv4
by construction.

Before flipping it, expect and budget for:

- a changed prefill numeric baseline for DSv4 CSA (the fix *removes* information,
  so eval scores may move down slightly — that is the correct direction);
- the early-exit fast path: it carries the same predicate, and it is currently
  never taken by DSv4 (`enable_early_exit` defaults to `False` at
  `streamindex_topk.py:681` and the DSv4 caller does not pass it), so turning
  early exit on for DSv4 *before* fixing the offset would make things worse, not
  better;
- the write side is unaffected — a compressed row's *content* and slot index are
  unchanged. This is purely about which rows a query is allowed to score.

## 6. Reproduction

Pure CPU, no device, seconds. Against the current tree (which already has
`causal_row_offset`, defaulting to 0):

```python
import jax.numpy as jnp
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    _visible_compressed_rows, _last_visible_compressed_row)

r = 4
# decode rows: the kv_len clamp makes the two conventions agree
assert all(
    min(L // r, int(_visible_compressed_rows(jnp.int32(L - 1), r, 0)))
    == min(L // r, L // r)
    for L in range(1, 200))

# prefill rows: 3 in 4 differ
L, kv = 32, 32 // r
diff = [p for p in range(L)
        if min(kv, int(_visible_compressed_rows(jnp.int32(p), r, 0)))
        != min(kv, (p + 1) // r)]
print(len(diff), "of", L)                       # -> 24 of 32
print(int(_last_visible_compressed_row(jnp.int32(8), 4, 0)),   # -> 2
      int(_last_visible_compressed_row(jnp.int32(8), 4, 3)))   # -> 1
```

Real output from that script:

```
=== decode rows (q_len == 1): today vs upstream, after the kv_len clamp ===
  decode rows differing over L=1..199: 0
=== prefill rows (single seq of L=32, every query position) ===
  positions differing: 24 of 32
   p= 0 today=1 upstream=0 extra_row=0 row_span_raw=(-4, 3)
   p= 4 today=2 upstream=1 extra_row=1 row_span_raw=(0, 7)
   p= 8 today=3 upstream=2 extra_row=2 row_span_raw=(4, 11)
  p=8 mask index today: 2 | at offset=3: 1
=== ratio 1 (GLM-5.2 DSA): do the two conventions coincide? ===
  identical for p=0..499: True
```

The committed tests for the parameter are in
`tests/kernels/test_dsv4_streamindex_topk.py` (the `causal_row_offset` block,
device-only) and `tests/models/vllm/test_dsv4_csa_smoke.py` (CPU).

## 7. What is not established here

- **No device run.** Everything above is arithmetic on the kernel's own pure-jnp
  helpers plus source reading. I have not measured what the fix does to DeepSeek-V4
  output quality, and I cannot: that needs a DSv4 checkpoint on a TPU host, which
  this branch's rung does not include.
- **Magnitude, not existence.** "How much does the leak change DSv4 prefill logits"
  is unanswered. The existence of the leak is arithmetic; its numerical size
  depends on the learned scores.
- **Ratio 128 (HCA) never reaches this kernel**, so the fix's scope really is CSA.
  I read that from `if self.compress_ratio == 4` at
  `v0.27.0:vllm/models/deepseek_v4/attention.py:284` and from the fact that the
  compressor path never calls `streamindex_topk`; I did not run an HCA config to
  confirm it dynamically.
- Whether upstream's GPU DeepSeek-V4 has a *separate* off-by-one of its own is out
  of scope. Six independent upstream sites agree with each other, which is the
  claim being made here — not that upstream is right in some absolute sense.
