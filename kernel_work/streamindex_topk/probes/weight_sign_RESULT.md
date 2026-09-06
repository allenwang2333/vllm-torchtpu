# P6 — `indexer_weights` is not non-negative, and E1 is dead on correctness

**probe:** Is `w` (`indexer_weights`, the per-head weight the kernel multiplies the relu'd scores by
at `streamindex_topk.py:333-335`) guaranteed `≥ 0` in production? E1 deletes that multiply entirely,
which is only legal if it can never flip a sign or change the ordering of the top-k.

**config:** a source question, answered from the vLLM / vllm-torchtpu / HuggingFace sources at their
current checkouts, and then **checked against the real GLM-5.2-FP8 checkpoint**. No TPU job. Ran as
a background research agent (`afd9214568401aa6a`) so it did not consume the pod.

**job:** none — this probe needs no hardware. The checkpoint half was read from
`gs://tpu-commons-ci/glm/GLM-5.2-FP8/` with **byte-range reads only**; nothing large was downloaded,
per the standing instruction.

## Answer: `w ≥ 0` is **NOT** guaranteed. It is false, in production, by a coin flip per element.

### The source half — no non-negativity anywhere on the path

Traced `weights_proj`'s output from the GEMM to the kernel argument, on all three code paths
(vllm-torchtpu, upstream vLLM, and the HF reference `modeling_glm_moe_dsa.py`). **There is no
activation, clamp, `relu`, `abs`, `exp`, `sigmoid`, `softplus` or `square` anywhere on it.** Only
positive *scalar* factors are applied, and every one of them is sign-preserving:

| file:line | what it shows |
| --- | --- |
| `mla_attention_op.py:245-248` | `w` is a raw slice of the `weights_proj` GEMM output — no activation |
| `mla_attention_op.py:276-280` | the only scalings are `softmax_scale`, `n_head ** -0.5`, `q_scale` — all positive scalars |
| `deepseek_v4_indexer.py:81-85` | passes `w` straight through to the kernel |
| `streamindex_topk.py:333-335` | **the `relu` is on the *score*; `w` multiplies *after* it** |
| `streamindex_topk.py:458-472` | `prepare_index_weights` only casts and zero-pads — no transform |
| `modeling_glm_moe_dsa.py:192, 239-244` | the author's own reference: `relu` on `scores` (line 240); `weights` (line 243) gets only `n_heads ** -0.5` |

Line 333-335 is the one that matters most. It is easy to read the `relu` in that expression as
guaranteeing non-negativity of the product, and it does not: the `relu` clamps the score, then `w`
multiplies the clamped result, so **a negative `w` produces a negative contribution to the head
sum**. Deleting the multiply would not merely rescale the ranking, it would change which entries are
positive.

### The checkpoint half — measured on the weights that actually ship

Read `model.layers.0` and `model.layers.1`'s `self_attn.indexer.weights_proj.weight` out of the real
**GLM-5.2-FP8** checkpoint:

| tensor | dtype | shape | fraction negative | min | max | rows all-non-negative |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `layers.0.self_attn.indexer.weights_proj.weight` | BF16 | `[32, 6144]` | **0.5002** | −1.20312 | 1.03125 | **0 of 32** |
| `layers.1.self_attn.indexer.weights_proj.weight` | BF16 | `[32, 6144]` | **0.5008** | −3.32812 | 2.81250 | **0 of 32** |

Half the entries are negative, the magnitudes are comparable in both directions, and **not one of
the 32 heads has an all-non-negative weight row**. This is an ordinary zero-centred learned
projection. There is no configuration of the model in which E1's premise holds.

### On the test-input question the brief flagged

`instruments/inputs.py:180` draws `w` from `uniform(−1.5, 1.5)`, and the brief was explicit that this
"is not evidence in either direction" — a test fixture's choice of distribution says what the fixture
author assumed, not what the model does. Correct, and I did not stop there; the checkpoint reading
above is the evidence. But the fixture is worth one note in the other direction:
`tests/kernels/test_dsv4_streamindex_topk.py:307` uses the same `uniform(-1.5, 1.5)` as a
**correctness** test. So an E1 implementation **would** have been caught by the existing test suite
rather than shipping silently. That is a small piece of good news about the yardstick, not about E1.

## Branch selected

**`combined.md` P6 fork (line 692): the `Not guaranteed` branch.**

> *"**Guaranteed `w ≥ 0`:** E1 is legal and C8 becomes a two-for-one."*
> *"**Not guaranteed:** take L13 only, and record E1 as rejected-on-correctness so it is not
> re-proposed."*

E1 is **rejected on correctness**, and this file is the record so that it is not re-proposed.

The branch's other half — *"take L13 only"* — is superseded by **P3**, which measured L13 and found
it **slower** than the code it would replace, by +2.86% at W=512 and +4.61% at W=128, consistently
signed across both widths. The broadcast Mosaic emits for `w[:, :, None]` is already free.

So **C8 is closed from both directions**: its E1 arm by this probe, its L13 arm by P3. That is worth
stating explicitly, because a candidate killed on only one of its two arms tends to come back
proposed as the other one.

## What E1 was worth, for the record

Since P3 measured it anyway, the price of the thing that cannot be had: deleting the broadcast *and*
the multiply is **13.70% faster at W=512** (1183.5 ns vs 1371.3 ns) and 29.28% faster at W=128
[`measured (j-efb89184-020f-495f-9d4b)`]. It was the largest single effect in P3. Recording it means
that if some future model *does* guarantee a non-negative indexer weight, the value of exploiting it
is already known and nobody pays for that measurement twice.

## Verdict

**PASS** — answered decisively, from source and from the shipping checkpoint, with the two agreeing.

**decided:** E1 rejected on correctness. C8 removed from the candidate list entirely (E1 here, L13 in
P3). Any future proposal that assumes a sign for `indexer_weights` must cite a checkpoint measurement,
not the `relu` at `streamindex_topk.py:333` — that `relu` is on the score, not on the weight.
