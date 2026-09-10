# probe 22 — correctness for the frozen variants

Task 3. The frozen 68 (`instruments/test_dsv4_sparsecore_topk.py`, md5
`b3241859e471fc87fa6e7a50e4ffec4b`), unmodified, no `-k`, one variant per chip, **plus** a
cross-module gate the 68 do not implement.

## Answer — job `j-188b0f9e-47cc-4908-9fae`, four variants

| variant | subject md5 | frozen 68 | xgate multiset | xgate bit-identical | verdict |
| --- | --- | --- | --- | --- | --- |
| `v11-retiled-output-emit` | `3dc249276fa68985e29f4da8e9c27fbd` | **68 passed** | 30/30 exact | **30/30** | ✅ clears its bit-exact gate |
| `v12-f32-input` | `408ef72e9afc935c928f2803764878cc` | **68 passed** | 30/30 exact | **30/30** | ✅ bit-exact, now **verified** |
| `v02-r1-shared-hist` | `c0ca54a66a549020cfc584e2d8d97abe` | **68 passed** | 30/30 exact | **30/30** | ✅ clears its bit-exact gate |
| `v03-fused-emit` | `7309435f82ab4b273abe22cbf8fcf502` | **46 failed, 22 passed** | **18/30 RED** | 12/30 | ⛔ **fails its own multiset-only gate** |

Runtimes 27–31 s per shard, all four on one pod. `pytest` and `xgate` ran on every shard regardless
of each other's result.

### `v03` is not a slot-order difference

Its gate was deliberately weakened to multiset/set-equality, because it is exact with a different
slot order. It fails **that** gate, in 18 of 30 cells, at both production `P=1` shapes and at `P=2`:

```
RED p1-b1024-n10240 heavy_ties   full   <-- row 0: index 1098907648 out of range for n=10240
RED p1-b1024-n10240 all_negative full   <-- row 0: returned 0 valid entries, expected 2048
RED p1-b64-n10240   random       full   <-- row 0: duplicate indices
RED p1-b1024-n10240 random       full   <-- row 0: -1 padding is not a suffix
```

`1098907648` is `0x41800000`, which is **float32 `16.0`** — and `heavy_ties` is
`rng.integers(0, 17)`, so `16.0` is the largest *score* in that distribution. The out-of-range
"index" is a score word. Likewise the `-2147483648` that fills whole rows at `p4-b16-n133120` is
`0x80000000`, the bit pattern of `-0.0`. **`v03`'s fused emit is landing key/score words in the
index array**, not permuting slots. Four failure modes in one variant — out-of-range values,
duplicates, non-suffix padding, and empty rows — are consistent with that one cause.

The 12 green cells are the degenerate ones and say why they are green: `strict_desc` (already
sorted, so the emit's order is the input's) and `all_neg_inf` (nothing is selected, so nothing is
emitted). A suite covering only those two would have shipped `v03`.

### `v11`, specifically

- `test_all_neg_inf_rows_return_no_selection[p1-b1024-n10240]` — the id the coordinator named as
  discriminating, because every landed word is the constant `-1` and a wrong stride shows as a
  wrong value — **PASSED**.
- `v11` is `P=1`-only for *performance*. Its `P>1` **correctness** is clean: `p2-b16-n10240` is
  exact and bit-identical in all 10 xgate cells, and every `p2`/`p4` id in the 68 passed. The known
  `P>1` regression is a relayout cost, not a wrong answer. Do not conflate the two.

## Jobs 2 and 3 — the remaining ranked arms, all green

`j-24463f36-f8ec-4c7b-a446` (four shards) and `j-f114bd8a-dbaf-4964-b6a5` (three). Every shard now
also proves **which body lowered**, by Pallas scope name, before the suite runs.

| arm | subject md5 | scope name observed | frozen 68 | xgate |
| --- | --- | --- | --- | --- |
| `v4-ragged` | `69c8834e130e23999e0284929230da3a` | `…_k2048_v4ragged` | 68 passed | 30/30 exact, 30/30 bit |
| `v4-c2048` | `88a67892e8c063b5d9fea96de72e99c4` | `…_k2048_v4c2048` | 68 passed | 30/30 exact, 30/30 bit |
| `v10` arity 2 | `f83332273f1cdc05995965c0405e8665` + `21f2d777419b3cf43ba940f004bd765a` | `…_k2048_sc2` | 68 passed | 30/30 exact, 30/30 bit |
| `v10` arity 16 | same pair | `…_k2048_sc16` | 68 passed | 30/30 exact, 30/30 bit |
| `v2` `V2_BIGSCRATCH` | `70f9873e2941bd94601c15cea1dd9d55` | `…_k2048` (**bare**) | 68 passed | 30/30 exact, 30/30 bit |
| `v4-control` = incumbent | `44353b347c96e0eedbd742c710d4540a` | `…_k2048` (bare) | 68 passed | 30/30 exact, 30/30 bit |
| `v10` control (`sample=0`) | `f83332273f1cdc05995965c0405e8665` | `…_k2048` (bare) | 68 passed | 30/30 exact, 30/30 bit |

Three of these are controls and each answers a distinct question. `v4-control` is the incumbent
driven through this harness end to end: it reproduces the `j-62275e52` baseline of 68/68 under the
`conftest` swap route, so a red elsewhere is the variant. `v10 control` is `v10`'s own file with the
sample body switched off: it clears the 16 hunks of flag plumbing, so a red sample arm would be
attributable to the sample body. `v4-control`'s md5 **is** the incumbent's, which is exactly why the
scope-name check is not optional there — the hash cannot tell "the ladder is wired" from
"everything resolved to the base".

⚠ **`v10` is bit-identical to the incumbent in all 30 cells**, including `heavy_ties` at both
production shapes. Its tie caveat is a real possibility, not an observed behaviour, on these inputs.

⛔ **`V2_BIGSCRATCH` is the one ranked arm whose activation cannot be checked by scope name.**
`sparsecore_topk_bigscratch.py:488` builds the same bare `sc_topk_direct_b{b}_n{n}_k{k}` the
incumbent does. Here the source md5 is sufficient (it is a distinct file, selected by path, with no
env default and no shared hash), but in the ranking pool two arms with the same scope name cannot be
told apart in a profile. Ask the `v2` builder for a `_v2bigscratch` suffix before that arm is
profiled.

⚠ **`conftest.py`'s `SUBJECT_GATE_OK` banner never reaches any log** — pytest captures conftest
stdout at import and discards it on a green run. The `tee` fix did not recover it. Activation
evidence therefore rests on `arm_check.log` (positive, per shard) and on `conftest` raising
otherwise; do not go looking for the banner.

## Why `xgate.py` exists — nothing in the frozen 68 compares a variant to the incumbent

`test_emit_pairs_is_bit_identical_to_the_incumbent` (`:295`) and `test_emit_pairs_survives_ragged_rows`
(`:351`) toggle `_reference_stage2_gather` **within the loaded module** — the module against itself
— and both are `TWO_STAGE_SHAPES` (`P>1`) only. Every other id is the module against a numpy
oracle. So a green 68 establishes **exactness** and establishes **nothing** about bit-identity to
the incumbent, which is the gate three of the four variants are quoted against.

`xgate.py` loads the pinned incumbent (`44353b34…`) and the variant as two distinct modules in one
process, runs byte-identical inputs, and reports both verdicts per cell over 3 shapes × 5 kinds ×
{full, ragged} = 30 cells. It imports `_scores`, `_mismatch`, `SEED` and `LANES` from the frozen
suite rather than retyping them. It asserts both modules resolve the same `P`. It carries a
negative control (one index corrupted to a real but losing column) that must go red — it did, on
all four shards. The incumbent never went red in any cell of any shard, which is the harness's own
clean bill.

## The subject swap, and why not `cp`

Four fanout shards share one container filesystem, so `cp variant.py` over the installed module in
one shard silently retargets the other three. `suite/conftest.py` instead binds
`sys.modules["vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk"]` before collection, selected by
`SCTOPK_SUBJECT` + `SCTOPK_SUBJECT_MD5`, **both required** — an unset variable raises rather than
falling back to whatever the image ships.

⚠ The banner it prints (`SUBJECT_GATE_OK`, and the installed copy's md5) was lost: `shard.sh` piped
pytest through `tail -110` and the banner is at the *top*. The swap is still proven, two ways:
`conftest.py` raises unless the md5 matches and `importlib.import_module(TARGET) is mod`, and all
four shards collected 68 ids without a collection error; and `v03` diverged wildly from the
incumbent, which is a live positive control that the binding reaches the kernel. Fixed for later
runs — `shard.sh`/`shard_gen.sh`/`shard_v10.sh` now `tee` the full log and print `head -12`.

## Method rules this run applied or produced

- **Compute the winners; do not read the file's comment about them.** `xgate` reports
  `n_negative_winners` per cell, and it measures the coordinator's third-pass correction directly:
  `strict_desc` at `p1-b1024-n10240` has **2,096,128 negative winners of 2,097,152 live** = 2047 of
  2048 per row. P=1 coverage of negative winners exists today. The 2-id `all_negative` addition is
  **dropped**, as instructed.
- **A file hash cannot prove which arm ran** when a variant selects on an env default or when two
  rungs share a hash (`v4`'s control md5 *is* the incumbent's). `arm_check.py` lowers the subject
  and requires the Pallas scope-name suffix — `_sc{arity}`, `_v4ragged`, `_v4c2048` — and aborts
  the shard before the suite when it is absent, because a green 68 against the wrong body is worse
  than a red one.
- **Gate the subject md5 inside the job.** The image tag is mutable; the installed copy read
  `1a89d451…` on `probes/18` against a nominal `fa63d6df`. Every shard prints and tests the
  subject, the suite and the pinned incumbent before doing anything.

## Files

| file | what |
| --- | --- |
| `suite/conftest.py` | the per-process subject swap |
| `xgate.py` | the cross-module gate; `--json` per shard |
| `arm_check.py` | scope-name arm activation; `--expect-suffix` / `--arity` |
| `shard.sh` | one single-file variant, no arm check (bit-exact gates) |
| `shard_gen.sh` | one single-file variant **with** a scope-suffix arm check |
| `shard_v10.sh` | `v10`: two files, both md5-gated, env default + arm check |
| `stage/` | exactly what was mapped to `/root/p22` |
| `run_cmd{,2,3}.txt` | the folded `RUN_CMD`s, `bash -n` checked, md5s read back |
| `j-188b0f9e.txt`, `j-24463f36.txt`, `j-f114bd8a.txt` | job logs |
| `outputs/<variant>/{junit.xml,xgate.json}` | synced artifacts |

`RUN_CMD` md5 read back from `cdk job recipe`, byte-identical after unfolding the YAML scalar:
job 1 `d05fb662a3f9a46eba8c8f6861bc062b`, job 2 `c97967f450c16522f5f1f0d0c542bc77`, job 3 `e5cd5d00cc579d816af1b51069f134f3`.
