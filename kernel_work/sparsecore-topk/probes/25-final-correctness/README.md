# probe 25 — correctness gate on the composed candidate

**Verdict: GREEN.** The composed candidate is **bit-identical to the incumbent**
in all 30 cross-gate cells, passes the frozen 68 unmodified, and both arm
checks confirm the composed body — not the incumbent's — is what lowered.

| | |
| --- | --- |
| job | `j-49ee915f-6906-43dc-ba3f`, tags `agent-jobs,final-composer` |
| created / complete | 2026-09-10T22:21:44Z / 22:31:40Z |
| status | `ExecutionComplete` / `Succeeded`, `tpu-fanout: 2/2 shards succeeded` |
| image | `…/vllm-torchtpu-hning_google_com:fa63d6df` |
| node | `gke-tpu-d716ab26-30z9`, tpu7x 2x2x1 |
| versions in-pod | jax 0.10.2, jaxlib 0.10.2, libtpu 0.0.44.1, torch 2.13.0+cpu |
| stage mapped | `stage/` → `/root/p25` |
| `RUN_CMD` md5 (local, no trailing newline) | `b39a92ee465aaed141f25f1d985b8f8e`, 899 chars |
| recipe read-back | YAML-unfolded, md5 **identical** to local |

## Subjects, gated in-job

A docker image tag is mutable, and `fa63d6df` was re-pushed mid-exercise in
this repo, so all three md5s are gated **twice**: once in `RUN_CMD` before
anything runs (`MAP_GATE_OK`), and again inside each shard
(`SHARD_GATE_OK <name>`). Both fired.

| file | md5 |
| --- | --- |
| `subjects/final/sparsecore_topk.py` | `109a12a572116c00c795ee170256db33` |
| `subjects/incumbent/sparsecore_topk.py` | `44353b347c96e0eedbd742c710d4540a` |
| `suite/test_dsv4_sparsecore_topk.py` (frozen) | `b3241859e471fc87fa6e7a50e4ffec4b` |

## Two shards, one chip each

| shard | subject | arm expectation |
| --- | --- | --- |
| `composed` | the candidate | `arm_check.py --expect-suffix _final` **and** `arm_check_final.py` |
| `control` | the pinned **incumbent**, byte-identical harness | `arm_check.py --expect-suffix NONE` |

The control shard exists so that a red on `composed` can be told from a harness
or image finding. Its `xgate` compares the incumbent to itself, so its 30/30
bit-identity is a **harness self-check, not evidence about the candidate**.

## Results

### Arm activation — by scope name in the lowered program

`ARM_CHECK_OK` on both shards, and `ARM_CHECK_FINAL_OK` on `composed`:

> every variant scope name over 7 shapes matches `_c<chunk>_final`, and the
> incumbent carries `'_final'` on none of them

All seven frozen-suite shapes, including the five that lower **two** kernels
(stage 1 and the `_keys` stage 2) — both halves carry the tag:

| shape | P | composed scope names |
| --- | --- | --- |
| b64-n10240-k2048 | 1 | `…b64_n10240_k2048_c2048_final` |
| b16-n10240-k2048 | 2 | `…b16_n4096_k2048_c2048_final`, `…b32_n5120_k2048_keys_c1280_final` |
| b16-n133120-k2048 | 4 | `…b16_n8192_k2048_c2048_final`, `…b64_n33280_k2048_keys_c1664_final` |
| b64-n133120-k2048 | 4 | `…b64_n8192_k2048_c2048_final`, `…b256_n33280_k2048_keys_c1664_final` |
| b16-n133120-k512 | 4 | `…b16_n2048_k512_c2048_final`, `…b64_n33280_k512_keys_c1664_final` |
| b1024-n133120-k2048 | 4 | `…b1024_n8192_k2048_c2048_final`, `…b4096_n33280_k2048_keys_c1664_final` |
| b1024-n10240-k2048 | 1 | `…b1024_n10240_k2048_c2048_final` |

The incumbent, read through the identical path, produced the bare
`sc_topk_direct_b*_n*_k*` names with no tag — the negative control on the
activation check.

`arm_check_final.py` is an addition to `arm_check.py`, not a replacement,
because `arm_check.py` matches with `any(nm.endswith(suffix))`: an `any` is
green when only one of a two-stage shape's two kernels is the new body, and
`endswith` has already failed spuriously in this exercise because `v2`
*prepends* its tag while `v4`/`v12` append. `arm_check_final.py` requires
**every** name over **all seven** shapes to match, by substring plus a
`_c\d+_final$` tail.

### Frozen suite — 68 unmodified, no `-k`

| shard | result | time |
| --- | --- | --- |
| `composed` | **68 passed**, 0 failed, 0 errors, 0 skipped | 24.45 s |
| `control` | **68 passed**, 0 failed, 0 errors, 0 skipped | 25.62 s |

Collection was checked separately (`--collect-only -q` → `68 tests collected`)
so a silently-empty selection cannot read as a pass.

### Cross-gate — the only thing that compares the candidate to the incumbent *file*

`xgate.py`, 3 shapes × 5 score kinds × {full, ragged} = 30 cells.

| | `composed` | `control` |
| --- | --- | --- |
| cells | 30 | 30 |
| exact top-k (multiset) | 30 | 30 |
| **bit-identical to the incumbent** | **30 / 30** | 30 / 30 |
| multiset failures | 0 | 0 |
| negative control | **`NEGATIVE_CONTROL_OK`** | `NEGATIVE_CONTROL_OK` |

**The negative control went RED, as required.** Corrupting one index to a real
but losing column made `bit_equal=False` and produced a multiset reason —
`max score difference np.float32(4.8411946), expected exactly 0.0` — so the
gate is capable of failing and the 30/30 above is evidence rather than a
tautology.

The `all_neg_inf` cells report `live=0`: every score is `-inf`, so there is
nothing selectable and both arms return all-`-1`. Those cells test the
`write_empty_rows` path, not selection.

**Bit-identity was the expectation and it held.** Every change in the composed
candidate is a relocation, a re-typing that round-trips, or an addressing
change on a conflict-resolving scatter — none of them can move a value. A
bit-identity failure would have been a real finding; there is none to report.

It also closes every value-mediated mechanism for the measurer: whatever the
composed candidate's timing turns out to be, it cannot be explained by the
kernel selecting different columns.

## Not covered

`_pick_chunk`'s small-chunk regime. Over 49 608 reachable `(b, n, k)`
combinations, 6 208 (12.5 %) resolve to a pipeline chunk < 128 words; the seven
frozen-suite shapes resolve to 2048 / 1664 / 1280, so none of them reaches it.
The extension point is `xgate.py`'s `SHAPES` table at `:54`. See
`../../final/README.md` §8 for why the file was not changed.

## Files

```
stage/                       what was mapped to /root/p25
  subjects/{final,incumbent}/sparsecore_topk.py
  suite/{conftest.py,test_dsv4_sparsecore_topk.py}     frozen, unmodified
  arm_check.py                                         copied verbatim from probe 22
  arm_check_final.py                                   new, paired all-shapes check
  xgate.py                                             copied verbatim from probe 22
  shard_final.sh                                       new
  fanout/tpu-fanout.sh
run_cmd.txt                  the single folded RUN_CMD line, bash -n clean
outputs/
  job.log                    full job log
  composed/{arm_check.log,arm_check_final.log,pytest.log,junit.xml,xgate.json}
  control/{arm_check.log,pytest.log,junit.xml,xgate.json}
```
