# P1 + P2 — the einsum reaches the MXU as fp8, and its f32 result is never materialised

**probe:** P1 — does the `compute_scores` einsum reach the MXU as fp8, or does Mosaic upcast the
operands to bf16? P2 — does Mosaic materialise the `(B, bq_sz·32, bkv_sz)` f32 einsum result as one
VMEM buffer, or tile the contraction?

**config:** `vllm-torchtpu` @ `4dff8488e0b137d636eea9117892ce9fef76f583`, branch
`annyan/streamidx-scores`. Working tree at submit carried only `?? kernel_work/streamindex_topk/step2/combined.md`
and `?? optimization_work/` — no tracked modifications. Image
`us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-annyan_google_com:13617694`.
Driver `probes/llo_capture.py`. `jax 0.10.2`, one tpu7x pod, two shards (one chip each).

**job:** `j-689b87cd-7460-4b90-bd87`, `JobStatus: Succeeded`, `tpu-fanout: 2/2 shards succeeded`,
both `rc=0`.

`RUN_CMD`, as read back from `cdk job recipe j-689b87cd-7460-4b90-bd87 --no-color`:

```
export PYTHONPATH=/workspace && bash /root/fanout/tpu-fanout.sh
  "python3 /workspace/kernel_work/streamindex_topk/probes/llo_capture.py
  --regime decode_prod --out $CDK_OUTPUT_DIR/decode_prod" "python3
  /workspace/kernel_work/streamindex_topk/probes/llo_capture.py --regime
  mixed_prod --out $CDK_OUTPUT_DIR/mixed_prod"
```

Dumps: `decode_prod` 10,834 jf files / 1031.0 MiB + 179 mosaic files / 85.6 MiB; `mixed_prod`
11,034 / 1103.6 MiB + 179 / 85.6 MiB. Retrieved with `cdk job sync-outputs
j-689b87cd-7460-4b90-bd87` to
`/home/annyan_google_com/cloud-devkit-data/job-outputs/j-689b87cd-7460-4b90-bd87/`.

> [!IMPORTANT]
> **Everything below is static compile-time analysis.** No number in this file is a timing.
> The driver prints a wall-clock only as proof the kernel executed; the two shards shared a host,
> and `xprof-llo/SKILL.md` is explicit that a fanned-out capture's wall-clock is not a performance
> number. It is not quoted here and must not be quoted anywhere.

## The reader chain had to be replaced, and that is a `references/` item

`.claude/skills/xprof-llo/setup.sh` **fails**: its pin
`https://github.com/vlasenkoalexey/xprof-cli.git` @ `da2b31a6447c67cb8f7084b404cbf36cee57687d`
returns `remote: Repository not found` under an authenticated `gh` (token scopes `gist, read:org,
repo, workflow`), and `xprof-cli` is not on PyPI either (`No solution found when resolving
dependencies`). So `get_llo_fit_summary` and `get_custom_call_mlir` were unavailable.

Worse, `setup.sh` **exits 0 on this failure** when its output is piped, because the clone error is
swallowed by the surrounding pipeline — an install that did nothing looks like an install that
worked. That is the failure mode the skill's own "Instruments that lie" section is about.

Both readers were replaced by reading the raw dumps, which is what they wrap:

| what the reader would have given | read instead from |
| --- | --- |
| `get_custom_call_mlir` | `*-post-infer-vector-layout.txt` and `*-post-apply-vector-layout.txt` in the `mosaic/` dump |
| `get_llo_fit_summary`'s VMEM figure | `*-06-post-rematerialize-allocations.txt` in the `jf/` dump — the literal allocation list, with `space=` and `size=` per entry |
| `get_llo_fit_summary`'s MXU matmul count / idle / spill fields | `*-71-final_bundles.txt` and `*-69-final_hlo-static-per-bundle-utilization.txt` |

This is strictly more direct than the reader — the allocation list is the compiler's own accounting
rather than a tool's summary of it — and it is the only route that works today.

## Which programs exist

Three `StreamIdxTC` LLO programs are compiled, one per band, and **the same three appear
byte-for-byte at both regimes** — their shapes come from `num_queries_per_block` and
`decode_req_batch_size`, not from the regime's token count:

| program | what it is | einsum, at `post-infer-vector-layout` |
| --- | --- | --- |
| `StreamIdxTC-m-bq_128-bkvp_2.1` | the mixed / prefill band, `bq_sz = 128` | one `tpu.matmul` |
| `StreamIdxTC-d-bq_1-bkvp_2.2` | the decode band, `decode_req_batch_size = 4` | four `tpu.matmul` (the seq-batch is unrolled) |
| `StreamIdxTC-d-bq_1-bkvp_2.3` | the decode-remainder band, one sequence | one `tpu.matmul` |

## P1 — answer: **native fp8. Nothing is upcast.**

At `post-infer-vector-layout`, the mixed band's matmul, verbatim (types only):

```
tpu.matmul ... : vector<4096x128xf8E4M3FN>, vector<2048x128xf8E4M3FN>,
                 vector<4096x2048xf32> -> vector<4096x2048xf32>
```

and the decode bands' are the same with `32x128` in place of `4096x128`. Both operands are
`f8E4M3FN`; the accumulator is `f32`. This is the native gen-7 pair `hardware-facts.md` lists
(`f32/bf16/f8e4m3fn/f8e5m2 × f8e4m3fn/f8e5m2`).

Confirmed again three levels lower, at the instruction opcode, in `*-71-final_bundles.txt`:

| program | `vmatmul` total | of which `bf16` | of which `f8` | of which `f32` |
| --- | ---: | ---: | ---: | ---: |
| `StreamIdxTC-m-bq_128-bkvp_2.1` | 1024 | **0** | **1024** | 0 |
| `StreamIdxTC-d-bq_1-bkvp_2.2` | 32 | **0** | **32** | 0 |
| `StreamIdxTC-d-bq_1-bkvp_2.3` | 8 | **0** | **8** | 0 |

Two distinct opcodes appear across all three, and both are fp8:
`vmatmul.mubr.f8e4m3fn.gmra.mrb` (1016) and `vmatmul.mubr.f8e4m3fn.vlgmr.msra.gmra.mrb` (48).
The weight push is `vmatpush3.f8e4m3.xpose.msra.mxu{0,1}`.

**The only `bf16` anywhere in the mixed band's Mosaic module is 2048 elements wide and is not on
the matmul path.** Exhaustively, the two occurrences are:

```
tpu.extf %bitcast : vector<1x2048xf8E8M0FNU> -> vector<1x2048xbf16>
tpu.extf %stack   : vector<1x1x2048xbf16>   -> vector<1x1x2048xf32>
```

That is the S1-a scale path — `streamindex_topk.py`'s `flat_bkv[:, head_dim:head_dim+1].T` bitcast
to `float8_e8m0fnu` and widened. It is a per-position dequant scale, not an einsum operand.

**Branch selected in `combined.md`'s P1 fork (line 687): the `fp8` branch.**
> *"**fp8:** mixed has 5.6× headroom and is the priority; C10's bf16-operand arm is a 1.86×
> regression and dies."*

Consequences, in `combined.md`'s own terms:
- The `mixed_prod` floor is **48.4 µs**, so the measured 269.5 µs is **18% achieved** — not 45%.
  Mixed is the priority, by a wide margin over decode's 62%.
- **C10's P8 arm (bf16 operands) is refuted, not merely unattractive.** It is not the status quo; it
  is a move away from a native fp8 matmul, and `combined.md` prices it at 1.86× slower.
- `report.md`'s "unsettleable from source" note on this 2.5× swing can be retired. It was settleable
  — from the *compiler's* output rather than the source's, which is exactly what P1 was for.

## P2 — answer: **tiled. Nothing near the f32 result is ever materialised in VMEM.**

Two independent readings agree.

**(a) `apply-vector-layout` splits the single logical result into eight N-tiles.** In the mixed
band, the one `tpu.matmul` at `post-infer-vector-layout` becomes **eight** at
`post-apply-vector-layout`, every one of them:

```
: vector<4096x256xf8E4M3FN>, vector<256x256xf8E4M3FN>,
  vector<4096x256xf32> -> vector<4096x256xf32>
```

— the `bkv_sz = 2048` output axis cut into 8 tiles of 256 (K shows as 256 because fp8 packs two
values per 16-bit lane slot). The contraction is tiled by the compiler, before any of this reaches
LLO.

**(b) The VMEM allocation list contains no such buffer.** This is the compiler's own accounting,
from `*-06-post-rematerialize-allocations.txt`. Complete, every `space=vmem` entry:

| program | alloc | size | shape | what it is |
| --- | --- | ---: | --- | --- |
| **`-m-bq_128-bkvp_2.1`** | 1 | 0.070 MiB | `u32[144,128]` | internal scratch |
| | 2 | 1.000 MiB | `s8[2,1,512,4,256]` | KV, double-buffered (`512 = bkv_p·256`) |
| | 3 | 1.000 MiB | `f8e4m3fn[2,1,128,32,128]` | q block, double-buffered |
| | 4 | 0.125 MiB | `f32[2,1,128,32]` | `indexer_weights`, double-buffered |
| | 5 | 2.000 MiB | `f32[2,128,16,128]` | `scores_block_x2`, double-buffered |
| | | **4.195 MiB** | | **total (4,399,104 B = 0x432000)** |
| **`-d-bq_1-bkvp_2.2`** | | **4.168 MiB** | largest is `s8[2,4,512,4,256]` at 4.000 MiB | |
| **`-d-bq_1-bkvp_2.3`** | | **1.095 MiB** | largest is `s8[2,1,512,4,256]` at 1.000 MiB | |

Identical at `decode_prod` and `mixed_prod` — as designed, `decode_prod` was the non-discriminating
control here and it behaved like one.

A materialised `(1, 128·32, 2048)` f32 would be **32.0 MiB** and the arm would read ≈37 MiB. It
reads **4.195 MiB**. Across all three programs the only spaces used at all are `vmem` (5 allocations),
`smem` (17, 5,164 B) and `sflag` (26, 132 B). There is no 32 MiB buffer anywhere.

**Where the intermediate actually lives:** vector registers, spilled and refilled against the
72 KiB `u32[144,128]` internal scratch. From the static per-bundle utilization of the mixed band —
11,602 bundles — `VSTORE:SPILL` issues 8,460 of the 8,716 total `VSTORE`s (97%), and `VLOAD:FILL`
issues 10,133 of the 11,539 total `VLOAD`s (88%).

**Branch selected in `combined.md`'s P2 fork (line 688): the `Tiled` branch.**
> *"**Tiled:** C4 is a no-op and drops off the list; take `bkv_p` to 8 at mixed directly, and §(b)'s
> mixed peaks fall from 37.5 MiB to ~6 MiB."*

The `~6 MiB` estimate is close. Measured, the only bkv_p-dependent allocation is the KV scratch
`s8[2, 1, bkv_p·256, 4, 256]`, at 1.000 MiB when `bkv_p = 2`. Taking `bkv_p` to 8 quadruples it to
4.000 MiB and leaves the other four untouched, so the mixed band's peak becomes
0.070 + 4.000 + 1.000 + 0.125 + 2.000 = **7.195 MiB** [`modelled`, by substitution into a `measured`
allocation list — the arm at `bkv_p = 8` was not compiled]. That is well inside the ~32 MiB default
scoped limit, so the `B · bq_sz · bkv_p ≤ ~500` inequality that §(b) built four areas' rankings on
**does not bind**.

Consequences:
- **C4 is a no-op and drops off the list.** So does its role as a prerequisite for C3-at-mixed
  and for C12.
- `bkv_p` at mixed can go to 8 directly — subject to P5', which value-checked `(4,4,4)` and passed.
- `scores_block_x2` at 2.000 MiB is the largest single allocation in the mixed band. Area 4's
  16-bit-scores ask (`combined.md` line 378, "`bo_scores` halves: −1 MiB mixed") is confirmed at
  exactly that: 2.000 → 1.000 MiB. It frees a real megabyte, but of a budget that is not scarce.

## Two things that surprised me and are not in `combined.md`

**1. The mixed band is VPU-saturated, not MXU-bound, in the static schedule.** Per-bundle slot
utilization against the machine's issue capacity (`MXU 2, XLU 2, VALU 4, VPOP 2, EUP 1, VLOAD 3,
VLOAD:FILL 3, VSTORE 2, VSTORE:SPILL 2, SALU 2`), over 11,602 bundles:

| unit | issues | bundles using it | slot utilization |
| --- | ---: | ---: | ---: |
| **VALU** | 43,931 | 97.0% | **94.66%** |
| VPOP | 8,738 | 55.7% | 37.66% |
| VSTORE | 8,716 | 54.4% | 37.56% |
| **VSTORE:SPILL** | 8,460 | 53.3% | 36.46% |
| **MXU** | 8,182 | 41.3% | 35.26% |
| VLOAD | 11,539 | 65.0% | 33.15% |
| **VLOAD:FILL** | 10,133 | 60.2% | 29.11% |
| XLU | 2,358 | 11.6% | 10.16% |
| SALU | 335 | 1.9% | 1.44% |
| EUP | 1 | 0.0% | 0.01% |

Fully-idle bundles: 7 of 11,602 (**0.1%**) — by the reference pair in `xprof-llo/SKILL.md`, near-zero
idle is the marker of a kernel that is kept busy, and this one is. But it is kept busy on the
**VPU**, at 94.66% of four issue slots, while the MXU sits at 35.26%. The decode bands are less
extreme (VALU 73.52% / 62.95%, MXU 5.49% / 4.80%).

This is a genuinely different picture from the one `combined.md` reasons from, which ranks mixed by
its distance from an **MXU/HBM** floor. If the mixed band is VPU-serialized, the levers that matter
are the ones that delete *elementwise* work between the matmul and the store — the relu, the weight
multiply, the head-sum, the scale, the mask — not the ones that feed the MXU better.

Three caveats, and they are load-bearing:
- This is a **static compile-time** schedule, not measured occupancy. It says what the compiler
  emitted, not what retired.
- `xprof-llo/SKILL.md` records that `get_llo_fit_summary` gave *the same* "VPU-serialized, MXU
  starved" verdict to two kernels 3.56× apart in speed, and that the ranked lever list would have
  sent you after the wrong one. Treat this table as a hypothesis to be measured, not a diagnosis.
- The spill/fill signal specifically ships self-labelled `[TRACK RECORD 0/5 — 0/5. Directly refuted
  twice]` in that same skill. **Do not open a candidate on spill reduction on the strength of this
  table.** It is recorded because it explains *how* P2's "not materialised" is realised.

**2. The absence of a VMEM buffer is not the absence of a cost.** P2's fork treats "tiled" as the
cheap branch. It is cheaper in *VMEM*, which is what the fork asked. But 18,593 spill-and-fill
issues is where the tiling went, and the VALU number above sits right next to it. C4 is dead as
written — there is no buffer to shrink — but the question C4 was gesturing at, whether the
intermediate is handled well, is *not* settled by this probe in the affirmative.

## Verdict

| probe | verdict | branch taken |
| --- | --- | --- |
| P1 | **PASS** — answered decisively, three independent readings | `fp8` |
| P2 | **PASS** — answered decisively, two independent readings | `Tiled` |

**decided:** the `mixed_prod` roofline is the fp8 one — 48.4 µs floor, 18% achieved, and mixed is
the priority. C10's bf16-operand arm is rejected. C4 is withdrawn; `bkv_p` at mixed is free to move
on VMEM grounds; `B · bq_sz · bkv_p ≤ ~500` does not bind.
