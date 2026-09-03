# R1c — where the 973.9 ns lives, and what the dump cannot tell us

**Reads after `R1.md` and `R1b.md`.**

R1b **measured** Option A's DMA-free `compute` arm at `bkv_p = 9` to be **973.9 ns/block** cheaper
than the incumbent's, and showed the halved payload contributes ≈ nothing to the *fetch* because
94.3 % of the incumbent's DMA is already hidden. R1b left open *why the compute arm is cheaper*,
with two candidates:

- **(T) the eliminated lane transpose.** The incumbent extracts its e8m0 scales as
  `flat_bkv[:, 128:129].T` (`streamindex_topk.py:267-268`) — a `(bkv_sz,1) -> (1,bkv_sz)` shuffle of
  a byte column sitting at **lane offset 128** inside a 256-lane record. Option A's scale rows are
  already lane-major, so the same extraction is a reshape.
- **(W) the halved payload width.** The incumbent's record is 256 lanes and it uses 128.

The stake as posed: under **(T)** a variant might capture the saving *without touching the cache
format*, which would strictly dominate Option A. Under **(W)** no such variant exists.

> ## Verdict — **settled by the fourth arm; read §7**
>
> **This block records what the first three arms alone could say. One paragraph in it — "The three
> arms cannot separate (T) from (W) at all" — is superseded by §7; the other three survive intact,
> including the byte-axis reading of (W).** The fourth arm has since been captured (job `j-a0566857-c6f2-4667-9eed`) and
> it breaks the collinearity: **(T) is confirmed and (W) is refuted on every instruction and
> code-size axis.** `w_reshape` — inc's 256-lane record with A's lane-major scale rows — carries
> inc's full byte traffic (9.79 MB/block, 2-sublane loads) and yet lands on `a_reshape` for XLU
> occupancy (**1.04 % against 1.05 %**, versus inc's 46.24 %), transposes (0), bundles (+1.3 % of
> the gap), spills (+0.5 %) and total ops (+0.9 %).
>

> **(T) is where the cost is.** Four independent readings agree: XLU bundle occupancy is
> **46.24 % in the incumbent against 1.05 % in both Option A arms**; the incumbent alone carries
> 288 `vector<128x128xi8>` tile transposes; the final-LLO delta is **+4,926 ops**, of which
> `vcombine` + `slane` cross-lane ops account for **5,760**; and the backward slice from
> `tpu.matmul` is **byte-identical across all three arms**, so the payload-consumption path costs
> the same everywhere.
>
> **(W) is narrowed, NOT refuted.** An earlier draft of this document claimed it was, on the
> grounds that `tpu.load` counts match (9,296 vs 9,368). **That reasoning was wrong and the
> artifact contradicts it.** Instruction *count* is not byte count: the incumbent's loads carry
> `sublanes [true,true,false...]` against Option A's `[true,false,...]`, so the incumbent reads
> **9,437,184 B per grid block against Option A's 4,755,456 B — a factor of 1.984** — in the same
> number of instructions. (W) costs **no extra vector instructions**; it costs **2× the VMEM read
> bytes and 2× the landing buffer**, and this static dump cannot price either.
>
> **The three arms cannot separate (T) from (W) at all.** Scale placement and record width are
> perfectly collinear across them: `inc` = {lane-128 scale, 256-lane record}, both A arms =
> {lane-major scale rows, 128-lane record}. **There is no arm with a 256-lane record and lane-major
> scale rows.** That one arm — compile-only, minutes on the existing harness — is the experiment
> that would settle this. **[run] It has since been run — §7.**
>
> **`R1b` §3.2's `[rev]` tilt toward payload width is NOT superseded.** On the load-width evidence
> it was right about the bytes; what it did not have was the XLU and transpose evidence showing the
> *instruction* cost is all on the scale path.

---

## 0. Provenance, and what class of number everything below is

| | |
|---|---|
| Job | **`j-a78a5081-7f73-42ad-b483`**, `JobStatus: Succeeded` |
| Shape | serial, single shard, **not** fanned out; three arms one after another |
| Driver | `probes/probe_llo_capture.py` (this branch) |
| Capture flags | `LIBTPU_INIT_ARGS="--xla_jf_dump_to=<d>/jf --xla_jf_dump_llo_text=true --xla_mosaic_dump_to=<d>/mosaic"`, set **above `import jax`** and above the probe import |
| Gate | `CAPTURE OK` printed for all three arms (non-empty `jf` and `mosaic`) |
| Volume | 35,735 files / 3.82 GB, via `cdk job sync-outputs j-a78a5081-7f73-42ad-b483` |
| jax | **0.10.2** for all three arms — deliberately *not* upgraded; see below |
| Reader | `xprof-cli` at `.claude/skills/xprof-llo/.venv_cli/bin/xprof-cli`, `--bypass_cache=True` on every comparison |
| Census | `probes/llo_census.py`, output saved as `probes/R1c_census.txt` |

**Arm identity is weaker than an earlier draft claimed.** `probe_kpack_consumer.source_md5()` hashes
**`streamindex_topk.py` from the image** (`b0099c0a27ad014fc814a1f543b32313`), not the probe. It
being equal across arms is guaranteed by construction — same image, same file, same process — and
is a check that could not have failed. What *does* tie the arms together: all three ran in one job
from one mapped source tree, and the dump's VMEM scratch shapes match `build_arm` exactly
(`2×4×2304×4×256` for `inc`, `2×4×2322×4×128` for both A arms).

> ### ⚠️ Everything in this document is **static compile-time analysis**
>
> Opcode counts, VLIW bundle counts, allocation sizes and occupancy percentages are read out of the
> compiler's own dump. **None of it is measured time.** The only measured numbers here are carried
> in from R1b and are marked. Nothing here goes to `/log-experiment`.
>
> **This job's wall-clock (~43 min) is meaningless** and is quoted nowhere as a result: it runs one
> iteration per arm purely to force compilation with dump flags on.

**On not upgrading jax to 0.11.0.** `/xprof-llo` requires jaxlib 0.11.0 for *trace-side* capture,
where 0.10.2 against libtpu 0.0.44.1 mismatches `PLUGIN_Profiler_Api`. This is a **dump-side**
capture: no profiler, only compiler flags writing files. Upgrading would risk changing the very
lowering being censused, which must match the jax 0.10.2 compilation R1b timed. The capture gate is
what makes this safe to assert rather than assume; it passed on all three arms.

**Run-order artifact.** `inc` ran first and its `mosaic/` directory holds 26 extra files — earlier
SparseCore and setup episodes, not kernel episodes. Each arm's kernel episode is 18 files: `inc`
`1788406475661658498`, `a_reshape` `1788407246630451635`, `a_concat` `1788407983335631851`.

---

## 1. Where the arms differ in the Mosaic IR

### 1.1 The transposes are author-written; the relayout in front of them is not

The **original** module (pass `-0001`, before any Mosaic pass) already contains, in `inc` only:

```
%transpose3A = tpu.transpose %slice3A_52, [1, 0] : vector<9216x1xi8> -> vector<1x9216xi8>   (× 4)
```

Zero in either A arm. These are `streamindex_topk.py:267`'s `.T`, one per sequence in the batch.
*(An earlier draft asserted "nobody put a transpose in the source; the layout mismatch summoned it."
That is false, and the test it rested on — zero `tpu.relayout` at `post-infer-vector-layout` — is a
check that could not have failed, since relayout-insertion is the next pass.)*

What the compiler adds is the **relayout that feeds them**. `tpu.relayout` by pass:

| pass | `inc` | `a_reshape` | `a_concat` |
|---|---|---|---|
| `-0010 post-infer-vector-layout` | 0 | 0 | 0 |
| `-0011 post-relayout-insertion` | **43** | **43** | **327** |
| `-0012 post-apply-vector-layout` | 0 | 0 | 0 |

Grouped by signature at `-0011`:

| signature | `inc` | `a_reshape` | `a_concat` |
|---|---|---|---|
| `vector<9216x128xi8>` in `8,{0,0},(4,128)` -> out `8,{0,0},(32,128)` — **the payload** | **4** | **4** | **4** |
| `vector<9216x1xi8>` in `8,{0,128},(4,128)` -> out `8,{0,0},(32,128)` — **the scale column** | **4** | 0 | 0 |
| `vector<9x2x4x128xi8>` in `8,{0,0},(4,128)` -> out `8,{0,0},(32,128)` — A's scale | 0 | **4** | 0 |
| `vector<1x128xi8>` in `8,{k,0},(4,128)`, k ∈ {0,1,2,3} — a_concat's scale | 0 | 0 | **288** |
| the f32 score-side relayouts, shared | 35 | 35 | 35 |
| the landing-buffer relayout, `1x1x1x4x256xi8` vs `…x128xi8` | 1 | 1 | 1 |

**The payload relayout is the same op in every arm** — same source type, same in-layout, same
out-layout, same count. Mosaic folds the `flat[:, :128]` lane slice into it; the incumbent does not
materialise a 256-lane *value* and then narrow it.

**The only structural difference is the scale**, and the in-layout names the mechanism:
**lane offset `{0,128}`** — the scale byte inside the 256-lane record — being relaid out to
lane-major `(32,128)` tiling so the author's transpose can consume it.

### 1.2 `a_concat` resolves R1b's count-vs-cost conflation

R1b reasoned that `a_concat` builds its scale operand a structurally very different way (72
materialised `(1,128)` rows concatenated along lanes) yet lands within 0.6 % of `a_reshape`, so the
scale path could not be what costs. That inference **conflated relayout count with relayout cost**.
`a_concat`'s 288 relayouts are `vector<1x128xi8>` — one vreg-row each, differing only in a sublane
offset. The incumbent's 4 are `vector<9216x1xi8>` column transposes at lane offset 128. 288 cheap
ops and 4 expensive ones is not a contradiction. **This part of R1b's `[rev]` correction does not
survive.**

---

## 2. The op census — and why the headline number in the first draft was wrong

### 2.1 The correct pass labels, and three different deltas

An earlier draft of this document reported "+45,983 ops at `post-apply-vector-layout`" and derived
a 100.2 % attribution from it. Both the label and the inference were wrong. `llo_census.py`'s
`pass_file()` matches by substring and takes the last hit, so `"post-apply-vector-layout"` resolves
to **`-0013 …-simplify`**, not `-0012`. And the MLIR at that stage is dominated by IR that never
executes. The three deltas, all `inc − a_reshape`:

| stage | `inc` | `a_reshape` | `a_concat` | Δ (inc − a_reshape) |
|---|---:|---:|---:|---:|
| `-0012 post-apply-vector-layout` | 185,797 | 131,657 | 133,953 | **+54,140** |
| `-0013 post-apply-vector-layout-simplify` | 114,653 | 68,670 | 71,258 | **+45,983** |
| **`-0018 post-finalize-llo`** — the last stage before codegen | **44,562** | **39,636** | **39,488** | **+4,926** |

**Quote `+4,926`.** It is the only one of the three that is close to executed code. The MLIR deltas
are ~9–11× larger and are not a cost model. At `-0018` the delta decomposes cleanly:

| op family at `-0018` | `inc` | `a_reshape` | `a_concat` | Δ |
|---|---:|---:|---:|---:|
| `vcombine*` (cross-lane combine) | 11,775 | 8,319 | 8,319 | **+3,456** = 12 / transpose |
| `slane*` (sublane movement) | 9,435 | 7,131 | 7,275 | **+2,304** = 8 / transpose |
| **their sum** | | | | **+5,760** = **20 / transpose**, against a net delta of +4,926 |

*Withdrawn from the earlier draft:* the claim that each transpose "lowers into 96 sublane_shuffle +
64 gather = 160 ops = 46,080 = 100.2 % of the delta". Those 96 and 64 were **defined** in the
script as the deltas divided by 288, so multiplying back and comparing to the delta is an identity,
not a check — it says only "the other op deltas net to −97". It was also read from a file that
still contains all 288 `tpu.transpose` ops, so those shuffles cannot be the transposes' lowering;
they are the relayout cascade *around* them.

### 2.2 The `-0013` census, with the dead-IR caveat attached

Reported because §1's structural reading is easier to see here, **not** as a cost model:

| op | `inc` | `a_reshape` | `a_concat` | Δ |
|---|---:|---:|---:|---:|
| `tpu.sublane_shuffle` | 56,167 | 28,519 | 30,031 | +27,648 |
| `tpu.gather` | 37,459 | 19,027 | 20,035 | +18,432 |
| `tpu.load` | **9,296** | **9,368** | **9,368** | **−72** |
| `tpu.bitcast_vreg` | 1,769 | 2,057 | 2,057 | −288 |
| `arith.select` | 331 | 619 | 331 | −288 |
| `tpu.roll_vectors` / `tpu.unroll_vectors` | 437 / 432 | 149 / 144 | 149 / 144 | +288 / +288 |
| `arith.mulf` / `maximumf` / `addf` / `cmpi` | 1440 / 1152 / 970 / 584 | identical | identical | **0** |
| **`tpu.transpose`** | **288** | **0** | **0** | **+288** |
| **`tpu.matmul`** | **144** | **144** | **144** | **0** |
| TOTAL | 114,653 | 68,670 | 71,258 | +45,983 |

Ops are matched in operator position (`%r = d.op` or a statement-leading `d.op`), which excludes
memref layout *attributes* like `tpu.tiled`. Three alternative regexes give the same Δ.

**`tpu.matmul` = 144 in all three arms**, same `f8e4m3` operands. The arms do the same arithmetic —
the comparison was never between a kernel and a cheaper kernel. The review of this document
strengthened that independently: a backward def-use slice from `tpu.matmul` returns **27,965 defs,
8,064 sublane_shuffles and 6,912 gathers, byte-identical in all three arms**. The
payload-consumption path costs the same everywhere.

### 2.3 `tpu.load` — what it does and does not show

**It does not show what an earlier draft claimed.** The counts match (9,296 vs 9,368) but the masks
do not:

```
inc  × 9,216 : tpu.load ... sublanes [true, true,  false×6] : memref<1x1x2304x4x256xi8, ...>
a_*  × 9,288 : tpu.load ... sublanes [true, false, false×6] : memref<1x1x2322x4x128xi8, ...>
```

Two enabled sublanes × 128 lanes × 4 packed bytes = **1,024 B per load** in `inc`; **512 B** in both
A arms. Per grid block: 9,216 × 1024 = **9,437,184 B** against 9,288 × 512 = **4,755,456 B**,
**1.984×** (256/129). That reproduces R1b §3.2's 9.4 MB / 4.76 MB figure to the byte. The −72 count delta is
not evidence of anything: A's buffer carries 18 extra scale rows (`2322` vs `2304`), × 4 = 72.

So the correct statement about the payload width is narrow and specific:

> **The 508 pad bytes ride in the same load instruction as the payload.** The width costs the
> incumbent **zero extra vector operations**. It costs **1.984× the payload read bytes** and
> **1.984× the landing buffer** (18.23 vs 9.3 MiB). Whether those extra bytes cost cycles — whether a masked
> 2-sublane load retires slower than a 1-sublane one — is a hardware question **this static dump
> cannot answer**, and it is the open channel through which (W) could still be real.

---

## 3. LLO fit summary — `tpu_custom_call.1`

*Static compile-time analysis. `bundles` is code size, not cycles.*

| field | `inc` | `a_reshape` | `a_concat` |
|---|---:|---:|---:|
| **XLU bundle occupancy** | **46.24 %** | **1.05 %** | **1.05 %** |
| VMEM allocated | 18.23 MiB | 9.30 MiB | 9.30 MiB |
| bundles (`spills.bundles` / `totals.total_bundles`) | 8,417 / 8,351 | 6,590 / 6,524 | 6,557 / 6,491 |
| spills + fills | 5,262 + 5,453 | 511 + 657 | 470 + 616 |
| VSTORE (of which `VSTORE:SPILL`) | 43.42 % (**42.83**) | 8.39 % (7.54) | 7.70 % (6.89) |
| VLOAD (of which `VLOAD:FILL`) | 88.12 % (**42.12**) | 82.35 % (9.36) | 82.26 % (8.88) |
| VALU | 96.04 % | 95.02 % | 94.98 % |
| MXU | 13.63 % | 17.48 % | 17.57 % |
| MXU matmuls | 144 | 144 | 144 |
| overlap(MXU+mem) | 13.5 % | 13.6 % | 13.3 % |
| mem-stall (MXU idle) | 77.7 % | 70.6 % | 70.4 % |
| verdict | VPU-serialized | VPU-serialized | VPU-serialized |

**XLU is the single best datum in the dump and the first draft omitted it.** 46.24 % against
1.05 % — a 44× gap on the cross-lane unit, which is exactly where a 128×128 byte transpose executes
and where nothing else in this kernel goes. Nothing about payload width would touch the XLU.

**Almost all of the incumbent's store traffic is spill traffic**: 42.83 of 43.42 percentage points,
and 42.12 of 88.12 on the load side is fills. The spill rate is 1.27/bundle against 0.18 — but note
`xprof-cli`'s own warning that this ratio *"shrinks when work is removed… never use either as a
grading metric"*. The spills co-vary with the scale path; per §4.1 this design cannot separate them
from record width.

**Two bundle fields exist and differ by exactly 66 in every arm.** The table names both. Ratios
below use `spills.bundles`.

| ratio | static bundles | R1b measured `compute` ns/block | discrepancy |
|---|---:|---:|---:|
| `inc` / `a_reshape` | 8,417 / 6,590 = 1.2772 | 4192.9 / 3219.0 = 1.3025 | 1.98 % |
| `a_concat` / `a_reshape` | 6,557 / 6,590 = 0.9950 | 3199.9 / 3219.0 = 0.9941 | 0.09 % |

**Do not lean on this.** It is two points, and this document contains its own counterexample:
`a_concat` has **+3.8 % IR** against `a_reshape` at `-0013` yet **−0.5 % bundles** and **−0.6 %
measured** — the op census gets the *sign* wrong there. That is the same op census §2.2 rests on,
which is why §2.1 quotes `-0018` instead.

---

## 4. What this means for Option A

### 4.1 The experiment that is missing, and it is cheap

> **[run] This section is history. The fourth arm it specifies has since been captured and it
> answered the question — see §7.** The text below is kept as written because it is the prediction
> the arm was designed against, and both of its branches were named before the result existed. The
> outcome was the first branch: the census landed at `a_reshape`'s. §4.1's own framing — "(T)
> accounts for all of the instruction-count delta; (W) for none of it, and for all of a byte delta
> the dump cannot price" — is exactly what §7 confirms, and it is narrower than "(T) accounts for
> the 973.9 ns". Prefer §4.1's wording.

**The three arms are collinear in the two hypotheses.** Every arm pairs scale placement with record
width: `inc` = {lane-128 scale, 256-lane record}, both A arms = {lane-major rows, 128-lane record}.
`a_concat` varies only the *reader-side formulation* of A's scale path, not the format. So the
strongest thing this dump can say is:

- (T) accounts for **all of the instruction-count delta** — XLU 44×, 288 transposes, +5,760
  cross-lane ops at `-0018`, matmul cone identical.
- (W) accounts for **none of it**, and for **all of a 1.985× VMEM-read-byte delta** that the dump
  cannot price.

**The missing fourth arm is a 256-lane record with the e8m0 scales broken out into separate
lane-major rows.** If its census and bundle count land at `a_reshape`'s, (T) is established and the
payload width is confirmed free. If they land at `inc`'s, the width is the mechanism after all. It
is a compile-only arm, minutes on `probe_llo_capture.py` as it stands, and it is the one thing the
current three cannot do. **Per the brief I have not built it** — but it, not a sixth kernel build,
is the next thing worth doing.

### 4.2 The reader-side variant: weaker than the first draft claimed

The saving is on the scale path — but *scale placement is itself part of the cache format*. The
expensive relayout exists because the byte sits at **lane offset 128 of a 256-lane `(4,128)`-tiled
record**, and any reader-side-only fix still has to get it out of there. So the first draft's
"a variant strictly dominates Option A" is **not established**. Candidates, honestly priced:

1. **Widen before transposing.** `streamindex_topk.py:267-268` runs `.T` on the **i8** slice, then
   bitcasts, then `astype(bf16)`. Reordering to widen first puts the transpose on `(16,128)` bf16
   tiling instead of `(32,128)` i8. Bit-exact by inspection (`astype` is elementwise, transpose is a
   permutation, so reordering cannot change a value) — an argument, not a test. **But it targets
   the 288 tile transposes, not the lane-128 relayout cascade that feeds them**, and the cascade is
   the larger part. Also note it is a change to **`streamindex_topk.py`, which this exercise may not
   modify** — if so, this candidate is not buildable here at all.
2. **Route the transpose through the MXU.** MXU sits at 13.63 % and every arm's verdict is
   *MXU starved*; moving a permutation off a 96 %-occupied VALU is the right direction. Requires
   candidate 1 first, and inherits its limits.
3. **Move the scale multiply onto the K operand.** `_compute_scores` applies the scale *after* the
   head-sum, as `s_summed * scale[1, bkv_sz]`, which is why it must be lane-major. Scaling the K
   rows before the matmul needs `scale[bkv_sz, 1]` — a **lane broadcast of a sublane-major column,
   which is free** and removes the transpose *and* the relayout outright. Algebraically exact (e8m0
   is unsigned so the scale is positive and commutes through `maximum(s, 0)`; the head-sum is
   linear) but **not bit-identical**, since the multiply moves from f32-after-accumulation to
   before it. Needs an accuracy check, not a bit-identity check. **This is the only candidate that
   actually removes the cost, and it is the only one that is genuinely reader-side.**
4. *(Ruled out)* A separate DMA gathering the scale column. A DMA copies rectangles; landing a
   `(bkv_sz,1)` slice leaves it sublane-major and the transpose is still owed.

**What it is worth.** If the whole `compute` delta were recovered: 973.9 ns/block × 16 blocks =
**−15.58 µs**. *(An earlier draft wrote −15.30 µs here. That figure is the **`full`**-mode delta,
−956.05 ns/block × 16, which is Option A's own measured net; the `compute` delta is 973.9 and the
difference is R1b §3.3's +17.8 ns/block wrong-sign fetch residual.)* Only candidate 3 could
plausibly reach the whole of it, and only candidate 3 changes numerics. This is a **static-analysis
ceiling, not a prediction and not a measurement.**

### 4.3 The final numbers, with R1b's four `[rev]` corrections carried in

| quantity | value | class |
|---|---|---|
| Baseline decode call, incumbent `bkv_p = 2`, ctx 8704 | **170.58 µs** device self time (`hlo_stats`) | measured |
| Decode call at `bkv_p = 9` | **143.39 µs** device / 145.95 µs chain slope | measured |
| Requirement bar, −13 % of 143.39 | **18.64 µs** | derived |
| Requirement bar, −22 % of 143.39 | **31.55 µs** | derived |
| **Option A's stake on top of `bkv_p = 9`** (`full` mode) | **−15.30 µs** — 10.7 % of the p=9 call | measured per-block, ×16 extrapolation |
| Its `compute`-only component | −973.9 ns/block = −15.58 µs | measured per-block, ×16 |
| **Shortfall against A's own −13 % floor** | **17.9 % short** | derived |
| **`[rev]`** Modelled HBM bound at `bkv_p = 9`: 151.0 MB / 3.70e12 = 40.81 µs = 48.4 % of the 84.279 µs stage A; halving it removes at most | **~20.4 µs** — **not** the ~15 µs mis-transferred from `bkv_p = 2` | **modelled**, never measured |
| **`[rev]`** Cross-arm contamination in the probe | **2.4–9.7 %** (arms 188.83–760.88 µs); the 9.7 % arm is `a_reshape/fetch/bkvp9`, which carries the −1323.8 ns delta | measured |
| Fetch already hidden behind compute at `bkv_p = 9` | **94.3 %** | measured |
| Consumer correctness | bit-identical **inside the probe**; **no named test job against the real kernel** | measured, in-probe only |
| Correctness status of the −15.30 µs | **⚠️ unverified** | — |

The ~20.4 µs modelled bound and the −15.30 µs measured stake are **two unrelated quantities**; the
near-coincidence with the mis-computed ~15 µs was exactly that. Never quote the modelled figure as
corroboration of the measured one.

### 4.4 The one-line answer

> **Option A as a whole is not recommended** — but on the arithmetic, not on the mechanism: it
> delivers −15.30 µs against its own −13 % floor of 18.64 µs, **17.9 % short**, with the writer-side
> change and the prefill exposure still unmeasured and the number still `⚠️ unverified`. The
> mechanism finding does **not** independently condemn it: the dump shows the cost is the scale
> placement, which is a cache-format property, so a reader-side variant that dominates Option A is
> **plausible but unproven** — candidate 3 in §4.2 is the only one that would, and it changes
> numerics. **Before spending anything else here, run the fourth arm in §4.1.**

**[run] The verdict above is left exactly as written and is not amended.** The fourth arm it asks
for has since been run (§7). On its own axis it is decisive: **on instructions and code size, all of
the incumbent's excess follows scale placement, not record width** — 98.7 % of the VLIW-bundle gap
and 100 % of the XLU gap. It does **not** convert into a split of the measured 973.9 ns/block, and
§7.2 says why. So the reader-side variant stays **"plausible but unproven"**, exactly as §4.4 has
it: candidate 3 is the only route to it, and §7.4 now prices against it a bf16 counter-cost of the
same order as the prize. **The verdict on Option A itself is unchanged**, because it never rested on
the mechanism: the 17.9 % arithmetic shortfall, the unmeasured writer-side change and prefill
exposure, and the `⚠️ unverified` status all stand.

---

## 5. Limitations

1. **All static.** No cycle counts. The bundle-ratio agreement in §3 is suggestive, is two points,
   and is contradicted in sign by `a_concat` in the same table.
2. **(T) and (W) are collinear across all three arms** (§4.1). "Confirmed / refuted" is a
   vocabulary *this three-arm design* cannot issue; "narrowed" is what it earns. **[run] The fourth
   arm in §7 breaks the collinearity and does earn "confirmed / refuted", on the instruction axis
   only.**
3. **The dump cannot price bytes.** Whether a masked 2-sublane load retires slower than a 1-sublane
   one decides whether (W)'s 4.68 MB/block of extra VMEM reads is free. Unanswered — **and still
   unanswered after §7**, which is a compile, not a timing run.
4. **`tpu.transpose` -> 20 LLO ops each is this lowering**, on jax 0.10.2 / libtpu 0.0.44.1, at this
   tiling. A different jaxlib may lower it better, which would shrink the prize.
5. **The probe's geometry, not the model's.** `bkv_p = 9`, ctx 8704, `SEQ_BATCH_SIZE = 4`,
   `BQ_SZ = 1`, `num_grid = 8`.
6. **No variant was built or measured.** §4.2's prices are arguments.
7. **Candidate 1 may be out of scope** (it edits `streamindex_topk.py`) and candidate 3 changes
   numerics. Neither has a test.
8. **The −15.30 µs is unchanged**: `provisional` / `⚠️ unverified`, a ×16 extrapolation from a probe
   covering 82.5 % of stage A, carrying the 2.4–9.7 % contamination band.
9. **Prefill was never measured**, here or in R1b. It remains the largest unquantified risk.
10. **This document's first draft was wrong in two load-bearing places** — it claimed (W) refuted on
    a load *count*, and derived a 100.2 % attribution that was an identity read from a mislabelled
    pass. Both were caught by `/critical-review` (§6) and are corrected above. The census script was
    corrected in the same pass.

---

## 6. Review

`/critical-review` was run on the first draft of this document in a clean context, against the raw
dump rather than this prose. Disposition: **`PARTIALLY CONFIRMED`**. It reproduced every LLO
fit-summary figure and every R1b figure carried into §4.3 exactly, and confirmed the census regex
against three independent alternatives. It found, and I verified against the artifact before
accepting each one:

| severity | finding | disposition |
|---|---|---|
| CRITICAL | "(W) refuted" is contradicted by the load masks — 1.985× bytes in equal instruction count | **accepted**, verdict and §2.3 rewritten; R1b's `[rev]` un-superseded |
| CRITICAL | the 46,080 / 100.2 % attribution is an identity, from a mislabelled pass, over mostly-dead IR | **accepted**, replaced with the `-0018` delta of +4,926 |
| HIGH | 4 `tpu.transpose` are author-written, present in pass `-0001` | **accepted**, §1.1 rewritten |
| HIGH | §4.2's ceiling is not supported by candidate 1, which reaches only the tile transposes | **accepted**, §4.2 repriced |
| HIGH | (T) and (W) are collinear; no arm separates them | **accepted**, §4.1 added with the missing fourth arm |
| MEDIUM | 973.9 × 16 = 15.58 µs; the 15.30 is the `full` delta | **accepted**, both now stated |
| MEDIUM | XLU 46.24 % vs 1.05 % omitted from §3 — the best evidence in the dump | **accepted**, added |
| MEDIUM | two bundle fields; a_concat's sign counterexample | **accepted**, both stated |
| LOW | `source_md5` hashes the kernel module, not the probe; no `matpush1` in `op_counts`; line 267-268 not 266-268 | **accepted**, all three fixed |
| LOW | correctness row labelled "measured" | **accepted**, qualified as in-probe only |

The review also produced two results by a method this document did not use — a def-use slice — that
**support** the direction of the finding: the backward cone from `tpu.matmul` is byte-identical
across all three arms, and the live cone exclusive to the transposes is ~104 % of the post-DCE
delta. Both are quoted in §2.2 with their provenance, and both rest on a hand-written regex parser
rather than an MLIR parser, which the reviewer flagged itself.

---

## 7. The fourth arm — the collinearity broken

§4.1 named the experiment the first three arms could not do and priced it as a compile, a census
and a paragraph. It has been run.

**Job `j-a0566857-c6f2-4667-9eed`**, `Succeeded`, serial single shard on the whole pod, not fanned
out. Same image as the predecessor, same `simple-run-command` recipe at `2x2x1`, `ENABLE_XPROF=0`,
same `/root/kw` mapping, jax **0.10.2** on all three arms, `CAPTURE OK` on all three, 36,330 files
/ 3.7 GB. Census: `probes/llo_census.py --arms inc,a_reshape,w_reshape`, saved as
`probes/R1c_census_w.txt`. **Still static compile-time analysis; the 41 min 53 s wall-clock is
meaningless and is not a result.**

### 7.1 What the arm is

`w_reshape`, layout `w = (257, 256)` in `probe_kpack_consumer.py`: **the incumbent's 256-lane
record with Option A's lane-major scale rows.** Rows 0..255 of each page carry inc's
payload bytes in inc's 256-lane record — 128 payload lanes, 128 dead — and row 256 holds that page's
1,024 e8m0 scales as `(4,256)`. Two qualifications review insisted on: the rows are **not**
byte-for-byte inc's, because the scale byte at lane 128 is left zero (`_unpack_w` reads row 256
instead); and `_unpack_w`'s payload is **not verbatim** `_unpack_inc`'s — it carries one extra
row-axis slice, `b4[:, :PSP]`, that inc does not have. The `flat[:, :D_IDX]` lane slice out of a
256-lane row *is* inc's. The scale line is `_unpack_a(how="reshape")`'s, a pure reshape with no lane
slice and no transpose. §7.6 item 4 charges the residual those two differences produce.

That fills the missing cell of the 2×2:

| | scale at lane 128 | lane-major scale rows |
|---|---|---|
| **256-lane record** | `inc` | **`w_reshape`** ← new |
| **128-lane record** | *(not built — not needed)* | `a_reshape`, `a_concat` |

**It is not a proposed cache format.** A 257th row per page holding 1,024 bytes that would fit in
the dead pad the record already carries is not a layout anyone would ship. It is a controlled
compile whose only job is to vary one factor.

**One gate has teeth, and it is not the one the arm was built with.** The job log carries three,
and review established that two of them could not have failed:

- `[llo] ran. checksum=2.63026688e+08 n_finite=34816` — **the load-bearing one.** `w_reshape`'s
  checksum is **bit-identical to `a_reshape`'s**, and `inc`'s 2.63026624e+08 is 64 away, which at
  2.63e8 is **4 f32 ULP** (ULP = 2⁴ there) — a reduction-order rounding difference, not a
  correctness one. Since each score is `Σ_slot s_summed[slot]·scale[slot]`, a permuted scale row
  would move that sum with near-certainty. This check could have failed and did not.
- `[llo] w slot-order check: True` — **a check that could not fail.** `check_slot_order_w` builds a
  fresh numpy array, writes `block[p // 256, p % 256] = p`, and asserts a row-major flatten equals
  `arange(1024)`. That is a tautology of numpy's reshape. It never calls `make_source` or
  `_unpack_w`, so the permutation it was written to catch would have sailed through it. It is
  reported here as run, not as evidence.
- `source_md5=b0099c0a27ad014fc814a1f543b32313` on all three arms — likewise guaranteed by
  construction, for the reason §0 already gives.

**A determinism control arrived for free.** `inc` and `a_reshape` reproduced the predecessor job
*exactly* — 18.23 MiB / 8,417 bundles / 5,262+5,453 spills and 9.3 MiB / 6,590 / 511+657 — so the
compilation is deterministic across jobs. §7.2 itself is single-job; what the control protects is
the tie between §7's `inc`/`a_reshape` columns and §3's, which came from the earlier job.

### 7.2 The discriminating table

For any metric M where M(inc) ≠ M(a_reshape), M(w) says which factor M follows. Position is
`(w − a_reshape) / (inc − a_reshape)`: **0.00 = follows the scale placement (T), 1.00 = follows the
record width (W).**

| metric | `inc` | `a_reshape` | `w_reshape` | position | follows |
|---|---:|---:|---:|---:|---|
| **XLU bundle occupancy** | **46.24 %** | **1.05 %** | **1.04 %** | **−0.000** | **scale placement** |
| `tpu.transpose` | 288 | 0 | 0 | +0.000 | scale placement |
| spills + fills | 10,715 | 1,168 | 1,218 | +0.005 | scale placement |
| `tpu.sublane_shuffle` | 56,167 | 28,519 | 28,771 | +0.009 | scale placement |
| total ops (`-0013`) | 114,653 | 68,670 | 69,093 | +0.009 | scale placement |
| `tpu.gather` | 37,459 | 19,027 | 19,243 | +0.012 | scale placement |
| VLIW bundles | 8,417 | 6,590 | 6,614 | +0.013 | scale placement |
| final-LLO ops (`-0018`) | 44,562 | 39,636 | 39,792 | +0.032 | scale placement |
| VSTORE | 43.42 % | 8.39 % | 8.75 % | +0.010 | scale placement |
| **VMEM read bytes / block** | **9,750,528** | **5,068,800** | **9,787,392** | **+1.008** | **record width** |
| **VMEM allocated** | **18.23 MiB** | **9.30 MiB** | **18.3 MiB** | **+1.008** | **record width** |
| `tpu.load` sublane mask | 2 | 1 | **2** | 1.00 | record width |
| *`tpu.load` count — excluded, see below* | *9,296* | *9,368* | *9,332* | *+0.500* | *confounded* |

The op-count rows, the VMEM-read-byte row and the `tpu.load` mask come from `probes/llo_census.py`
(saved as `R1c_census_w.txt`); XLU, VSTORE, spills, bundles and VMEM allocated come from
`xprof-cli get_llo_fit_summary` (saved as `R1c_fit_w.txt`).
**XLU and VSTORE are `nonzero_bundle_pct`**, not `occupancy_pct` — on
`occupancy_pct` XLU reads 27.87 / 0.64 / 0.64 and the position is identical. Bundle counts are
`spills.bundles`; `totals.total_bundles` runs exactly 66 lower in every arm (8,351 / 6,524 / 6,548)
and gives the same position to three decimals.

**One census row is excluded, and it is the one that would look worst: `tpu.load` count sits at
+0.500, exactly the middle.** It is excluded because it is confounded, not because of where it
landed — `w`'s 257th row per page adds 36 loads by construction, which is all of the distance from
`inc`'s 9,296, and a load count carries no byte information anyway (§2.3). The mask, which does, is
the row above it. Two further fit-summary metrics are not tabulated and also fall outside the band;
both point *away* from the record width: `VLOAD` 88.12 / 82.35 / 81.16 → **−0.206** and
`mem-stall(MXU idle)` 77.7 / 70.6 / 69.5 → **−0.155**, i.e. `w` is marginally *better* than
`a_reshape` on each.

With those on the table: every instruction and code-size metric sits within 3.2 % of `a_reshape`,
and every byte and footprint metric sits at or just past `inc`. The separation is clean in both
directions.

**So, on the axis the instrument actually measures:** the record width costs **1.984× the payload
read traffic** (9,437,184 B vs 4,755,456 B on the page memref; 1.924× counting every load),
**1.984× the landing buffer** (18,874,368 B vs 9,510,912 B; `w`'s 257th row takes it to
18,948,096 B) and **1.3 % of the VLIW-bundle gap**. Scale placement accounts for the other
**98.7 % of the bundle gap** and **100 % of the XLU gap**.

**What that does not license is a split of R1b's measured 973.9 ns/block**, and an earlier draft of
this section wrote exactly that — "R1b's measured 973.9 ns/block *is* the scale placement". Review
struck it and was right to. Bundles are code size; converting a bundle share into a share of a
nanosecond delta needs bundles ∝ time *and* the byte channel priced at zero, and this document
supplies neither. On the first, §3's only calibration point — `inc` vs `a_reshape` — puts the
static bundle ratio at 1.2772 against a measured 1.3025, **a 1.98 % error, larger than the 1.3 %
effect being resolved**. On the second, §7.3 is about to say the byte channel cannot be priced at
all. **The defensible claim is the instruction-axis one: there, all of the excess follows scale
placement.** How the 973.9 ns divides between the instruction axis and the byte axis is
**unmeasured** — the fourth arm was compiled, never timed.

### 7.3 What (W) is still allowed to be

The fourth arm does not make the extra 4.7 MB/block free — it makes them **invisible to a static
dump**, which was already this document's position and remains its honest limit. `w_reshape` reads
inc's bytes and compiles to `a_reshape`'s code; whether those bytes cost wall-clock is a question
only a timing run answers, and I have not run one.

**An earlier draft closed this section by saying the byte cost "is not the 973.9 ns".** That is the
inference the paragraph above forbids, and it was circular: it used a code-size attribution, which
is silent about bytes by construction, to prove that bytes do not matter. Struck. The honest
statement is that **the byte channel is unbounded in both directions by this dump** — it could be
0 ns/block or it could be most of the 973.9 — and the only thing the fourth arm establishes about
it is that it buys no instructions.

### 7.4 The reader-side variant — what it is worth, and what it costs

(T) dominates, so per the stake as posed the reader-side variant is the live option. Two things
`w_reshape` does and does not establish about it:

- **It does establish that the instruction cost is reachable without the cache format.** Reaching
  `a_reshape`'s scale path while keeping the incumbent's record recovers **98.7 % of the VLIW-bundle
  gap and 100 % of the XLU gap**. If the whole 973.9 ns/block were instruction-side, a variant that
  gets the scale operand lane-major without a transpose would be worth **−15.58 µs** on the
  `bkv_p = 9` decode call. *That is a static ceiling on a measured-then-extrapolated quantity —
  973.9 ns/block is measured, ×16 is extrapolation, and the "essentially all of it" is a static
  attribution the dump cannot make (§7.2). It is not a prediction and not a measurement, and the
  floor under it could be well below −15.58 µs if the byte channel is real.*
- **It does not establish that a reader-side route exists.** `w_reshape` reaches lane-major scales
  by **changing the cache format** (adding a row). It is a discriminator, not a variant.

Of §4.2's four candidates, the fourth arm changes the ranking:

**Candidate 3 is now the only one worth considering** — move the scale multiply onto the K operand.
`_compute_scores` applies the scale after the head-sum as `s_summed * scale[1, bkv_sz]`, which is
the *reason* the scale must be lane-major. Scaling the K rows before the matmul needs
`scale[bkv_sz, 1]`, a **lane broadcast of a sublane-major column** — free, no transpose, no
relayout, and **no cache-format change**. Algebraically exact: e8m0 is an unsigned exponent so the
scale is positive and commutes through `maximum(s, 0)`, and the head-sum is linear.

Candidates 1 and 2 (widen before transposing; route the transpose through the MXU) are **withdrawn
as recommendations**. Both edit `streamindex_topk.py`, which is out of scope for this exercise
regardless of what the dump says, and both only make the transpose cheaper where candidate 3
removes it.

**What candidate 3 costs, part 1: it promotes the matmul to bf16, and that was measured at
+9.80 % on decode stage A.** This is the finding that most changes §7.4, and an earlier draft of
this section missed it entirely — it priced candidate 3's cost as "changes rounding" and stopped.
The shipping K operand is `float8_e4m3fn` (`streamindex_topk.py:265-266`; the census reports
`tpu.matmul … dtypes f8e4m3`). e4m3 tops out at 448 with a 4-bit exponent, while e8m0 spans
2⁻¹²⁷…2¹²⁷, so **a scaled K row cannot stay in e4m3 without saturating** — the scaled operand has
to be widened, which is why the rounding argument below says *bf16*-before-accumulation. Widening
the operand promotes the score einsum to bf16, and that is Option F, which `results.md` §11.2
**measured** (`j-8c300d39-2612-469d-8602`, min r² 0.999994):

| stage-A chain slope | incumbent | `f_bf16` |
|---|---:|---:|
| decode b64/ctx8704 | 110.00 µs | 120.78 µs **+9.80 %** |
| prefill q1024/ctx9216 | 246.12 µs | 257.53 µs **+4.63 %** |

Two things make this the right number to be worried by and the wrong number to subtract. Right:
`f_bf16` was measured **with HBM traffic held byte-for-byte constant**, which is exactly candidate
3's situation (the cache stays fp8; the widen is in-kernel), and §11.3 attributes the prefill
+4.63 % *to the matmul alone* — `vmatmul` 1,024 → 2,048 at an unchanged VPU census. Wrong: it was
measured at b64/ctx8704 with a different stage-A geometry from R1c's `bkv_p = 9` probe, so per this
repo's own rule about numbers measured elsewhere it does not transfer as a literal. What transfers
is the **mechanism and the order of magnitude: a counter-cost in (0, +9.80 %] on decode, against a
prize whose ceiling is 10.7 % of the same call.** Candidate 3 could plausibly net to zero or worse.

**What candidate 3 costs, part 2: it is not bit-identical.** The multiply moves from
f32-after-accumulation to bf16-before-accumulation, so it changes rounding. A checksum comparison is
the wrong instrument and would fail on a correct implementation.

**So the cheapest next step is not the accuracy check — it is a compile.** Before anyone writes
candidate 3, build the compile-only arm that folds the scale onto a bf16 K operand and census its
`vmatmul` count against the incumbent's 144 `tpu.matmul` / f8e4m3. That is the same class of
experiment §7 just ran, minutes on `probe_llo_capture.py` as it stands, and it decides whether the
−15.58 µs ceiling has a −10 µs floor underneath it. **I have not built it.**

**The right check, named and not run.** `results.md` §11 settles what the statistic must be: an
absolute recall-vs-f64 gate **fails on the incumbent itself** — at rank 512 of 2048 the true
boundary gaps are ~2.5e-04, below the kernel's own scoring noise, so which of two effectively-tied
items returns is arbitrary. **Arm-vs-arm index agreement against the incumbent is the right
statistic**, and it is what every option in that document was judged by. Concretely, the check
would follow job `j-5d02a60e-2266-477c-82d6`'s pattern:

1. Spy on the module-global `sparsecore_topk` inside a trace of the **shipping**
   `streamindex_topk.__wrapped__`, so the comparison is against the real kernel and not a
   transcription.
2. Run the scale-on-K variant and the incumbent on identical inputs, ≥3 seeds, and compare
   **returned top-k indices**, not scores.
3. Classify every disagreement. A disagreement is benign when it is a sign flip at a near-tie —
   `gap_f64 > 0`, `gap_f32 < 0` — with **max |gap|/σ ≤ 4**, the threshold `results.md` §11 recorded
   before any number existed. **Note that this predicate needs an f64 reference for each swapped
   pair**, so the run computes one even though the *gate* is arm-vs-arm; step 2 alone is not
   executable. And note what the incumbent's figures are and are not: **10 of 2,560 disagreeing,
   all 31 across three seeds sign flips, max |gap|/σ of 0.49 / 0.58 / 0.66 are §11's
   incumbent-**vs-f64** numbers**, not arm-vs-arm ones. They are the right order-of-magnitude
   yardstick for "how much disagreement is normal here"; they are not a bar the variant's
   arm-vs-arm count can be compared to directly.
4. Localize three ways, as §11 did, so a stage-B regression cannot hide inside a stage-A pass:
   `_effective_row_lengths` against the oracle, and stage B expected `EXACT`.

That is a `run-command-in-cdk` job of the kind §11 already ran, not a benchmark. **I have not run
it and I have not built the variant.**

The check has real teeth against candidate 3's actual failure modes — a permuted scale, a wrong
broadcast axis or fp8 saturation would all move scores far past the boundary gap and show up as
many disagreements well above the incumbent's observed 0.66. Its limitation is that it is an
*accuracy* gate on a change whose principal risk is a *performance* regression, which is what the
`vmatmul` census above is for.

### 7.5 Option A

**Unchanged, and §4.4 stands verbatim.** The fourth arm was never able to move it: A's shortfall is
arithmetic — −15.30 µs against its own −13 % floor of 18.64 µs, 17.9 % short — and the mechanism
was never what condemned it.

What the fourth arm changes is what can be said about the *alternative*, and less than an earlier
draft of this paragraph claimed. It **does not** show that candidate 3 dominates Option A. It shows
that the incumbent's excess *instructions* are a scale-placement cost, that a reader-side route to
them exists in principle, and that candidate 3 is the only named candidate on that route. Against
that: candidate 3 carries a measured-elsewhere bf16 counter-cost of the same order as the prize
(§7.4), the split of the 973.9 ns between instructions and bytes is unmeasured (§7.2, §7.3), and
the variant has not been built, timed or accuracy-checked. **The alternative is a live lead with a
known counter-cost, not a better option than A.**

### 7.6 Limitations specific to §7

1. **Static, all of it.** The fourth arm was compiled and censused, never timed. Nothing in §7.2 is
   a cycle count.
2. **A static dump cannot price bytes** (§7.3). (W) is refuted as an instruction-cost mechanism and
   is untested as a bandwidth one.
3. **`w_reshape` is not a shippable layout** and was never intended as one.
4. **The +1.3 % code-size residual** is the 256-lane record's real instruction cost — the
   `flat[:, :D_IDX]` lane slice plus the extra row. Small, but not zero, and it is charged to the
   width in §7.2.
5. **Candidate 3 is named, not built, not measured and not accuracy-checked.** Its algebraic
   exactness is an argument from the properties of e8m0 and `maximum`, not a test.
6. **The −15.58 µs ceiling assumes candidate 3 recovers what `w_reshape` recovers.** They reach the
   same place by different routes and that assumption is untested. It is also a hybrid quantity:
   measured (973.9 ns/block) × extrapolated (×16) × static (the attribution). Calling it "a static
   ceiling" flattens three provenances into one.
7. **The 98.7 / 1.3 split is a share of VLIW bundles, not of the 973.9 ns.** §7.2 says so; it is
   repeated here because the earlier draft of this document made exactly that slip in four places
   and it is the easiest one to make again.
8. **Bundles are a poor proxy at this resolution.** The only calibration point in the document
   (§3, `inc` vs `a_reshape`) is off by 1.98 %, which is larger than the 1.3 % residual the
   discriminator resolves. The *direction* of §7.2 is not in doubt — the metrics separate by two
   orders of magnitude, not by 1.3 % — but the third significant figure is not meaningful.
9. **Candidate 3's bf16 counter-cost is a repo number, not one measured here.** `results.md` §11.2's
   +9.80 % / +4.63 % were measured at b64/ctx8704 and q1024/ctx9216, geometries that are not R1c's.
   The saturation argument that forces the widen (e4m3 max 448 vs e8m0's range) is inference from
   dtype ranges, not a test.
10. **The `tpu.load` count row is excluded from §7.2** on a confounding argument, and the excluded
    row sits at exactly +0.500.

### 7.7 Review of §7

`/critical-review` in a clean context, against the raw dump, the job log and the recipe read-back.
Disposition **`PARTIALLY CONFIRMED`**. It reproduced every discriminator position, every census
figure and every fit-summary figure exactly, re-ran the reader itself (three different results for
three arms — the negative control on `--bypass_cache`), independently re-derived the determinism
control from job `j-a78a5081`'s own dump, and re-ran `llo_census.py` to a byte-identical artifact.
What it broke was the layer above the numbers.

| severity | finding | disposition |
|---|---|---|
| HIGH | "98.7 % **of the 973.9 ns** is scale placement" maps a code-size share onto a measured time delta; §3's only bundles-vs-time calibration point is off by 1.98 %, larger than the 1.3 % being resolved | **accepted.** §7.2's closing sentence struck and replaced; echoes in §7.3, §7.5 and the §4.4 `[run]` note rewritten; limitations 7–8 added |
| HIGH | §7.3's last sentence retracted its own first: it used the code-size attribution to prove bytes do not matter | **accepted.** Struck; §7.3 now says the byte channel is unbounded in both directions |
| HIGH | Candidate 3 forces the K operand to bf16 (e4m3 max 448 vs e8m0's 2⁻¹²⁷…2¹²⁷), which is Option F — **measured at +9.80 % decode / +4.63 % prefill** in `results.md` §11.2, with HBM held constant. §7.4 priced its cost as "changes rounding" | **accepted, and it is the largest change to §7.** §7.4 now carries the counter-cost, the caveat that it was measured at another geometry, and a compile-only `vmatmul` census as the cheap next step. §7.5 downgraded from "candidate 3 dominates A" to "a live lead with a known counter-cost" |
| HIGH | Two of §7.1's three gates could not have failed: `check_slot_order_w` is a tautology of numpy's reshape and never calls `make_source` or `_unpack_w`; `source_md5` is guaranteed by construction | **accepted.** §7.1 rewritten to credit the one gate with teeth — the checksum |
| MEDIUM | "There is no metric in the middle" was achieved by dropping the `tpu.load` count row, which sits at exactly +0.500 | **accepted.** Row restored to the table, marked confounded, with the confound stated; `VLOAD` −0.206 and mem-stall −0.155 disclosed too |
| MEDIUM | §7.4's "figures to beat or match" are §11's incumbent-vs-**f64** numbers, borrowed as a bar for an arm-vs-arm gate; and step 3's predicate needs an f64 reference step 2 does not mention | **accepted.** Step 3 rewritten |
| MEDIUM | `llo_census.py`'s byte model was 4× low for every non-`x4x` load (f32 loads, `memref<1x32x128xi8>`) | **accepted — a real bug in my code.** Fixed to a uniform `nsub × 512 B`; `R1c_census_w.txt` regenerated. Common-mode, so **every position is unchanged**; absolute per-block totals rose by 233,472 B each and the all-loads ratio corrects 1.97× → 1.924× |
| LOW | Checksum gap is 4 f32 ULP, not 2 | accepted |
| LOW | 1.985× should be 1.984× (256/129) | accepted, here and in §2.3 |
| LOW | "byte-for-byte inc's record" and "`_unpack_inc`'s verbatim" are both literally false | accepted, §7.1 qualified |
| LOW | "the cross-job comparison in §7.2" — §7.2 is single-job | accepted |
| LOW | ~45 min is 41 m 53 s | accepted |

The reviewer's own steelman against itself is worth recording: its HIGH findings 1 and 2 are one
objection to four sentences in a document that states the correct scope in five other places, and
its strongest substantive finding (the bf16 counter-cost) is an inference from dtype ranges rather
than a measurement of candidate 3. I have kept all of them anyway — the four sentences were the
ones a reader would quote, and a counter-cost of the prize's own order is worth a paragraph even
when it is inferred. On the central question it checked hardest, it found nothing: `w_reshape` is
the arm it claims to be, and every number reproduced.
