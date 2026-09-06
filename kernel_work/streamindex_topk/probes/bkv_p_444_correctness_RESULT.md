# P5′ — `bkv_p = 4` at mixed is numerically identical to the shipped `bkv_p = 2`

**probe:** the surviving correctness shard of P5. `probes/bkv_p_prod.py` swept `num_kv_pages_per_block`
for *speed* and landed `(4, 4, 4)` as a win, but the `(4, 4, 4)` tuple was **never value-checked** —
only the shipped `(2, 2, 2)` was. Does raising `bkv_p` change the top-k the kernel returns?

This is one shard, attached to another job rather than given a job of its own, as instructed. The
speed half of P5 was already answered and was **not** re-run.

**config:** `vllm-torchtpu` @ `4dff8488e0b137d636eea9117892ce9fef76f583`, branch
`annyan/streamidx-scores`; working tree at submit `?? kernel_work/streamindex_topk/step2/combined.md`,
`?? optimization_work/` only. Image `…/vllm-torchtpu-annyan_google_com:13617694`. One tpu7x pod, two
shards. `probes/bkv_p_prod.py --mode correct` rebinds `S.NUM_KV_PAGES_PER_BLOCK` from the probe
rather than editing `instruments/shapes.py`, which is read-only.

**job:** `j-e13fc627-a8c5-4ef4-bd5e`, `JobStatus: Succeeded`, `tpu-fanout: 2/2 shards succeeded`,
both `rc=0`.

`RUN_CMD`, as read back from `cdk job recipe j-e13fc627-a8c5-4ef4-bd5e --no-color`:

```
export PYTHONPATH=/workspace && bash /root/fanout/tpu-fanout.sh "python3
  /workspace/kernel_work/streamindex_topk/probes/bkv_p_prod.py --mode correct
  --tuples 4,4,4 --regimes mixed_prod,decode_prod" "python3
  /workspace/kernel_work/streamindex_topk/probes/bkv_p_prod.py --mode control
  --tuples 4,4,4 --regimes mixed_prod"
```

The second shard is the **negative control**, and it is the reason this is a real check rather than
a green tick: it perturbs the input so the oracle comparison *must* fail, confirming the check can
detect a wrong answer at this tuple. A correctness arm whose negative control was not also run at
the same configuration is a check that has not been shown to have teeth.

## Measured (job `j-e13fc627-a8c5-4ef4-bd5e`)

Verbatim from the shard logs:

```
[mixed_prod  tuple=(4, 4, 4)] PASS -- all 1024 rows a valid top-2048 vs the float64 oracle.
    worst_rank=2098 of allowed 2304, min_recall=0.99512 of required 0.99
[decode_prod tuple=(4, 4, 4)] PASS -- all 64 rows a valid top-2048 vs the float64 oracle.
    worst_rank=2109 of allowed 2304, min_recall=0.99561 of required 0.99
[mixed_prod  tuple=(4, 4, 4)] NEGATIVE CONTROL OK -- fired on 1024 of 1024 rows.
    worst_rank=8159 of allowed 2304, min_recall=0.42578 of required 0.99
```

## Against the shipped tuple

| regime | metric | `bkv_p = (2,2,2)`, from `bkv_p_RESULT.md` | `bkv_p = (4,4,4)`, this job | identical? |
| --- | --- | ---: | ---: | :---: |
| `mixed_prod` | `worst_rank` (allowed 2304) | 2098 | 2098 | **yes** |
| `mixed_prod` | `min_recall` (required 0.99) | 0.99512 | 0.99512 | **yes** |
| `decode_prod` | `worst_rank` (allowed 2304) | 2109 | 2109 | **yes** |
| `decode_prod` | `min_recall` (required 0.99) | 0.99561 | 0.99561 | **yes** |
| `mixed_prod` | negative control `worst_rank` | 8159 | 8159 | **yes** |
| `mixed_prod` | negative control `min_recall` | 0.42578 | 0.42578 | **yes** |

**Bit-identical on all six figures**, including both negative-control margins. Not "within
tolerance" — the same numbers.

That is the strong form of the answer and it is the expected one on inspection: `bkv_p` sets how many
KV pages one inner iteration consumes, i.e. how the reduction is *tiled*, and the accumulation is in
f32 either way. Changing the tiling changes the DMA schedule and the loop trip count, not the
arithmetic. The measurement confirms it rather than assuming it, which is the point — a tiling change
that *did* perturb the accumulation order would show up here as a small margin drift, and none
appears.

**`bkv_p` is numerically neutral.** Any candidate whose only mechanism is moving `bkv_p` needs no
correctness argument of its own beyond this file.

## Branch selected

**`combined.md` P5 fork (line 691): the `Match, and mixed moves` branch.**

> *"**Match, and mixed moves:** C3 generalises and jumps the ranking."*

C3 generalises. And it does so with more room than the fork assumed, because **P2 removed the VMEM
constraint that was expected to cap it**: the mixed band's measured peak at `bkv_p = 2` is 4.195 MiB,
of which only the KV scratch `s8[2, 1, bkv_p·256, 4, 256]` scales with `bkv_p`. At `bkv_p = 8` the
mixed peak is 7.195 MiB [`modelled`, substituted into a `measured` allocation list — the
`bkv_p = 8` arm was not compiled], nowhere near the ceiling, so `combined.md`'s
`B · bq_sz · bkv_p ≤ ~500` inequality does not bind.

What this file does **not** establish: it value-checks `(4, 4, 4)`, not `(8, 8, 8)`. A candidate that
takes `bkv_p` to 8 inherits the *argument* that tiling is numerically neutral — which is now backed by
a measurement at one factor-of-two step — but not a measurement at 8. Given how cheap this shard is
(it rode along on another job), wave 4 should attach `--tuples 8,8,8` to whatever job it runs next
rather than reasoning from this one.

## Verdict

**PASS** — both regimes pass at `(4, 4, 4)`, the negative control fires on 1024 of 1024 rows at the
same tuple, and the margins are bit-identical to the shipped configuration.

**decided:** `bkv_p` is numerically neutral at least through 4. C3 generalises to mixed and jumps the
ranking. `bkv_p = 8` remains unverified and should have a shard attached to the next job.
