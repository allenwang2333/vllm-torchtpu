# GDN v3 BATCHED decode: torn SSM state at 64 decodes

`race_check.py` runs the real Mosaic `fused_conv1d_gdn` kernel (no interpret
mode) on a tpu7x chip. It uses the pooled state layout of one DP rank of
Qwen3.5-397B-A17B. It compares every output and every written SSM state with a
float64 reference. It does not stop at the first bad state. For each bad state
it labels every wrong element by the value it holds:

| label | meaning |
|---|---|
| `stale` | the sequence's own state before the call (the write did not land) |
| `untouched` | the block's old bytes, for a fresh sequence |
| `pre[+d]` | the state that sequence `i+d` read, bit for bit |
| `new[+d]` | the state that sequence `i+d` should have written |
| `other` | none of these |

`row_map_kib` shows where the wrong bytes sit inside the 4 MiB state, by
1 KiB pool row.

Each case runs `--calls` times (default 3). Every call rebuilds the inputs and
the pool from the case name's seed, so all calls of a case get identical
inputs. Different digests across calls mean the kernel is nondeterministic.

## The suite

| case | decodes | decode_tile_size | BATCHED pipeline steps |
|---|---|---|---|
| `dec64_t64_r64` | 64 | 4 | 16 |
| `dec70_t128_mixed` | 70 (8 fresh) + 2 prefill | 4 | 18 |
| `dec37_t64` | 37 (1 fresh) | 4 | 10 |
| `dec7_t16` | 7 | 4 | 2 |
| `dec64_t64_r64_tile1` | 64 | 1 | 64 |
| `dec64_t64_r64_tile2` | 64 | 2 | 32 |
| `steps2_dec8` .. `steps8_dec32` | 8, 12, 16, 32 | 4 | 2, 3, 4, 8 |

`--suite quick` runs only `dec64_t64_r64` and `dec7_t16`. `--suite boundary`
runs 48, 56, 64 and 128 decodes at tile 4 (12, 14, 16, 32 steps) and 128
decodes at tile 2 (64 steps), with other seeds.

A call is `FAIL` when any state is off by more than 2e-2 of its max |ref|,
any conv state is not exact, any output is off by more than 2e-2, or the call
wrote bytes outside the sequences' blocks. On device, fp32 rounding alone
gives a state error of about 5e-4 for decodes.

## Run it on a tpu7x VM

From a clean checkout of the commit under test (the script only imports
`src/vllm_torchtpu` from the path you give it):

```bash
git clone https://github.com/vllm-project/vllm-torchtpu && cd vllm-torchtpu
git checkout 2b881a2cd50a25fb125ed9df405e70ff2ea579ad
# The repro is not in that commit; copy this folder next to the checkout.
cp -r /path/to/repro_golden_race /tmp/repro_golden_race

python3 -m venv /tmp/gdnrace && . /tmp/gdnrace/bin/activate
pip install jax==0.11.2 jaxlib==0.11.2 libtpu==0.0.47 numpy ml_dtypes

python3 /tmp/repro_golden_race/race_check.py . --calls 3 --out /tmp/golden.jsonl
```

It needs no vLLM install: it stubs the one vLLM setting the pool adapter
reads. It prints one `CALL` line per call and appends the full record,
including each bad state's classification, to `--out`. Expect about 5 minutes,
most of it compiles.

To test another revision, check it out (or `git worktree add`) and pass its
path instead of `.`.

## Run it through cdk

From a checkout that has this folder and both revisions in its history:

```bash
cd tools/kernel_evolve/gdn_v3/repro_golden_race
python3 submit_cdk.py --image <image with jax/jaxlib 0.11.2, libtpu 0.0.47> \
    --tag golden-race-check \
    --arm golden=2b881a2cd50a25fb125ed9df405e70ff2ea579ad \
    --arm fixed=<revision to compare>
```

It prints the job id. The job is one single-host tpu7x pod (2x2x1). The arms
run one after another, each in its own python process. `--dry-run` prints the
recipe and the pod script and submits nothing.

Read the result:

```bash
cdk job log <job-id> | grep -E '^(=== ARM|CALL|RESULT)'
cdk job sync-outputs <job-id>     # <arm>.jsonl and <arm>.log under outputs/
```

## What to look for

The pipeline in `outer_kernel` gives each state ref two VMEM slots. jax's
`emit_pipeline` issues, in step `s`: `copy_in(s+1)`, `wait_in(s)`, the body,
`copy_out(s)`, `wait_out(s-1)`. So in step `s+1` it starts `copy_in(s+2)`
into the slot that `copy_out(s)` may still be reading. If the read loses, the
pool block of the sequence at tile position `k` of tile `s` gets bytes of the
state that tile `s+2` read for its position `k`: label `pre[+2*tile]`, which is
`pre[+8]` at the default tile of 4.

What that predicts, and what to check in a failing record:

- every wrong element is `pre[+8]`, bit for bit, and `other` is 0;
- no victim sits in the last two tiles (they have no `s+2`);
- no victim's `+8` partner is a fresh sequence (it reads nothing);
- conv states and outputs stay correct (tile `s+2` computes from what it read);
- digests differ between calls of the same case.

The tear is timing dependent. On tpu7x with jax 0.11.2 / libtpu 0.0.47 we saw
it at tile 4 from about 14 pipeline steps up, and not at all at tile 1 or 2.
A passing call is bit-identical to a kernel without the race.
