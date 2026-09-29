# Two Shot qualification

These portable workers measure the maintained kernel through separate standalone
JAX and production Torch cohorts. Run them only on an allocated TPU7x `2x2x4`
slice, using the serving environment and compiler flags. Install this checkout
before launch. The runner records physical rank order, package versions, compiler
flags, and implementation and harness fingerprints. It never chooses a device
order on behalf of the launcher.

## Worker commands

For Torch, launch 32 workers with the site's normal Torch TPU environment,
including `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and device visibility. The native
arm calls `torch.distributed.all_reduce`; the candidate uses the same bridge as
K3's KDA output projection. Standalone JAX timings cannot admit serving shapes.

```bash
python scripts/collectives/two_shot_benchmark.py run \
  --framework torch --output runs/torch-smoke --smoke --save-executables
python scripts/collectives/two_shot_benchmark.py run \
  --framework torch --output runs/torch-reload \
  --reload-executables runs/torch-smoke
python scripts/collectives/two_shot_benchmark.py run \
  --framework torch --output runs/torch-primary --correctness-extras
```

Repeat the full command in a **fresh process launch**, with a new output and
compile-cache directory, to produce `runs/torch-confirmation`. Keep runtime flags
and rank placement identical. Output paths must be empty. `--batches 4` selects a
diagnostic subset; incomplete or smoke runs cannot qualify the serving manifest.
Serialized executables are trusted local artifacts and are checked against the
same source, runtime, and placement identity before loading.

For a four-process JAX launch, each process gets its own process ID:

```bash
python scripts/collectives/two_shot_benchmark.py run \
  --framework jax --output runs/jax-primary --distributed \
  --coordinator-address "$COORDINATOR" --num-processes 4 \
  --process-id "$PROCESS_ID" --correctness-extras
```

Initialize JAX distributed execution before enumerating devices. Use a second
fresh JAX launch for confirmation. JAX and Torch results remain separate even
when device placement matches.

## Evidence and admission

Full runs cover BF16 `[B,7168]`, where `B={1,2,4,8,16,32,64}`. Optional extras check
FP32 and odd widths. Before timing, every rank is checked against a NumPy FP64
reference over independently constructed, rank-distinct inputs, including changed
inputs and cancellation. Candidate error bounds allow FP32 accumulation plus one
final dtype rounding; native bounds permit low-precision reduction. The Torch
path also checks eager execution. Compiled
chains of lengths 1, 2, 3, 20, and 60 exercise repeated odd/even calls. Additional
protocol lifecycle tests live in `tests/kernels/test_two_shot_tpu.py`.

Timing uses five warmups and 25 paired trials of dependent chains of lengths 20
and 60. Arm order and low/high order alternate. Each chain consumes its preceding
full-output checksum behind an optimization barrier. All workers synchronize
before timing; the slowest process duration is taken after timing. Raw per-process
durations and signed per-trial slopes are retained. The reported slope includes
the common checksum/dependency work; nothing is silently subtracted.

Saved optimized HLO must contain exactly the requested number of native
all-reduces or Two Shot calls, without substituting the other implementation,
and retain checksum reductions between calls. The audit follows data operands from
each collective output into the next collective input, including parameter use
inside fusion bodies; control dependencies and unused fusion arguments do not
establish a data-dependent chain. The final collective must contribute to the
executable output. Before timing and again during admission, all processes must
have identical canonical collective dependency graphs for each arm and chain
length. Graph hashes retain operation kinds, shapes/layouts, constants, semantic
attributes, fusion bodies, and ordered data dependencies; instruction names and
compiler metadata are excluded. Full HLO artifacts are separately hashed.
Missing, unexpected, or divergent execution fails the run. Compile duration is separate from steady-state timing. Partial
results remain incomplete and cannot be admitted.

```bash
python scripts/collectives/two_shot_benchmark.py qualify \
  --primary runs/torch-primary/results.json \
  --confirmation runs/torch-confirmation/results.json \
  --output runs/qualification.json --report runs/qualification-report.json
python scripts/collectives/two_shot_benchmark.py run \
  --framework torch --output runs/torch-auto \
  --manifest runs/qualification.json
```

Admission recomputes statistics from raw samples, rechecks all-rank correctness,
and hashes and re-audits optimized HLO. Both independent Torch launches must save
strictly more than 5% and 0.5 microseconds, with candidate p90 below native p10.
The report retains rejected shapes. If no shape qualifies, the manifest has no
entries. The independent `--manifest` run executes the actual serving selector,
checks its emitted HLO, and adds unqualified `B=3` as a native fallback probe; it
does not feed admission. Review evidence before copying a qualified manifest into
`src/vllm_torchtpu/distributed/two_shot_qualification.json`.

These operator measurements do not establish a serving improvement. The PR must
also report the prescribed cold-start K3 A/B latency campaign and fixed GSM8K
accuracy comparison with identical assets and settings.

## Actual vLLM bucket and cache probe

`verify_torch_buckets.py` uses the real vLLM decorator, bucket backend, TPU compiler
adaptor, and Torch TPU compiler. It loads the canonical K3 model configuration
without weights and gives its small reduction module a synthetic admission list
containing only `B=1`. This exercises candidate `B=1` and native `B=3` even before
any performance admission. It does not alter the packaged qualification manifest.

Launch each command as 32 Torch TPU workers using the same rank layout:

```bash
python scripts/collectives/verify_torch_buckets.py \
  --model "$MODEL" --order eligible-first --phase cold \
  --cache-root runs/buckets-eligible-cache --output runs/buckets-eligible-cold
python scripts/collectives/verify_torch_buckets.py \
  --model "$MODEL" --order eligible-first --phase reload \
  --cache-root runs/buckets-eligible-cache --output runs/buckets-eligible-reload
```

Repeat with `--order native-first` and a separate cache root to reverse the initial
trace. The reload command must be a fresh worker launch. Cold runs require exactly
one compiled executable for each bucket and audit its actual optimized HLO.
Reloads must restore both AOT artifacts with zero tracing and zero TPU backend
compilations. Every rank checks changed inputs, cancellation, output shape, and
dtype through both buckets before and after warmup. Runtime/compiler sources must
remain unchanged between cold and reload launches. This probe complements the
operator harness's independent per-shape compilation; it tests the same traced
module and the production per-bucket specialization and cache machinery.
