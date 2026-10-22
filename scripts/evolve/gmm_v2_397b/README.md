# gmm_v2 campaign scorer: Qwen3.5-397B-A17B-FP8, AgentX c8, TP8+EP8, MTP k=3

| file | what it is |
|---|---|
| `bench.py` | the frozen scorer (`perfkit scorer run` interface): gmm_v2 microbenchmark, one TensorCore |
| `faults.py` | calibration arms: `golden_aa` (golden relabelled), `wrong_kernel`, `slow_kernel` (also the served perturbation, with a call-chain marker) |
| `calibrate.py` | judges calibration jobs and measures the noise floor from the `golden_aa` arm |
| `validity.py` | frozen validity adapter (`perfkit validity run`): AOT at all 30 serving shapes + emit_pipeline trip counts, CPU only |
| `test_bench.py` | CPU checks of `bench.py` against golden (env patch, served LIBTPU_INIT_ARGS, op names) |
| `served_recipe.py` | rebuilds the served recipe of record from the job's rendered JobSet |
| `jobsets/scorer-jobset.yml` | the JobSet `perfkit cluster` renders for one scoring job (golden + one candidate, frozen configs) |
| `jobsets/served-recipe-of-record.yml` | the served recipe of record (cdk job j-e8281eaa-7238-47b0-be0c), cdk additions stripped |
| `jobsets/served-phased-profile.yml` | the same with the phased profiler on, for the kernel breakdown |

Score one candidate (each arm root is the kernel directory, so its `tree` is the
git subtree of `src/vllm_torchtpu/kernels/megablox`):

    perfkit cluster score --bench scripts/evolve/gmm_v2_397b/bench.py \
      --image us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-caojx_google_com@sha256:78108de22c693ad17d2547c82076f34863752feda979e670d5a2230993e24cb2 \
      --jax 0.10.2 --libtpu 0.0.47 --arm-path . --source-dir . --kernel-dir . \
      --arm "golden=$GOLDEN/src/vllm_torchtpu/kernels/megablox:control" \
      --arm "cand=$WORKTREE/src/vllm_torchtpu/kernels/megablox" --candidate cand \
      --config decode_gate_up --config decode_down --config prefill_gate_up --config prefill_down

Calibrate (three jobs, then judge):

    python scripts/evolve/gmm_v2_397b/faults.py "$GOLDEN" /tmp/faults
    perfkit cluster submit ... --arm "golden=...:control" --arm golden_aa=/tmp/faults/golden_aa \
      --arm wrong_kernel=/tmp/faults/wrong_kernel --arm slow_kernel=/tmp/faults/slow_kernel
    python scripts/evolve/gmm_v2_397b/calibrate.py rep1.json rep2.json rep3.json --tolerance 0.08 \
      --weights decode_gate_up=0.45,decode_down=0.45,prefill_gate_up=0.05,prefill_down=0.05
