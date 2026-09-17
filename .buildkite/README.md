# Buildkite pipelines

`pipeline_tests.yml` runs tests in `tests/` marked `cpu_test` on `cpu_64_core` and excludes
them from the general TPU jobs. PR steps exclude `nightly` tests; nightly
steps include them. Both use the CI image built by `pipeline_build.yml`.
The CPU steps set `CPU_ONLY=1` to run Docker without privileged mode or device
mounts and select the JAX CPU backend. Host model caches and test results
use `$HOME/.cache/vllm-torchtpu-ci` on CPU agents and `/mnt/disks/persist`
on TPU agents.

For a test file that runs entirely without accelerator hardware, add
`pytestmark = pytest.mark.cpu_test` after its imports. Use
`@pytest.mark.cpu_test` for individual tests in mixed files. Module imports
must work on CPU because pytest collects tests before applying marker filters.
Mocks that use `torch.tpu` must supply that object on CPU hosts.

Run the CPU tests locally with the project's test dependencies installed:

```bash
JAX_PLATFORMS=cpu pytest tests/ -m cpu_test
```

## Perf benchmarks

Tests marked `benchmark` gate on measured throughput rather than
correctness. They live in `pipeline_perf.yml` rather than
`pipeline_tests.yml`, so a slow sweep does not serialize behind the general
suite, and the general nightly step excludes them
(`-m '... and not benchmark'`) so they do not run twice. A benchmark step
declares `artifact_paths: perf_eval_results/**/*`; write results under
`/perf_eval_results` inside the container and `run_in_docker.sh` copies
them back to the workspace for upload.

`tests/benchmarks/test_structured_decoding_nightly.py` is the current
example: it drives
[`benchmarks/benchmark_structured_decoding.py`](../benchmarks/benchmark_structured_decoding.py)
and compares each grammar against the unconstrained baseline measured in
the same run. Override the gate with `SD_MIN_THROUGHPUT_RATIO`.

Its step uploads the results twice: `structured_decoding_benchmark.json`
under the fixed name the test asserts on, and a
`structured_decoding_benchmark_<YYYY-MM-DD>.json` copy so a downloaded
artifact still says which nightly produced it. The step archives the JSON
before re-raising a failed gate, so a regression's numbers survive the red
build. To rehearse the step before it reaches a nightly, run it from
`pipeline_dev.yml` on your branch (see that file's header).
