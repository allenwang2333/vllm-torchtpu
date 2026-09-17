# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Nightly structured decoding benchmark on TPU.

Runs `benchmarks/benchmark_structured_decoding.py` end to end against a
real engine and gates on the result, so a change that makes grammar
masking expensive fails the nightly instead of quietly costing tokens/sec.

The gate is a *ratio*, not an absolute number. Each structured workload is
compared against the unconstrained baseline measured in the same process,
on the same chips, moments earlier; that cancels out agent-to-agent
variation, model changes and compiler drift, which an absolute tokens/sec
floor would keep tripping over. What it still catches is the failure mode
this suite exists for: masking work that scales with the batch instead of
staying in the fast paths -- for example, the packed bitmask being
materialized densely per step, or the unconstrained fast path in
`StructuredDecodingManager.prepare_structured_decoding_input` being lost
so every step pays a host round trip.

The full result set (throughput, TPOT, per-point mask bytes) is written as
JSON for artifact upload, so a regression can be diffed against previous
nightlies rather than just read off a failed assertion.

Correctness of the masks themselves is not retested here; that is
`tests/entrypoints/llm/test_structured_output.py`.

Marked `nightly` (too slow for PRs) and `benchmark` (so the general
nightly step can exclude it -- see `.buildkite/pipeline_tests.yml`).
"""

import json
import os
import sys
import tempfile
import time
import weakref
from pathlib import Path

import pytest
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

import benchmark_structured_decoding as bsd  # noqa: E402

pytestmark = [pytest.mark.nightly, pytest.mark.benchmark]

# Small enough that the sweep fits a nightly slot, and already pulled into
# the CI model cache by test_structured_output.py.
MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_MODEL_LEN = 1024
BATCH_SIZES = (1, 8, 32)
OUTPUT_LEN = 128
NUM_WARMUP = 1
NUM_RUNS = 3

STRUCTURED_SCHEMAS = ("flat_json", "complex_ast")
BASELINE_SCHEMA = "unconstrained"

# Share of the unconstrained baseline's throughput a structured workload
# must keep. Deliberately loose: this is a cliff detector, not a
# fine-grained perf tracker (use the uploaded JSON for trends). Grammar
# masking on the fast paths costs a few percent.
#
# The gap widens with batch size rather than staying constant, because the
# unconstrained baseline scales better than the masked path: a measured
# Qwen3-0.6B TP=1 sweep came in at ~0.81 for batch 1 and 8, but 0.515
# (flat_json) and 0.506 (complex_ast) at batch 32. A 0.50 floor leaves that
# largest cell 1-3% of headroom, so ordinary run-to-run noise would trip it
# and report a regression that is not one. 0.40 still catches the failure
# this gate exists for -- masking ceasing to be incremental, which costs
# far more than a few percent -- without alarming on noise.
MIN_THROUGHPUT_RATIO = float(os.environ.get("SD_MIN_THROUGHPUT_RATIO", "0.4"))

# run_in_docker.sh mounts this and copies it back for artifact upload.
RESULTS_DIR = os.environ.get("SD_BENCHMARK_RESULTS_DIR", "/perf_eval_results")


def _point_id(batch_size: int, schema: str) -> str:
    return f"{schema}-bs{batch_size}"


def _results_dir() -> Path:
    """Artifact directory, falling back to a temp dir off the CI agent."""
    candidate = Path(RESULTS_DIR)
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path(tempfile.mkdtemp(prefix="sd_benchmark_"))
    return candidate


@pytest.fixture(scope="module")
def bench_args():
    """CLI defaults for the sweep, built through the real parser."""
    return bsd.parse_args(
        [
            "--model",
            MODEL_NAME,
            "--batch-sizes",
            *[str(size) for size in BATCH_SIZES],
            "--output-len",
            str(OUTPUT_LEN),
            "--num-warmup",
            str(NUM_WARMUP),
            "--num-runs",
            str(NUM_RUNS),
            "--schemas",
            BASELINE_SCHEMA,
            *STRUCTURED_SCHEMAS,
            "--max-model-len",
            str(MAX_MODEL_LEN),
        ]
    )


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture, so hand out a weakref.proxy to let the
    # engine be collected on teardown (same pattern as
    # test_structured_output.py).
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=max(BATCH_SIZES),
        tensor_parallel_size=1,
    )

    yield weakref.proxy(llm)

    del llm
    cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def results(llm, bench_args):
    """Run the whole sweep once and publish it as JSON.

    Module scoped on purpose: the engine build and the first compile
    dominate, so re-running per assertion would multiply the nightly cost
    for no extra signal.
    """
    try:
        vocab_size = llm.llm_engine.model_config.get_vocab_size()
    except AttributeError:
        vocab_size = None

    points = []
    for batch_size in BATCH_SIZES:
        for schema in bench_args.schemas:
            points.append(
                bsd.benchmark_point(
                    llm, bsd.SCHEMAS[schema], batch_size, bench_args, vocab_size
                )
            )

    metadata = {
        "model": MODEL_NAME,
        "output_len": OUTPUT_LEN,
        "num_runs": NUM_RUNS,
        "tensor_parallel_size": 1,
        "min_throughput_ratio": MIN_THROUGHPUT_RATIO,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = _results_dir() / "structured_decoding_benchmark.json"
    path.write_text(json.dumps(bsd.results_to_json(points, metadata), indent=2))

    # The table is the thing a human opens the job log for.
    print(f"\nStructured decoding benchmark ({MODEL_NAME}):\n")
    print(bsd.format_results_table(points))
    print(f"\nWrote {path}")

    by_point = {_point_id(point.batch_size, point.schema): point for point in points}
    return by_point


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("schema", (BASELINE_SCHEMA,) + STRUCTURED_SCHEMAS)
def test_point_generated_tokens(results, batch_size, schema):
    """Every point must have actually produced the tokens it was timed on.

    Guards the ratios below: a workload that silently generated nothing
    would otherwise report an enormous throughput and pass.
    """
    point = results[_point_id(batch_size, schema)]

    assert point.generated_tokens == batch_size * OUTPUT_LEN
    assert point.throughput_tokens_per_s > 0
    assert point.tpot_ms > 0


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("schema", STRUCTURED_SCHEMAS)
def test_structured_throughput_vs_baseline(results, batch_size, schema):
    """A grammar must not cost more than half the baseline throughput."""
    point = results[_point_id(batch_size, schema)]
    baseline = results[_point_id(batch_size, BASELINE_SCHEMA)]

    ratio = point.throughput_tokens_per_s / baseline.throughput_tokens_per_s

    assert ratio >= MIN_THROUGHPUT_RATIO, (
        f"{schema} at batch {batch_size} kept only {ratio:.2%} of the "
        f"unconstrained baseline "
        f"({point.throughput_tokens_per_s:.1f} vs "
        f"{baseline.throughput_tokens_per_s:.1f} tok/s), below the "
        f"{MIN_THROUGHPUT_RATIO:.0%} floor. Structured decoding is no "
        "longer incremental over plain sampling: check that the bitmask "
        "fast paths still hit and that the mask is not being "
        "materialized densely per step."
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_recursive_grammar_stays_close_to_flat_grammar(results, batch_size):
    """A deep recursive schema costs more host-side bitmask work than a
    flat one, but it is the same masking path -- the gap should be small.
    A blow-up here points at per-request work that scales with grammar
    complexity leaking into the per-step path."""
    flat = results[_point_id(batch_size, "flat_json")]
    recursive = results[_point_id(batch_size, "complex_ast")]

    ratio = recursive.throughput_tokens_per_s / flat.throughput_tokens_per_s

    assert ratio >= MIN_THROUGHPUT_RATIO, (
        f"complex_ast at batch {batch_size} kept only {ratio:.2%} of "
        f"flat_json's throughput ({recursive.throughput_tokens_per_s:.1f} "
        f"vs {flat.throughput_tokens_per_s:.1f} tok/s)"
    )


def test_results_json_is_written_for_artifact_upload(results):
    """The nightly step uploads this file; a silent write failure would
    leave the run with a pass and no numbers."""
    path = _results_dir() / "structured_decoding_benchmark.json"
    payload = json.loads(path.read_text())

    assert payload["metadata"]["model"] == MODEL_NAME
    expected = len(BATCH_SIZES) * (1 + len(STRUCTURED_SCHEMAS))
    assert len(payload["results"]) == expected
    for entry in payload["results"]:
        assert entry["throughput_tokens_per_s"] > 0
        # Packed-vs-dense mask accounting needs the vocab size; its absence
        # means the engine stopped exposing model_config.
        assert entry["mask_memory"] is not None
