# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark structured (grammar-constrained) decoding on TPU.

Structured decoding costs two things on top of plain sampling, and this
script measures both:

* a host-side step (``SD:PrepareInput``) that maps the scheduler's packed
  int32 bitmask rows onto the runner's chunk-local batch rows and pushes
  them to the device, and
* a compiled device step (``SD:MaskLogits``) that unpacks the bitmask and
  drives the disallowed logits to ``-inf`` before sampling.

Both are emitted as ``TraceAnnotation`` scopes by
``vllm_torchtpu/runner/structured_decoding_manager.py``, so a profiled run
can attribute time to them directly instead of leaving the cost buried in
the sampling block. ``--enable-trace-report`` turns that on: the script
drives ``LLM.start_profile()`` / ``LLM.stop_profile()`` around one extra
iteration, then reads the Chrome-format trace the workers wrote and reports
per-invocation latency for each ``SD:`` scope.

Three built-in workloads cover the range of grammars the masking path sees:

* ``unconstrained`` -- no grammar at all, the baseline the other two are
  compared against.
* ``flat_json``     -- a flat key/value object, the common extraction shape.
* ``complex_ast``   -- a recursive expression-tree schema (``$ref`` back into
  ``$defs``, ``anyOf`` branches), which keeps the grammar FSM in states with
  large, frequently changing allowed-token sets.

Example:

    python benchmarks/benchmark_structured_decoding.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --batch-sizes 1 8 32 64 \\
        --output-len 512 \\
        --schemas unconstrained flat_json complex_ast \\
        --enable-trace-report

The module keeps the workload definitions, the trace aggregation and the
reporting free of TPU and vLLM imports so they can be unit tested on CPU;
only ``run_suite`` needs a device.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import gzip
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Callable, Iterable, Sequence

# Scopes emitted by StructuredDecodingManager. The runner appends
# "#k=v,...#" metadata to the scope name, so these are matched as prefixes.
SD_SCOPES: tuple[str, ...] = (
    "SD:PrepareInput",
    "SD:MaskLogits",
    "SD:PrepareSpecInput",
    "SD:MaskSpecLogits",
)

# Reported in the summary table; the spec scopes only appear when the run
# also enables speculative decoding, so they stay out of the fixed columns.
PRIMARY_SD_SCOPES: tuple[str, ...] = ("SD:PrepareInput", "SD:MaskLogits")

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 8, 32, 64)


@dataclasses.dataclass(frozen=True)
class SchemaSpec:
    """One benchmark workload.

    ``json_schema`` is None for the unconstrained baseline; everything else
    about the request (prompt, sampling) is identical across workloads so a
    row-to-row delta isolates the grammar cost.
    """

    name: str
    prompt: str
    description: str
    json_schema: dict[str, Any] | None = None

    @property
    def is_structured(self) -> bool:
        return self.json_schema is not None


FLAT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "age": {
            "type": "integer"
        },
        "email": {
            "type": "string"
        },
        "is_active": {
            "type": "boolean"
        },
        "role": {
            "type": "string",
            "enum": ["admin", "user", "guest"]
        },
    },
    "required": ["name", "age", "email", "is_active", "role"],
}

# Recursive arithmetic/comparison AST. `Node` is either a leaf (number or
# identifier) or another `Expression`, so the grammar can descend without a
# fixed bound -- the stress case for bitmask generation and masking.
COMPLEX_AST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "$defs": {
        "Expression": {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["+", "-", "*", "/", "==", "<", ">"],
                },
                "left": {
                    "$ref": "#/$defs/Node"
                },
                "right": {
                    "$ref": "#/$defs/Node"
                },
            },
            "required": ["op", "left", "right"],
        },
        "Node": {
            "anyOf": [
                {
                    "type": "number"
                },
                {
                    "type": "string"
                },
                {
                    "$ref": "#/$defs/Expression"
                },
            ]
        },
    },
    "properties": {
        "ast": {
            "$ref": "#/$defs/Expression"
        },
        "description": {
            "type": "string"
        },
    },
    "required": ["ast", "description"],
}

SCHEMAS: dict[str, SchemaSpec] = {
    "unconstrained":
    SchemaSpec(
        name="unconstrained",
        prompt="Describe the architecture of a modern TPU inference server: ",
        description="baseline, no grammar",
    ),
    "flat_json":
    SchemaSpec(
        name="flat_json",
        prompt=("Extract the user record as JSON with name, age, email, "
                "is_active and role: "),
        description="flat key/value object",
        json_schema=FLAT_JSON_SCHEMA,
    ),
    "complex_ast":
    SchemaSpec(
        name="complex_ast",
        prompt=("Represent the expression ((a + 2) * (b - 3)) == c as a "
                "nested JSON AST with a description: "),
        description="recursive expression tree ($ref / anyOf)",
        json_schema=COMPLEX_AST_SCHEMA,
    ),
}


@dataclasses.dataclass
class ScopeStats:
    """Aggregated timings for one ``SD:`` trace scope."""

    scope: str
    count: int
    total_ms: float
    mean_ms: float
    p50_ms: float
    p99_ms: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class MaskMemory:
    """Bytes moved per step by the packed bitmask vs a dense mask.

    The runner ships ``ceil(vocab/32)`` int32 words per row and unpacks them
    inside the compiled kernel. The alternative -- materializing the mask on
    the host -- would move a full row per request, which is what
    ``dense_bool_bytes`` / ``dense_logits_bytes`` price out.
    """

    packed_bytes: int
    dense_bool_bytes: int
    dense_logits_bytes: int

    @property
    def savings_vs_bool(self) -> float:
        return 1.0 - self.packed_bytes / self.dense_bool_bytes

    @property
    def savings_vs_logits(self) -> float:
        return 1.0 - self.packed_bytes / self.dense_logits_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "packed_bytes": self.packed_bytes,
            "dense_bool_bytes": self.dense_bool_bytes,
            "dense_logits_bytes": self.dense_logits_bytes,
            "savings_vs_bool": self.savings_vs_bool,
            "savings_vs_logits": self.savings_vs_logits,
        }


ScopeStatsMap = dict[str, "ScopeStats"]


@dataclasses.dataclass
class BenchmarkResult:
    """One (batch size, schema) point."""

    batch_size: int
    schema: str
    latency_s: float
    generated_tokens: int
    latencies_s: list[float] = dataclasses.field(default_factory=list)
    scope_stats: ScopeStatsMap = dataclasses.field(default_factory=dict)
    trace_window_s: float | None = None
    mask_memory: MaskMemory | None = None

    @property
    def throughput_tokens_per_s(self) -> float:
        if self.latency_s <= 0:
            return 0.0
        return self.generated_tokens / self.latency_s

    @property
    def tpot_ms(self) -> float:
        """Mean time per output token, in ms, across the whole batch."""
        tokens_per_request = self.generated_tokens / max(self.batch_size, 1)
        if tokens_per_request <= 0:
            return 0.0
        return self.latency_s * 1e3 / tokens_per_request

    @property
    def sd_overhead_pct(self) -> float:
        """Share of the traced window spent inside the ``SD:`` scopes.

        ``SD:PrepareInput`` and ``SD:MaskLogits`` are disjoint sibling
        scopes, so summing them does not double count.
        """
        if not self.trace_window_s or self.trace_window_s <= 0:
            return 0.0
        total_ms = sum(stats.total_ms
                       for scope, stats in self.scope_stats.items()
                       if scope in PRIMARY_SD_SCOPES)
        return 100.0 * total_ms / (self.trace_window_s * 1e3)

    def as_dict(self) -> dict[str, Any]:
        scopes = {
            scope: stats.as_dict()
            for scope, stats in self.scope_stats.items()
        }
        memory = self.mask_memory.as_dict() if self.mask_memory else None
        return {
            "batch_size": self.batch_size,
            "schema": self.schema,
            "latency_s": self.latency_s,
            "latencies_s": self.latencies_s,
            "generated_tokens": self.generated_tokens,
            "throughput_tokens_per_s": self.throughput_tokens_per_s,
            "tpot_ms": self.tpot_ms,
            "trace_window_s": self.trace_window_s,
            "sd_overhead_pct": self.sd_overhead_pct,
            "scope_stats": scopes,
            "mask_memory": memory,
        }


def estimate_mask_memory(vocab_size: int,
                         num_reqs: int,
                         logits_dtype_bytes: int = 2) -> MaskMemory:
    """Per-step mask bytes for `num_reqs` rows of `vocab_size` logits."""
    if vocab_size <= 0 or num_reqs <= 0:
        raise ValueError("vocab_size and num_reqs must be positive")
    num_words = math.ceil(vocab_size / 32)
    return MaskMemory(
        packed_bytes=num_reqs * num_words * 4,
        dense_bool_bytes=num_reqs * vocab_size,
        dense_logits_bytes=num_reqs * vocab_size * logits_dtype_bytes,
    )


def scope_of(event_name: str, scopes: Sequence[str] = SD_SCOPES) -> str | None:
    """Map a raw trace event name to one of `scopes`, else None.

    ``TraceAnnotation`` renders as ``"SD:MaskLogits#num_reqs=8,...#"``, so
    the match is on the prefix up to the metadata delimiter. Longer scope
    names are tried first so ``SD:MaskSpecLogits`` is not swallowed by a
    prefix of ``SD:MaskLogits`` should the names ever overlap.
    """
    for scope in sorted(scopes, key=len, reverse=True):
        if event_name == scope or event_name.startswith(scope + "#"):
            return scope
    return None


def aggregate_scope_durations(
    events: Iterable[tuple[str, float]],
    scopes: Sequence[str] = SD_SCOPES,
) -> dict[str, ScopeStats]:
    """Aggregate `(event_name, duration_us)` pairs per ``SD:`` scope."""
    buckets: dict[str, list[float]] = {}
    for name, duration_us in events:
        scope = scope_of(name, scopes)
        if scope is None:
            continue
        buckets.setdefault(scope, []).append(duration_us / 1e3)

    stats: dict[str, ScopeStats] = {}
    for scope, samples in buckets.items():
        ordered = sorted(samples)
        stats[scope] = ScopeStats(
            scope=scope,
            count=len(ordered),
            total_ms=sum(ordered),
            mean_ms=statistics.fmean(ordered),
            p50_ms=_percentile(ordered, 0.50),
            p99_ms=_percentile(ordered, 0.99),
        )
    return stats


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted sequence."""
    if not ordered:
        return 0.0
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def parse_chrome_trace(path: str,
                       max_retries: int = 5,
                       retry_delay: float = 3.0) -> list[tuple[str, float]]:
    """Read `(name, dur_us)` for the complete events in a Chrome trace.

    Accepts the plain and gzipped files the torch profiler writes. Only
    duration events (``ph == "X"``) carry a ``dur``; instant and metadata
    events are skipped. Retries if the file is still being written by a
    worker process.
    """
    payload = None
    for attempt in range(max_retries):
        try:
            opener: Callable[..., Any] = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt") as handle:
                payload = json.load(handle)
            break
        except (json.JSONDecodeError, OSError, EOFError, gzip.BadGzipFile) as exc:
            if isinstance(exc, FileNotFoundError):
                return []
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"[warn] Skipping unparseable trace {path}: {exc}")
                return []

    if payload is None:
        return []

    raw_events = (payload.get("traceEvents", [])
                  if isinstance(payload, dict) else payload)
    events: list[tuple[str, float]] = []
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        if event.get("ph") != "X" or "dur" not in event:
            continue
        name = event.get("name")
        if isinstance(name, str):
            events.append((name, float(event["dur"])))
    return events


def find_trace_files(trace_dir: str) -> list[str]:
    """Every Chrome trace under `trace_dir`, as deduped real paths.

    The profiler writes one file per worker, nested under a per-capture run
    directory, so this globs recursively.
    """
    patterns = ("**/*.pt.trace.json", "**/*.pt.trace.json.gz",
                "**/*.trace.json", "**/*.trace.json.gz")
    found: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(trace_dir, pattern),
                              recursive=True):
            found.add(os.path.realpath(path))
    return sorted(found)


def collect_trace_events(
        trace_dir: str,
        exclude: Iterable[str] = (),
) -> list[tuple[str, float]]:
    """Read the Chrome traces under `trace_dir`, skipping `exclude`.

    Every capture in a run shares one ``torch_profiler_dir``, so a point
    passes the files that already existed before it armed the profiler;
    what is left is exactly what its own iteration produced.
    """
    skip = {os.path.realpath(path) for path in exclude}
    events: list[tuple[str, float]] = []
    for path in find_trace_files(trace_dir):
        if path in skip:
            continue
        events.extend(parse_chrome_trace(path))
    return events


def format_results_table(results: Sequence[BenchmarkResult]) -> str:
    """Render the summary table, grouped by batch size."""
    header = (f"{'Batch':>5} | {'Schema':<14} | {'Tok/s':>10} | "
              f"{'TPOT (ms)':>10} | {'PrepareInput':>13} | "
              f"{'MaskLogits':>11} | {'SD %':>6}")
    rule = "-" * len(header)
    lines = [header, rule]

    last_batch: int | None = None
    for result in results:
        if last_batch is not None and result.batch_size != last_batch:
            lines.append(rule)
        last_batch = result.batch_size
        prepare = result.scope_stats.get("SD:PrepareInput")
        mask = result.scope_stats.get("SD:MaskLogits")
        lines.append(f"{result.batch_size:>5} | {result.schema:<14} | "
                     f"{result.throughput_tokens_per_s:>10.2f} | "
                     f"{result.tpot_ms:>10.2f} | "
                     f"{_fmt_scope(prepare):>13} | "
                     f"{_fmt_scope(mask):>11} | "
                     f"{result.sd_overhead_pct:>5.2f}%")
    return "\n".join(lines)


def _fmt_scope(stats: ScopeStats | None) -> str:
    return "-" if stats is None else f"{stats.mean_ms:.3f}"


def results_to_json(results: Sequence[BenchmarkResult],
                    metadata: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable payload for CI regression tracking."""
    return {
        "metadata": metadata,
        "results": [result.as_dict() for result in results],
    }


def build_sampling_params(spec: SchemaSpec, output_len: int,
                          temperature: float, ignore_eos: bool) -> Any:
    """Build vLLM ``SamplingParams`` for one workload.

    Imported lazily so the workload definitions and the reporting helpers
    stay importable (and unit testable) without vLLM installed.
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    structured_outputs = (StructuredOutputsParams(
        json=spec.json_schema) if spec.is_structured else None)
    return SamplingParams(
        temperature=temperature,
        max_tokens=output_len,
        ignore_eos=ignore_eos,
        structured_outputs=structured_outputs,
    )


def _generated_tokens(outputs: Iterable[Any]) -> int:
    return sum(
        len(completion.token_ids) for output in outputs
        for completion in output.outputs)


def run_workload(llm: Any, spec: SchemaSpec, batch_size: int,
                 sampling_params: Any) -> tuple[float, int]:
    """One ``generate`` call; returns (wall seconds, generated tokens)."""
    prompts = [spec.prompt] * batch_size
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    latency = time.perf_counter() - start
    return latency, _generated_tokens(outputs)


def benchmark_point(
    llm: Any,
    spec: SchemaSpec,
    batch_size: int,
    args: argparse.Namespace,
    vocab_size: int | None = None,
) -> BenchmarkResult:
    """Warm up, measure, and (optionally) profile one (batch, schema)."""
    sampling_params = build_sampling_params(spec, args.output_len,
                                            args.temperature, args.ignore_eos)

    for _ in range(args.num_warmup):
        run_workload(llm, spec, batch_size, sampling_params)

    # Keep each latency paired with the tokens that run actually produced:
    # without --ignore-eos a grammar can terminate at different lengths from
    # run to run, and dividing one run's tokens by another's time would
    # invent throughput that was never measured.
    samples: list[tuple[float, int]] = []
    for _ in range(args.num_runs):
        samples.append(run_workload(llm, spec, batch_size, sampling_params))
    samples.sort(key=lambda sample: sample[0])
    median_latency, median_tokens = samples[(len(samples) - 1) // 2]

    result = BenchmarkResult(
        batch_size=batch_size,
        schema=spec.name,
        latency_s=median_latency,
        generated_tokens=median_tokens,
        latencies_s=[latency for latency, _ in samples],
    )

    if vocab_size:
        result.mask_memory = estimate_mask_memory(vocab_size, batch_size)

    if args.enable_trace_report and spec.is_structured:
        # The engine writes every capture under the one profiler directory
        # it was built with, so the previous points' traces are already
        # there; snapshot them and read only what this capture adds.
        os.makedirs(args.trace_dir, exist_ok=True)
        before = find_trace_files(args.trace_dir)
        llm.start_profile()
        window, _ = run_workload(llm, spec, batch_size, sampling_params)
        llm.stop_profile()
        # The workers flush asynchronously on stop_profile(); give the files
        # a moment to land before reading them.
        time.sleep(args.trace_flush_s)
        result.trace_window_s = window
        result.scope_stats = aggregate_scope_durations(
            collect_trace_events(args.trace_dir, exclude=before))

    return result


def run_suite(args: argparse.Namespace) -> list[BenchmarkResult]:
    """Build the engine once and sweep (batch size x schema)."""
    from vllm import LLM

    specs = [SCHEMAS[name] for name in args.schemas]
    max_num_seqs = max(args.batch_sizes)

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    if args.enable_trace_report:
        llm_kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": args.trace_dir,
        }
    llm = LLM(**llm_kwargs)

    vocab_size = None
    try:
        vocab_size = llm.llm_engine.model_config.get_vocab_size()
    except AttributeError:
        pass

    results: list[BenchmarkResult] = []
    for batch_size in args.batch_sizes:
        for spec in specs:
            print(f"[bench] batch_size={batch_size} schema={spec.name} "
                  f"({spec.description})")
            results.append(
                benchmark_point(llm, spec, batch_size, args, vocab_size))
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Benchmark structured decoding throughput, latency and "
                     "SD: trace attribution on TPU."))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-sizes",
                        type=int,
                        nargs="+",
                        default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument(
        "--schemas",
        type=str,
        nargs="+",
        default=["unconstrained", "flat_json", "complex_ast"],
        choices=sorted(SCHEMAS),
        help="Workloads to run; 'unconstrained' is the baseline.")
    parser.add_argument("--num-warmup",
                        type=int,
                        default=1,
                        help="Untimed iterations; the first pays XLA "
                        "compilation and grammar compilation.")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep generating to --output-len so every step exercises the "
        "masking path and token counts are comparable across schemas.")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--enable-trace-report",
        action="store_true",
        help="Capture a profiled iteration per structured workload and "
        "report per-invocation SD: scope latency.")
    parser.add_argument("--trace-dir", type=str, default="sd_traces")
    parser.add_argument(
        "--trace-flush-s",
        type=float,
        default=10.0,
        help="Seconds to wait for workers to flush trace files.")
    parser.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help="Path to write the markdown summary table.")
    parser.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args(argv)
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        parser.error("--batch-sizes must be positive")
    if args.output_len <= 0:
        parser.error("--output-len must be positive")
    if args.num_runs <= 0:
        parser.error("--num-runs must be positive")
    # Preserve the caller's order but drop duplicates.
    args.schemas = list(dict.fromkeys(args.schemas))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_suite(args)

    table = format_results_table(results)
    print()
    print(table)

    if args.output_markdown:
        with open(args.output_markdown, "w") as handle:
            handle.write(table + "\n")
        print(f"\nWrote {args.output_markdown}")

    if args.output_json:
        metadata = {
            "model": args.model,
            "output_len": args.output_len,
            "num_runs": args.num_runs,
            "tensor_parallel_size": args.tensor_parallel_size,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(args.output_json, "w") as handle:
            json.dump(results_to_json(results, metadata), handle, indent=2)
        print(f"\nWrote {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
