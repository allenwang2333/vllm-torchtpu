# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmarks/benchmark_structured_decoding.py.

These cover the parts of the benchmark that must stay correct without a
TPU: the workload catalog, the Chrome-trace parsing and ``SD:`` scope
aggregation, the derived throughput/latency metrics, and the two report
formats. The device path (``run_suite``) is exercised with a fake engine so
the orchestration -- warmup, repeat count, profiled iteration -- is checked
without loading a model.
"""

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

import benchmark_structured_decoding as bsd  # noqa: E402

pytestmark = pytest.mark.cpu_test


def _write_capture(run_dir: Path, filename: str,
                   events: list[tuple[str, float]]) -> Path:
    """Write one worker's Chrome trace into a capture run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    path.write_text(
        json.dumps({
            "traceEvents": [{
                "ph": "X",
                "name": name,
                "dur": duration
            } for name, duration in events]
        }))
    return path


class TestSchemaCatalog:

    def test_expected_workloads_are_registered(self):
        assert set(
            bsd.SCHEMAS) == {"unconstrained", "flat_json", "complex_ast"}

    def test_only_the_baseline_is_unconstrained(self):
        assert not bsd.SCHEMAS["unconstrained"].is_structured
        assert bsd.SCHEMAS["flat_json"].is_structured
        assert bsd.SCHEMAS["complex_ast"].is_structured

    def test_schemas_are_json_serializable_objects(self):
        for name in ("flat_json", "complex_ast"):
            schema = bsd.SCHEMAS[name].json_schema
            assert schema["type"] == "object"
            assert schema["required"]
            # Round-trips, so xgrammar receives exactly what is declared.
            assert json.loads(json.dumps(schema)) == schema

    def test_complex_ast_is_actually_recursive(self):
        """The stress case only stresses anything if Node can nest."""
        defs = bsd.COMPLEX_AST_SCHEMA["$defs"]
        node_branches = defs["Node"]["anyOf"]
        assert {"$ref": "#/$defs/Expression"} in node_branches
        expression = defs["Expression"]["properties"]
        assert expression["left"] == {"$ref": "#/$defs/Node"}
        assert expression["right"] == {"$ref": "#/$defs/Node"}

    def test_every_workload_has_a_prompt(self):
        for spec in bsd.SCHEMAS.values():
            assert spec.prompt.strip()
            assert spec.description.strip()


class TestScopeMatching:

    def test_bare_scope_name_matches(self):
        assert bsd.scope_of("SD:MaskLogits") == "SD:MaskLogits"

    def test_annotated_scope_name_matches(self):
        name = ("SD:PrepareInput#num_reqs=2,padded_num_reqs=4,"
                "cur_start_idx=0,cur_end_idx=2#")
        assert bsd.scope_of(name) == "SD:PrepareInput"

    def test_spec_scope_is_not_swallowed_by_the_shorter_name(self):
        assert bsd.scope_of("SD:MaskSpecLogits#num_reqs=1#") == \
            "SD:MaskSpecLogits"

    def test_unrelated_event_is_ignored(self):
        assert bsd.scope_of("aten::copy_") is None
        # A scope name without the metadata delimiter is not a prefix match:
        # it belongs to some other annotation.
        assert bsd.scope_of("SD:MaskLogitsExtra") is None


class TestAggregateScopeDurations:

    def test_groups_and_summarizes_per_scope(self):
        events = [
            ("SD:PrepareInput#num_reqs=8#", 100.0),
            ("SD:PrepareInput#num_reqs=8#", 300.0),
            ("SD:MaskLogits#num_reqs=8#", 50.0),
            ("aten::mul", 9999.0),
        ]

        stats = bsd.aggregate_scope_durations(events)

        assert set(stats) == {"SD:PrepareInput", "SD:MaskLogits"}
        prepare = stats["SD:PrepareInput"]
        assert prepare.count == 2
        # Durations are microseconds in the trace, milliseconds in the report.
        assert prepare.total_ms == pytest.approx(0.4)
        assert prepare.mean_ms == pytest.approx(0.2)
        assert stats["SD:MaskLogits"].mean_ms == pytest.approx(0.05)

    def test_percentiles_use_nearest_rank(self):
        events = [("SD:MaskLogits", float(us))
                  for us in range(1, 101)]  # 1..100 us

        stats = bsd.aggregate_scope_durations(events)["SD:MaskLogits"]

        assert stats.count == 100
        assert stats.p50_ms == pytest.approx(0.050)
        assert stats.p99_ms == pytest.approx(0.099)

    def test_no_matching_events_yields_empty_stats(self):
        assert bsd.aggregate_scope_durations([("aten::add", 1.0)]) == {}


class TestChromeTraceParsing:

    def _write_trace(self, path: Path, events, gzipped: bool = False):
        payload = {"traceEvents": events}
        if gzipped:
            with gzip.open(path, "wt") as handle:
                json.dump(payload, handle)
        else:
            path.write_text(json.dumps(payload))

    def test_reads_duration_events_only(self, tmp_path):
        trace = tmp_path / "worker0.pt.trace.json"
        self._write_trace(
            trace,
            [
                {
                    "ph": "X",
                    "name": "SD:MaskLogits#num_reqs=4#",
                    "dur": 42.0
                },
                # Instant and metadata events carry no duration.
                {
                    "ph": "i",
                    "name": "SD:MaskLogits"
                },
                {
                    "ph": "M",
                    "name": "process_name"
                },
            ],
        )

        assert bsd.parse_chrome_trace(str(trace)) == [
            ("SD:MaskLogits#num_reqs=4#", 42.0)
        ]

    def test_reads_gzipped_traces(self, tmp_path):
        trace = tmp_path / "worker0.pt.trace.json.gz"
        self._write_trace(trace, [{
            "ph": "X",
            "name": "SD:PrepareInput",
            "dur": 7.0
        }],
                          gzipped=True)

        assert bsd.parse_chrome_trace(str(trace)) == [("SD:PrepareInput", 7.0)]

    def test_collects_every_worker_trace_in_the_directory(self, tmp_path):
        """TP>1 writes one trace per worker; all of them must be counted."""
        (tmp_path / "rank_0").mkdir()
        (tmp_path / "rank_1").mkdir()
        self._write_trace(tmp_path / "rank_0" / "a.pt.trace.json", [{
            "ph": "X",
            "name": "SD:MaskLogits",
            "dur": 10.0
        }])
        self._write_trace(tmp_path / "rank_1" / "b.pt.trace.json.gz",
                          [{
                              "ph": "X",
                              "name": "SD:MaskLogits",
                              "dur": 30.0
                          }],
                          gzipped=True)

        stats = bsd.aggregate_scope_durations(
            bsd.collect_trace_events(str(tmp_path)))

        assert stats["SD:MaskLogits"].count == 2
        assert stats["SD:MaskLogits"].total_ms == pytest.approx(0.04)

    def test_empty_directory_is_not_an_error(self, tmp_path):
        assert bsd.collect_trace_events(str(tmp_path)) == []
        assert bsd.find_trace_files(str(tmp_path)) == []

    def test_excluded_files_are_skipped(self, tmp_path):
        """Consecutive points share one profiler directory, so a point must
        only read the traces its own capture added."""
        self._write_trace(tmp_path / "old.pt.trace.json", [{
            "ph": "X",
            "name": "SD:MaskLogits",
            "dur": 999.0
        }])
        before = bsd.find_trace_files(str(tmp_path))
        self._write_trace(tmp_path / "new.pt.trace.json", [{
            "ph": "X",
            "name": "SD:MaskLogits",
            "dur": 20.0
        }])

        events = bsd.collect_trace_events(str(tmp_path), exclude=before)

        assert events == [("SD:MaskLogits", 20.0)]


class TestMaskMemory:

    def test_packed_bitmask_is_eight_times_smaller_than_a_bool_mask(self):
        memory = bsd.estimate_mask_memory(vocab_size=32_000, num_reqs=8)

        assert memory.packed_bytes == 8 * (32_000 // 32) * 4
        assert memory.dense_bool_bytes == 8 * 32_000
        assert memory.savings_vs_bool == pytest.approx(0.875)
        # bf16 logits: 2 bytes per token, so 16x.
        assert memory.savings_vs_logits == pytest.approx(0.9375)

    def test_vocab_not_a_multiple_of_32_rounds_up_to_a_whole_word(self):
        memory = bsd.estimate_mask_memory(vocab_size=50, num_reqs=1)
        assert memory.packed_bytes == 2 * 4

    @pytest.mark.parametrize("vocab_size,num_reqs", [(0, 4), (64, 0), (-1, 1)])
    def test_rejects_non_positive_shapes(self, vocab_size, num_reqs):
        with pytest.raises(ValueError):
            bsd.estimate_mask_memory(vocab_size, num_reqs)


class TestBenchmarkResultMetrics:

    def test_throughput_and_tpot(self):
        result = bsd.BenchmarkResult(batch_size=8,
                                     schema="flat_json",
                                     latency_s=2.0,
                                     generated_tokens=1024)

        assert result.throughput_tokens_per_s == pytest.approx(512.0)
        # 1024 tokens / 8 requests = 128 tokens per request over 2s.
        assert result.tpot_ms == pytest.approx(15.625)

    def test_zero_latency_does_not_divide_by_zero(self):
        result = bsd.BenchmarkResult(batch_size=1,
                                     schema="flat_json",
                                     latency_s=0.0,
                                     generated_tokens=0)
        assert result.throughput_tokens_per_s == 0.0
        assert result.tpot_ms == 0.0

    def test_sd_overhead_sums_the_primary_scopes_only(self):
        result = bsd.BenchmarkResult(
            batch_size=8,
            schema="flat_json",
            latency_s=1.0,
            generated_tokens=512,
            trace_window_s=1.0,
            scope_stats=bsd.aggregate_scope_durations([
                ("SD:PrepareInput", 4_000.0),
                ("SD:MaskLogits", 6_000.0),
                # Spec scopes are reported but excluded from the headline
                # percentage, which would otherwise double count.
                ("SD:MaskSpecLogits", 90_000.0),
            ]),
        )

        assert result.sd_overhead_pct == pytest.approx(1.0)

    def test_overhead_is_zero_without_a_traced_window(self):
        result = bsd.BenchmarkResult(batch_size=1,
                                     schema="flat_json",
                                     latency_s=1.0,
                                     generated_tokens=10)
        assert result.sd_overhead_pct == 0.0


class TestReporting:

    def _results(self):
        baseline = bsd.BenchmarkResult(batch_size=8,
                                       schema="unconstrained",
                                       latency_s=1.0,
                                       generated_tokens=4096)
        structured = bsd.BenchmarkResult(
            batch_size=8,
            schema="flat_json",
            latency_s=1.1,
            generated_tokens=4096,
            trace_window_s=1.1,
            scope_stats=bsd.aggregate_scope_durations([
                ("SD:PrepareInput#num_reqs=8#", 80.0),
                ("SD:MaskLogits#num_reqs=8#", 95.0),
            ]),
            mask_memory=bsd.estimate_mask_memory(32_000, 8),
        )
        return [baseline, structured]

    def test_table_has_a_row_per_point(self):
        table = bsd.format_results_table(self._results())
        lines = [line for line in table.splitlines() if "|" in line]

        assert lines[0].strip().startswith("Batch")
        assert any("unconstrained" in line for line in lines)
        assert any("flat_json" in line for line in lines)

    def test_table_shows_a_dash_when_a_scope_was_not_traced(self):
        table = bsd.format_results_table(self._results())
        unconstrained_row = next(line for line in table.splitlines()
                                 if "unconstrained" in line)
        assert unconstrained_row.count("-") >= 2

    def test_json_payload_round_trips(self):
        payload = bsd.results_to_json(self._results(), {"model": "test"})
        restored = json.loads(json.dumps(payload))

        assert restored["metadata"] == {"model": "test"}
        assert len(restored["results"]) == 2
        structured = restored["results"][1]
        assert structured["schema"] == "flat_json"
        assert structured["throughput_tokens_per_s"] > 0
        assert structured["scope_stats"]["SD:MaskLogits"]["count"] == 1
        assert structured["mask_memory"]["savings_vs_bool"] > 0

    def test_json_payload_tolerates_a_point_without_trace_data(self):
        payload = bsd.results_to_json(self._results()[:1], {})
        assert payload["results"][0]["scope_stats"] == {}
        assert payload["results"][0]["mask_memory"] is None


class TestArgParsing:

    def test_defaults_match_the_documented_sweep(self):
        args = bsd.parse_args([])
        assert args.batch_sizes == [1, 8, 32, 64]
        assert args.schemas == ["unconstrained", "flat_json", "complex_ast"]
        assert args.output_len == 512
        assert args.ignore_eos is True
        assert args.enable_trace_report is False

    def test_explicit_sweep_is_honored(self):
        args = bsd.parse_args([
            "--model", "my/model", "--batch-sizes", "2", "4", "--output-len",
            "16", "--schemas", "flat_json", "--enable-trace-report"
        ])

        assert args.model == "my/model"
        assert args.batch_sizes == [2, 4]
        assert args.schemas == ["flat_json"]
        assert args.enable_trace_report is True

    def test_duplicate_schemas_are_collapsed_in_order(self):
        args = bsd.parse_args(
            ["--schemas", "flat_json", "unconstrained", "flat_json"])
        assert args.schemas == ["flat_json", "unconstrained"]

    def test_ignore_eos_can_be_turned_off(self):
        assert bsd.parse_args(["--no-ignore-eos"]).ignore_eos is False

    @pytest.mark.parametrize("argv", [
        ["--batch-sizes", "0"],
        ["--output-len", "0"],
        ["--num-runs", "0"],
        ["--schemas", "does_not_exist"],
    ])
    def test_invalid_arguments_exit(self, argv):
        with pytest.raises(SystemExit):
            bsd.parse_args(argv)


class FakeLLM:
    """Minimal stand-in for vLLM's ``LLM`` offline API."""

    def __init__(self, tokens_per_request: int = 4):
        self.tokens_per_request = tokens_per_request
        self.generate_calls: list[int] = []
        self.profile_started = 0
        self.profile_stopped = 0

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.generate_calls.append(len(prompts))
        completion = SimpleNamespace(
            token_ids=list(range(self.tokens_per_request)))
        return [SimpleNamespace(outputs=[completion]) for _ in prompts]

    def start_profile(self):
        self.profile_started += 1

    def stop_profile(self):
        self.profile_stopped += 1


class TestBenchmarkPoint:

    def _args(self, **overrides):
        args = bsd.parse_args([])
        args.num_warmup = 1
        args.num_runs = 2
        args.trace_flush_s = 0.0
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    @pytest.fixture(autouse=True)
    def stub_sampling_params(self, monkeypatch):
        """``build_sampling_params`` imports vLLM, which needs no exercise
        here: the fake engine ignores the object."""
        monkeypatch.setattr(bsd, "build_sampling_params",
                            lambda *a, **kw: object())

    def test_warmup_iterations_are_not_measured(self):
        llm = FakeLLM()
        args = self._args()

        result = bsd.benchmark_point(llm, bsd.SCHEMAS["flat_json"], 4, args)

        # 1 warmup + 2 measured, each with batch_size prompts.
        assert llm.generate_calls == [4, 4, 4]
        assert len(result.latencies_s) == 2
        assert result.generated_tokens == 4 * 4

    def test_mask_memory_is_reported_when_the_vocab_is_known(self):
        result = bsd.benchmark_point(FakeLLM(),
                                     bsd.SCHEMAS["flat_json"],
                                     2,
                                     self._args(),
                                     vocab_size=32_000)
        assert result.mask_memory is not None
        assert result.mask_memory.packed_bytes > 0

    def test_no_profiling_without_the_flag(self):
        llm = FakeLLM()
        bsd.benchmark_point(llm, bsd.SCHEMAS["flat_json"], 2, self._args())
        assert llm.profile_started == 0
        assert llm.profile_stopped == 0

    def test_unconstrained_baseline_is_never_profiled(self, tmp_path):
        """No grammar means no SD: scopes, so the extra iteration would
        only cost time."""
        llm = FakeLLM()
        args = self._args(enable_trace_report=True,
                          trace_dir=str(tmp_path / "traces"))

        result = bsd.benchmark_point(llm, bsd.SCHEMAS["unconstrained"], 2,
                                     args)

        assert llm.profile_started == 0
        assert result.scope_stats == {}

    def test_structured_workload_captures_and_parses_a_trace(self, tmp_path):
        trace_root = tmp_path / "traces"
        llm = FakeLLM()
        args = self._args(enable_trace_report=True, trace_dir=str(trace_root))

        # The real engine writes into the one torch_profiler_dir it was
        # built with, during stop_profile(); emulate that.
        def fake_stop_profile():
            llm.profile_stopped += 1
            _write_capture(trace_root / "run_1", "rank_0.pt.trace.json", [
                ("SD:PrepareInput#num_reqs=2#", 120.0),
                ("SD:MaskLogits#num_reqs=2#", 80.0),
            ])

        llm.stop_profile = fake_stop_profile

        result = bsd.benchmark_point(llm, bsd.SCHEMAS["flat_json"], 2, args)

        assert llm.profile_started == 1
        assert llm.profile_stopped == 1
        assert result.trace_window_s is not None
        assert result.scope_stats["SD:PrepareInput"].mean_ms == \
            pytest.approx(0.12)
        assert result.scope_stats["SD:MaskLogits"].mean_ms == \
            pytest.approx(0.08)

    def test_earlier_points_traces_are_not_counted_again(self, tmp_path):
        """Every point captures into the same directory, so a stale trace
        left by the previous point must not inflate this one."""
        trace_root = tmp_path / "traces"
        _write_capture(trace_root / "run_1", "rank_0.pt.trace.json",
                       [("SD:MaskLogits#num_reqs=2#", 5_000.0)])
        llm = FakeLLM()
        args = self._args(enable_trace_report=True, trace_dir=str(trace_root))

        def fake_stop_profile():
            llm.profile_stopped += 1
            _write_capture(trace_root / "run_2", "rank_0.pt.trace.json",
                           [("SD:MaskLogits#num_reqs=2#", 80.0)])

        llm.stop_profile = fake_stop_profile

        result = bsd.benchmark_point(llm, bsd.SCHEMAS["flat_json"], 2, args)

        stats = result.scope_stats["SD:MaskLogits"]
        assert stats.count == 1
        assert stats.mean_ms == pytest.approx(0.08)
