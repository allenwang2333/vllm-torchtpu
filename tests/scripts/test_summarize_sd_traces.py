# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the structured-decoding nightly gate.

Covers scripts/vllm/benchmarking/summarize_sd_traces.py (trace parsing, stale
trace exclusion, mask memory, block efficiency, result-file enrichment), the
SD metrics it feeds to check_regression.py, and the two
qwen3-0.6b-structured-* configs. All of it runs without a TPU.
"""

import gzip
import importlib.util
import json
import pathlib
import subprocess

import pytest

pytestmark = pytest.mark.cpu_test

_BENCH_DIR = pathlib.Path(
    __file__).resolve().parents[2] / "scripts" / "vllm" / "benchmarking"


def _load(name):
    spec = importlib.util.spec_from_file_location(name,
                                                  _BENCH_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sds = _load("summarize_sd_traces")
check_regression = _load("check_regression")


def _write_trace(path, events, gz=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
        "traceEvents": [{
            "ph": "X",
            "name": name,
            "dur": dur_us
        } for name, dur_us in events]
    })
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        path.write_text(payload)
    return path


class TestScopeMatching:

    def test_bare_scope_name_matches(self):
        assert sds.scope_of("SD:MaskLogits") == "SD:MaskLogits"

    def test_annotated_scope_name_matches(self):
        assert sds.scope_of(
            "SD:PrepareInput#num_reqs=8,vocab=151936#") == "SD:PrepareInput"

    def test_lookalike_prefix_is_not_matched(self):
        assert sds.scope_of("SD:MaskLogitsSpec#num_reqs=8#") is None

    def test_unrelated_event_is_ignored(self):
        assert sds.scope_of("aten::matmul") is None


class TestTraceParsing:

    def test_reads_complete_events_only(self, tmp_path):
        path = tmp_path / "rank0.pt.trace.json"
        path.write_text(
            json.dumps({
                "traceEvents": [
                    {
                        "ph": "X",
                        "name": "SD:MaskLogits#n=1#",
                        "dur": 250
                    },
                    {
                        "ph": "B",
                        "name": "SD:MaskLogits#n=1#",
                        "dur": 999
                    },
                    {
                        "ph": "X",
                        "name": "aten::add",
                        "dur": 5
                    },
                ]
            }))
        durations = sds.scope_durations_ms(sds.load_events(path))
        assert durations["SD:MaskLogits"] == [0.25]
        assert durations["SD:PrepareInput"] == []

    def test_reads_gzipped_traces(self, tmp_path):
        path = _write_trace(tmp_path / "rank0.pt.trace.json.gz",
                            [("SD:PrepareInput#n=2#", 1000)],
                            gz=True)
        assert sds.scope_durations_ms(
            sds.load_events(path))["SD:PrepareInput"] == [1.0]

    def test_empty_directory_yields_no_means(self, tmp_path):
        assert sds.collect_scope_means(tmp_path, set()) == {}

    def test_earlier_cells_traces_are_not_counted_again(self, tmp_path):
        """Every cell profiles into one directory; a stale trace from the
        previous cell must not leak into this cell's mean."""
        stale = _write_trace(tmp_path / "c1" / "r0.pt.trace.json",
                             [("SD:MaskLogits#", 5000)])
        _write_trace(tmp_path / "c8" / "r0.pt.trace.json",
                     [("SD:MaskLogits#", 80)])
        means = sds.collect_scope_means(tmp_path, {stale})
        assert means["SD:MaskLogits"] == pytest.approx(0.08)

    def test_uses_one_rank_instead_of_pooling_tp_ranks(self, tmp_path):
        _write_trace(tmp_path / "r0.pt.trace.json",
                     [("SD:MaskLogits#", 100)] * 3)
        _write_trace(tmp_path / "r1.pt.trace.json", [("SD:MaskLogits#", 900)])
        means = sds.collect_scope_means(tmp_path, set())
        assert means["SD:MaskLogits"] == pytest.approx(0.1)

    def test_corrupt_trace_is_skipped(self, tmp_path):
        (tmp_path / "bad.pt.trace.json").write_text("{not json")
        _write_trace(tmp_path / "ok.pt.trace.json",
                     [("SD:PrepareInput#", 400)])
        assert sds.collect_scope_means(tmp_path, set()) == {
            "SD:PrepareInput": pytest.approx(0.4)
        }


class TestMaskMemory:

    def test_packed_bitmask_vs_dense(self):
        mem = sds.mask_memory(num_reqs=8, vocab_size=151_936)
        assert mem["packed_bytes"] == 8 * (151_936 // 32) * 4
        assert mem["savings_vs_bool"] == pytest.approx(0.875)
        # bf16 logits: the figure #1222 reported on TPU.
        assert mem["savings_vs_logits"] == pytest.approx(0.9375)

    def test_vocab_not_a_multiple_of_32_rounds_up(self):
        assert sds.mask_memory(num_reqs=1, vocab_size=33)["packed_bytes"] == 8

    @pytest.mark.parametrize("num_reqs,vocab_size", [(0, 10), (1, 0),
                                                     (-1, 10)])
    def test_rejects_non_positive_shapes(self, num_reqs, vocab_size):
        with pytest.raises(ValueError):
            sds.mask_memory(num_reqs, vocab_size)


class TestBlockEfficiency:

    def test_is_one_without_spec_decode(self):
        assert sds.block_efficiency(
            "vllm:num_generation_tokens_total 4096\n") == 1.0

    def test_uses_accepted_over_drafts_across_labels(self):
        text = "\n".join([
            "# HELP vllm:spec_decode_num_drafts_total Number of drafts.",
            'vllm:spec_decode_num_drafts_total{engine="0"} 60',
            'vllm:spec_decode_num_drafts_total{engine="1"} 40',
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 150',
            'vllm:spec_decode_num_accepted_tokens_total{engine="1"} 130',
        ])
        # (150 + 130) accepted / (60 + 40) drafts, plus the bonus token.
        assert sds.block_efficiency(text) == pytest.approx(3.8)


class TestEnrichResultFile:

    def test_adds_sd_metrics_to_the_measured_cell(self, tmp_path):
        result = tmp_path / "isl128_osl128_c8.json"
        result.write_text(json.dumps({
            "completed": 128,
            "median_tpot_ms": 3.1
        }))
        traces = tmp_path / "profile"
        stale = _write_trace(traces / "old" / "r0.pt.trace.json",
                             [("SD:MaskLogits#", 9000)])
        _write_trace(
            traces / "new" / "r0.pt.trace.json",
            [("SD:PrepareInput#", 120), ("SD:MaskLogits#", 80)],
        )
        exclude = tmp_path / "before.txt"
        exclude.write_text(f"{stale}\n")

        rc = sds.main([
            "--result-file",
            str(result),
            "--trace-dir",
            str(traces),
            "--exclude-list",
            str(exclude),
            "--num-reqs",
            "8",
            "--vocab-size",
            "151936",
        ])

        data = json.loads(result.read_text())
        assert rc == 0
        assert data["median_tpot_ms"] == 3.1
        assert data["sd_prepare_input_mean_ms"] == pytest.approx(0.12)
        assert data["sd_mask_logits_mean_ms"] == pytest.approx(0.08)
        assert data["sd_mask_memory"]["savings_vs_bool"] == pytest.approx(
            0.875)

    def test_unreachable_metrics_endpoint_is_not_fatal(self, tmp_path):
        result = tmp_path / "isl128_osl128_c1.json"
        result.write_text("{}")
        rc = sds.main([
            "--result-file",
            str(result),
            "--num-reqs",
            "1",
            "--vocab-size",
            "32",
            "--metrics-url",
            "http://127.0.0.1:9/metrics",
        ])
        assert rc == 0
        assert "block_efficiency" not in json.loads(result.read_text())


class TestRegressionGate:
    _BASE = {
        "isl128_osl128_c8": {
            "completed": 128,
            "median_tpot_ms": 3.1,
            "sd_prepare_input_mean_ms": 0.12,
            "sd_mask_logits_mean_ms": 0.08,
        }
    }

    def test_sd_metrics_are_gated(self):
        gated = {m.result_field for m in check_regression.GATED_METRICS}
        assert {
            "sd_prepare_input_mean_ms", "sd_mask_logits_mean_ms",
            "block_efficiency"
        } <= gated

    def test_passes_within_tolerance(self):
        results = {
            "isl128_osl128_c8":
            dict(self._BASE["isl128_osl128_c8"], sd_mask_logits_mean_ms=0.085)
        }
        assert check_regression.check_perf(results, self._BASE,
                                           tolerance=0.10) == 0

    def test_fails_when_the_mask_kernel_slows_down(self):
        results = {
            "isl128_osl128_c8":
            dict(self._BASE["isl128_osl128_c8"], sd_mask_logits_mean_ms=0.2)
        }
        assert check_regression.check_perf(results, self._BASE,
                                           tolerance=0.10) == 1

    def test_baselines_without_sd_fields_are_unaffected(self):
        base = {"isl128_osl128_c8": {"completed": 128, "median_tpot_ms": 3.1}}
        results = {
            "isl128_osl128_c8": {
                "completed": 128,
                "median_tpot_ms": 3.1
            }
        }
        assert check_regression.check_perf(results, base, tolerance=0.05) == 0

    def test_calibration_picks_up_sd_fields(self, tmp_path):
        for run in ("a", "b"):
            d = tmp_path / run
            d.mkdir()
            (d / "isl128_osl128_c8.json").write_text(
                json.dumps(self._BASE["isl128_osl128_c8"]))
        baseline = check_regression.calibrate_perf(
            [tmp_path / "a", tmp_path / "b"])
        assert baseline["isl128_osl128_c8"][
            "sd_mask_logits_mean_ms"] == pytest.approx(0.08)


class TestConfigs:

    @pytest.mark.parametrize(
        "config,schema_key",
        [
            ("qwen3-0.6b-structured-flat-json", "properties"),
            ("qwen3-0.6b-structured-complex-ast", "$defs"),
        ],
    )
    def test_config_sends_a_structured_outputs_grammar(self, config,
                                                       schema_key):
        """Source the config the way run_benchmarks.sh does and check the
        request body it produces is a real structured_outputs grammar."""
        script = (
            f'SCRIPT_DIR="{_BENCH_DIR}"; source "$SCRIPT_DIR/configs/{config}.sh"; '
            'printf "%s\\n%s\\n%s\\n" "$CAPTURE_PROFILE" "$SD_TRACE_SUMMARY" "$SD_VOCAB_SIZE"; '
            'printf "%s" "$BENCHMARK_EXTRA_BODY"')
        out = subprocess.run(["bash", "-c", script],
                             check=True,
                             capture_output=True,
                             text=True).stdout
        capture, summary, vocab, body = out.split("\n", 3)
        assert (capture, summary) == ("1", "1")
        assert int(vocab) > 0
        grammar = json.loads(body)["structured_outputs"]["json"]
        assert grammar["type"] == "object"
        assert schema_key in grammar

    def test_complex_ast_grammar_is_recursive(self):
        body = json.loads((_BENCH_DIR / "structured_schemas" /
                           "complex_ast.json").read_text())
        defs = body["structured_outputs"]["json"]["$defs"]
        assert {
            "$ref": "#/$defs/Node"
        } in defs["Expression"]["properties"].values()
        assert {"$ref": "#/$defs/Expression"} in defs["Node"]["anyOf"]
