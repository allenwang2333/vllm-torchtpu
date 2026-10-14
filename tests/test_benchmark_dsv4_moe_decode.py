# SPDX-License-Identifier: Apache-2.0
"""Host-only tests; also runnable via unittest without TPU pytest fixtures."""

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_dsv4_moe_decode.py"
SPEC = importlib.util.spec_from_file_location("decode_bench", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


class DecodeBenchmarkTest(unittest.TestCase):
    def test_preflight_fails_closed_on_sudo_failure_or_device_owner(self):
        from subprocess import CompletedProcess

        for result in (
            CompletedProcess([], 1, "", "sudo: a password is required"),
            CompletedProcess([], 0, "1234", "/dev/vfio/0: python"),
        ):
            with (
                patch.object(bench.glob, "glob", return_value=["/dev/vfio/0"]),
                patch.object(bench.subprocess, "run", return_value=result),
                self.assertRaises(RuntimeError),
            ):
                bench.preflight()

    def test_timing_statistics_use_all_samples(self):
        stats = bench.statistics([1, 2, 3, 4])
        self.assertEqual(stats["p50_ms"], 2.5)
        self.assertEqual(stats["mean_ms"], 2.5)
        self.assertAlmostEqual(stats["p90_ms"], 3.7)
        for values in ([], [0], [float("nan")], [-1]):
            with self.assertRaises(ValueError):
                bench.statistics(values)

    def test_error_gate_detects_one_bad_token_and_nonfinite_output(self):
        reference = np.ones((256, 8))
        actual = reference.copy()
        actual[3] *= 1.1
        result = bench.errors(actual, reference)
        self.assertLess(result["relative_l2"], bench.REL_BOUND)
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["per_token_relative_l2"][3], 0.1)
        actual[3, 0] = float("nan")
        with self.assertRaises(ValueError):
            bench.errors(actual, reference)

    def test_zero_reference_does_not_hide_absolute_error(self):
        self.assertTrue(bench.errors(np.zeros((2, 4)), np.zeros((2, 4)))["passed"])
        self.assertFalse(bench.errors(np.ones((2, 4)), np.zeros((2, 4)))["passed"])

    def test_load_distributions_and_real_scoring_semantics(self):
        shape = bench.Shape()
        bias = np.linspace(-0.2, 0.2, shape.experts).astype(np.float32)
        for distribution in ("balanced", "skewed"):
            x, logits = bench.make_inputs(shape, 64, bias, 41, distribution)
            ids, weights = bench.cpu_routes(logits, bias, shape.topk)
            self.assertEqual(x.shape, (64, 4096))
            self.assertEqual(ids.shape, (64, 6))
            self.assertTrue(all(len(set(row)) == 6 for row in ids))
            np.testing.assert_allclose(weights.sum(1), 1.5, rtol=1e-6)
            counts = np.bincount(ids.ravel(), minlength=shape.experts)
            self.assertEqual(counts.sum(), 64 * 6)
            if distribution == "balanced":
                self.assertLessEqual(counts.max() - counts.min(), 1)
            else:
                self.assertEqual(counts[32:].sum(), 0)

    def test_tensor_reader_seeks_only_selected_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data = np.arange(6, dtype=np.float32).reshape(2, 3)
            header = json.dumps(
                {"wanted": dict(dtype="F32", shape=[2, 3], data_offsets=[3, 27])}
            ).encode()
            (path / "shard.safetensors").write_bytes(
                struct.pack("<Q", len(header))
                + header
                + b"pad"
                + data.tobytes()
                + b"tail"
            )
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"wanted": "shard.safetensors"}})
            )
            reader = bench.TensorReader(path)
            np.testing.assert_array_equal(reader.read("wanted"), data)
            self.assertEqual(
                reader.records["wanted"]["sha256"], bench.digest(data.tobytes())
            )
            with self.assertRaises(KeyError):
                reader.read("other")

    def test_reject_different_model_semantics(self):
        _, config = bench.configuration(None, False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config["swiglu_limit"] = 7
            (path / "config.json").write_text(json.dumps(config))
            with self.assertRaises(NotImplementedError):
                bench.configuration(path, False)


if __name__ == "__main__":
    unittest.main()
