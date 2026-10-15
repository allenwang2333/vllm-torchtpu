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
"""Standalone CPU interpreter / TPU tests for the V4.1 v2 CSA2 kernel.

Run directly with Python to avoid initializing vLLM and TorchTPU on CPU.
On a TPU JAX backend the same tests compile and execute the real kernel.
"""

import importlib.util
import pathlib
import unittest

import jax
import jax.numpy as jnp
import numpy as np

_SOURCE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src/vllm_torchtpu/kernels/deepseek_v41/csa2.py"
)
_SPEC = importlib.util.spec_from_file_location("dsv41_v2_csa2", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
csa2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(csa2)


def _inputs(lengths=(6,), starts=None, *, padding=0, heads=8, seed=17):
    """Generate consistent overlapping SWA windows for independent requests."""
    starts = tuple(0 for _ in lengths) if starts is None else starts
    rng = np.random.default_rng(seed)
    tokens = sum(lengths) + padding
    q = jnp.asarray(
        rng.standard_normal((tokens, heads, 512), dtype=np.float32) * 0.3,
        jnp.bfloat16,
    )
    main_kv = jnp.asarray(
        rng.standard_normal((tokens, 512, 4, 128), dtype=np.float32),
        jnp.float8_e4m3fn,
    )
    main_scale = jnp.asarray(rng.uniform(0.01, 0.8, (tokens, 512)), jnp.float32)
    swa = np.zeros((tokens, 128, 512), dtype=jnp.bfloat16)
    positions = np.zeros(tokens, np.int32)
    boundaries = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    for b, (count, first_position) in enumerate(zip(lengths, starts)):
        if count == 0:
            continue
        first = max(0, first_position - 127)
        sequence_length = first_position + count - first
        values = rng.standard_normal((sequence_length, 512), dtype=np.float32)
        values = values.astype(jnp.bfloat16)
        for i in range(count):
            t, p = int(boundaries[b]) + i, first_position + i
            left, width = max(0, p - 127), min(p + 1, 128)
            swa[t, :width] = values[left - first : p + 1 - first]
            positions[t] = p
    theta = positions[:, None] * np.power(10000, -np.arange(32)[None, :] / 32)
    rope_cache = np.zeros((tokens, 2, 128), np.float32)
    rope_cache[:, 0, :64] = 1
    rope_cache[:, 0, 64:] = np.repeat(np.cos(theta), 2, axis=-1)
    rope_cache[:, 1, 64::2] = np.sin(theta)
    rope_cache[:, 1, 65::2] = -np.sin(theta)
    return dict(
        q=q,
        main_kv=main_kv,
        main_kv_scale=main_scale,
        swa_kv=jnp.asarray(swa),
        starts=jnp.asarray(boundaries),
        positions=jnp.asarray(positions),
        sink=jnp.asarray(rng.uniform(-2, 3, heads), jnp.float32),
        rope_cache=jnp.asarray(rope_cache),
    )


def _reference(inputs, *, compress_ratio=2, scale=512**-0.5):
    """Per-token oracle: use that token's own SWA, concatenate both branches."""
    data = {k: np.asarray(v) for k, v in inputs.items()}
    result = np.zeros(data["q"].shape, np.float32)
    for t in range(int(data["starts"][-1])):
        position = int(data["positions"][t])
        ng = min((position + 1) // compress_ratio, 512)
        nl = min(position + 1, 128)
        main = data["main_kv"][t].astype(np.float32).reshape(512, 512)
        main = main * data["main_kv_scale"][t, :, None]
        swa = data["swa_kv"][t].astype(np.float32)
        # Model the specified BF16 KV rounding BEFORE QK, not scaling logits/P.
        kv = (
            np.concatenate((main[:ng], swa[:nl]))
            .astype(jnp.bfloat16)
            .astype(np.float32)
        )
        logits = data["q"][t].astype(np.float32) @ kv.T * np.float32(scale)
        logits = np.concatenate((logits, data["sink"][:, None]), axis=1)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = exp / exp.sum(axis=1, keepdims=True)
        p = p[:, :-1].astype(jnp.bfloat16).astype(np.float32)
        out = p @ kv
        x, y = out[:, 448::2].copy(), out[:, 449::2].copy()
        cos = data["rope_cache"][t, 0, 64::2]
        sin = data["rope_cache"][t, 1, 64::2]
        out[:, 448::2], out[:, 449::2] = x * cos + y * sin, y * cos - x * sin
        result[t] = out.astype(jnp.bfloat16).astype(np.float32)
    return result


class Csa2Test(unittest.TestCase):
    def run_kernel(self, inputs, **kwargs):
        return np.asarray(
            csa2.csa2(**inputs, **kwargs, interpret=jax.default_backend() != "tpu")
        ).astype(np.float32)

    def assert_reference(self, inputs, *, bq=6, compress_ratio=2, scale=512**-0.5):
        actual = self.run_kernel(
            inputs, bq=bq, compress_ratio=compress_ratio, scale=scale
        )
        expected = _reference(inputs, compress_ratio=compress_ratio, scale=scale)
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-2)
        relative_l2 = np.linalg.norm(actual - expected) / max(
            float(np.linalg.norm(expected)), 1e-12
        )
        self.assertLess(relative_l2, 0.006)
        np.testing.assert_array_equal(actual[int(inputs["starts"][-1]) :], 0)
        return actual

    def test_group_sizes_tails_and_window_transition(self):
        inputs = _inputs(
            lengths=(0, 9, 2, 0, 7), starts=(0, 123, 0, 0, 1021), padding=3
        )
        for bq in (1, 6, 8, 16):
            with self.subTest(bq=bq):
                self.assert_reference(inputs, bq=bq)

    def test_early_windows_across_groups(self):
        self.assert_reference(_inputs(lengths=(14,), starts=(0,), padding=1), bq=6)

    def test_pv_pipeline_capacity_and_tails(self):
        # Ten MRB slots fit H96; H104/H128 must use the six-slot schedule.
        # Exercise single-Q startup, both tail parities and a BQ rollover.
        for heads in (96, 104, 128):
            with self.subTest(heads=heads):
                inputs = _inputs(
                    lengths=(0, 1, 2, 3, 5, 17),
                    starts=(0, 0, 127, 2048, 2048, 2048),
                    padding=2,
                    heads=heads,
                )
                self.assert_reference(inputs, bq=16)

    def test_48_requests_six_queries(self):
        inputs = _inputs(
            lengths=(6,) * 48, starts=tuple(b * 64 for b in range(48)), heads=64
        )
        self.assert_reference(inputs, bq=6)

    def test_swa_block_reuse_and_mtp_boundaries(self):
        # Cover cache rotation, a BQ group requiring three logical blocks,
        # and a prefetch that must be drained after the final partial group.
        for position, length, bq in ((2048, 136, 6), (0, 270, 16), (2048, 129, 6)):
            with self.subTest(position=position, length=length, bq=bq):
                inputs = _inputs(lengths=(length,), starts=(position,))
                expected = self.assert_reference(inputs, bq=bq)
                values = np.array(inputs["swa_kv"])
                anchor = max(0, position - 127)
                block_count = (position + length - anchor + 127) // 128 + 1
                representatives = {
                    min(length - 1, max(0, anchor + 128 * block + 127 - position))
                    for block in range(block_count)
                }
                # A block comes from its final available Q's window. Other
                # gathered copies are redundant and must never be consumed.
                for token in range(length):
                    if token not in representatives:
                        values[token] = 448
                poisoned = dict(inputs, swa_kv=jnp.asarray(values))
                np.testing.assert_array_equal(
                    self.run_kernel(poisoned, bq=bq), expected
                )

    def test_request_isolation(self):
        inputs = _inputs(lengths=(5, 7, 0), starts=(128, 128, 0), padding=2)
        expected = self.run_kernel(inputs)
        altered = dict(inputs, swa_kv=inputs["swa_kv"].at[5:12].set(448))
        actual = self.run_kernel(altered)
        np.testing.assert_array_equal(actual[:5], expected[:5])
        self.assertGreater(np.max(np.abs(actual[5:12] - expected[5:12])), 1)

    def test_static_shape_metadata_changes(self):
        for lengths, starts in (((3, 0, 7), (127, 0, 4)), ((0, 8, 2), (0, 1020, 1))):
            self.assert_reference(_inputs(lengths=lengths, starts=starts, padding=2))

    def test_compression_and_attention_scale(self):
        inputs = _inputs(lengths=(7,), starts=(0,))
        for ratio in (1, 2, 4):
            with self.subTest(ratio=ratio):
                self.assert_reference(inputs, compress_ratio=ratio, scale=0.125)

    def test_sink_and_inverse_rope_analytic(self):
        inputs = _inputs(lengths=(1,), starts=(0,))
        inputs["q"] = jnp.zeros_like(inputs["q"])
        inputs["swa_kv"] = jnp.ones_like(inputs["swa_kv"]) * 2
        inputs["sink"] = jnp.zeros_like(inputs["sink"])
        rope_cache = np.zeros((1, 2, 128), np.float32)
        rope_cache[:, 0, :64] = 1
        rope_cache[:, 1, 64::2] = 1
        rope_cache[:, 1, 65::2] = -1
        inputs["rope_cache"] = jnp.asarray(rope_cache)
        actual = self.run_kernel(inputs)
        # One valid SWA KV of value 2, one zero-logit sink: output is 1.
        expected = np.ones(actual.shape, np.float32)
        expected[:, :, 449::2] *= -1
        np.testing.assert_array_equal(actual, expected)
        inputs["sink"] = jnp.full_like(inputs["sink"], 1000)
        np.testing.assert_array_equal(self.run_kernel(inputs), 0)

    def test_zero_main_scale_preserves_swa(self):
        inputs = _inputs(lengths=(2,), starts=(128,))
        inputs["main_kv_scale"] = jnp.zeros_like(inputs["main_kv_scale"])
        actual = self.assert_reference(inputs)
        self.assertGreater(np.max(np.abs(actual)), 0)
        inputs["swa_kv"] = jnp.zeros_like(inputs["swa_kv"])
        np.testing.assert_array_equal(self.run_kernel(inputs), 0)

    def test_all_gemms_use_bf16_and_metadata_uses_smem(self):
        inputs = _inputs(lengths=(1,))
        traced = jax.make_jaxpr(lambda **x: csa2.csa2(**x, interpret=True))(**inputs)
        dots, calls, mxu_ops = [], [], []

        def walk(value):
            if isinstance(value, jax.extend.core.ClosedJaxpr):
                walk(value.jaxpr)
            elif isinstance(value, jax.extend.core.Jaxpr):
                for eqn in value.eqns:
                    if eqn.primitive.name == "dot_general":
                        dots.append(eqn)
                    if eqn.primitive.name == "pallas_call":
                        calls.append(eqn)
                    if eqn.primitive.name in (
                        "matmul_push_rhs",
                        "matmul_acc_lhs",
                        "matmul_pop",
                    ):
                        mxu_ops.append(eqn)
                    for child in eqn.params.values():
                        walk(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    walk(child)
            elif isinstance(value, dict):
                for child in value.values():
                    walk(child)

        walk(traced)
        self.assertTrue(dots)
        for dot in dots:
            self.assertEqual([v.aval.dtype for v in dot.invars], [jnp.bfloat16] * 2)
            self.assertEqual(dot.outvars[0].aval.dtype, jnp.float32)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].params["grid_mapping"].num_index_operands, 2)
        refs = calls[0].params["jaxpr"].invars[:2]
        for ref in refs:
            self.assertEqual(str(ref.aval.memory_space), "smem")

        dots.clear()
        calls.clear()
        walk(jax.make_jaxpr(lambda **x: csa2.csa2(**x, interpret=False))(**inputs))
        self.assertFalse(dots)
        self.assertEqual(len(calls), 1)
        self.assertEqual({op.params["mxu_index"] for op in mxu_ops}, {0, 1})
        self.assertEqual(
            {op.primitive.name for op in mxu_ops},
            {"matmul_push_rhs", "matmul_acc_lhs", "matmul_pop"},
        )
        for op in mxu_ops:
            if op.primitive.name == "matmul_pop":
                self.assertEqual(op.outvars[0].aval.dtype, jnp.float32)
            else:
                self.assertEqual(op.invars[0].aval.dtype, jnp.bfloat16)

    def test_bf16_query_precision(self):
        inputs = _inputs(lengths=(1,), starts=(128,))
        inputs["q"] = jnp.full_like(inputs["q"], 1.03125)
        actual = self.assert_reference(inputs)
        quantized = dict(
            inputs, q=inputs["q"].astype(jnp.float8_e4m3fn).astype(jnp.bfloat16)
        )
        self.assertGreater(np.max(np.abs(actual - self.run_kernel(quantized))), 1e-4)

    def test_swa_keeps_bf16_precision(self):
        inputs = _inputs(lengths=(1,), starts=(0,))
        inputs["q"] = jnp.zeros_like(inputs["q"])
        inputs["sink"] = jnp.zeros_like(inputs["sink"])
        inputs["swa_kv"] = jnp.full_like(inputs["swa_kv"], 1.03125)
        # One SWA key plus the sink gives P=1/2; FP8 would lose 0.03125.
        np.testing.assert_array_equal(self.run_kernel(inputs), 0.515625)

    def test_bad_contract(self):
        original = _inputs(lengths=(1,))
        for name, value in (
            ("q", original["q"].astype(jnp.float8_e4m3fn)),
            ("main_kv", original["main_kv"].reshape(1, 512, 512)),
            ("main_kv_scale", original["main_kv_scale"].astype(jnp.bfloat16)),
            ("swa_kv", original["swa_kv"].astype(jnp.float8_e4m3fn)),
            ("swa_kv", original["swa_kv"].astype(jnp.float16)),
            ("starts", jnp.zeros((1,), jnp.int32)),
            ("positions", jnp.zeros((1,), jnp.float32)),
            ("rope_cache", original["rope_cache"][:, 0, :64]),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.run_kernel(dict(original, **{name: value}))
        for kwargs in ({"bq": 0}, {"bq": 17}, {"bq": True}, {"compress_ratio": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_kernel(original, **kwargs)

    def test_empty_requests_and_padding(self):
        for padding in (0, 9):
            inputs = _inputs(lengths=(0, 0), padding=padding)
            np.testing.assert_array_equal(self.run_kernel(inputs), 0)


if __name__ == "__main__":
    unittest.main()
