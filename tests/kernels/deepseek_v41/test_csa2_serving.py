# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for the actual serving cache-to-CSA2 bridge."""

import importlib.util
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "csa2_oracle", Path(__file__).with_name("test_csa2.py")
)
assert _spec is not None and _spec.loader is not None
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


def serving_inputs(inputs):
    """Encode native BF16 SWA records, including invalid cache overlay rows."""
    positions = np.asarray(inputs["positions"])
    lower, upper = np.asarray(inputs["swa_lower"]), np.asarray(inputs["swa_upper"])
    starts = np.asarray(inputs["starts"])
    swa = np.asarray(inputs["swa_kv"])
    width = 192 if swa.shape[1] == 256 else 128
    records = np.full((len(positions), width), -1, np.int32)
    rows = []
    for begin, end in zip(starts[:-1], starts[1:]):
        addresses = {}
        for t in range(begin, end):
            for column in range(width):
                logical = positions[t] - 127 + column if width == 128 else column
                if lower[t] <= logical < upper[t]:
                    if logical not in addresses:
                        addresses[logical] = len(rows)
                        base = max(0, int(positions[t]) - 127) if width == 128 else 0
                        rows.append(swa[t, logical - base])
                    records[t, column] = addresses[logical]
    rows.extend(np.full((32, 512), np.nan, dtype=jnp.bfloat16))
    bits = np.asarray(rows, dtype=jnp.bfloat16).view(np.uint16).astype(np.uint32)
    pool = (bits[:, ::2] | (bits[:, 1::2] << 16)).view(np.int32).reshape(-1, 128)
    counts = np.asarray(inputs["main_lengths"])
    main_records = np.where(
        np.arange(512)[None, :] < counts[:, None], np.arange(512)[None, :], -1
    )
    main_records[int(starts[-1]) :] = -1
    return dict(
        q=inputs["q"],
        swa_pool=jnp.asarray(pool),
        main_pool=inputs["main_kv"].reshape(-1, 128),
        rope_pool=inputs["main_rope"].reshape(-1, 128),
        swa_records=jnp.asarray(records),
        main_records=jnp.asarray(main_records, jnp.int32),
        pages=jnp.zeros((len(starts) - 1, 1), jnp.int32),
        query_start_loc=inputs["starts"],
        sink=inputs["sink"],
        scale=512**-0.5,
        page_size=4,
        max_requests=max(1, int(np.sum(np.diff(starts) > 0))),
        precompacted=True,
        gathered_nope=inputs["main_kv"],
        gathered_rope=inputs["main_rope"],
        positions=inputs["positions"],
    )


class Csa2ServingTest(unittest.TestCase):
    def test_native_cache_bridge(self):
        from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
            sparse_fp8_attention_jax,
        )

        for noncausal in (False, True):
            with self.subTest(noncausal=noncausal):
                inputs, dense = oracle._packed_inputs(noncausal=noncausal, heads=64)
                args = serving_inputs(inputs)
                arrays = {k: v for k, v in args.items() if isinstance(v, jax.Array)}
                static = {k: v for k, v in args.items() if k not in arrays}
                run = jax.jit(
                    lambda data, static=static: sparse_fp8_attention_jax(
                        **data, **static
                    )
                )
                actual = np.asarray(run(arrays)).astype(np.float32)
                expected = oracle._packed_reference(inputs, dense)
                np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-2)
                self.assertLess(
                    np.linalg.norm(actual - expected) / np.linalg.norm(expected), 0.006
                )


if __name__ == "__main__":
    unittest.main()
