"""Diagnostic (session 5): does the SWA cache stay correct across MANY
sequential single-token decode steps (matching the real 49-decode-step DSV4
run), not just the 3 steps the existing test_dsv4_mla_swa.py covers?

Reuses test_dsv4_mla_swa.py's own reference implementation and cache-compare
machinery -- just drives many more, smaller (batch_size=1) decode steps.
"""
import os
import sys

src_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src_path)
tests_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, tests_path)

for key in list(sys.modules.keys()):
    if key == "vllm_torchtpu" or key.startswith("vllm_torchtpu."):
        del sys.modules[key]

import jax
import jax.numpy as jnp
import numpy as np

from tests.kernels.test_dsv4_mla_swa import CorrectnessTest


class ManyStepTest(CorrectnessTest):

    def setUp(self):
        # Copy of CorrectnessTest.setUp, but batch_size=1 and sliding_window
        # matching the real DeepSeek-V4-Flash checkpoint config (128), so
        # this run (15 prompt + 49 decode = 64 total positions) never
        # triggers window eviction -- pure append/pack behavior, same as
        # the real failing run.
        self.rng = np.random.default_rng(1234)
        self.rng_key = jax.random.PRNGKey(1234)

        self.kv_dtype = jnp.bfloat16
        self.q_dtype = jnp.bfloat16
        from vllm_torchtpu.kernels.deepseek_v4.mla_swa import get_dtype_packing
        self.kv_packing = get_dtype_packing(self.kv_dtype)

        self.batch_size = 1
        self.num_heads = 128
        self.head_dim = 512
        self.lkv_dim = 640
        self.sliding_window = 128
        self.page_size = 16
        self.attention_sinks = jnp.array(
            self.rng.random(size=(self.num_heads,), dtype=np.float32) *
            300.0 + 200.0)

        def cdiv(a, b):
            return (a + b - 1) // b

        self.ref_pages_per_seq = cdiv(128, self.page_size)
        self.ref_page_indices = jnp.arange(self.batch_size *
                                            self.ref_pages_per_seq,
                                            dtype=jnp.int32)
        head_dim = self.head_dim
        self.ref_cache = jnp.zeros(
            (
                self.batch_size * self.ref_pages_per_seq,
                self.page_size // self.kv_packing,
                self.kv_packing,
                head_dim,
            ),
            dtype=self.kv_dtype,
        )

        self.sw_physical_page_size = self.page_size + 4
        self.swc_cache = jnp.zeros(
            (
                self.batch_size * self.ref_pages_per_seq,
                self.sw_physical_page_size // self.kv_packing,
                get_dtype_packing(jnp.uint8),
                640,
            ),
            dtype=jnp.uint8,
        )
        self.swc_page_indices = self.ref_page_indices
        self.kv_lens = jnp.zeros((self.batch_size,), dtype=jnp.int32)

    def test_many_decode_steps(self):
        print(f"JAX Backend: {jax.default_backend()}")

        # Step 0: 15-token prefill (matches the real DSV4_PROMPT run).
        new_kv_lens = jnp.array([15], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 15], dtype=jnp.int32)
        self.kv_lens = self.kv_lens + new_kv_lens
        total_tokens = 15
        q = self.gen_random((total_tokens, self.num_heads, self.head_dim),
                             self.q_dtype)
        new_kv = self.gen_random((total_tokens, self.head_dim), self.kv_dtype)
        distribution = jnp.array([0, 0, self.batch_size], dtype=jnp.int32)
        print("=== step 0 (prefill, 15 tokens) ===")
        self.run_and_compare_outputs(q, new_kv, self.kv_lens, cu_q_lens,
                                      distribution)

        # Steps 1..49: single-token decode, checked after EVERY step.
        for step in range(1, 50):
            new_kv_lens = jnp.array([1], dtype=jnp.int32)
            cu_q_lens = jnp.array([0, 1], dtype=jnp.int32)
            self.kv_lens = self.kv_lens + new_kv_lens
            total_tokens = 1
            q = self.gen_random(
                (total_tokens, self.num_heads, self.head_dim), self.q_dtype)
            new_kv = self.gen_random((total_tokens, self.head_dim),
                                      self.kv_dtype)
            distribution = jnp.array([self.batch_size, self.batch_size,
                                       self.batch_size],
                                      dtype=jnp.int32)
            print(f"=== decode step {step} (position={int(self.kv_lens[0]) - 1}) ===")
            self.run_and_compare_outputs(q, new_kv, self.kv_lens, cu_q_lens,
                                          distribution)
        print("ALL 50 STEPS PASSED — cache stayed correct across the full "
              "decode length matching the real failing run.")


if __name__ == "__main__":
    import unittest
    unittest.main()
