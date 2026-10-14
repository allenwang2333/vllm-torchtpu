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
"""Grouped-output tests for SparseCore MLA attention (`merge_group_rows`)."""

import os
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout

pytestmark = pytest.mark.tpu_test

NOPE_DIM = 512
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
NUM_HEADS = 64
TOPK = 2048
TOPK_CAP = 512
CHUNK_SIZE = 256
KERNEL_BATCH_SIZE = 16

FAULT_ENV_VAR = "MLA_GROUPED_TEST_FAULT"
# Explicit mapping from each distinct in-kernel fault injection on
# `sparse_attention_chunk` to the single check ID that must catch it.
FAULT_EXPECTED_CHECK: dict[str, str] = {
    "write_offset_off_by_one_batch": "check_group_output_bitwise",
    "missing_group_carry_o_in": "check_group_output_bitwise",
    "corrupted_group_carry_lse": "check_group_lse_bitwise",
    "unmasked_empty_row_lse": "check_empty_real_row_masked",
    "group_lse_not_initialized": "check_unreached_padding_lse_neg_inf",
    "nan_real_row": "check_real_row_finite",
}


def _build_served_case(
    rng: np.random.Generator,
    *,
    query_tokens: int,
    seq_q: tuple[int, ...],
    overflow: bool,
) -> dict[str, Any]:
    """Construct a served-shape SparseCore MLA problem with padded groups."""
    page_size = 256
    cache_tokens = 8192
    pages = cache_tokens // page_size
    k_scale = 0.75

    def quantized_bytes(dim: int) -> np.ndarray:
        values = rng.random((pages, page_size, dim), dtype=np.float32) * 2.0 - 1.0
        fp8 = jnp.asarray(values / k_scale).astype(jnp.float8_e4m3fn)
        return np.asarray(jax.lax.bitcast_convert_type(fp8, jnp.uint8))

    nope_bytes = quantized_bytes(NOPE_DIM)
    rope_bytes = quantized_bytes(ROPE_DIM)
    tensorcore_nope = nope_bytes.reshape(pages, page_size, 4, 128)
    tensorcore_rope = np.pad(rope_bytes, ((0, 0), (0, 0), (0, ROPE_DIM))).reshape(
        pages, page_size // 4, 4, 128
    )
    sparsecore_nope = jnp.asarray(native_sc_cache_layout.pack_nope(tensorcore_nope))
    sparsecore_rope = jnp.asarray(
        native_sc_cache_layout.pack_rope_banded(tensorcore_rope)
    )

    num_seqs = len(seq_q)
    base_pt = rng.permutation(pages).astype(np.int32)
    inverse_pt = np.argsort(base_pt)
    page_table = jnp.concatenate(
        [jnp.asarray(base_pt, jnp.int32)]
        + [
            jnp.asarray(rng.permutation(pages), jnp.int32)
            for _ in range(max(num_seqs - 1, 0))
        ]
    )

    physical = np.stack(
        [
            rng.choice(cache_tokens, size=TOPK, replace=False)
            for _ in range(query_tokens)
        ]
    ).astype(np.int32)
    logical = (
        inverse_pt[physical // page_size] * page_size + physical % page_size
    ).astype(np.int32)

    counts = np.clip(rng.normal(128, 36, size=query_tokens), 1, 380).astype(np.int32)
    counts[0] = 0  # single empty real row in group 0
    counts[16:32] = 0  # full 16-token grid step owning nothing in group 0
    counts[32:48] = 512  # full 512-entry window-0 step
    real_tokens = int(sum(seq_q))
    if real_tokens > 300:
        counts[260] = 0  # empty real row in second 256-token chunk
    if overflow:
        counts[1] = 1025  # multi-window overflow row in group 0
        if real_tokens > 300:
            counts[270] = 1100  # multi-window overflow row in second chunk
        counts[min(real_tokens - 1, query_tokens - 1)] = (
            1500  # overflow at last real row
        )
    else:
        counts = np.minimum(counts, TOPK_CAP)

    live = np.arange(TOPK)[None, :] < counts[:, None]
    topk_raw = jnp.where(jnp.asarray(live), jnp.asarray(logical, jnp.int32), -1)
    owns = np.asarray(jnp.any(topk_raw >= 0, axis=-1))

    q_full = rng.normal(size=(query_tokens, NUM_HEADS, HEAD_DIM)).astype(np.float32)
    q_nope = jnp.asarray(q_full[..., :NOPE_DIM], dtype=jnp.bfloat16)
    q_pe = jnp.asarray(q_full[..., NOPE_DIM:], dtype=jnp.bfloat16)

    cu_q_lens = jnp.asarray(np.concatenate([[0], np.cumsum(seq_q)]), jnp.int32)
    num_decode = sum(1 for n in seq_q if n == 4)
    distribution = jnp.array([num_decode, num_decode, num_seqs], jnp.int32)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "sparsecore_nope": sparsecore_nope,
        "sparsecore_rope": sparsecore_rope,
        "topk_raw": topk_raw,
        "owns": owns,
        "counts": counts,
        "page_table": page_table,
        "cu_q_lens": cu_q_lens,
        "distribution": distribution,
        "real_tokens": real_tokens,
        "k_scale": k_scale,
    }


def _run_with_in_kernel_fault(fn: Any, *, fault: str) -> Any:
    """Execute `fn()` while injecting `fault` into `sparse_attention_chunk`."""
    if fault not in FAULT_EXPECTED_CHECK:
        raise ValueError(
            f"Unknown fault {fault!r}; expected one of {tuple(FAULT_EXPECTED_CHECK)}"
        )

    orig_chunk = sparse_mla.sparse_attention_chunk

    def _faulty_chunk(
        q: jax.Array,
        cache_kv_nope: jax.Array,
        cache_kv_rope: jax.Array,
        kv_lens: jax.Array,
        physical_indices: jax.Array | None,
        lse: jax.Array | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        start_pos = kwargs.get("start_pos", 0)
        out_rows = kwargs.get("out_rows", 0)
        out_row_offset = kwargs.get("out_row_offset", 0)

        if fault == "write_offset_off_by_one_batch":
            # On the second chunk of group 0 (start_pos=256, end_pos=512,
            # out_rows=512), shift out_row_offset by one kernel batch (+16 rows).
            # This stays inside static bounds (16 <= 256 and 512 <= 16 + 512) so
            # the kernel executes and writes rows [256, 512) into [240, 496).
            if out_rows >= 2 * CHUNK_SIZE and start_pos == out_row_offset + CHUNK_SIZE:
                kwargs["out_row_offset"] = out_row_offset + KERNEL_BATCH_SIZE
        elif fault == "missing_group_carry_o_in":
            # Drop the carry-over `o_in` on the second chunk of group 0 so the
            # first chunk's rows [0, 256) in group 0 are lost.
            if out_rows >= 2 * CHUNK_SIZE and start_pos == out_row_offset + CHUNK_SIZE:
                kwargs["o_in"] = None
        elif (
            fault == "corrupted_group_carry_lse"
            and out_rows >= 2 * CHUNK_SIZE
            and start_pos == out_row_offset + CHUNK_SIZE
            and lse is not None
        ):
            # Perturb chunk 0's finite owning LSE carry entering chunk 1 of
            # group 0, leaving empty-row -inf sentinels and outputs intact.
            lse = jnp.where(jnp.isfinite(lse), lse + 1.0, lse)
        elif fault == "unmasked_empty_row_lse":
            # Disable `lse_mask_empty` on chunk 0 so empty real rows (counts == 0)
            # receive a finite dummy-key LSE instead of -inf.
            if start_pos == 0:
                kwargs["lse_mask_empty"] = False
        elif (
            fault == "group_lse_not_initialized"
            and out_rows > 0
            and out_row_offset == out_rows
            and lse is not None
            and kwargs.get("o_in") is None
        ):
            # Zero-initialize group 1's lse buffer (instead of -inf) so unreached
            # padding rows [grid_valid_rows:] remain 0.0 while real rows match.
            lse = jnp.zeros_like(lse)
        elif fault == "nan_real_row" and start_pos == 0:
            # Inject NaN into active real query row 5 entering the kernel.
            q = q.at[5].set(jnp.nan)

        return orig_chunk(
            q,
            cache_kv_nope,
            cache_kv_rope,
            kv_lens,
            physical_indices,
            lse,
            *args,
            **kwargs,
        )

    with patch.object(sparse_mla, "sparse_attention_chunk", side_effect=_faulty_chunk):
        return fn()


def _assert_grouped_matches_reference(
    outs: tuple[jax.Array, ...],
    lses: tuple[jax.Array, ...],
    ref_out: jax.Array,
    ref_lse: jax.Array,
    *,
    query_tokens: int,
    merge_group_rows: int,
    real_tokens: int,
    owns: np.ndarray,
    lse_mask_empty: bool = True,
) -> None:
    """Validate grouped `(outs, lses)` against `merge_group_rows=0` outputs."""
    assert isinstance(outs, tuple), f"outs must be a tuple, got {type(outs)}"
    assert isinstance(lses, tuple), f"lses must be a tuple, got {type(lses)}"
    num_groups = query_tokens // merge_group_rows
    assert len(outs) == num_groups, (
        f"expected {num_groups} output groups, got {len(outs)}"
    )
    assert len(lses) == num_groups, f"expected {num_groups} lse groups, got {len(lses)}"

    ref_out_f32 = np.asarray(ref_out[:real_tokens], dtype=np.float32)
    if not np.isfinite(ref_out_f32).all():
        raise AssertionError(
            "check_real_row_finite: reference real rows contain non-finite values"
        )

    ref_out_bits = np.asarray(jax.lax.bitcast_convert_type(ref_out, jnp.uint16))
    ref_lse_np = np.asarray(ref_lse, dtype=np.float32)
    ref_lse_bits = ref_lse_np.view(np.uint32)

    saw_partial_group = False
    saw_fully_padded_group = False

    for g in range(num_groups):
        out_g = outs[g]
        lse_g = lses[g]
        assert out_g.shape == (
            merge_group_rows,
            ref_out.shape[1],
            ref_out.shape[2],
        ), f"group {g} output shape mismatch: {out_g.shape}"
        assert lse_g.shape == (
            merge_group_rows,
            ref_lse.shape[1],
        ), f"group {g} lse shape mismatch: {lse_g.shape}"

        g_start = g * merge_group_rows
        valid_rows = min(max(real_tokens - g_start, 0), merge_group_rows)
        grid_valid_rows = min(
            ((valid_rows + KERNEL_BATCH_SIZE - 1) // KERNEL_BATCH_SIZE)
            * KERNEL_BATCH_SIZE,
            merge_group_rows,
        )
        if 0 < valid_rows < merge_group_rows:
            saw_partial_group = True
        elif valid_rows == 0:
            saw_fully_padded_group = True

        out_g_f32 = np.asarray(out_g, dtype=np.float32)
        lse_g_np = np.asarray(lse_g, dtype=np.float32)

        if valid_rows > 0:
            if (
                not np.isfinite(out_g_f32[:valid_rows]).all()
                or np.isnan(lse_g_np[:valid_rows]).any()
            ):
                raise AssertionError(
                    f"check_real_row_finite: group {g} real rows [:{valid_rows}] "
                    "contain NaN/Inf output or NaN lse"
                )
            owns_g = owns[g_start : g_start + valid_rows]

            out_g_bits = np.asarray(
                jax.lax.bitcast_convert_type(out_g[:grid_valid_rows], jnp.uint16)
            )
            want_out_bits = ref_out_bits[g_start : g_start + grid_valid_rows]
            if not np.array_equal(out_g_bits, want_out_bits):
                raise AssertionError(
                    f"check_group_output_bitwise: group {g} rows "
                    f"[{g_start}:{g_start + grid_valid_rows}] differ from reference"
                )

            if owns_g.any() and not np.isfinite(lse_g_np[:valid_rows][owns_g]).all():
                raise AssertionError(
                    f"check_real_row_finite: group {g} owning real rows "
                    "contain non-finite lse"
                )
            if (
                lse_mask_empty
                and (~owns_g).any()
                and (
                    not np.all(lse_g_np[:valid_rows][~owns_g] == -np.inf)
                    or not np.all(out_g_f32[:valid_rows][~owns_g] == 0.0)
                )
            ):
                raise AssertionError(
                    f"check_empty_real_row_masked: group {g} empty real rows "
                    "do not have lse == -inf and zero output"
                )

            lse_g_bits = lse_g_np[:grid_valid_rows].view(np.uint32)
            want_lse_bits = ref_lse_bits[g_start : g_start + grid_valid_rows]
            if not np.array_equal(lse_g_bits, want_lse_bits):
                raise AssertionError(
                    f"check_group_lse_bitwise: group {g} lse rows "
                    f"[{g_start}:{g_start + grid_valid_rows}] differ from reference"
                )

        if lse_mask_empty and grid_valid_rows < merge_group_rows:
            unreached_lse = lse_g_np[grid_valid_rows:]
            if not np.all(unreached_lse == -np.inf):
                raise AssertionError(
                    f"check_unreached_padding_lse_neg_inf: group {g} unreached rows "
                    f"[{grid_valid_rows}:{merge_group_rows}] not all lse == -inf"
                )

    assert saw_partial_group, (
        f"Expected at least one partially padded group for real_tokens={real_tokens}, "
        f"merge_group_rows={merge_group_rows}"
    )
    assert saw_fully_padded_group, (
        f"Expected at least one fully padded group for real_tokens={real_tokens}, "
        f"query_tokens={query_tokens}, merge_group_rows={merge_group_rows}"
    )


class SparseMlaGroupedOutputsTest(parameterized.TestCase):
    """Verify `merge_group_rows` grouped outputs on the served SparseCore path."""

    def _run_served(
        self,
        case: dict[str, Any],
        *,
        merge_group_rows: int = 0,
        mixed_fused_prep: bool = True,
        bypass_outer_jit: bool = False,
        fault: str | None = None,
    ) -> Any:
        active_fault = fault or os.environ.get(FAULT_ENV_VAR, "").strip() or None
        fn = (
            sparse_mla.sparse_ragged_paged_attention.__wrapped__
            if (bypass_outer_jit or active_fault is not None)
            else sparse_mla.sparse_ragged_paged_attention
        )
        call_fn = lambda: fn(  # noqa: E731
            case["q_nope"],
            case["sparsecore_nope"],
            case["sparsecore_rope"],
            case["topk_raw"],
            case["page_table"],
            case["cu_q_lens"],
            case["distribution"],
            sm_scale=0.125,
            k_scale=case["k_scale"],
            return_lse=True,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            gather_and_attention_chunk_size=CHUNK_SIZE,
            topk_cap=TOPK_CAP,
            small_chunk_batch_size=4,
            small_chunk_tokens=32,
            q_pe=case["q_pe"],
            lse_lane_dense=True,
            lse_mask_empty=True,
            mixed_live_gather=True,
            mixed_live_widths=(256, 384),
            mixed_body_levers=sparse_mla.MIXED_BODY_DEFAULT,
            mixed_fused_prep=mixed_fused_prep,
            merge_group_rows=merge_group_rows,
        )
        if active_fault is not None and merge_group_rows > 0:
            return _run_with_in_kernel_fault(call_fn, fault=active_fault)
        return call_fn()

    @parameterized.product(
        (
            {
                "query_tokens": 1024,
                "seq_q": (4, 4, 292),
                "mixed_fused_prep": True,
            },
            {
                "query_tokens": 1024,
                "seq_q": (4, 4, 292),
                "mixed_fused_prep": False,
            },
            {
                "query_tokens": 2048,
                "seq_q": (4, 4, 605),
                "mixed_fused_prep": True,
            },
            {
                "query_tokens": 4096,
                "seq_q": (4, 4, 1349),
                "mixed_fused_prep": True,
            },
        ),
        merge_group_rows=(256, 512),
        overflow=(False, True),
    )
    def test_grouped_outputs_bit_identical_to_single_output(
        self,
        query_tokens: int,
        seq_q: tuple[int, ...],
        mixed_fused_prep: bool,
        merge_group_rows: int,
        overflow: bool,
    ) -> None:
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore MLA grouped-output test requires a TPU")

        seed = (
            20261014
            + query_tokens
            + merge_group_rows
            + int(overflow)
            + 7 * int(mixed_fused_prep)
        )
        case = _build_served_case(
            np.random.default_rng(seed),
            query_tokens=query_tokens,
            seq_q=seq_q,
            overflow=overflow,
        )
        real_tokens = case["real_tokens"]
        self.assertGreater(real_tokens % merge_group_rows, 0)
        self.assertLessEqual(
            (real_tokens // merge_group_rows + 1) * merge_group_rows,
            query_tokens - merge_group_rows,
        )

        ref_out, ref_lse = jax.block_until_ready(
            self._run_served(
                case, merge_group_rows=0, mixed_fused_prep=mixed_fused_prep
            )
        )
        self.assertEqual(ref_out.shape, (query_tokens, NUM_HEADS, NOPE_DIM))
        self.assertEqual(ref_lse.shape, (query_tokens, NUM_HEADS))

        # Positive control: attention output moved away from input q_nope.
        owns = case["owns"][:real_tokens]
        self.assertTrue(owns.any() and (~owns).any())
        diff_from_q = np.abs(
            np.asarray(ref_out[:real_tokens][owns], dtype=np.float32)
            - np.asarray(case["q_nope"][:real_tokens][owns], dtype=np.float32)
        ).max()
        self.assertGreater(diff_from_q, 2e-2)

        outs, lses = jax.block_until_ready(
            self._run_served(
                case,
                merge_group_rows=merge_group_rows,
                mixed_fused_prep=mixed_fused_prep,
            )
        )
        _assert_grouped_matches_reference(
            outs,
            lses,
            ref_out,
            ref_lse,
            query_tokens=query_tokens,
            merge_group_rows=merge_group_rows,
            real_tokens=real_tokens,
            owns=case["owns"],
            lse_mask_empty=True,
        )

        # Positive control for overflow fix-up in grouped mode: perturbing a window-2
        # top-k slot (>= TOPK_CAP) of the last real row (in the partially padded group)
        # must move that group's output and lse.
        if overflow and mixed_fused_prep:
            ov_row = real_tokens - 1
            ov_group, ov_local = divmod(ov_row, merge_group_rows)
            moved_case = dict(case)
            moved_case["topk_raw"] = (
                case["topk_raw"]
                .at[ov_row, TOPK_CAP + 7]
                .set((case["topk_raw"][ov_row, TOPK_CAP + 7] + 13) % 8192)
            )
            moved_outs, moved_lses = jax.block_until_ready(
                self._run_served(
                    moved_case,
                    merge_group_rows=merge_group_rows,
                    mixed_fused_prep=mixed_fused_prep,
                )
            )
            self.assertFalse(
                np.array_equal(
                    np.asarray(
                        jax.lax.bitcast_convert_type(
                            moved_outs[ov_group][ov_local], jnp.uint16
                        )
                    ),
                    np.asarray(
                        jax.lax.bitcast_convert_type(
                            outs[ov_group][ov_local], jnp.uint16
                        )
                    ),
                )
            )
            self.assertFalse(
                np.array_equal(
                    np.asarray(moved_lses[ov_group][ov_local]).view(np.uint32),
                    np.asarray(lses[ov_group][ov_local]).view(np.uint32),
                )
            )

    @parameterized.named_parameters(
        *(
            dict(testcase_name=fault, fault=fault, expected_check=check_id)
            for fault, check_id in FAULT_EXPECTED_CHECK.items()
        )
    )
    def test_fault_injection_control_fails(
        self, fault: str, expected_check: str
    ) -> None:
        """Each in-kernel fault in `sparse_attention_chunk` fails its check ID."""
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore MLA grouped-output test requires a TPU")

        query_tokens = 2048
        merge_group_rows = 512
        case = _build_served_case(
            np.random.default_rng(20261050),
            query_tokens=query_tokens,
            seq_q=(4, 4, 605),
            overflow=False,
        )
        ref_out, ref_lse = jax.block_until_ready(
            self._run_served(case, merge_group_rows=0, mixed_fused_prep=True)
        )
        fault_outs, fault_lses = jax.block_until_ready(
            self._run_served(
                case,
                merge_group_rows=merge_group_rows,
                mixed_fused_prep=True,
                fault=fault,
            )
        )
        with self.assertRaisesRegex(AssertionError, rf"^{expected_check}:"):
            _assert_grouped_matches_reference(
                fault_outs,
                fault_lses,
                ref_out,
                ref_lse,
                query_tokens=query_tokens,
                merge_group_rows=merge_group_rows,
                real_tokens=case["real_tokens"],
                owns=case["owns"],
                lse_mask_empty=True,
            )

    def test_sparse_attention_chunk_direct_bounds_and_faults(self) -> None:
        """Directly test `sparse_attention_chunk` bounds and in-kernel faults."""
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore MLA grouped-output test requires a TPU")

        query_tokens = 1024
        merge_group_rows = 512
        case = _build_served_case(
            np.random.default_rng(20261088),
            query_tokens=query_tokens,
            seq_q=(4, 4, 292),
            overflow=False,
        )

        captured: dict[str, Any] = {}
        orig_chunk = sparse_mla.sparse_attention_chunk

        def _capture_chunk(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("start_pos") == CHUNK_SIZE:
                captured["args"] = args
                captured["kwargs"] = dict(kwargs)
            return orig_chunk(*args, **kwargs)

        with patch.object(
            sparse_mla, "sparse_attention_chunk", side_effect=_capture_chunk
        ):
            ref_out, ref_lse = jax.block_until_ready(
                self._run_served(case, merge_group_rows=0, mixed_fused_prep=True)
            )
            clean_outs, clean_lses = jax.block_until_ready(
                self._run_served(
                    case,
                    merge_group_rows=merge_group_rows,
                    mixed_fused_prep=True,
                    bypass_outer_jit=True,
                )
            )

        self.assertIn("args", captured)
        c_args = list(captured["args"])
        c_kwargs = dict(captured["kwargs"])

        for bad_kwargs, pattern in (
            (
                {"out_row_offset": CHUNK_SIZE + KERNEL_BATCH_SIZE},
                "fall outside the group buffer",
            ),
            (
                {"out_row_offset": -KERNEL_BATCH_SIZE},
                "fall outside the group buffer",
            ),
            ({"out_rows": 0, "out_row_offset": 0}, "o_in .* needs out_rows"),
        ):
            with self.assertRaisesRegex(ValueError, pattern):
                sparse_mla.sparse_attention_chunk(*c_args, **{**c_kwargs, **bad_kwargs})

        bad_args_no_lse = list(c_args)
        bad_args_no_lse[5] = None
        with self.assertRaisesRegex(ValueError, "out_rows needs an lse buffer"):
            sparse_mla.sparse_attention_chunk(*bad_args_no_lse, **c_kwargs)

        # In-bounds shifted `out_row_offset = 16` with finite zero-filled `o_in`
        # executes on TPU and fails `check_group_output_bitwise`.
        o_in_finite = jnp.where(
            jnp.isfinite(c_kwargs["o_in"]),
            c_kwargs["o_in"],
            jnp.zeros_like(c_kwargs["o_in"]),
        )
        shifted_o, shifted_lse = jax.block_until_ready(
            sparse_mla.sparse_attention_chunk(
                *c_args,
                **{
                    **c_kwargs,
                    "o_in": o_in_finite,
                    "out_row_offset": KERNEL_BATCH_SIZE,
                },
            )
        )
        shifted_out_0 = sparse_mla.prepare_outputs(shifted_o, NUM_HEADS, NOPE_DIM)
        shifted_lse_0 = shifted_lse[:, :NUM_HEADS]
        with self.assertRaisesRegex(AssertionError, r"^check_group_output_bitwise:"):
            _assert_grouped_matches_reference(
                (shifted_out_0, *clean_outs[1:]),
                (shifted_lse_0, *clean_lses[1:]),
                ref_out,
                ref_lse,
                query_tokens=query_tokens,
                merge_group_rows=merge_group_rows,
                real_tokens=case["real_tokens"],
                owns=case["owns"],
                lse_mask_empty=True,
            )

        # Zeroing chunk 1's slice of the group lse fails
        # `check_unreached_padding_lse_neg_inf`.
        zero_lse_args = list(c_args)
        zero_lse_args[5] = c_args[5].at[CHUNK_SIZE:].set(0.0)
        _, bad_lse = jax.block_until_ready(
            sparse_mla.sparse_attention_chunk(*zero_lse_args, **c_kwargs)
        )
        with self.assertRaisesRegex(
            AssertionError, r"^check_unreached_padding_lse_neg_inf:"
        ):
            _assert_grouped_matches_reference(
                clean_outs,
                (bad_lse[:, :NUM_HEADS], *clean_lses[1:]),
                ref_out,
                ref_lse,
                query_tokens=query_tokens,
                merge_group_rows=merge_group_rows,
                real_tokens=case["real_tokens"],
                owns=case["owns"],
                lse_mask_empty=True,
            )

    @parameterized.named_parameters(
        dict(
            testcase_name="return_lse_false",
            merge_group_rows=256,
            overrides={"return_lse": False},
        ),
        dict(
            testcase_name="lse_mask_empty_false",
            merge_group_rows=256,
            overrides={"lse_mask_empty": False},
        ),
        dict(
            testcase_name="not_chunk_multiple",
            merge_group_rows=128,
            overrides={},
        ),
        dict(
            testcase_name="does_not_divide_tokens",
            merge_group_rows=768,
            overrides={},
        ),
    )
    def test_merge_group_rows_validation(
        self, merge_group_rows: int, overrides: dict[str, Any]
    ) -> None:
        """Invalid `merge_group_rows` combinations must raise `ValueError`."""
        num_tokens = 1024
        q_nope = jnp.zeros((num_tokens, NUM_HEADS, NOPE_DIM), jnp.bfloat16)
        q_pe = jnp.zeros((num_tokens, NUM_HEADS, ROPE_DIM), jnp.bfloat16)
        nope = jnp.zeros((16, 256, 128), jnp.uint32)
        rope = jnp.zeros((16, 64, 128), jnp.uint32)
        topk = jnp.full((num_tokens, TOPK), -1, jnp.int32)
        page_table = jnp.arange(16, dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 512], jnp.int32)
        distribution = jnp.array([0, 0, 1], jnp.int32)

        kwargs: dict[str, Any] = {
            "cache_layout": sparse_mla.SPARSECORE_CACHE_LAYOUT,
            "gather_and_attention_chunk_size": CHUNK_SIZE,
            "topk_cap": TOPK_CAP,
            "q_pe": q_pe,
            "lse_lane_dense": True,
            "lse_mask_empty": True,
            "return_lse": True,
            "merge_group_rows": merge_group_rows,
            **overrides,
        }
        with self.assertRaisesRegex(ValueError, "merge_group_rows"):
            sparse_mla.sparse_ragged_paged_attention(
                q_nope,
                nope,
                rope,
                topk,
                page_table,
                cu_q_lens,
                distribution,
                **kwargs,
            )


if __name__ == "__main__":
    absltest.main()
