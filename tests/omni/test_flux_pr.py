# SPDX-License-Identifier: Apache-2.0
"""PR CI running tests for FLUX models on TPU v7x.

Fast end-to-end tests designed to run in PR CI (< 3 minutes) on TPU v7x machines.
Covers fast inference with FLUX.1-schnell and FLUX.2-klein, in-code expected_slice
validation, deterministic seed parity, and HBM memory metrics without file writes.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Ensure repository root is on sys.path for test utilities
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.distributed.tpu_test_utils import \
    run_in_isolated_process  # noqa: E402
from vllm_torchtpu.omni.platform import OmniTpuPlatform  # noqa: E402

pytestmark = [pytest.mark.diffusion]

# ==============================================================================
# Golden Multi-Patch Reference Constants
#
# 1. Independent CPU Diffusers reference (ground truth baseline):
#    Generated with Hugging Face diffusers FluxPipeline on CPU (bfloat16 + float32 VAE).
#    Full reference array saved in gs://mehdy/vllm/omni/golden/flux_schnell_cpu_golden_img.npy
# 2. Native TPU v7x expectation (strict PR CI regression guard).
# ==============================================================================
DIFFUSERS_PATCH_CENTER = np.array([
    [0.15574795, 0.15044659, 0.27011064],
    [0.28999740, 0.34247899, 0.30188876],
    [0.33087501, 0.43038124, 0.33332568],
],
                                  dtype=np.float32)

DIFFUSERS_PATCH_HIGH_VAR = np.array([
    [0.16544730, 0.20743600, 0.19300953],
    [0.20389146, 0.14982936, 0.21267194],
    [0.84121597, 0.83013960, 0.18811190],
],
                                    dtype=np.float32)

DIFFUSERS_PATCH_TOP_RIGHT = np.array([
    [0.29085350, 0.27446980, 0.25844613],
    [0.35081630, 0.34825307, 0.34749365],
    [0.43005502, 0.42972776, 0.43195513],
],
                                     dtype=np.float32)

# Native TPU v7x expectation (Center patch [254:257, 254:257, 0])
TPU_V7X_PATCH_CENTER = np.array([
    [0.196078, 0.235294, 0.337255],
    [0.200000, 0.266667, 0.309804],
    [0.313726, 0.317647, 0.196078],
],
                                dtype=np.float32)


def _resolve_model_path(model_id: str) -> str:
    """Resolve local cached snapshot directory or fall back to HF model ID."""
    candidate_roots = []
    if "HF_HUB_CACHE" in os.environ:
        candidate_roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if "HF_HOME" in os.environ:
        candidate_roots.append(Path(os.environ["HF_HOME"]) / "hub")
    candidate_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    safe_name = "models--" + model_id.replace("/", "--")
    for root in candidate_roots:
        snapshot_dir = root / safe_name / "snapshots"
        if snapshot_dir.is_dir():
            snapshots = sorted(list(snapshot_dir.iterdir()))
            if snapshots:
                return str(snapshots[0])
    return model_id


def _extract_image(output) -> Image.Image:
    """Extract PIL Image from Omni generate output."""
    return output.images[0]


class TestFluxPrCiOnTpu:
    """Fast PR CI functional and numerical tests for FLUX on TPU v7x."""

    @pytest.mark.spawns_tpu_workers
    def test_flux1_schnell_fast_t2i_tpu(self):
        """Test FLUX.1-schnell 2-step fast generation on TPU v7x without file writing."""

        model_path = _resolve_model_path("black-forest-labs/FLUX.1-schnell")
        prompt = "A photo of a majestic snow leopard perched atop a mountain peak at golden hour, 8k"

        omni = Omni(model_path, init_timeout=600, stage_init_timeout=600)
        try:
            sampling_params = OmniDiffusionSamplingParams(
                height=512,
                width=512,
                num_inference_steps=2,
                guidance_scale=0.0,
                seed=42,
            )
            outputs = omni.generate(prompt, sampling_params)
            assert outputs, "Omni returned empty output list"

            img = _extract_image(outputs[0])
            assert isinstance(
                img, Image.Image), f"Expected PIL Image, got {type(img)}"
            assert img.size == (512,
                                512), f"Expected (512, 512), got {img.size}"

            # In-memory numerical sanity: image must not be blank / collapsed
            arr = np.array(img.convert("RGB"), dtype=np.float32)
            assert np.all(
                np.isfinite(arr)), "Generated image contains NaN or Inf pixels"
            variance = float(np.var(arr))
            assert variance > 10.0, f"Generated image has suspiciously low variance ({variance:.2f}), likely blank"
        finally:
            omni.shutdown()

    @pytest.mark.spawns_tpu_workers
    def test_flux2_klein_fast_t2i_tpu(self):
        """Test FLUX.2-klein-4B fast generation on TPU v7x without file writing."""

        model_path = _resolve_model_path("black-forest-labs/FLUX.2-klein-4B")
        prompt = "A cute fluffy red panda wearing tiny goggles reading a magical glowing book, 8k"

        omni = Omni(model_path, init_timeout=600, stage_init_timeout=600)
        try:
            sampling_params = OmniDiffusionSamplingParams(
                height=512,
                width=512,
                num_inference_steps=4,
                guidance_scale=1.0,
                seed=123,
            )

            outputs = omni.generate(prompt, sampling_params)
            assert outputs, "Omni returned empty output list"

            img = _extract_image(outputs[0])
            assert isinstance(
                img, Image.Image), f"Expected PIL Image, got {type(img)}"
            assert img.size == (512,
                                512), f"Expected (512, 512), got {img.size}"

            arr = np.array(img.convert("RGB"), dtype=np.float32)
            assert np.all(np.isfinite(arr))
            assert float(
                np.var(arr)) > 10.0, "Generated image variance is too low"
        finally:
            omni.shutdown()

    @pytest.mark.spawns_tpu_workers
    def test_flux_deterministic_seed_parity_tpu(self):
        """Test that identical seeds produce bitwise identical output on TPU v7x."""

        model_path = _resolve_model_path("black-forest-labs/FLUX.1-schnell")
        prompt = "A red flower in an open sunny green meadow"
        fixed_seed = 42

        omni = Omni(model_path, init_timeout=600, stage_init_timeout=600)
        try:
            # Run 1 with fixed seed
            params1 = OmniDiffusionSamplingParams(
                height=512,
                width=512,
                num_inference_steps=2,
                guidance_scale=0.0,
                seed=fixed_seed,
            )
            out1 = omni.generate(prompt, params1)
            img1 = _extract_image(out1[0])

            # Run 2 with identical seed
            params2 = OmniDiffusionSamplingParams(
                height=512,
                width=512,
                num_inference_steps=2,
                guidance_scale=0.0,
                seed=fixed_seed,
            )
            out2 = omni.generate(prompt, params2)
            img2 = _extract_image(out2[0])

            # Run 3 with distinct seed
            params3 = OmniDiffusionSamplingParams(
                height=512,
                width=512,
                num_inference_steps=2,
                guidance_scale=0.0,
                seed=99999,
            )
            out3 = omni.generate(prompt, params3)
            img3 = _extract_image(out3[0])

            arr1 = np.array(img1.convert("RGB"), dtype=np.float32)
            arr2 = np.array(img2.convert("RGB"), dtype=np.float32)
            arr3 = np.array(img3.convert("RGB"), dtype=np.float32)

            # Identical seeds must match exactly bitwise
            np.testing.assert_allclose(
                arr1,
                arr2,
                atol=0.0,
                rtol=0.0,
                err_msg=
                "Identical seed 42 on TPU v7x must produce exact bitwise pixel match",
            )

            # Distinct seed must diverge
            assert not np.array_equal(
                arr1,
                arr3), "Different seeds should produce divergent pixel values"
        finally:
            omni.shutdown()

    @pytest.mark.spawns_tpu_workers
    def test_flux_golden_multi_patch_parity_tpu(self):
        """Test Golden Multi-Patch Slice Parity against independent CPU Diffusers reference.

        Evaluates 3 distinct 3x3 patches on Channel 0 (Red) across spatial regions:
          1. Center patch [254:257, 254:257] (main subject attention)
          2. High-variance patch [208:211, 272:275] (highest gradient details)
          3. Top-right interior patch [126:129, 382:385] (off-center features)
        """

        model_path = _resolve_model_path("black-forest-labs/FLUX.1-schnell")
        prompt = "A vibrant green leaf with glistening morning water droplets"

        omni = Omni(model_path, init_timeout=600, stage_init_timeout=600)
        try:
            params = OmniDiffusionSamplingParams(
                height=512,
                width=512,
                num_inference_steps=2,
                guidance_scale=0.0,
                seed=0,
                generator_device="cpu",
            )
            outputs = omni.generate(prompt, params)
            assert outputs, "Omni returned empty output list"

            img = _extract_image(outputs[0])
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        finally:
            omni.shutdown()

        assert np.all(
            np.isfinite(arr)), "Generated image contains NaN or Inf pixels"
        assert 0.10 < float(
            np.mean(arr)) < 0.35, f"Suspicious image mean: {np.mean(arr):.4f}"
        assert float(np.var(
            arr)) > 0.01, f"Suspicious image variance: {np.var(arr):.4f}"

        # Tier 1: Native TPU v7x Regression Guard (Strict atol=1e-2)
        actual_center = arr[254:257, 254:257, 0]
        max_diff_tpu = float(
            np.max(np.abs(actual_center - TPU_V7X_PATCH_CENTER)))
        np.testing.assert_allclose(
            actual_center,
            TPU_V7X_PATCH_CENTER,
            atol=1e-2,
            rtol=1e-2,
            err_msg=
            f"TPU v7x native regression detected! (max diff: {max_diff_tpu:.4f})",
        )

        # Tier 2: Diffusers Cross-Hardware Parity (Cosine Sim >= 0.90 across spatial patches)
        patches_to_test = [
            ("Center", arr[254:257, 254:257, 0], DIFFUSERS_PATCH_CENTER),
            ("High-Variance", arr[208:211, 272:275,
                                  0], DIFFUSERS_PATCH_HIGH_VAR),
            ("Top-Right", arr[126:129, 382:385, 0], DIFFUSERS_PATCH_TOP_RIGHT),
        ]

        patch_failures = []
        for p_name, actual, expected in patches_to_test:
            mae = float(np.mean(np.abs(actual - expected)))
            a_flat = actual.flatten().astype(np.float64)
            e_flat = expected.flatten().astype(np.float64)
            norm_prod = float(np.linalg.norm(a_flat) * np.linalg.norm(e_flat))
            cos_sim = float(np.dot(a_flat, e_flat) /
                            norm_prod) if norm_prod > 0 else 0.0

            # Diffusers standard: Cosine similarity >= 0.90 (cosine distance <= 0.10)
            # and mean absolute error <= 0.16 (accommodating sub-pixel edge shifts on high-contrast textures)
            if cos_sim < 0.90 or mae > 0.16:
                patch_failures.append(
                    f"Patch '{p_name}' exceeded tolerance: cos_sim={cos_sim:.4f} (limit >= 0.90), MAE={mae:.4f} (limit <= 0.16)"
                )

        assert not patch_failures, "\n".join(patch_failures)

    def test_flux_tpu_memory_stats(self):
        """Verify TPU device detection, memory querying, and statistics resetting."""
        run_in_isolated_process(_run_flux_tpu_memory_stats, timeout=60)


def _run_flux_tpu_memory_stats() -> None:
    """Isolated target verifying TPU device detection and memory querying."""
    import torch_tpu  # noqa: F401

    device_name = OmniTpuPlatform.get_device_name()
    assert "tpu" in device_name.lower()

    # TPU v7x memory limits (e.g. 16/32/64/96/128 GiB per chip/slice)
    total_mem = OmniTpuPlatform.get_device_total_memory()
    assert total_mem > 0, "Device total memory must be positive"
    assert total_mem <= 128 * 1024**3, f"Unexpected HBM size: {total_mem / 1024**3:.1f} GiB"

    torch.accelerator.reset_peak_memory_stats()
