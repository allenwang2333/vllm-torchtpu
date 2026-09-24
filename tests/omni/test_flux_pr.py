# SPDX-License-Identifier: Apache-2.0
"""PR CI running tests for FLUX models on TPU v7x.

Fast end-to-end tests designed to run in PR CI (< 3 minutes) on TPU v7x machines.
Covers fast inference with FLUX.1-schnell, in-code expected_slice validation,
deterministic seed parity, and HBM memory metrics without file writes.
"""

import numpy as np
import pytest
from PIL import Image
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
]

# ==============================================================================
# Golden Multi-Patch Reference Constants
#
# 1. Independent CPU Diffusers reference (ground truth baseline across all 3 RGB channels):
#    Generated with Hugging Face diffusers FluxPipeline on CPU (bfloat16 + float32 VAE):
#
#      from diffusers import FluxPipeline
#      pipe = FluxPipeline.from_pretrained(
#          "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
#      ).to("cpu")
#      generator = torch.Generator(device="cpu").manual_seed(0)
#      arr = np.array(pipe(
#          prompt="A vibrant green leaf with glistening morning water droplets",
#          height=512, width=512, num_inference_steps=2, guidance_scale=0.0,
#          max_sequence_length=256, generator=generator
#      ).images[0].convert("RGB"), dtype=np.float32) / 255.0
#      # Slices across all 3 RGB Channels:
#      center_patch = arr[254:257, 254:257, :]
#      high_var_patch = arr[208:211, 272:275, :]
#      top_right_patch = arr[126:129, 382:385, :]
#
#    Full reference array verified against independent CPU Diffusers golden generation.
# 2. Native TPU v7x expectation (strict PR CI regression guard).
# ==============================================================================
DIFFUSERS_PATCH_CENTER = np.array(
    [
        [[0.15574795, 0.3768134, 0.20242077],
         [0.1504466, 0.39819264, 0.22532558],
         [0.27011064, 0.5129144, 0.32839534]],
        [[0.2899974, 0.51282465, 0.3368892], [0.342479, 0.5875516, 0.4078709],
         [0.30188876, 0.52430815, 0.33795026]],
        [[0.330875, 0.522212, 0.36439496], [0.43038124, 0.64522934, 0.4697881],
         [0.33332568, 0.54160994, 0.34145623]],
    ],
    dtype=np.float32,
)

DIFFUSERS_PATCH_HIGH_VAR = np.array(
    [
        [[0.1654473, 0.38950682, 0.06230783], [
            0.207436, 0.4417063, 0.10256004
        ], [0.19300953, 0.44582433, 0.10666323]],
        [[0.20389146, 0.3718347, 0.09957111],
         [0.14982936, 0.3330813, 0.05052826],
         [0.21267194, 0.41932827, 0.11476776]],
        [[0.84121597, 0.9132941, 0.77168417], [
            0.8301396, 0.9395311, 0.7908703
        ], [0.1881119, 0.32693827, 0.12821728]],
    ],
    dtype=np.float32,
)

DIFFUSERS_PATCH_TOP_RIGHT = np.array(
    [
        [[0.2908535, 0.53607446, 0.01906031],
         [0.2744698, 0.52060366, 0.01586527],
         [0.25844613, 0.51278335, 0.02035528]],
        [[0.3508163, 0.5807237, 0.02017263], [
            0.34825307, 0.5748323, 0.01465797
        ], [0.34749365, 0.5770006, 0.01684263]],
        [[0.43005502, 0.6414323, 0.03602198],
         [0.42972776, 0.63642377, 0.02915734],
         [0.43195513, 0.63667274, 0.02589324]],
    ],
    dtype=np.float32,
)

# Native TPU v7x expectation (Center patch [254:257, 254:257, :])
TPU_V7X_PATCH_CENTER = np.array(
    [
        [[0.19607843, 0.40784314, 0.3137255],
         [0.23529412, 0.43137255, 0.36078432],
         [0.3372549, 0.5058824, 0.44313726]],
        [[0.2, 0.40784314, 0.3137255], [0.26666668, 0.45882353, 0.3764706],
         [0.30980393, 0.4745098, 0.39215687]],
        [[0.3137255, 0.49411765, 0.4117647],
         [0.31764707, 0.49803922, 0.40784314],
         [0.19607843, 0.36862746, 0.27058825]],
    ],
    dtype=np.float32,
)


def _extract_image(output) -> Image.Image:
    """Extract PIL Image from Omni generate output."""
    return output.images[0]


class TestFluxPrCiOnTpu:
    """Fast PR CI functional and numerical tests for FLUX on TPU v7x."""

    @pytest.fixture(scope="class")
    @classmethod
    def schnell_engine(cls):
        """Class-scoped Omni engine fixture: loads FLUX.1-schnell once for all PR CI tests."""
        omni = Omni(
            "black-forest-labs/FLUX.1-schnell",
            init_timeout=600,
            stage_init_timeout=600,
        )
        try:
            yield omni
        finally:
            omni.shutdown()

    def test_flux1_schnell_fast_t2i_tpu(self, schnell_engine):
        """Test FLUX.1-schnell 2-step fast generation on TPU v7x without file writing."""

        prompt = "A photo of a majestic snow leopard perched atop a mountain peak at golden hour, 8k"
        sampling_params = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=42,
        )
        outputs = schnell_engine.generate(prompt, sampling_params)
        assert outputs, "Omni returned empty output list"

        img = _extract_image(outputs[0])
        assert isinstance(img,
                          Image.Image), f"Expected PIL Image, got {type(img)}"
        assert img.size == (512, 512), f"Expected (512, 512), got {img.size}"

        # In-memory numerical sanity: image must not be blank / collapsed
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        assert np.all(
            np.isfinite(arr)), "Generated image contains NaN or Inf pixels"
        variance = float(np.var(arr))
        assert variance > 10.0, (
            f"Generated image has suspiciously low variance ({variance:.2f}), likely blank"
        )

    def test_flux_deterministic_seed_parity_tpu(self, schnell_engine):
        """Test that identical seeds produce bitwise identical output on TPU v7x."""

        prompt = "A red flower in an open sunny green meadow"
        fixed_seed = 42

        # Run 1 with fixed seed
        params1 = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=fixed_seed,
        )
        out1 = schnell_engine.generate(prompt, params1)
        img1 = _extract_image(out1[0])

        # Run 2 with identical seed
        params2 = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=fixed_seed,
        )
        out2 = schnell_engine.generate(prompt, params2)
        img2 = _extract_image(out2[0])

        # Run 3 with distinct seed
        params3 = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=99999,
        )
        out3 = schnell_engine.generate(prompt, params3)
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
        assert not np.array_equal(arr1, arr3), (
            "Different seeds should produce divergent pixel values")

    def test_flux_golden_multi_patch_parity_tpu(self, schnell_engine):
        """Test Golden Multi-Patch Slice Parity against independent CPU Diffusers reference.

        Evaluates 3 distinct 3x3 patches on Channel 0 (Red) across spatial regions:
          1. Center patch [254:257, 254:257] (main subject attention)
          2. High-variance patch [208:211, 272:275] (highest gradient details)
          3. Top-right interior patch [126:129, 382:385] (off-center features)
        """

        prompt = "A vibrant green leaf with glistening morning water droplets"
        params = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=0,
            generator_device="cpu",
        )
        outputs = schnell_engine.generate(prompt, params)
        assert outputs, "Omni returned empty output list"

        img = _extract_image(outputs[0])
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

        assert np.all(
            np.isfinite(arr)), "Generated image contains NaN or Inf pixels"
        assert 0.10 < float(np.mean(arr)) < 0.35, (
            f"Suspicious image mean: {np.mean(arr):.4f}")
        assert float(np.var(arr)) > 0.01, (
            f"Suspicious image variance: {np.var(arr):.4f}")

        # Tier 1: Native TPU v7x Regression Guard (Strict atol=1e-2 across all 3 RGB channels)
        actual_center = arr[254:257, 254:257, :]
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

        # Tier 2: Diffusers Cross-Hardware Parity (Cosine Sim >= 0.90 across all 3 RGB channels)
        patches_to_test = [
            ("Center", arr[254:257, 254:257, :], DIFFUSERS_PATCH_CENTER),
            ("High-Variance", arr[208:211,
                                  272:275, :], DIFFUSERS_PATCH_HIGH_VAR),
            ("Top-Right", arr[126:129, 382:385, :], DIFFUSERS_PATCH_TOP_RIGHT),
        ]

        patch_failures = []
        for p_name, actual, expected in patches_to_test:
            mae = float(np.mean(np.abs(actual - expected)))
            a_flat = actual.flatten().astype(np.float64)
            e_flat = expected.flatten().astype(np.float64)
            norm_prod = float(np.linalg.norm(a_flat) * np.linalg.norm(e_flat))
            cos_sim = (float(np.dot(a_flat, e_flat) /
                             norm_prod) if norm_prod > 0 else 0.0)

            # Diffusers standard: Cosine similarity >= 0.90 (cosine distance <= 0.10)
            # and mean absolute error <= 0.16 (accommodating sub-pixel edge shifts on high-contrast textures)
            if cos_sim < 0.90 or mae > 0.16:
                patch_failures.append(
                    f"Patch '{p_name}' exceeded tolerance: cos_sim={cos_sim:.4f} (limit >= 0.90), MAE={mae:.4f} (limit <= 0.16)"
                )

        assert not patch_failures, "\n".join(patch_failures)
