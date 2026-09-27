# SPDX-License-Identifier: Apache-2.0
"""PR CI running tests for FLUX models on TPU v7x.

Fast end-to-end tests designed to run in PR CI (< 3 minutes) on TPU v7x machines.
Covers fast inference with FLUX.1-schnell, golden-image parity with a CPU diffusers
reference, and deterministic seed parity, without file writes.
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
# Golden reference: the same config run through diffusers on CPU, reduced to
# what the test compares. Regenerate with diffusers 0.40.0:
#
#   import torch
#   from diffusers import FluxPipeline
#
#   torch.set_num_threads(8)
#   pipe = FluxPipeline.from_pretrained(
#       "black-forest-labs/FLUX.1-schnell",
#       revision="741f7c3ce8b383c54771c7003378a50191e9efe9",
#       torch_dtype=torch.bfloat16,
#   )
#   # float32 VAE (bf16 convolutions are very slow on CPU); cast the latents.
#   pipe.vae.to(torch.float32)
#   decode = pipe.vae.decode
#   pipe.vae.decode = lambda z, *args, **kwargs: decode(z.float(), *args, **kwargs)
#   img = pipe(
#       prompt="A vibrant green leaf with glistening morning water droplets",
#       height=512,
#       width=512,
#       num_inference_steps=2,
#       guidance_scale=0.0,
#       max_sequence_length=512,
#       generator=torch.Generator(device="cpu").manual_seed(0),
#       output_type="np",
#   ).images[0]
#   DIFFUSERS_THUMBNAIL = _thumbnail(img).round(3)
#   DIFFUSERS_HF_ENERGY = round(_high_freq_energy(img), 5)
# ==============================================================================
DIFFUSERS_THUMBNAIL = np.array(
    [
        [
            [0.191, 0.145, 0.151, 0.123, 0.092, 0.072, 0.070, 0.168],
            [0.124, 0.134, 0.100, 0.069, 0.058, 0.140, 0.389, 0.321],
            [0.061, 0.059, 0.028, 0.029, 0.230, 0.426, 0.538, 0.120],
            [0.026, 0.081, 0.025, 0.100, 0.402, 0.503, 0.259, 0.112],
            [0.033, 0.031, 0.015, 0.335, 0.473, 0.377, 0.099, 0.097],
            [0.025, 0.029, 0.115, 0.419, 0.507, 0.059, 0.085, 0.075],
            [0.015, 0.040, 0.232, 0.462, 0.149, 0.109, 0.134, 0.078],
            [0.041, 0.062, 0.267, 0.205, 0.343, 0.449, 0.387, 0.128],
        ],
        [
            [0.421, 0.419, 0.431, 0.393, 0.316, 0.262, 0.237, 0.326],
            [0.357, 0.361, 0.314, 0.286, 0.266, 0.348, 0.587, 0.517],
            [0.164, 0.143, 0.101, 0.125, 0.387, 0.622, 0.676, 0.307],
            [0.082, 0.120, 0.063, 0.151, 0.570, 0.653, 0.417, 0.293],
            [0.098, 0.061, 0.050, 0.456, 0.639, 0.496, 0.302, 0.289],
            [0.092, 0.070, 0.165, 0.605, 0.642, 0.165, 0.260, 0.227],
            [0.077, 0.158, 0.400, 0.631, 0.306, 0.292, 0.288, 0.187],
            [0.160, 0.246, 0.419, 0.333, 0.517, 0.616, 0.562, 0.253],
        ],
        [
            [0.076, 0.142, 0.161, 0.090, 0.022, 0.012, 0.011, 0.012],
            [0.121, 0.122, 0.074, 0.030, 0.015, 0.033, 0.139, 0.030],
            [0.006, 0.008, 0.009, 0.009, 0.130, 0.071, 0.101, 0.015],
            [0.006, 0.020, 0.007, 0.028, 0.125, 0.186, 0.053, 0.020],
            [0.015, 0.011, 0.008, 0.119, 0.183, 0.091, 0.092, 0.051],
            [0.014, 0.015, 0.017, 0.117, 0.093, 0.021, 0.070, 0.049],
            [0.010, 0.050, 0.079, 0.094, 0.040, 0.023, 0.015, 0.010],
            [0.043, 0.092, 0.064, 0.034, 0.021, 0.071, 0.043, 0.009],
        ],
    ]
)
DIFFUSERS_HF_ENERGY = 0.02016

# Tolerances come from measurements against the reference (thumbnail MAE,
# minimum per-channel thumbnail Pearson, high-frequency energy ratio):
#   TPU v7x                          0.008  0.992  0.98-1.05
#   same recipe on a different CPU   0.006  0.995  1.00
#   max_sequence_length=256          0.073  0.623  0.97
#   another seed                     0.167  0.060  1.40
#   brightness +0.03                 0.030  0.992  0.98
#   R/B channels swapped             0.085  0.581  0.98
#   5x5 box blur                     0.008  0.992  0.18
#   Gaussian noise, sigma 0.02       0.008  0.991  3.73
THUMBNAIL_MAE_MAX = 0.02
THUMBNAIL_PEARSON_MIN = 0.95
HF_ENERGY_RATIO_RANGE = (0.8, 1.25)

# ==============================================================================


def _extract_image(output) -> Image.Image:
    """Extract PIL Image from Omni generate output."""
    return output.images[0]


def _thumbnail(img: np.ndarray) -> np.ndarray:
    """Area-downsample a (512, 512, 3) image to a channel-first (3, 8, 8) grid.

    Averaging 64x64 blocks removes pixel-level drift between backends but keeps
    layout, color and brightness.
    """
    return img.reshape(8, 64, 8, 64, 3).mean(axis=(1, 3)).transpose(2, 0, 1)


def _high_freq_energy(img: np.ndarray) -> float:
    """Mean absolute Laplacian: drops with blur, rises with noise or blocking."""
    laplacian = (
        4 * img[1:-1, 1:-1]
        - img[:-2, 1:-1]
        - img[2:, 1:-1]
        - img[1:-1, :-2]
        - img[1:-1, 2:]
    )
    return float(np.abs(laplacian).mean())


class TestFluxPrCiOnTpu:
    """Fast PR CI functional and numerical tests for FLUX on TPU v7x."""

    @pytest.fixture(scope="class")
    def schnell_engine(self):
        """Class-scoped Omni engine fixture: loads FLUX.1-schnell once per class."""
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
        """Test FLUX.1-schnell 2-step fast generation on TPU v7x without file writes."""

        prompt = (
            "A photo of a majestic snow leopard perched atop a mountain peak at "
            "golden hour, 8k"
        )
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
        assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}"
        assert img.size == (512, 512), f"Expected (512, 512), got {img.size}"

        # In-memory numerical sanity: image must not be blank / collapsed
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        variance = float(np.var(arr))
        assert variance > 10.0, (
            f"Generated image has suspiciously low variance ({variance:.2f}), "
            "likely blank"
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
        np.testing.assert_array_equal(
            arr1,
            arr2,
            err_msg="Identical seed 42 on TPU v7x must produce a bitwise pixel match",
        )

        # Distinct seed must diverge
        assert not np.array_equal(arr1, arr3), (
            "Different seeds should produce divergent pixel values"
        )

    def test_flux_golden_thumbnail_parity_tpu(self, schnell_engine):
        """Compare the golden image with the independent CPU diffusers reference.

        The 8x8 thumbnail catches layout, color and brightness errors; the
        high-frequency energy of the full image catches blur, noise and blocking.
        Full-resolution pixels are not compared: with 2-step bf16 sampling,
        single pixels differ by up to 0.83 even between CPU runs on two machines.
        """

        prompt = "A vibrant green leaf with glistening morning water droplets"
        params = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=0.0,
            # T5 pads the prompt to this length without a mask, so it changes the
            # image and must match the reference.
            max_sequence_length=512,
            seed=0,
            generator_device="cpu",
        )
        outputs = schnell_engine.generate(prompt, params)
        assert outputs, "Omni returned empty output list"

        img = _extract_image(outputs[0])
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

        assert 0.10 < float(np.mean(arr)) < 0.35, (
            f"Suspicious image mean: {np.mean(arr):.4f}"
        )
        assert float(np.var(arr)) > 0.01, (
            f"Suspicious image variance: {np.var(arr):.4f}"
        )

        thumb = _thumbnail(arr)
        mae = float(np.abs(thumb - DIFFUSERS_THUMBNAIL).mean())
        pearson = min(
            float(np.corrcoef(thumb[c].ravel(), DIFFUSERS_THUMBNAIL[c].ravel())[0, 1])
            for c in range(3)
        )
        hf_ratio = _high_freq_energy(arr) / DIFFUSERS_HF_ENERGY
        lo, hi = HF_ENERGY_RATIO_RANGE

        summary = (
            f"thumbnail MAE={mae:.4f} (max {THUMBNAIL_MAE_MAX}), "
            f"min channel Pearson={pearson:.4f} (min {THUMBNAIL_PEARSON_MIN}), "
            f"high-frequency energy ratio={hf_ratio:.3f} (range {lo}-{hi})"
        )
        print(summary)
        assert (
            mae <= THUMBNAIL_MAE_MAX
            and pearson >= THUMBNAIL_PEARSON_MIN
            and lo <= hf_ratio <= hi
        ), f"TPU image diverged from the CPU diffusers reference: {summary}"
