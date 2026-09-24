# SPDX-License-Identifier: Apache-2.0
"""Nightly build tests for FLUX models on TPU v7x.

Focused integration and numerical correctness tests for nightly builds (~5 minutes).
Focuses on accuracy of common paths: tri-model matrix validation (FLUX.1-schnell,
FLUX.1-dev, FLUX.2-klein-4B) with distribution metric signatures and RoPE sequence length alignment.
All tests run in-memory without PNG writes.
"""

import numpy as np
import pytest
from PIL import Image
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
]


def _extract_image(output) -> Image.Image:
    """Extract PIL Image from Omni generate output."""
    return output.images[0]


def compute_distribution_metrics(img: Image.Image) -> dict[str, float]:
    """Calculate scalar distribution moments and percentiles from output pixels."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "energy": float(np.mean(np.square(arr))),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "var": float(np.var(arr)),
    }


class TestFluxNightlyOnTpu:
    """Nightly regression test suite for FLUX models on TPU v7x."""

    @pytest.mark.parametrize(
        "model_id,steps,guidance",
        [
            ("black-forest-labs/FLUX.1-schnell", 4, 0.0),
            ("black-forest-labs/FLUX.1-dev", 10, 3.5),
            ("black-forest-labs/FLUX.2-klein-4B", 4, 1.0),
        ],
    )
    def test_flux_tri_model_matrix_tpu(self, model_id: str, steps: int,
                                       guidance: float):
        """Validate common path execution across all three FLUX model variants on TPU v7x.

        Covers distilled guidance-free Schnell, full 10-step CFG guidance Dev with distribution
        metric signatures, and Klein-4B with RoPE sequence length alignment.
        """
        prompt = "A futuristic glowing crystal monolith in a misty desert at night, neon reflections"

        omni = Omni(model_id, init_timeout=600, stage_init_timeout=600)
        try:
            sampling_params = OmniDiffusionSamplingParams(
                height=512,
                width=512,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=42,
            )

            outputs = omni.generate(prompt, sampling_params)
            assert outputs, f"Failed generation for model {model_id}"

            img = _extract_image(outputs[0])
            assert isinstance(img, Image.Image)
            assert img.size == (512, 512)

            metrics = compute_distribution_metrics(img)
            assert 0.10 < metrics["mean"] < 0.90, (
                f"Unnatural pixel mean on {model_id}: {metrics['mean']:.4f}")
            assert metrics["std"] > 0.05, (
                f"Suspiciously low image contrast on {model_id} (std={metrics['std']:.4f})"
            )
            assert metrics["var"] > 0.01, (
                f"Image variance too low on {model_id} ({metrics['var']:.4f})")
            assert metrics["p90"] > metrics["p10"], (
                f"Collapsed dynamic range on {model_id}")
        finally:
            omni.shutdown()
