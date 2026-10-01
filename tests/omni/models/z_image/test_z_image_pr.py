# SPDX-License-Identifier: Apache-2.0
"""PR CI tests for Z-Image model family on TPU.

Single-engine end-to-end smoke test for Z-Image-Turbo (512x512, 4 steps) plus
deterministic seed parity, without file writes. The model is resolved through
the HF cache (``HF_HOME``) like the FLUX PR tests.

Temporarily also marked ``nightly`` so the per-PR omni step (``-m 'not nightly
and omni'``) does not download Z-Image-Turbo from Hugging Face on every PR.
"""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("vllm_omni", reason="vllm_omni is required for Z-Image tests")

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
    # TODO(rishabhmanoj): drop once Tongyi-MAI/Z-Image-Turbo is staged in the
    # CI golden model bucket (HF_HOME) so this can run in the per-PR omni step.
    pytest.mark.nightly,
]

Z_IMAGE_TURBO = "Tongyi-MAI/Z-Image-Turbo"


def _extract_image(output) -> Image.Image:
    """Extract PIL Image from Omni generate output."""
    return output.images[0]


class TestZImagePrCiOnTpu:
    """Fast PR CI functional and numerical tests for Z-Image-Turbo on TPU."""

    @pytest.fixture(scope="class")
    def turbo_engine(self):
        """Class-scoped Omni engine fixture: loads Z-Image-Turbo once per class."""
        omni = Omni(
            Z_IMAGE_TURBO,
            init_timeout=900,
            stage_init_timeout=900,
        )
        try:
            yield omni
        finally:
            omni.shutdown()

    def test_z_image_turbo_fast_t2i_tpu(self, turbo_engine):
        """Test Z-Image-Turbo fast generation and deterministic seed parity on TPU."""
        prompt = (
            "A cinematic portrait of a red panda astronaut in space, "
            "detailed fur, soft rim light, 4k"
        )
        params1 = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=0.0,
            seed=42,
        )
        out1 = turbo_engine.generate(prompt, params1)
        assert out1, "Omni returned empty output list"
        img1 = _extract_image(out1[0])
        assert isinstance(img1, Image.Image)
        assert img1.size == (512, 512)

        arr1 = np.asarray(img1.convert("RGB"), dtype=np.float32) / 255.0
        assert 0.05 < float(np.mean(arr1)) < 0.95, (
            f"Suspicious image mean: {np.mean(arr1):.4f}"
        )
        assert float(np.var(arr1)) > 0.005, (
            f"Suspiciously low variance ({np.var(arr1):.4f}), likely blank"
        )

        # Replay with identical seed must match bitwise.
        out2 = turbo_engine.generate(prompt, params1)
        img2 = _extract_image(out2[0])
        arr2 = np.asarray(img2.convert("RGB"), dtype=np.float32) / 255.0
        np.testing.assert_array_equal(
            arr1,
            arr2,
            err_msg="Identical seed 42 on TPU must produce a bitwise pixel match",
        )

        # Distinct seed must diverge.
        params3 = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=0.0,
            seed=99999,
        )
        out3 = turbo_engine.generate(prompt, params3)
        img3 = _extract_image(out3[0])
        arr3 = np.asarray(img3.convert("RGB"), dtype=np.float32) / 255.0
        assert not np.array_equal(arr1, arr3), (
            "Different seeds should produce divergent pixel values"
        )
