# SPDX-License-Identifier: Apache-2.0
"""Nightly build tests for Z-Image models on TPU.

Integration and numerical correctness tests across the Z-Image model family:
- Tongyi-MAI/Z-Image-Turbo (distilled 8-step guidance-free)
- Tongyi-MAI/Z-Image (base 28-step with Classifier-Free Guidance)

Models are resolved through the HF cache (``HF_HOME``) like the FLUX tests.

Every generation runs in a fresh interpreter (see ``_generate_once``): the
single-chip uniproc executor opens the TPU *in-process* and PJRT keeps
``/dev/vfio/0`` until that process exits, so a multi-chip run launched
afterwards from the same pytest process cannot acquire chip 0.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("vllm_omni", reason="vllm_omni is required for Z-Image tests")

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
]

Z_IMAGE_TURBO = "Tongyi-MAI/Z-Image-Turbo"
Z_IMAGE_BASE = "Tongyi-MAI/Z-Image"

PROMPT = (
    "A photorealistic landscape of majestic mountains and a serene lake "
    "at sunset, golden hour reflections"
)

# Same-seed parity of ulysses_degree=4 against the single-chip image.
# "advanced_uaa" pads 30 heads to 32 and runs bf16 all-to-all collectives, so
# bit-exactness is not expected. Measured on v6e-8 (Turbo, 8 steps, seed 42):
# PSNR 41.4 dB, MAE 0.0029. Thresholds leave >10 dB / ~3x headroom.
SP_PARITY_PSNR_MIN_DB = 30.0
SP_PARITY_MAE_MAX = 0.01

# Executed with ``python -c`` in a child interpreter; argv[1] is a JSON config.
_GENERATE_CHILD_SCRIPT = """
import json, sys
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

cfg = json.loads(sys.argv[1])
omni = Omni(
    cfg["model_id"], init_timeout=1800, stage_init_timeout=1800, **cfg["omni_kwargs"]
)
try:
    params = OmniDiffusionSamplingParams(**cfg["params"])
    outputs = omni.generate(cfg["prompt"], params)
    assert outputs, "empty generation output"
    outputs[0].images[0].save(cfg["out_path"])
finally:
    omni.shutdown()
"""

# Per-generation child timeout. The parity test runs two children back to
# back, so 2 x 800 s stays under the CI-wide pytest ``--timeout=1800`` and a
# hung worker is reaped here (killing the child releases its TPU chips) rather
# than by pytest-timeout, which would leave the child running. Measured sp=1 +
# sp=4 including model load and compile is ~8 min total on a v6e-8. Until the
# Z-Image weights are staged in the CI model cache, the first child's budget
# also has to cover the Hugging Face download (~20 GB per model).
_GENERATE_TIMEOUT_S = 800


def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def compute_distribution_metrics(img: Image.Image) -> dict[str, float]:
    """Calculate scalar distribution moments and percentiles from output pixels."""
    arr = _to_array(img)
    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "var": float(np.var(arr)),
    }


def _assert_plausible_image(img: Image.Image, label: str) -> None:
    assert isinstance(img, Image.Image)
    assert img.size == (1024, 1024)
    metrics = compute_distribution_metrics(img)
    assert 0.10 < metrics["mean"] < 0.90, (
        f"Unnatural pixel mean on {label}: {metrics['mean']:.4f}"
    )
    assert metrics["var"] > 0.01, (
        f"Image variance too low on {label} ({metrics['var']:.4f})"
    )
    assert metrics["p90"] > metrics["p10"], f"Collapsed dynamic range on {label}"


def _sampling_params(steps: int, guidance: float) -> dict:
    return {
        "height": 1024,
        "width": 1024,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "seed": 42,
    }


def _generate_once(
    model_id: str, params: dict, out_path: Path, **omni_kwargs
) -> Image.Image:
    """Generate one image in a fresh interpreter and return it.

    A child process (rather than ``Omni`` in-process) is required so that the
    TPU device nodes opened by a single-chip run are released before the next
    run starts; see the module docstring.
    """
    cfg = {
        "model_id": model_id,
        "prompt": PROMPT,
        "params": params,
        "omni_kwargs": omni_kwargs,
        "out_path": str(out_path),
    }
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", _GENERATE_CHILD_SCRIPT, json.dumps(cfg)],
        timeout=_GENERATE_TIMEOUT_S,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Generation for {model_id} ({omni_kwargs}) exited with {proc.returncode}"
    )
    assert out_path.is_file(), f"Generation for {model_id} produced no image"
    return Image.open(out_path)


class TestZImageNightlyOnTpu:
    """Nightly regression test suite for Z-Image models on TPU."""

    @pytest.mark.parametrize(
        "model_id,steps,guidance",
        [
            (Z_IMAGE_TURBO, 8, 0.0),
            (Z_IMAGE_BASE, 28, 3.5),
        ],
        ids=["z-image-turbo", "z-image-base"],
    )
    def test_z_image_model_matrix_tpu(
        self, tmp_path: Path, model_id: str, steps: int, guidance: float
    ):
        """Validate common path execution across Z-Image-Turbo and Base on TPU."""
        img = _generate_once(
            model_id, _sampling_params(steps, guidance), tmp_path / "img.png"
        )
        _assert_plausible_image(img, model_id)

    def test_z_image_turbo_ulysses_sp4_parity_tpu(self, tmp_path: Path):
        """Same-seed parity of ``ulysses_degree=4`` against a single-chip run.

        This is the only test that exercises the Z-Image multi-chip path that
        the TPU platform enables: 30 heads are not divisible by 4, so strict
        Ulysses mode would raise and the run only succeeds because
        ``OmniTpuPlatform.init_diffusion_model_runner_runtime`` promoted
        ``ulysses_mode`` to ``"advanced_uaa"`` (``ulysses_degree=2`` divides 30
        and would silently stay in strict mode). It needs a >= 4 chip runner;
        on the single-chip Omni CI shape it is skipped.
        """
        from vllm_torchtpu.omni.platform import OmniTpuPlatform

        if OmniTpuPlatform.get_device_count() < 4:
            pytest.skip("Requires at least 4 TPU chips for ulysses_degree=4")

        params = _sampling_params(steps=8, guidance=0.0)
        ref = _to_array(_generate_once(Z_IMAGE_TURBO, params, tmp_path / "sp1.png"))
        sp4_img = _generate_once(
            Z_IMAGE_TURBO, params, tmp_path / "sp4.png", ulysses_degree=4
        )
        _assert_plausible_image(sp4_img, f"{Z_IMAGE_TURBO} (sp=4)")
        sp4 = _to_array(sp4_img)

        mae = float(np.mean(np.abs(sp4 - ref)))
        mse = float(np.mean((sp4 - ref) ** 2))
        psnr = float("inf") if mse == 0 else 10.0 * float(np.log10(1.0 / mse))
        # Printed (not just asserted) so nightly logs record the measured margin.
        print(f"[z-image sp=4 parity] PSNR={psnr:.2f} dB MAE={mae:.4f}", flush=True)
        assert psnr >= SP_PARITY_PSNR_MIN_DB, (
            f"sp=4 vs sp=1 PSNR {psnr:.2f} dB below {SP_PARITY_PSNR_MIN_DB} dB "
            f"(MAE={mae:.4f})"
        )
        assert mae <= SP_PARITY_MAE_MAX, (
            f"sp=4 vs sp=1 MAE {mae:.4f} above {SP_PARITY_MAE_MAX} (PSNR={psnr:.2f} dB)"
        )
