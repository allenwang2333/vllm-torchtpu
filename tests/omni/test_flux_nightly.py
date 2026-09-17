# SPDX-License-Identifier: Apache-2.0
"""Nightly build tests for FLUX models on TPU v7x.

Focused integration and numerical correctness tests for nightly builds (~5 minutes).
Focuses on accuracy of common paths: full guidance-distilled FLUX.1-dev denoise loops
with distribution metric signatures, tri-model matrix validation (FLUX.1-schnell,
FLUX.1-dev, FLUX.2-klein-4B), and overlapped CPU Diffusers vs. TPU Omni parity.
All tests run in-memory without PNG writes.
"""

import gc
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from diffusers import FluxPipeline
from PIL import Image
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.diffusion,
    pytest.mark.spawns_tpu_workers,
]


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
            snapshots = sorted(snapshot_dir.iterdir())
            if snapshots:
                return str(snapshots[0])
    return model_id


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


def compute_image_parity(img1: Image.Image,
                         img2: Image.Image) -> dict[str, float]:
    """Compute mathematical parity and error metrics (PSNR, MAE, MSE) between two images."""
    arr1 = np.asarray(img1.convert("RGB"), dtype=np.float32) / 255.0
    arr2 = np.asarray(img2.convert("RGB"), dtype=np.float32) / 255.0

    assert arr1.shape == arr2.shape, f"Shape mismatch: {arr1.shape} vs {arr2.shape}"

    diff = arr1 - arr2
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(np.square(diff)))
    psnr = float("inf") if mse == 0.0 else float(20.0 *
                                                 np.log10(1.0 / np.sqrt(mse)))

    return {"mae": mae, "mse": mse, "psnr": psnr}


class _CpuFluxReferenceWorker:
    """Spawns an isolated background CPU process to compute a 64x64 FLUX reference.

    Runs concurrently with earlier TPU tests to overlap CPU wall-clock latency.
    """

    def __init__(self,
                 model_id: str = "black-forest-labs/FLUX.1-schnell",
                 prompt: str = "A red ball on a white plane",
                 seed: int = 42,
                 steps: int = 1,
                 height: int = 64,
                 width: int = 64,
                 max_sequence_length: int = 128) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.seed = seed
        self.steps = steps
        self.height = height
        self.width = width
        self.max_sequence_length = max_sequence_length
        self.start_wall_time = time.perf_counter()

        self._ctx = multiprocessing.get_context("spawn")
        self._queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=self._worker_target,
            args=(self.model_id, self.prompt, self.seed, self.steps,
                  self.height, self.width, self.max_sequence_length,
                  self._queue),
        )
        self._process.start()

    @staticmethod
    def _worker_target(model_id: str, prompt: str, seed: int, steps: int,
                       height: int, width: int, max_sequence_length: int,
                       result_queue: Any) -> None:
        t0 = time.perf_counter()
        model_path = _resolve_model_path(model_id)
        pipe = FluxPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16).to("cpu")
        t_load = time.perf_counter() - t0

        t1 = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
        )
        t_infer = time.perf_counter() - t1
        arr = np.array(output.images[0].convert("RGB"),
                       dtype=np.float32) / 255.0
        t_total = time.perf_counter() - t0

        timings = {
            "load_seconds": t_load,
            "infer_seconds": t_infer,
            "total_cpu_seconds": t_total,
        }
        result_queue.put((arr, timings))

    def get_result(self,
                   timeout: float = 600
                   ) -> tuple[np.ndarray, dict[str, float]]:
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            self.cleanup()
            raise AssertionError(
                f"Background CPU FLUX job timed out after {timeout}s")
        if self._process.exitcode != 0:
            self.cleanup()
            raise AssertionError(
                f"Background CPU FLUX job failed (exitcode={self._process.exitcode})"
            )

        arr, timings = self._queue.get(timeout=5)
        self.cleanup()
        return arr, timings

    def cleanup(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._queue.close()


class TestFluxNightlyOnTpu:
    """Nightly regression test suite for FLUX models on TPU v7x."""

    @pytest.fixture(scope="class", autouse=True)
    def _background_cpu_job(self, request):
        """Start CPU generation in background if parity test is scheduled to run."""
        test_names = [item.name for item in request.session.items]
        worker = None
        if any("cpu_vs_tpu_parity" in name for name in test_names):
            worker = _CpuFluxReferenceWorker(
                model_id="black-forest-labs/FLUX.1-schnell",
                prompt="A red ball on a white plane",
                seed=42,
                steps=1,
                height=64,
                width=64,
            )
        yield worker
        if worker is not None:
            worker.cleanup()

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
        model_path = _resolve_model_path(model_id)
        prompt = "A futuristic glowing crystal monolith in a misty desert at night, neon reflections"

        omni = Omni(model_path, init_timeout=600, stage_init_timeout=600)
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
            assert 0.10 < metrics[
                "mean"] < 0.90, f"Unnatural pixel mean on {model_id}: {metrics['mean']:.4f}"
            assert metrics[
                "std"] > 0.05, f"Suspiciously low image contrast on {model_id} (std={metrics['std']:.4f})"
            assert metrics[
                "var"] > 0.01, f"Image variance too low on {model_id} ({metrics['var']:.4f})"
            assert metrics["p90"] > metrics[
                "p10"], f"Collapsed dynamic range on {model_id}"
        finally:
            omni.shutdown()

    def test_flux_cpu_vs_tpu_parity(self, _background_cpu_job):
        """Validate cross-platform numerical parity between CPU Diffusers and TPU Omni."""
        assert _background_cpu_job is not None, "Background CPU worker was not initialized"

        print("\n[CPU Reference] Awaiting background CPU generation result...",
              flush=True)
        t_wait_start = time.perf_counter()
        arr_cpu, timings = _background_cpu_job.get_result(timeout=600)
        wait_seconds = time.perf_counter() - t_wait_start

        print(
            f"\n[CPU Reference Timing]\n"
            f"  Model:             {_background_cpu_job.model_id}\n"
            f"  Resolution:        {_background_cpu_job.height}x{_background_cpu_job.width}, Steps: {_background_cpu_job.steps}\n"
            f"  CPU Load Time:     {timings['load_seconds']:.2f}s\n"
            f"  CPU Infer Time:    {timings['infer_seconds']:.2f}s\n"
            f"  Total CPU Time:    {timings['total_cpu_seconds']:.2f}s\n"
            f"  Wait in this test: {wait_seconds:.2f}s (overlapped with previous TPU tests)\n",
            flush=True,
        )

        model_path = _resolve_model_path(_background_cpu_job.model_id)
        omni_tpu = Omni(model_path, init_timeout=600, stage_init_timeout=600)
        try:
            out_tpu = omni_tpu.generate(
                _background_cpu_job.prompt,
                OmniDiffusionSamplingParams(
                    height=_background_cpu_job.height,
                    width=_background_cpu_job.width,
                    num_inference_steps=_background_cpu_job.steps,
                    guidance_scale=0.0,
                    seed=_background_cpu_job.seed,
                    generator_device="cpu",
                    max_sequence_length=(
                        _background_cpu_job.max_sequence_length),
                ),
            )
            img_tpu = _extract_image(out_tpu[0])
        finally:
            omni_tpu.shutdown()
            del omni_tpu
            gc.collect()

        img_cpu = Image.fromarray((arr_cpu * 255).astype(np.uint8))
        parity = compute_image_parity(img_tpu, img_cpu)

        print(
            f"[Parity Metrics]\n"
            f"  PSNR: {parity['psnr']:.2f} dB\n"
            f"  MAE:  {parity['mae']:.4f}\n"
            f"  MSE:  {parity['mse']:.6f}\n",
            flush=True,
        )

        assert parity[
            "psnr"] >= 20.0, f"PSNR vs CPU baseline failed: {parity['psnr']:.2f} dB < 20.0 dB"
        assert parity[
            "mae"] <= 0.15, f"MAE vs CPU baseline failed: {parity['mae']:.4f} > 0.15"
