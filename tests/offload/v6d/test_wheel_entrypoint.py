# SPDX-License-Identifier: Apache-2.0
"""Built-wheel contract for v6d TPU DMA provider discovery."""

import subprocess
import sys
import textwrap
from pathlib import Path


def test_built_wheel_exposes_lazy_v6d_tpu_dma_entrypoint(tmp_path):
    root = Path(__file__).resolve().parents[3]
    wheel_dir = tmp_path / "wheel"
    install_dir = tmp_path / "install"
    wheel_dir.mkdir()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel_path, ) = wheel_dir.glob("*.whl")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_dir),
            str(wheel_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    probe = textwrap.dedent("""
        import importlib.metadata
        import sys

        install_dir = sys.argv[1]
        sys.path.insert(0, install_dir)
        (distribution,) = importlib.metadata.distributions(path=[install_dir])
        entry_points = [
            entry_point
            for entry_point in distribution.entry_points
            if entry_point.group == "v6d.offload_providers"
        ]
        assert [(entry_point.name, entry_point.value)
                for entry_point in entry_points] == [
            ("tpu.tpu_dma",
             "vllm_torchtpu.offload.v6d.provider:get_provider")
        ]
        assert "tpu_sync.api.torch.kv_cache_offloader" not in sys.modules
        provider = entry_points[0].load()()
        assert provider.api_version == 1
        assert "tpu_sync.api.torch.kv_cache_offloader" not in sys.modules
    """)
    subprocess.run(
        [sys.executable, "-c", probe,
         str(install_dir)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
