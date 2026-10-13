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
"""Tests for GCS and local torch-tpu wheel override scripts."""

import hashlib
import os
import pathlib
import subprocess

_REPO = pathlib.Path(__file__).resolve().parents[2]

_FAKE_GCLOUD = """#!/bin/bash
set -euo pipefail
echo "gcloud $*" >> "$FAKE_DIR/gcloud.log"
if [ "$1" = "auth" ] && [ "$2" = "print-access-token" ]; then
  echo "fake-access-token"
  exit 0
fi
if [ "$1" = "storage" ] && [ "$2" = "cp" ]; then
  src="$3"
  dst="$4"
  filename="$(basename "$src")"
  mkdir -p "$dst"
  echo "fake wheel content from $src" > "${dst%/}/$filename"
  exit 0
fi
exit 1
"""

_FAKE_DOCKER = """#!/bin/bash
set -euo pipefail
echo "docker $*" >> "$FAKE_DIR/docker.log"
if [ "$1" = "build" ]; then
  # Record any staged wheels present during docker build
  context_dir="${!#}"
  if [ -d "$context_dir/.staged_override_wheels" ]; then
    ls -1 "$context_dir/.staged_override_wheels" >> "$FAKE_DIR/staged_wheels.log" || true
  fi
fi
exit 0
"""

_FAKE_BUILDKITE_AGENT = """#!/bin/bash
set -euo pipefail
echo "buildkite-agent $*" >> "$FAKE_DIR/buildkite.log"
if [ "$1" = "meta-data" ] && [ "$2" = "get" ]; then
  key="$3"
  if [ -f "$FAKE_DIR/meta_$key" ]; then
    cat "$FAKE_DIR/meta_$key"
  fi
  exit 0
fi
if [ "$1" = "meta-data" ] && [ "$2" = "set" ]; then
  key="$3"
  val="$4"
  printf '%s' "$val" > "$FAKE_DIR/meta_$key"
  exit 0
fi
exit 0
"""


def _setup_stubs(tmp_path: pathlib.Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        "gcloud": _FAKE_GCLOUD,
        "docker": _FAKE_DOCKER,
        "buildkite-agent": _FAKE_BUILDKITE_AGENT,
    }
    for name, body in stubs.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_DIR": str(tmp_path),
    }


def test_build_override_image_with_gcs_uri(tmp_path: pathlib.Path):
    env = _setup_stubs(tmp_path)
    gcs_uri = "gs://ci-wheels-tpu/torch_tpu-0.2.0.whl"
    result = subprocess.run(
        [
            "bash",
            str(_REPO / "docker" / "build_override_image.sh"),
            "-b",
            "vllm-torchtpu-local:dev",
            "-t",
            "vllm-torchtpu-local:dev-override",
            "-w",
            gcs_uri,
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    gcloud_log = (tmp_path / "gcloud.log").read_text()
    assert f"gcloud storage cp {gcs_uri}" in gcloud_log

    staged_log = (tmp_path / "staged_wheels.log").read_text().splitlines()
    assert "torch_tpu-0.2.0.whl" in staged_log

    docker_log = (tmp_path / "docker.log").read_text()
    assert "--build-arg USE_LOCAL_WHEELS=1" in docker_log
    assert "--build-arg OVERRIDE_SOURCE=vllm-torchtpu-local:dev" in docker_log

    assert not (_REPO / ".staged_override_wheels").exists()


def test_build_override_image_with_local_wheel(tmp_path: pathlib.Path):
    env = _setup_stubs(tmp_path)
    local_whl = tmp_path / "torch_tpu-local-cp312.whl"
    local_whl.write_text("fake local wheel")

    result = subprocess.run(
        [
            "bash",
            str(_REPO / "docker" / "build_override_image.sh"),
            "-b",
            "vllm-torchtpu-local:dev",
            "-t",
            "vllm-torchtpu-local:dev-override",
            "-w",
            str(local_whl),
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    staged_log = (tmp_path / "staged_wheels.log").read_text().splitlines()
    assert "torch_tpu-local-cp312.whl" in staged_log

    docker_log = (tmp_path / "docker.log").read_text()
    assert "--build-arg USE_LOCAL_WHEELS=1" in docker_log
    assert not (_REPO / ".staged_override_wheels").exists()


def test_setup_docker_env_with_gcs_wheel_override(tmp_path: pathlib.Path):
    env = _setup_stubs(tmp_path)
    gcs_uri = "gs://ci-wheels-tpu/torch_tpu-0.2.0.whl"
    expected_hash = hashlib.sha256(gcs_uri.encode()).hexdigest()[:8]

    (tmp_path / "meta_VLLM_COMMIT_HASH").write_text("deadbeef1234")
    env["BUILDKITE_COMMIT"] = "cafebabe5678"
    env["TORCH_TPU_WHEEL_GCS_URI"] = gcs_uri

    result = subprocess.run(
        [
            "bash",
            str(_REPO / ".buildkite" / "scripts" / "setup_docker_env.sh"),
            "dev",
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    expected_tag = (
        "us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/"
        f"vllm-torchtpu-ci/vllm-torchtpu:cafebabe5678-deadbeef1234-whl-{expected_hash}"
    )

    ci_image_tag = (tmp_path / "meta_CI_IMAGE_TAG").read_text()
    assert ci_image_tag == expected_tag

    docker_log = (tmp_path / "docker.log").read_text()
    assert "docker manifest inspect" not in docker_log
    assert f"Dockerfile -t {expected_tag}" in docker_log
    assert f"Dockerfile.override -t {expected_tag}" in docker_log
    assert f"docker push {expected_tag}" in docker_log
