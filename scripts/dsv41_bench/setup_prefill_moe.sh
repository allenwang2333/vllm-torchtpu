#!/usr/bin/env bash
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

# Run in the base JAX image with vLLM sources and the verified TorchTPU wheel mounted.
set -euo pipefail
ulimit -c 0
cd "$(dirname "$0")/../.."
export TMPDIR=/tmp/tpu_logs/tmp PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache UV_CACHE_DIR=/tmp/tpu_logs/uv-cache
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 VLLM_TARGET_DEVICE=empty TORCH_DEVICE_BACKEND_AUTOLOAD=0
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0
export PYTHONPATH="$PWD/src:$PWD"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$ARTIFACT_LOCAL_DIR/setup"
exec > >(tee "$ARTIFACT_LOCAL_DIR/setup/setup.log") 2>&1
test "$(git -C /workspace/vllm rev-parse HEAD)" = e37e51dd246cf421c554e7d6f53e178e3fd29085
python3 -m venv /tmp/tpu_logs/venv
# shellcheck disable=SC1091
source /tmp/tpu_logs/venv/bin/activate
python -m pip install -q --upgrade pip setuptools wheel setuptools_scm packaging ninja cmake pybind11
python -m pip install -q torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
python - <<'PY'
import hashlib
import subprocess
import sys
import tomllib
from pathlib import Path

wheel = Path('/wheels/torch-tpu/torch_tpu-0.2.0rc1-cp312-cp312-manylinux_2_31_x86_64.whl')
expected = '915bf99d453e33b46933ad9baa37af24b930dcafc5e015a75f9f2b7943087d38'
actual = hashlib.sha256(wheel.read_bytes()).hexdigest()
if actual != expected:
    raise RuntimeError(f'TorchTPU wheel hash mismatch: {actual}')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', str(wheel)], check=True)
requirements = set()
for root in (Path('/workspace/vllm'), Path.cwd()):
    requirements.update(tomllib.loads((root / 'pyproject.toml').read_text())['build-system']['requires'])
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *sorted(requirements)], check=True)
PY
VLLM_VERSION_OVERRIDE=0.1.dev1+ge37e51dd2.empty python -m pip install -q --no-build-isolation -e /workspace/vllm
python - <<'PY'
import tomllib
from pathlib import Path
from scripts.dsv41_bench.bench_prefill_moe import RUNTIME_PINS

requirements = tomllib.loads(Path('pyproject.toml').read_text())['project']['dependencies']
requirements = [r for r in requirements if not r.startswith(('vllm @', 'torch-tpu=='))]
requirements += [f'{name}=={version}' for name, version in RUNTIME_PINS.items() if name != 'torch-tpu']
requirements += ['runai-model-streamer[gcs]==0.17.0', 'runai-model-streamer-gcs==0.17.0',
                 'frozendict==2.4.7', 'immutabledict==4.3.1', 'tensorboard==2.21.0']
Path('/tmp/tpu_logs/requirements.txt').write_text('\n'.join(requirements) + '\n')
PY
python -m pip install -q -r /tmp/tpu_logs/requirements.txt
python -m pip install -q --no-deps --no-build-isolation -e .
python -m pip freeze > "$ARTIFACT_LOCAL_DIR/setup/pip-freeze.txt"
python -m pip check | tee "$ARTIFACT_LOCAL_DIR/setup/pip-check.txt"
unset TORCH_DEVICE_BACKEND_AUTOLOAD
export VLLM_TARGET_DEVICE=tpu
exec bash scripts/dsv41_bench/run_prefill_moe.sh
