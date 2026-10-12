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

# One host of the Kimi-K3 megakernel pre-shard (kimi-k3-preshard.yaml).
#
# `python -m tools.megakernel preshard` expects a Slurm allocation: it derives
# the host from SLURM_NODEID and re-launches itself with srun unless it is
# told it already is a worker. Here each indexed pod is one host: it converts
# the eight ranks of its index, and index 0 also writes the prefill
# projection cache. Nothing touches the chips; the pod holds them only to
# land on a TPU host, which is where the gcsfuse mounts are.
#
# Resumable: a rank whose index.json exists is skipped, so a run that hits
# the deadline is finished by submitting the same workload again.

set -euo pipefail

cd /workspace/vllm-torchtpu

K3_CHECKPOINT="${K3_CHECKPOINT:-/ckpt/k3}"
K3_PRESHARD_DIR="${K3_PRESHARD_DIR:-/cache/hf/hub/presharded/kimi-k3-tp32}"
K3_PRESHARD_IN_FLIGHT="${K3_PRESHARD_IN_FLIGHT:-4}"
INDEX="${JOB_COMPLETION_INDEX:-0}"

# Skip the srun re-launch; no TPU for a host-side conversion.
export MEGAKERNEL_TOOL_WORKER=1
export JAX_PLATFORMS=cpu
export SLURM_NODEID="${INDEX}"

log() { echo "[k3_preshard host ${INDEX} $(date -u +%H:%M:%S)] $*"; }

log "checkpoint ${K3_CHECKPOINT}, output ${K3_PRESHARD_DIR}"
if [[ ! -f "${K3_CHECKPOINT}/model.safetensors.index.json" ]]; then
  log "ERROR: no model.safetensors.index.json under ${K3_CHECKPOINT}" >&2
  ls -la "${K3_CHECKPOINT}" >&2 || true
  exit 2
fi
mkdir -p "${K3_PRESHARD_DIR}"

ranks=()
for ((r = 8 * INDEX; r < 8 * INDEX + 8; r++)); do
  rank_dir=$(printf '%s/rank%02d' "${K3_PRESHARD_DIR}" "${r}")
  if [[ -f "${rank_dir}/index.json" ]]; then
    log "rank ${r}: index.json present, skipping"
    continue
  fi
  # A partial rank from an earlier attempt: convert_rank rewrites every file,
  # so nothing to clean.
  ranks+=("${r}")
done

if (( ${#ranks[@]} > 0 )); then
  rank_list=$(IFS=,; echo "${ranks[*]}")
  log "converting ranks ${rank_list} (${K3_PRESHARD_IN_FLIGHT} in flight)"
  python -m tools.megakernel preshard k3 \
    --checkpoint "${K3_CHECKPOINT}" \
    --output "${K3_PRESHARD_DIR}" \
    --ranks "${rank_list}" \
    --in-flight "${K3_PRESHARD_IN_FLIGHT}"
else
  log "all eight ranks of this host already converted"
fi

if [[ "${INDEX}" == "0" ]]; then
  cache="${K3_PRESHARD_DIR}/prefill-sp-cache"
  if [[ -f "${cache}/layout.json" ]]; then
    log "prefill-sp-cache present, skipping"
  else
    # write_prefill refuses an existing directory; layout.json is written
    # last, so a directory without it is a partial attempt.
    rm -rf "${cache}"
    log "writing prefill projection cache"
    python -m tools.megakernel preshard prefill \
      --checkpoint "${K3_CHECKPOINT}" \
      --output "${K3_PRESHARD_DIR}"
  fi
  {
    echo "source checkpoint: ${K3_CHECKPOINT}"
    echo "vllm-torchtpu commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "buildkite build: ${BUILDKITE_BUILD_URL:-unknown}"
    echo "written: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${K3_PRESHARD_DIR}/PROVENANCE.txt"
fi

# The files of this host's ranks against their index: every family's bytes
# on disk equal its declared shape and dtype.
python - "${K3_PRESHARD_DIR}" "${INDEX}" <<'PY'
import json, os, sys
import numpy as np

root, index = sys.argv[1], int(sys.argv[2])
bad = 0
for rank in range(8 * index, 8 * index + 8):
    rank_dir = os.path.join(root, f"rank{rank:02d}")
    spec = json.load(open(os.path.join(rank_dir, "index.json")))
    assert spec["rank"] == rank, (rank, spec["rank"])
    dense_end = 0
    for name, fam in spec["families"].items():
        nbytes = int(np.prod(fam["shape"])) * np.dtype(
            "uint16" if fam["dtype"] == "bfloat16" else fam["dtype"]).itemsize
        if "files" in fam:
            layers = fam["shape"][0]
            per_layer = nbytes // layers
            for f, (lo, hi) in zip(fam["files"], fam["layer_ranges"]):
                size = os.path.getsize(os.path.join(rank_dir, f))
                want = per_layer * (hi - lo)
                if size != want:
                    bad += 1
                    print(f"rank {rank} {name} {f}: {size} bytes, want {want}")
        else:
            dense_end = max(dense_end, fam["offset"] + nbytes)
    size = os.path.getsize(os.path.join(rank_dir, "dense.bin"))
    if size != dense_end:
        bad += 1
        print(f"rank {rank} dense.bin: {size} bytes, want {dense_end}")
    print(f"rank {rank}: {len(spec['families'])} families checked")
if index == 0:
    cache = os.path.join(root, "prefill-sp-cache")
    layout = json.load(open(os.path.join(cache, "layout.json")))
    for name, fam in layout["families"].items():
        want = int(np.prod(fam["shape"])) * 2  # bfloat16
        size = os.path.getsize(os.path.join(cache, fam["file"]))
        if size != want:
            bad += 1
            print(f"prefill-sp-cache {name}: {size} bytes, want {want}")
    print(f"prefill-sp-cache: {len(layout['families'])} families checked")
sys.exit(1 if bad else 0)
PY

log "done"
