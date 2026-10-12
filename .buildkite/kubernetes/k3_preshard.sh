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
# projection cache and the DSpark speculator's shards, then verifies a
# sample rank against the checkpoints. Nothing touches the chips; the pod
# holds them only to land on a TPU host, which is where the gcsfuse mounts
# are.
#
# Resumable: a rank whose index.json exists is skipped, so a run that hits
# the deadline is finished by submitting the same workload again; the same
# resubmission adds a missing prefill cache or DSpark directory.

set -euo pipefail

cd /workspace/vllm-torchtpu

K3_CHECKPOINT="${K3_CHECKPOINT:-/ckpt/k3}"
K3_PRESHARD_DIR="${K3_PRESHARD_DIR:-/cache/hf/hub/presharded/kimi-k3-tp32}"
K3_PRESHARD_IN_FLIGHT="${K3_PRESHARD_IN_FLIGHT:-4}"
# The DSpark head's snapshot; the loader names the shard directory after the
# first 8 characters of this directory's name, so it must be the revision.
K3_DSPARK_DIR="${K3_DSPARK_DIR:-/cache/hf/hub/models--RedHatAI--Kimi-K3-speculator.dspark/snapshots/38a88101e0d46bb22134b9da340f381b954d40d4}"
K3_DSPARK_REV=$(basename "${K3_DSPARK_DIR}")
export K3_DSPARK_REV
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
  if [[ ! -f "${K3_DSPARK_DIR}/model.safetensors" ]]; then
    log "ERROR: no model.safetensors under ${K3_DSPARK_DIR}" >&2
    ls -la "${K3_DSPARK_DIR}" >&2 || true
    exit 2
  fi
  dspark_out="${K3_PRESHARD_DIR}/dspark-${K3_DSPARK_REV:0:8}"
  if [[ -f "${dspark_out}/index.json" ]]; then
    log "${dspark_out} present, skipping"
  else
    # save_kernel_shards rewrites every rank file and writes index.json
    # last, so a directory without it is a partial attempt.
    log "writing DSpark shards from ${K3_DSPARK_DIR}"
    python -m tools.megakernel preshard dspark \
      --draft "${K3_DSPARK_DIR}" \
      --output "${K3_PRESHARD_DIR}"
  fi
  # Rebuilds rank 5 (layers 1, 46, 92 and the DSpark shards) from the two
  # checkpoints and compares it byte for byte with what is on disk.
  log "verifying rank 5 against the checkpoints"
  python -m tools.megakernel preshard verify \
    --against "${K3_PRESHARD_DIR}" \
    --checkpoint "${K3_CHECKPOINT}" \
    --draft "${K3_DSPARK_DIR}"
  {
    echo "source checkpoint: ${K3_CHECKPOINT} (gs://tpu-commons-ci/moonshootai/kimi/k3)"
    echo "dspark checkpoint: ${K3_DSPARK_DIR} (RedHatAI/Kimi-K3-speculator.dspark @ ${K3_DSPARK_REV})"
    echo "tool: python -m tools.megakernel preshard k3 / prefill / dspark / verify"
    echo "vllm-torchtpu commit: ${K3_PRESHARD_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
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
    dspark = os.path.join(root, "dspark-" + os.environ["K3_DSPARK_REV"][:8])
    index_json = json.load(open(os.path.join(dspark, "index.json")))
    want = int(index_json["bytes_per_rank"])
    for rank in range(int(index_json["ranks"])):
        size = os.path.getsize(os.path.join(dspark, f"rank{rank:02d}.bin"))
        if size != want:
            bad += 1
            print(f"{dspark} rank{rank:02d}.bin: {size} bytes, want {want}")
    print(f"{os.path.basename(dspark)}: {index_json['ranks']} ranks x {want} bytes, "
          f"{len(index_json['arrays'])} arrays checked")
sys.exit(1 if bad else 0)
PY

log "done"
