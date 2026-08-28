#!/bin/bash
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

set -euo pipefail

# Every docker removal below is best-effort and tolerates failure. An agent
# runs one job at a time, but run_in_docker.sh starts the test container with
# `docker run --rm`, and the daemon removes it asynchronously -- after
# `docker run` has already returned and the job has finished. So the next job
# on that agent can reach this cleanup while the previous container is still
# being torn down: `docker ps -a` still lists it, and `docker rm -f` then
# fails with "removal ... is already in progress". Under `set -e` that aborts
# the caller before any test runs, turning housekeeping into a red build.
# A resource that is already being removed is the outcome we wanted.
TARGET_IMAGES=("vllm-torchtpu" "torchtpu-vllm-local")

# Report free space either side of the cleanup. A job that dies with "No space
# left on device" reports a path inside the container, not the host filesystem
# that actually filled, so without this there is nothing in the log to tell the
# candidate disks apart. The docker root is wherever the daemon is configured to
# live, not necessarily /var/lib/docker, and /mnt/disks/persist only exists on
# agents with a data disk attached.
report_disk() {
  local when="$1" docker_root
  echo "=== Disk usage (${when} cleanup) ==="
  docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)
  df -h "${docker_root:-/var/lib/docker}" 2>/dev/null || true
  if [[ -d /mnt/disks/persist ]]; then
    df -h /mnt/disks/persist 2>/dev/null || true
    # The HF cache is append-only across builds, so its size is the number
    # worth watching on the agents that have it. Bounded because it can hold
    # enough files for the walk itself to be slow.
    timeout 60 du -sh /mnt/disks/persist/models 2>/dev/null || true
  fi
}

report_disk before

echo "=== Starting Docker resource cleanup ==="
echo "Target images: ${TARGET_IMAGES[*]}"

for IMG in "${TARGET_IMAGES[@]}"; do
  echo "----------------------------------------"
  echo "Cleaning up images matching: ${IMG}"

  # Get image IDs matching either exact name or registry repository name
  # e.g., "vllm-torchtpu" or ".../.../vllm-torchtpu"
  OLD_IMAGES=$(docker images --format '{{.Repository}} {{.ID}}' | awk -v img="${IMG}" '$1 == img || $1 ~ "/"img"$" {print $2}' | sort -u)

  if [[ -n "$OLD_IMAGES" ]]; then
    echo "Found matching images. Checking for dependent containers..."

    TOTAL_CONTAINERS=""
    for img_id in $OLD_IMAGES; do
      TOTAL_CONTAINERS="$TOTAL_CONTAINERS $(docker ps -a -q --filter "ancestor=$img_id")"
    done

    CLEANED_CONTAINERS=$(echo "$TOTAL_CONTAINERS" | tr ' ' '\n' | grep -v '^$' | sort -u || true)
    if [[ -n "$CLEANED_CONTAINERS" ]]; then
      echo "Removing leftover containers using ${IMG} image(s)..."
      echo "$CLEANED_CONTAINERS" | xargs -r docker rm -f || true
    fi

    echo "Removing old ${IMG} image(s)..."
    echo "$OLD_IMAGES" | xargs -r docker rmi -f || true
  else
    echo "No images matching ${IMG} found to clean up."
  fi
done

echo "----------------------------------------"
echo "Pruning old Docker build cache..."
# Bounded: a prune blocks while an earlier one is still in flight, so a job
# killed mid-prune leaves the next one waiting on the daemon indefinitely.
# Hanging the job outright is a worse failure than the disk pressure this is
# here to relieve, and a partial prune still reclaims what it got through.
timeout 300 docker builder prune -f || true

report_disk after

echo "=== Cleanup complete ==="
