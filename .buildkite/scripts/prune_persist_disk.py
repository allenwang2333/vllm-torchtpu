#!/usr/bin/env python3
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
"""Evicts least recently used (LRU) model caches from persistent disk if usage > 65%."""

import os
from pathlib import Path
import shutil

PERSIST_DIR = Path("/mnt/disks/persist")
THRESHOLD_PERCENT = 65


def get_disk_usage_percent(path: Path) -> int:
  try:
    stat = os.statvfs(path)
    total_blocks = stat.f_blocks
    free_blocks = stat.f_bfree
    if total_blocks == 0:
      return 0
    return int(((total_blocks - free_blocks) / total_blocks) * 100)
  except OSError as e:
    print(f"Error checking disk usage for {path}: {e}")
    return 0


def prune_persist_disk() -> None:
  if not PERSIST_DIR.exists():
    return

  # 1. Clean legacy compilation caches
  for legacy in [
      PERSIST_DIR / "tpu_jax_cache",
      PERSIST_DIR / "torchtpu_cache",
  ]:
    if legacy.exists():
      print(f"Removing legacy cache: {legacy}")
      shutil.rmtree(legacy, ignore_errors=True)

  usage = get_disk_usage_percent(PERSIST_DIR)
  print(
      f"Current persistent disk usage: {usage}% (Threshold:"
      f" {THRESHOLD_PERCENT}%)"
  )
  if usage <= THRESHOLD_PERCENT:
    return

  print(
      f"Usage exceeds {THRESHOLD_PERCENT}%. Evicting least recently used"
      " models..."
  )

  # 2. Gather all HuggingFace model directories
  model_dirs = []
  for base in [PERSIST_DIR / "models" / "hub", PERSIST_DIR / "models"]:
    if base.exists():
      for entry in base.iterdir():
        if entry.is_dir() and entry.name.startswith("models--"):
          model_dirs.append(entry)

  # Sort by access time (st_atime), oldest first
  model_dirs.sort(key=lambda d: d.stat().st_atime)

  # 3. Evict iteratively until usage drops to or below threshold
  for model_dir in model_dirs:
    if get_disk_usage_percent(PERSIST_DIR) <= THRESHOLD_PERCENT:
      break
    print(f"Evicting LRU model directory: {model_dir}")
    shutil.rmtree(model_dir, ignore_errors=True)

  final_usage = get_disk_usage_percent(PERSIST_DIR)
  print(f"Persistent disk usage after prune: {final_usage}%")


if __name__ == "__main__":
  prune_persist_disk()
