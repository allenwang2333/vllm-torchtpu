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

# The one list of environment variables a Buildkite step may set and expect to
# reach the serving container, whichever driver starts it.
#
# It used to be written out twice -- once in run_in_docker.sh (single host) and
# once in run_multihost.sh (ray/mp) -- and the two drifted: a knob added to the
# multihost list was silently absent on v7x-8, so the same pipeline env gave a
# different server depending on the queue. USE_PHASED_PROFILER is how that was
# found; it was in the multihost list and not the single-host one, and a
# single-host profiling run quietly produced an unphased 1.3 GB capture instead
# of the decode window it asked for.
#
# Add names here and nowhere else. Unset or empty names are skipped, so an
# entry costs nothing when it is not in use.
#
# 2026-08-31: the *serving-config* overrides that `scripts/vllm/benchmarking/
# configs/*.sh` read with `${NAME:-default}` were absent from this list, so a
# pipeline step setting ASYNC_SCHEDULING / KV_CACHE_DTYPE / MAX_MODEL_LEN /
# NUM_PROMPTS / K3_* got the config's default and no warning. A step that asks
# for async and silently serves synchronously produces a plausible number,
# which is the worst possible failure. They are here now.
#
# 2026-08-31: a SECOND kind of drift, one level deeper than the one above.
# run_multihost.sh hard-codes `-e VLLM_DISABLE_COMPILE_CACHE=1`
# (CONTAINER_ENV_COMMON); run_in_docker.sh does not set it at all. Neither
# reads it from the pipeline, so this list could not have caught it. The
# consequence is that **a v7x-8 rehearsal runs a different compilation regime
# from the pod job it is rehearsing**: the single-host arm serves from the
# cached AOT artifact and the pod arm compiles cold. It is how BK #322 arm B
# failed with ShapeSpecializationError on v7x-8 for a change that would have
# booted fine on the pod. Adding the name here lets a pipeline make the two
# match; the multihost default is deliberately left as it is, because every
# pod number on record was measured with the cache off.
#
# Audited 2026-08-30: the container launchers are run_in_docker.sh,
# run_multihost.sh (head) and scripts/multihost/run_cluster{,_mp}.sh (workers,
# via PASSTHROUGH_ENV_STR) -- all covered here. run_gpc_multihost.sh is a
# fourth `docker run` and is deliberately *not* wired up: it starts a fixed
# GPC correctness topology whose script sets its own knobs inside the
# container. A pipeline env var will not reach that suite; that is its
# contract, not an oversight, but do not assume otherwise.
# shellcheck disable=SC2034  # consumed by the sourcing driver
PASSTHROUGH_ENV_NAMES=(
  ASYNC_SCHEDULING
  BENCHMARK_WARMUP_RUNS
  K3_COMPILE_SIZES
  K3_CONCURRENCY
  KV_CACHE_DTYPE
  MAX_MODEL_LEN
  NUM_PROMPTS
  ONEHOT_MOE_PERMUTE_THRESHOLD
  TPU_ACCELERATOR_TYPE
  TPU_GMM_DECODE_BUCKET_ROWS
  TPU_MOE_EXPERT_LOAD_BUCKET
  TPU_MOE_EXPERT_LOAD_DIR
  TPU_MOE_EXPERT_LOAD_EVERY
  TPU_MOE_EXPERT_LOAD_RANKS
  TPU_MOE_EXPERT_LOAD_SLOTS
  TPU_MOE_HIERARCHICAL_EP
  TPU_MOE_SKIP_PADDED_TOKENS
  TPU_MULTIHOST_BACKEND
  TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL
  USE_MOE_SPARSE_CORE
  USE_PHASED_PROFILER
  VLLM_DISABLE_COMPILE_CACHE
  VLLM_ENGINE_READY_TIMEOUT_S
  VLLM_XLA_CHECK_RECOMPILATION
)

# Populates PASSTHROUGH_ENV_ARGS with `-e NAME=value` for every name that is set.
build_passthrough_env_args() {
  PASSTHROUGH_ENV_ARGS=()
  local name
  for name in "${PASSTHROUGH_ENV_NAMES[@]}"; do
    if [ -n "${!name:-}" ]; then
      PASSTHROUGH_ENV_ARGS+=(-e "${name}=${!name}")
    fi
  done
}
