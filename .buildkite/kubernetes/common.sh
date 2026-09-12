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

# What a kube step runs: which image, and what of the step's environment
# crosses into the workload pod.
#
# Sourced, not run: it exports WORKLOAD_IMAGE and leaves the launcher's
# environment flags in `env_args`. Both wrappers need the identical set - a
# step that moves between a single-host shape and a slice must not quietly
# change image or lose its Hugging Face token - and the reasoning below is long
# enough that keeping two copies of it in step would guarantee they diverge.
#
#   source .buildkite/kubernetes/common.sh
#   /opt/launcher/launch ... "${env_args[@]}" ...

# The same tag setup_docker_env.sh pushed, built the same way rather than read
# back from build metadata, because that is how every other script in this repo
# names the image and two ways of naming it is one too many. The launcher
# resolves it to a digest at submission, so a run is pinned to the bytes this
# build pushed even if a later build moves the tag.
IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
WORKLOAD_IMAGE="${WORKLOAD_IMAGE:-${IMAGE_REPO}:${BUILDKITE_COMMIT:-latest}}"
export WORKLOAD_IMAGE

# What run_in_docker.sh passes as a fixed -e on every container. Exported here
# rather than named in the pipeline because none of it varies by step, and
# named in FORWARD below because exporting alone would only set them for the
# launcher: the launcher forwards a variable when a --env names it.
export FORCE_COLOR="1"
export TQDM_MININTERVAL="30"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU="0.0.0"
export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME="oauth2accesstoken"
export UV_NO_CACHE="1"

# HF_HOME is deliberately absent: the launcher's Job already points it at the
# fleet's model bucket, where run_in_docker.sh has to name a path on the VM's
# data disk. Setting it here would override a mount these scripts know nothing
# about.

# Names a step sets that no convention would find. A name the step has not set
# is skipped rather than injected empty, which is what run_in_docker.sh's
# `${VAR:+-e VAR=...}` expansions buy and what several of these depend on.
FORWARD=(
  # The fixed set exported above.
  FORCE_COLOR TQDM_MININTERVAL SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU
  UV_INDEX_TORCH_TPU_REGISTRY_USERNAME UV_NO_CACHE
  # Unset on a kube agent, and naming an unset variable is how the launcher is
  # asked for it: HF_TOKEN is in its env_secrets registry, so it reads it from
  # Secret Manager and puts it in the pod. On bare metal run_in_docker.sh
  # writes it into /etc/environment and passes it by value.
  HF_TOKEN
  # Benchmark and eval selection.
  BENCHMARK_WARMUP_RUNS RUN_CODE_EVAL RUN_TYPE MODEL_PATH MODEL_URI
  SERVED_MODEL_NAME RUN_ROOT VLLM_ENGINE_READY_TIMEOUT_S
  TPU_ACCELERATOR_TYPE TPU_MULTIHOST_BACKEND TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL
  USE_MOE_SPARSE_CORE P4D2_BIND_HOST PROXY_PORT
  # Set by bootstrap.sh on a nightly build; several benchmark configs branch on
  # it, and a step's `if:` cannot pass it into the pod for them.
  NIGHTLY CREATED_BY
  # bktec enumerates examples with a plain `pytest --collect-only`, which does
  # not see the -m filter in the plugin's test-cmd, so the steps that shard one
  # file set this to keep the split over what actually runs.
  PYTEST_ADDOPTS
  # The suite token bktec uploads results with. Unset above, so this asks the
  # launcher for it: it is in the env_secrets registry, the same way HF_TOKEN
  # is.
  BUILDKITE_ANALYTICS_TOKEN
  # What a slice's pods are asked to run, read by multihost_entry.sh. Unset for
  # a single-host step, and so skipped.
  MULTIHOST_ARGS
)
# BQ_PROJECT_ID, BQ_TABLE and GCP_INSTANCE_NAME are not forwarded. This lane
# runs beside the bare-metal one against the same table, and two rows per night
# for one measurement is worse than none: results leave the pod as Buildkite
# artifacts instead.

# Every BUILDKITE_* the agent set, enumerated rather than listed. There are
# around a hundred, most of them build metadata a test may read, and a
# hand-maintained list of them drifts.
#
# What the sweep carries that matters:
#
#   BUILDKITE_AGENT_ACCESS_TOKEN and _ENDPOINT let the workload call
#   buildkite-agent itself. It is the only thing that can see its own output
#   files: unlike bare metal, where run_in_docker.sh bind-mounts
#   /perf_eval_results into the agent's checkout for `artifact_paths` to glob,
#   a pod shares no filesystem with the agent. Not a widening of trust - the
#   step's own commands already run with this token, from the same repo as the
#   test.
#
#   BUILDKITE_TEST_ENGINE_* is the suite slug and runner configuration the
#   `tests` plugin exports for bktec. bktec runs inside the workload, so
#   without them a suite runs and reports nothing. The two credentials the
#   plugin exports beside them are not carried; see below.
#
#   BUILDKITE_PARALLEL_JOB and _COUNT are how bktec knows which shard it is.
#   They are unset on a step without `parallelism:`, and unset names are
#   skipped.
#
# Five are held back:
#
#   BUILDKITE_COMMAND is this step's own script. Forwarding it into the
#   container invites something to re-execute the launcher from inside the
#   workload it launched.
#
#   BUILDKITE_PLUGINS is a JSON blob describing agent-side plugins, which mean
#   nothing in the pod and can be large enough to bloat the pod spec.
#
#   BUILDKITE_AGENT_JOB_API_SOCKET, _TOKEN and BUILDKITE_OIDC_TOKEN_PATH are
#   all paths on the agent host. run_in_docker.sh can forward them because it
#   bind-mounts what they point at; a workload pod is on another cluster
#   entirely, so none of them can ever resolve. Forwarding is worse than
#   omitting: absent, the CLI says "socket empty or undefined", which names the
#   problem; present and dangling, it fails as a connection or open error.
#   Nothing depends on them: `buildkite-agent oidc request-token` is an API
#   call made with BUILDKITE_AGENT_ACCESS_TOKEN, which the sweep does carry.
FORWARD_DENY="BUILDKITE_COMMAND BUILDKITE_PLUGINS BUILDKITE_AGENT_JOB_API_SOCKET BUILDKITE_AGENT_JOB_API_TOKEN BUILDKITE_OIDC_TOKEN_PATH"

# The `tests` plugin's pre-command mints one OIDC token on the agent and
# exports it as both of these. On bare metal that is fine, because bktec starts
# seconds later. Here the clock starts before Kueue admits the workload, before
# a node scales up, before a CI image is pulled cold and before the model
# compiles - and the token's default lifetime is two hours, which a step whose
# own timeout is four can spend entirely on waiting. bktec then fails at the
# end of a run that already did the work, which is the most expensive place to
# discover an expired credential.
#
# Unset rather than deny-listed: the launcher prefers its own environment to
# the registry, so leaving either set here would forward the stale token and
# never reach Secret Manager.
#
# Uploads take the suite token from the fleet registry, named in FORWARD above.
# It does not expire, and it authenticates against the suite directly, so a
# result upload no longer depends on the suite's OIDC policy listing this
# pipeline. Test splitting has no such token - it wants an org API token, which
# the fleet does not hold - so it is left unset on purpose: bktec then mints
# its own OIDC token from inside the pod, at the moment it needs one.
unset BUILDKITE_ANALYTICS_TOKEN BUILDKITE_TEST_ENGINE_API_ACCESS_TOKEN

# What lets bktec mint that token in the pod. `buildkite-agent oidc
# request-token` refuses by default unless it can register the token it is
# about to print with the agent's log redactor, and the redactor lives behind
# the Job API socket - a path on the agent host, which is why the sweep denies
# it. So the guard fires on exactly the one caller it cannot help.
#
# Safe here because the token is never printed. bktec captures stdout into its
# own configuration, and the pod's log reaches Buildkite through the launcher,
# which forwards lines rather than evaluating them. The redactor would be
# guarding against a leak that has no path to occur.
export BUILDKITE_AGENT_OIDC_REQUEST_TOKEN_SKIP_TOKEN_REDACTION=true

env_args=()
_forward_seen=" "
for _name in $(compgen -v | grep '^BUILDKITE_' | sort); do
  case " $FORWARD_DENY " in *" $_name "*) continue ;; esac
  env_args+=(--env "$_name")
  _forward_seen="$_forward_seen$_name "
done
for _name in "${FORWARD[@]}"; do
  case "$_forward_seen" in *" $_name "*) continue ;; esac
  env_args+=(--env "$_name")
  _forward_seen="$_forward_seen$_name "
done
unset _name _forward_seen
