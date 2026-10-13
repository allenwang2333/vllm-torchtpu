#!/usr/bin/env bash
# Diagnostic, not a measurement: one cdk job that serves golden with the phased
# profiler armed for one sweep after the timed ones, and prints the job id.
#
#   trace.sh <label>
#
# The traces (decode_only/, prefill_only/ per rank) land in the job's out.tgz
# under out/profile/golden. Same pins, caps and staging as score.sh.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
rel=${here#"$repo"/}
# shellcheck source=pins.sh
source "$here/pins.sh"
label=$1
export RUN_TAG=${RUN_TAG:-$label} RECIPE_NAME=${RECIPE_NAME:-$USER-gdn-v3-397b-serve-scorer}
stage=$(mktemp -d "${TMPDIR:-/tmp}/gdn-v3-397b-stage.XXXXXX")
bash "$here/stage.sh" "$stage" "$(git -C "$repo" rev-list -1 HEAD -- "$rel")" "golden@$GOLDEN"
bash "$here/submit.sh" "$label" "$stage" ARMS="golden:control" \
  SCORER_FLAGS="$SCORER_FLAGS --profile-dir /scratch/out/profile"
