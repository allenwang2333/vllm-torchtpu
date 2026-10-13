#!/usr/bin/env bash
# The frozen scorer command of the gdn/v3 Qwen3.5-397B campaign: stage the arms,
# submit one cdk job, wait for it, and print the scorer's JSON on stdout.
#
#   score.sh <label> cand@<rev>[:<patch>] [aa] [<name>@<rev>[:<patch>] ...]
#
# Golden (the control, run first) is always the frozen commit below; "aa" adds a
# second copy of golden as the A/A arm, run second so that a candidate which
# hangs the TPU cannot take the A/A measurement down with it. Everything but the JSON goes to stderr;
# with no result the script prints nothing and exits non-zero. RUN_TAG (default
# the label) tags the job; submit.sh caps live jobs per RUN_TAG at BUDGET (3).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
rel=${here#"$repo"/}

# shellcheck source=pins.sh
source "$here/pins.sh"

label=$1; shift
export RUN_TAG=${RUN_TAG:-$label} RECIPE_NAME=${RECIPE_NAME:-$USER-gdn-v3-397b-serve-scorer}
scorer_rev=$(git -C "$repo" rev-list -1 HEAD -- "$rel")
specs=("golden@$GOLDEN"); arms="golden:control"; cand=""
for a in "$@"; do
  [ "$a" = aa ] && { specs+=("aa@$GOLDEN"); arms+=" aa"; }
done
for a in "$@"; do
  [ "$a" = aa ] && continue
  specs+=("$a"); arms+=" ${a%%@*}"; [ -n "$cand" ] || cand=${a%%@*}
done
stage=$(mktemp -d "${TMPDIR:-/tmp}/gdn-v3-397b-stage.XXXXXX")
bash "$here/stage.sh" "$stage" "$scorer_rev" "${specs[@]}"
echo "scorer rev $scorer_rev; arms: $arms" >&2
jid=$(bash "$here/submit.sh" "$label" "$stage" ARMS="$arms" CANDIDATE="$cand" \
  SCORER_FLAGS="$SCORER_FLAGS")
echo "submitted $jid" >&2

cdkj() {
  local codes
  codes=$(cdk agent-letter 2>/dev/null | grep -oE 'ack [0-9]+/10: [0-9A-F]+' \
    | sed 's/.*: //' | paste -sd-)
  cdk --agent-code="$codes" "$@"
}
# Queueing for a non-spot node can take hours on a busy cluster; a 4-arm job runs ~3.5 h.
deadline=$(( $(date +%s) + ${WAIT_S:-43200} ))
while :; do
  st=$(cdkj job desc "$jid" 2>/dev/null | sed -n 's/^job_status: *//p' | head -1)
  case "$st" in Succeeded|Failed|Deleted) break ;; esac
  [ "$(date +%s)" -lt "$deadline" ] || { echo "$jid still ${st:-unknown} after WAIT_S; not deleted" >&2; exit 1; }
  sleep 180
done
echo "$jid ended: $st" >&2
# The newest pod's score.json: a re-created pod re-runs every arm.
obj=$(gsutil ls -l "gs://cloud-devkit/jobs/$jid/outputs/**/score.json" 2>/dev/null \
  | grep score.json | sort -k2 | tail -1 | awk '{print $3}')
[ -n "$obj" ] || { echo "$jid left no score.json" >&2; exit 1; }
out=$(gsutil cat "$obj")
[ -n "$out" ] || { echo "$jid: empty score.json" >&2; exit 1; }
printf '%s\n' "$out"
