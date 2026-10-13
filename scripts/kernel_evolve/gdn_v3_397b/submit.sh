#!/usr/bin/env bash
# Submit one serve-scorer job and print its id on stdout.
#
#   submit.sh <label> <stage-dir> ARMS="golden:control cand aa" SCORER_FLAGS="..." [KEY=VALUE ...]
#
# Required environment: IMAGE (pinned by digest), EXPECT_JAX and EXPECT_LIBTPU
# (the versions the pod refuses to run without), WEIGHTS_BUCKET, WEIGHTS_PREFIX,
# RUN_TAG (counts this campaign's or run's jobs). BUDGET (default 3) caps the
# jobs carrying RUN_TAG that may be live at once; the submitting user's own
# Running+Pending jobs are capped at 5 as well, and a job is only submitted while
# a Ready single-host tpu7x node is free; it waits up to SLOT_WAIT_S for all
# three. DEADLINE_S (default 21600) force-kills the job. The recipe is registered in
# the same step as the submit, because other sessions rewrite recipes.yml.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
label=$1; stage=$2; shift 2
: "${IMAGE:?}" "${EXPECT_JAX:?}" "${EXPECT_LIBTPU:?}" "${WEIGHTS_BUCKET:?}" "${WEIGHTS_PREFIX:?}" "${RUN_TAG:?}"
budget=${BUDGET:-3}
cdk_dir=${CDK_DIR:-$HOME/cloud-devkit}
recipe=${RECIPE_NAME:-$USER-gdn-v3-397b-serve-scorer}

cdkj() {
  local codes
  codes=$(cdk agent-letter 2>/dev/null | grep -oE 'ack [0-9]+/10: [0-9A-F]+' \
    | sed 's/.*: //' | paste -sd-)
  [ "$(echo "$codes" | tr '-' '\n' | grep -c .)" = 10 ] || { echo "cdk ack codes missing" >&2; return 1; }
  cdk --agent-code="$codes" "$@"
}

# Prints "<live jobs tagged RUN_TAG> <my Running+Pending jobs> <busy> <ready>",
# where busy/ready count the Ready non-spot single-host tpu7x nodes and those holding a job.
caps() {
  local d rc
  d=$(mktemp -d)
  # Files, not environment variables: the job list outgrows one argument's size limit.
  cdkj job list -a -A -n 300 -o json > "$d/rows" && cdkj usage -o json > "$d/nodes" \
    && RUN_TAG=$RUN_TAG python3 - "$d/rows" "$d/nodes" <<'PY'
import json, os, sys
rows, nodes = (json.load(open(p)) for p in sys.argv[1:3])
assert isinstance(rows, list) and isinstance(nodes, list), "cdk did not return JSON arrays"
full = os.environ["USER"]; me = {full, full.split("_")[0], full.split("_")[0] + "_google_com"}
live = [r for r in rows if r.get("job_status") in ("Running", "Pending", "InQueue")]
run = [r for r in live if os.environ["RUN_TAG"] in (r.get("tags") or [])]
mine = [r for r in live if r.get("user") in me and r.get("job_status") in ("Running", "Pending")]
# The recipe keeps off spot nodes, so only the other single-host pools are slots.
pool = lambda n: n.get("node_pool", "")
ready = [n for n in nodes if "singlehost" in pool(n) and "spot" not in pool(n)
         and n.get("status") == "Ready"]
assert ready, "cdk usage lists no Ready single-host node"
busy = sum(1 for n in ready if any(p.get("job_id") for p in n.get("pods") or []))
print(len(run), len(mine), busy, len(ready))
PY
  rc=$?
  rm -rf "$d"
  return $rc
}
# Wait for a slot under all three caps (SLOT_WAIT_S, default 6 h), never past one.
slot_deadline=$(( $(date +%s) + ${SLOT_WAIT_S:-21600} ))
while :; do
  read -r run mine busy ready < <(caps) || { echo "cannot count the caps" >&2; exit 1; }
  if [ "$run" -lt "$budget" ] && [ "$mine" -lt 5 ] && [ "$busy" -lt "$ready" ]; then break; fi
  msg="jobs tagged $RUN_TAG $run/$budget, $USER's $mine/5, single-host nodes busy $busy/$ready"
  [ "$(date +%s)" -lt "$slot_deadline" ] || { echo "NO SLOT after SLOT_WAIT_S: $msg" >&2; exit 2; }
  echo "waiting for a slot: $msg" >&2
  sleep "${SLOT_POLL_S:-120}"
done
echo "slot: jobs tagged $RUN_TAG $run/$budget, $USER's $mine/5, nodes busy $busy/$ready" >&2

rel=recipes/experimental/$USER/$recipe/values.yml
install -D -m 644 "$here/cdk-recipe.yaml" "$cdk_dir/$rel"
python3 - "$cdk_dir/recipes.yml" "$recipe" "$USER" "$rel" <<'PY'
import re, sys
path, name, owner, rel = sys.argv[1:5]
lines = open(path).read().splitlines(keepends=True)
starts = [i for i, l in enumerate(lines) if l.startswith("- ")]
head = "".join(lines[: starts[0]]) if starts else "".join(lines)
keep = []
for n, i in enumerate(starts):
    j = starts[n + 1] if n + 1 < len(starts) else len(lines)
    block = "".join(lines[i:j])
    if re.match(rf"- name:\s*{re.escape(name)}\s*$", lines[i].rstrip("\n")):
        continue
    keep.append(block if block.endswith("\n") else block + "\n")
entry = f'- name: {name}\n  owner: "{owner}"\n  k8s_file: {rel}\n  require_gcs_mount: true\n'
open(path, "w").write(head + "".join(keep) + entry)
PY

out=$(cdkj job create "$recipe" -t "$RUN_TAG,agent-jobs,${AI_AGENT:-agent},$label" --ttl 3600 \
  --active-deadline-seconds "${DEADLINE_S:-21600}" --map-dir "$stage:tpu-job:/mapped/stage" \
  IMAGE="$IMAGE" EXPECT_JAX="$EXPECT_JAX" EXPECT_LIBTPU="$EXPECT_LIBTPU" WEIGHTS_BUCKET="$WEIGHTS_BUCKET" WEIGHTS_PREFIX="$WEIGHTS_PREFIX" "$@" 2>&1) \
  || { echo "$out" | sed -E 's/--agent-code=[^ ]+/--agent-code=<redacted>/' >&2; exit 1; }
echo "$out" | sed -E 's/--agent-code=[^ ]+/--agent-code=<redacted>/' >&2
jid=$(echo "$out" | grep -oE 'j-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}' | head -1)
[ -n "$jid" ] || { echo "no job id in the cdk output" >&2; exit 1; }
echo "$jid"
