#!/usr/bin/env bash
# Build a stage directory for run_job.sh from revisions of this repository.
#
#   stage.sh <out-dir> <scorer-rev> <name>@<rev>[:<patch>] ...
#
# Each arm is `git archive <rev>` with <patch> applied, and its expected tree
# is the tree git itself computes for that content. The scorer files come from
# <scorer-rev>, so the job runs the committed scorer, not a working copy.
set -euo pipefail
out=$1; scorer_rev=$2; shift 2
here=scripts/kernel_evolve/gdn_v3_397b
repo=$(git rev-parse --show-toplevel)
rm -rf "$out"; mkdir -p "$out/arms" "$out/scorer"
git -C "$repo" archive "$scorer_rev" "$here" \
  | tar -x -C "$out/scorer" --strip-components=3
cp "$out/scorer/run_job.sh" "$out/run_job.sh"
: > "$out/arms.manifest"
for spec in "$@"; do
  name=${spec%%@*}; rest=${spec#*@}; rev=${rest%%:*}; patch=""
  [ "$rest" != "$rev" ] && patch=${rest#*:}
  mkdir -p "$out/arms/$name"
  git -C "$repo" archive "$rev" | tar -x -C "$out/arms/$name"
  idx=$(mktemp); rm -f "$idx"
  GIT_INDEX_FILE=$idx git -C "$repo" read-tree "$rev"
  if [ -n "$patch" ]; then
    git -C "$out/arms/$name" apply "$(realpath "$patch")"
    GIT_INDEX_FILE=$idx git -C "$repo" apply --cached "$(realpath "$patch")"
  fi
  tree=$(GIT_INDEX_FILE=$idx git -C "$repo" write-tree); rm -f "$idx"
  echo "$name $tree" >> "$out/arms.manifest"
  echo "arm $name: $(git -C "$repo" rev-parse "$rev") patch ${patch:-none} tree $tree" >&2
done
