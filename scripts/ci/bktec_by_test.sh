#!/bin/bash
# Runs bktec over the step's individual tests rather than its files.
#
# bktec plans a file by its recorded duration in the shared tpu-ci suite, which
# includes the tests other steps run from that file and this step deselects, so
# its shards come out uneven. A test's own duration does not have that problem,
# and the step's -m filter (in PYTEST_ADDOPTS) keeps deselected tests out of the
# list altogether. Arguments are the paths and options to collect from.
set -uo pipefail

selectors=$(mktemp)
# Only test IDs: some modules print while they are collected, and a printed line
# that mentions a test would otherwise reach pytest as an argument.
pytest --collect-only -q "$@" | grep -E '^tests/[^ ]+\.py::' > "$selectors"
rc=${PIPESTATUS[0]}
# A collection error would otherwise leave its tests off the list unnoticed.
# 5 is pytest's "nothing collected", which the caller tolerates.
if [ "$rc" -ne 0 ]; then
  echo "pytest --collect-only exited $rc" >&2
  exit "$rc"
fi
echo "$(wc -l < "$selectors") tests to plan"
BUILDKITE_TEST_ENGINE_SELECTOR_FILE="$selectors" exec bktec run
