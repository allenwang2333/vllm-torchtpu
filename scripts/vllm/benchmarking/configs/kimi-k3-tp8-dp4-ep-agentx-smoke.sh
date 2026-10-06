#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by k3_1p1d.sh
# TEMPORARY, for hardware bring-up. Not for merge.
#
# The AgentX pair with one ladder point instead of two. The SERVER is identical
# - the ladder reaches no serve flag, and MAX_NUM_SEQS, the compile buckets, gmu
# and the context are inherited - so this shakes out the same deployment in a
# fraction of the replay time.
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/kimi-k3-tp8-dp4-ep-agentx.sh"
AGENTX_CONC_LIST="1"
