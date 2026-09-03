#!/bin/bash
# Shard 0 does double duty: the three one-layer perf arms (cheap -- `pre1` is
# the arm that has to reproduce R2's 170.474 us before any of the rest can be
# read), and then the Option D equivalence tests.
#
# Perf runs first on purpose. If the pytest half overruns the job it must not
# be what costs us the measurement.
bash /root/optd/d_perf_shard.sh pre1 dcond1 d1
A=$?
bash /root/optd/d_correctness.sh optd
B=$?
echo "===== SHARD0 SUMMARY perf_failed_steps=$A correctness_rc=$B ====="
exit $((A + B))
