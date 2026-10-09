#!/bin/bash
# One forge scorer arm, run by the step of the pipeline perfkit/buildkite.py
# writes into the arm's commit. It serves a benchmark config, sweeps it, and
# fingerprints the server's output while the server is still up.
#
# Usage: scripts/forge/run_arm.sh <config> <results-dir> [<extra sweeps>]
#
# Into <results-dir>, beside what run_benchmarks.sh writes:
#   iteration_tokens.log  the server's step counters every 15 seconds, and once
#                         more when the sweep ends, before anything else runs
#   rep<N>/               each extra sweep, against the same server
#   numerics*.json/.out   the output fingerprints, as `perfkit numerics capture`
#                         writes them
# Only rank 0's profile traces are kept: they are what forge reads, and every
# rank's would make one arm's artifacts gigabytes.
#
# The profiled sweep is ended once rank 0 has captured the phases the config
# scores, FORGE_SCORED_PHASES (both decode-only and prefill-heavy steps when it
# is unset): the phased profiler takes each phase once, and GLM-5.2's twin
# then ran its profiled sweep for another 1m44s with nothing left to capture.
# run_benchmarks.sh does not fail a measured run over its profile run.
set -u
config=$1
out=$2
extra=${3:-0}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bench=./scripts/vllm/benchmarking
port=${PORT:-8000}
mkdir -p "$out"

snapshot() {
    python3 "$here/capture.py" counters --port "$port" >>"$out/iteration_tokens.log" 2>/dev/null
}

(while sleep 15; do snapshot; done) &
scraper=$!

captured() {
    grep -q "Worker_DP0_EP0.*Profiling for $1 phase finished" "$out/server.log" 2>/dev/null
}
scored=${FORGE_SCORED_PHASES:-decode_only prefill_heavy}
all_captured() {
    local phase
    for phase in $scored; do
        captured "$phase" || return 1
    done
}
(
    while sleep 10; do
        if all_captured; then
            if pkill -f -- "bin/vllm bench serve .*--profile"; then
                # Said first: the server can take longer to write the profile
                # out than the arm takes to finish, and the arm ends this loop.
                echo "forge: the scored phases are captured; the profiled sweep was ended"
                curl -s -m 60 -X POST "http://localhost:$port/stop_profile" >/dev/null 2>&1
            fi
            break
        fi
    done
) &
stopper=$!
trap 'kill "$scraper" "$stopper" 2>/dev/null; bash "$bench/cleanup_server.sh"' EXIT

"$bench/run_benchmarks.sh" --config "$config" --results-dir "$out" --keep-alive --port "$port"
rc=$?
kill "$scraper" "$stopper" 2>/dev/null
wait "$scraper" "$stopper" 2>/dev/null
# The reading the step shares are taken from: the sweep is over and nothing
# else has run on the server yet.
snapshot

if [ "$rc" -eq 0 ]; then
    # The server is warm by now, so the extra sweeps take no warm-up of their own.
    for k in $(seq 1 "$extra"); do
        BENCHMARK_WARMUP_RUNS=0 "$bench/run_benchmarks.sh" --config "$config" \
            --results-dir "$out/rep$k" --host localhost --port "$port" || rc=$?
    done
    suite="$here/suite.json"
    python3 "$here/capture.py" capture --port "$port" --suite "$suite" \
        --out "$out/numerics.json" >"$out/numerics.out" || rc=$?
    python3 "$here/capture.py" capture --port "$port" --suite "$suite" --no-logprobs \
        --out "$out/numerics-nologprobs.json" >"$out/numerics-nologprobs.out" || rc=$?
    python3 "$here/capture.py" capture --port "$port" --suite "$suite" --no-logprobs \
        --batch 64 --out "$out/numerics-batch.json" >"$out/numerics-batch.out" || rc=$?
fi

find "$out" -name '*.xplane.pb' ! -name 'rank0_*' -delete 2>/dev/null
exit "$rc"
