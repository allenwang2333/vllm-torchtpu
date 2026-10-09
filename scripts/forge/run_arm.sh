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
trap 'kill "$scraper" 2>/dev/null; bash "$bench/cleanup_server.sh"' EXIT

"$bench/run_benchmarks.sh" --config "$config" --results-dir "$out" --keep-alive --port "$port"
rc=$?
kill "$scraper" 2>/dev/null
wait "$scraper" 2>/dev/null
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
