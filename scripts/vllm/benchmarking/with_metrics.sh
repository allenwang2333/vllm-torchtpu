#!/bin/bash
# Usage: with_metrics.sh <results-dir> <command...>
#
# Run a benchmark flow while recording the server's histogram of tokens per
# engine step (vllm:iteration_tokens_total) every 15 seconds into
# <results-dir>/iteration_tokens.log. A layer run reads it to count the steps
# that carried a prefill chunk and the decode-only ones: a step of no more
# tokens than the engine runs sequences carried no prefill.
#
# Afterwards only rank 0's profile traces are kept. They are what a layer run
# reads, and every rank's would make each build's artifacts gigabytes.
set -u
out=$1
shift
mkdir -p "$out"
port=${PORT:-8000}
(
    while true; do
        snap=$(python3 - "$port" <<'EOF' 2>/dev/null
import sys, urllib.request
text = urllib.request.urlopen(f"http://localhost:{sys.argv[1]}/metrics", timeout=5).read().decode()
print("\n".join(l for l in text.splitlines() if l.startswith("vllm:iteration_tokens_total_bucket")))
EOF
        )
        if [ -n "$snap" ]; then
            { echo "# $(date +%s)"; echo "$snap"; } >>"$out/iteration_tokens.log"
        fi
        sleep 15
    done
) &
scraper=$!
"$@"
rc=$?
kill "$scraper" 2>/dev/null
find "$out" -name '*.xplane.pb' ! -name 'rank0_*' -delete 2>/dev/null
exit "$rc"
