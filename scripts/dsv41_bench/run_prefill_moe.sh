#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
ulimit -c 0
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:$PWD"
export VLLM_TARGET_DEVICE=tpu VLLM_PLUGINS=torchtpu SKIP_JAX_PRECOMPILE=1

: "${ARTIFACT_LOCAL_DIR:?Falcon artifact directory is required}"
: "${MOE_RUN_NAME:?Set a unique run name}"
run_local="/workspace/moe-results/$MOE_RUN_NAME"
run_artifact="$ARTIFACT_LOCAL_DIR"
test ! -e "$run_local"
# One benchmark per Falcon experiment: analyzers read the artifact root.
for entry in manifest.json profiling rank-{0..7}; do
    test ! -e "$run_artifact/$entry"
done
mkdir -p "$run_local" "$run_artifact"

publish() {
    code=$?
    trap - EXIT
    printf '%s\n' "$code" > "$run_local/exit-status.txt"
    if [ "$code" -eq 0 ]; then
        if python3 - "$run_local" <<'PY'
import sys
from pathlib import Path
from scripts.dsv41_bench.bench_prefill_moe import export_falcon_artifact

run = Path(sys.argv[1])
export_falcon_artifact(run, run / "artifact")
PY
        then
            # Publish the root manifest only after every evidence file is copied.
            if ! cp -R "$run_local/artifact/profiling" \
                "$run_local/artifact/"rank-* "$run_artifact/" \
                || ! cp "$run_local/artifact/manifest.json" "$run_artifact/manifest.json"; then
                echo "ARTIFACT_PUBLISH_FAILED source=$run_local" >&2
                code=1
            fi
        else
            echo "ARTIFACT_CONTRACT_FAILED source=$run_local" >&2
            code=1
        fi
    fi
    if [ "$code" -ne 0 ]; then
        printf '%s\n' "$code" > "$run_local/exit-status.txt"
        failure_dir="$run_artifact/failed/$MOE_RUN_NAME"
        if ! mkdir -p "$failure_dir" || ! cp -R "$run_local/." "$failure_dir/"; then
            echo "ARTIFACT_FAILURE_LOG_PUBLISH_FAILED source=$run_local" >&2
            exit 1
        fi
    fi
    echo "RUN_FINISHED exit=$code artifact=$run_artifact"
    exit "$code"
}
trap publish EXIT

python3 - "$run_local" <<'PY'
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
from scripts.dsv41_bench.bench_prefill_moe import check_runtime

out = Path(sys.argv[1])
(out / 'python-packages.txt').write_text(''.join(
    f'{name}=={version}\n' for name, version in sorted(
        (dist.metadata['Name'], dist.version) for dist in md.distributions()
    )
))
manifest = json.loads(Path('.dsv41_bench/source-files.json').read_text())
for name, expected in manifest['files'].items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Source hash mismatch: {name}')
versions = check_runtime()
(out / 'runtime-versions.json').write_text(json.dumps(versions, indent=2))
(out / 'source-files.json').write_text(json.dumps(manifest, indent=2))
print('SOURCE_AND_RUNTIME_OK', flush=True)
PY

run_capture() {
    phase=$1
    warmup=$2
    steps=$3
    shift 3
    mkdir -p "$run_local/$phase"
    echo "PHASE_START $phase global_tokens=16384 local_tokens=2048 tp=1 dp=8 ep=8 token_layout=sharded_no_replicas"
    local status=0
    timeout --signal=TERM --kill-after=30s 1200s \
        python3 -u -m scripts.dsv41_bench.bench_prefill_moe \
        --global-tokens 16384 --tp-size 1 --dp-size 8 --warmup "$warmup" \
        --profile-steps "$steps" --seed 0 \
        --result-dir "$run_local/$phase/results" \
        --profile "$run_local/$phase/xprof" \
        --golden-dir "$run_local/golden" "$@" \
        > "$run_local/$phase/bench.log" 2>&1 || status=$?
    if [ "$status" -ne 0 ]; then
        echo "PHASE_FAILED $phase exit=$status"
        tail -n 200 "$run_local/$phase/bench.log"
        if [ -f "$run_local/$phase/results/workers.log" ]; then
            tail -n 300 "$run_local/$phase/results/workers.log"
        fi
        return "$status"
    fi
    test -s "$run_local/$phase/results/device-metrics.json"
    echo "PHASE_COMPLETE $phase"
}

run_capture smoke 1 1 --update-golden
run_capture baseline 5 5 --verify --dump-llo "$run_local/baseline/compiler"
echo "BASELINE_DEVICE_CAPTURE_OK"
