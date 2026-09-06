#!/bin/bash
# [branch-local] Per-host runner for the 112-MiB TP32 all-reduce microbench
# (scripts/benchmark_torchtpu_all_reduce_112m.py) on the tpu7x-32 pod.
#
# Executed inside every host's container by run_multihost.sh --backend
# torchrun, which exports NODE_RANK / NNODES / MASTER_ADDR / NPROC_PER_NODE /
# TORCH_TPU_TOPOLOGY / TORCH_TPU_SLICEBUILDER_ADDRESSES (the same env the
# Ray/mp executors hand vLLM workers). Runs a matrix of LIBTPU / script
# variants; every variant is its own torchrun rendezvous on its own port, so
# a failed variant on one host fails that variant on all hosts and the loop
# moves on. Rank 0 (this host when NODE_RANK=0) writes the xplane profile.
set -uo pipefail

# Multi-host (run_multihost.sh --backend torchrun exports NNODES/NODE_RANK/
# MASTER_ADDR/NPROC_PER_NODE + the mesh env) or single-host (run_in_docker.sh:
# nothing exported; the torch_tpu single-host launcher supplies the mesh env
# and every local device joins one --standalone rendezvous).
if [ -z "${NNODES:-}" ] || [ "${NNODES}" = "1" ]; then
    eval "$(python3 -m torch_tpu._internal.distributed.launchers.singlehost_wrapper 2>/dev/null \
        | grep -E '^(TORCH_TPU_TOPOLOGY|TORCH_TPU_SLICEBUILDER_ADDRESSES|WORLD_SIZE)=')"
    export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES
    SINGLE_HOST=1
    NNODES=1
    NODE_RANK=0
    MASTER_ADDR=127.0.0.1
    NPROC="${WORLD_SIZE:?singlehost_wrapper did not report WORLD_SIZE}"
else
    SINGLE_HOST=0
    NODE_RANK="${NODE_RANK:?}"
    MASTER_ADDR="${MASTER_ADDR:?}"
    NPROC="${NPROC_PER_NODE:-8}"
fi
OUT="${ALLREDUCE_OUT:-/perf_eval_results/allreduce}"
SCRIPT=/root/torchtpu-vllm/scripts/benchmark_torchtpu_all_reduce_112m.py
ITER="${ALLREDUCE_ITERATIONS:-100}"
# Optional variant selection: positional args (from run_multihost.sh) or
# ALLREDUCE_VARIANTS, space-separated names; empty = run every variant.
VARIANTS="${*:-${ALLREDUCE_VARIANTS:-}}"
want() { [ -z "$VARIANTS" ] || [[ " $VARIANTS " == *" $1 "* ]]; }
mkdir -p "$OUT"

# What vllm_torchtpu/env_override.py injects for tpu7x before libtpu loads
# (the benchmark script imports torch_tpu directly, bypassing it).
TT_SERVING_FLAGS="--xla_tpu_use_dynamic_smem_negotiation=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=67108864"
FORCE_TC_FLAGS="--xla_tpu_use_dynamic_smem_negotiation=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=1073741824 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=1073741824"
SC_HIER_FLAGS="$TT_SERVING_FLAGS --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true"

echo "[allreduce] node=$NODE_RANK/$NNODES single_host=$SINGLE_HOST master=$MASTER_ADDR nproc=$NPROC topology=${TORCH_TPU_TOPOLOGY:-unset} variants=${VARIANTS:-all}"
echo "[allreduce] slicebuilder=${TORCH_TPU_SLICEBUILDER_ADDRESSES:-unset}"
python3 -c "import torch, torch_tpu; print('[allreduce] torch', torch.__version__)" 2>/dev/null | grep allreduce || true
pip show torch-tpu 2>/dev/null | grep -i '^Version' | sed 's/^/[allreduce] torch-tpu /' || true
pip show libtpu 2>/dev/null | grep -i '^Version' | sed 's/^/[allreduce] libtpu /' || true

run_variant() {
    # run_variant <name> <master_port> <LIBTPU_INIT_ARGS> [script args...]
    local name="$1" port="$2" libtpu="$3"
    shift 3
    mkdir -p "$OUT/$name"
    echo "=== [allreduce] variant=$name node=$NODE_RANK port=$port"
    echo "=== [allreduce] LIBTPU_INIT_ARGS='$libtpu' args='$*'"
    local launch
    if [ "$SINGLE_HOST" = "1" ]; then
        launch=(--standalone --nproc-per-node="$NPROC")
    else
        launch=(--nnodes="$NNODES" --nproc-per-node="$NPROC" --node-rank="$NODE_RANK"
                --master-addr="$MASTER_ADDR" --master-port="$port")
    fi
    LIBTPU_INIT_ARGS="$libtpu" python3 -m torch.distributed.run "${launch[@]}" \
        "$SCRIPT" --warmups 10 --iterations "$ITER" --expected-world-size 0 \
        --profile-dir "$OUT/$name/profile" --hlo-dump-dir "$OUT/$name/hlo" "$@" \
        2>&1 | tee "$OUT/$name/node${NODE_RANK}.log"
    local rc="${PIPESTATUS[0]}"
    grep -h "TORCHTPU_ALL_REDUCE_RESULT" "$OUT/$name/node${NODE_RANK}.log" \
        | sed 's/^.*TORCHTPU_ALL_REDUCE_RESULT //' > "$OUT/$name/results_node${NODE_RANK}.jsonl" || true
    grep -h "TORCHTPU_ALL_REDUCE_VERIFY" "$OUT/$name/node${NODE_RANK}.log" \
        | sed 's/^.*TORCHTPU_ALL_REDUCE_VERIFY //' > "$OUT/$name/verify_node${NODE_RANK}.txt" || true
    local n
    n=$(wc -l < "$OUT/$name/results_node${NODE_RANK}.jsonl" 2>/dev/null || echo 0)
    echo "=== [allreduce] variant=$name node=$NODE_RANK rc=$rc result_records=$n"
    if [ "$NODE_RANK" = "0" ] && [ "$n" -gt 0 ]; then
        python3 - "$OUT/$name/results_node0.jsonl" "$name" << 'PYEOF'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("rank") == 0:
        print(f"[allreduce] RESULT {sys.argv[2]}: median={r['median_ms']:.3f} ms p90={r['p90_ms']:.3f} p99={r['p99_ms']:.3f} min={r['min_ms']:.3f} "
              f"algo={r['algorithm_gbps']:.1f} GB/s bus={r['ring_equivalent_bus_gbps']:.1f} GB/s msg={r['message_mib']:.0f} MiB world={r['world_size']} "
              f"mode={r.get('mode', 'all_reduce')} out={r.get('output_shape')}")
PYEOF
    fi
    return "$rc"
}

# 1. Baseline = the reported repro: raw torch_tpu, libtpu defaults, no
#    TPU_*_BOUNDS. If the slice refuses to initialize without the bounds env
#    from the original recipe, retry once with it and keep it for the rest.
if want baseline && ! run_variant baseline 30201 ""; then
    echo "[allreduce] baseline failed without TPU_*_BOUNDS; retrying with the recipe's bounds env"
    export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
    export TPU_HOST_BOUNDS=1,1,4
    run_variant baseline_bounds 30211 "" || true
fi

# 2. What vLLM serving actually gets (SC offload threshold 64 MiB => this
#    112 MiB all-reduce is offloaded to SparseCore under vllm_torchtpu).
want tt_serving_flags && run_variant tt_serving_flags 30202 "$TT_SERVING_FLAGS" || true

# 3. Force the collective to stay on TensorCore (threshold 1 GiB).
want force_tensorcore && run_variant force_tensorcore 30203 "$FORCE_TC_FLAGS" || true

# 4. The flag from the original ask, on top of the serving flags.
want sc_hierarchical_ar && run_variant sc_hierarchical_ar 30204 "$SC_HIER_FLAGS" || true

# 5. Topology-aware group from torch.tpu.topology_aware_mesh.
want topology_aware && run_variant topology_aware 30205 "" --topology-aware || true

# 6. Two 56-MiB operands instead of one 112-MiB operand.
want chunks2 && run_variant chunks2 30206 "" --chunks=2 || true

# 7. Decode-sized message ([16, 7168] = 224 KiB) for the latency floor.
want decode_16tok && run_variant decode_16tok 30207 "" --tokens 16 || true

# 8. Explicit reduce-scatter + all-gather (functional collectives) instead
#    of one all-reduce: does the compiler's all-reduce beat the manual
#    decomposition, and how do the two phases split the time?
want rs_ag && run_variant rs_ag 30208 "" --mode rs_ag || true

# 9./10. Each phase alone (reduce-scatter of the 112 MiB operand; all-gather
#    of the 3.5 MiB per-rank shard back to 112 MiB).
want rs_only && run_variant rs_only 30209 "" --mode rs_only || true
want ag_only && run_variant ag_only 30210 "" --mode ag_only || true

# 11. Same decomposition on 56-MiB halves.
want rs_ag_chunks2 && run_variant rs_ag_chunks2 30212 "" --mode rs_ag --chunks=2 || true

# 12. All-reduce with SparseCore collective offload switched off (the XLA
#    option tpu_platform.py used for the unified KV pool), to obtain the
#    TensorCore all-reduce the offload-min-size flags never reached.
want tc_offload_off && run_variant tc_offload_off 30213 "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false" || true

# 13+. Correctness: structured input, every element on every rank vs a CPU
#    float64 reference; fp32 runs are exact-summable, bf16 shows rounding.
port=30220
for m in all_reduce rs_ag rs_only ag_only; do
    want verify_${m}_f32 && run_variant verify_${m}_f32 $port "" --mode $m --dtype float32 --verify --iterations 5 --warmups 1 || true
    port=$((port + 1))
    want verify_${m}_bf16 && run_variant verify_${m}_bf16 $port "" --mode $m --verify --iterations 5 --warmups 1 || true
    port=$((port + 1))
done

echo "[allreduce] node=$NODE_RANK all variants done"
if [ "$NODE_RANK" = "0" ]; then
    echo "[allreduce] ===== SUMMARY (rank 0) ====="
    grep -h "^\[allreduce\] RESULT" "$OUT"/*/node0.log 2>/dev/null || true
    grep -h "\[allreduce\] RESULT" "$OUT"/*/node0.log 2>/dev/null | sed 's/.*\[allreduce\] //' > "$OUT/summary.txt" || true
    echo "[allreduce] xplane traces:"
    find "$OUT" -name '*.xplane.pb' 2>/dev/null || true
fi
exit 0
