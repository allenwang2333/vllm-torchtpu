#!/bin/bash
# Repeated-restart A/B matrix for cheap serving knobs (no code changes).
#
# One invocation = one agent = one host, so every cell it produces is directly
# comparable. Each cell is a FULL server lifetime: fresh `vllm serve`, N
# benchmark repeats, optional completion capture and gsm8k, then teardown.
# That separates the two variances an A/B needs to be judged against:
#
#   within-server  = spread of the N repeats inside one cell
#   across-restart = spread of the cell means for one arm
#
# Arms are given as an ordered list so a caller can write A,B,B,A: with the
# arms interleaved, monotone drift over the job cannot alias onto the arm
# (see lesson 7df107efaae9).
#
# Usage:
#   k3_knob_matrix.sh --config kimi-k3-sliced8-tp8-ep --results-dir DIR \
#       --arms base,flagset,flagset,base [--reps 3] [--probe] [--gsm8k 100] \
#       [--no-log-dedup]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME="kimi-k3-sliced8-tp8-ep"
RESULTS_DIR=""
ARMS=""
REPS=3
PORT=8000
PROBE=0
GSM8K_LIMIT=0
GSM8K_CELLS=""
NO_LOG_DEDUP=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)       CONFIG_NAME="$2"; shift 2 ;;
    --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
    --arms)         ARMS="$2"; shift 2 ;;
    --reps)         REPS="$2"; shift 2 ;;
    --port)         PORT="$2"; shift 2 ;;
    --probe)        PROBE=1; shift ;;
    --no-log-dedup) NO_LOG_DEDUP=1; shift ;;
    --gsm8k)        GSM8K_LIMIT="$2"; shift 2 ;;
    --gsm8k-cells)  GSM8K_CELLS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done
[ -n "$ARMS" ] || { echo "ERROR: --arms is required"; exit 2; }
RESULTS_DIR="${RESULTS_DIR:-/tmp/k3_knob_matrix}"
mkdir -p "$RESULTS_DIR"

# The four flags from gh/agent/k3-b1-tpot-optimization-backup's launcher
# (scripts/k3/launch_kimi_k3_tp32_ep.sh), measured there one at a time on
# dummy weights at TP32/ISL=1 and never A/B'd together.
F_LAYER_SCHED="--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true"
F_ASYNC_MERGER="--xla_tpu_enable_async_collective_merger=true"
F_DECOMPOSE_AR="--xla_tpu_decompose_all_reduce_bidirectional_communication=true"
F_MULTI_OVERLAP="--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true"

# Prints the `KEY=VALUE` overrides that define an arm, one per line.
arm_env() {
  case "$1" in
    base)     ;;
    flagset)  echo "LIBTPU_INIT_ARGS=$F_LAYER_SCHED $F_ASYNC_MERGER $F_DECOMPOSE_AR $F_MULTI_OVERLAP" ;;
    f_sched)  echo "LIBTPU_INIT_ARGS=$F_LAYER_SCHED" ;;
    f_merge)  echo "LIBTPU_INIT_ARGS=$F_ASYNC_MERGER" ;;
    f_decomp) echo "LIBTPU_INIT_ARGS=$F_DECOMPOSE_AR" ;;
    f_overlap) echo "LIBTPU_INIT_ARGS=$F_MULTI_OVERLAP" ;;
    fp8kv)    echo "KV_CACHE_DTYPE=fp8" ;;
    # Only for kimi-k3-sliced8-tp8-tunehit: moves pages_per_seq 128 -> 129,
    # the one MLA tuning-key field that can be changed without changing the
    # workload. tunehit selects the table row, tunemiss falls back.
    tunehit)  echo "MAX_MODEL_LEN=32768" ;;
    tunemiss) echo "MAX_MODEL_LEN=33024" ;;
    # Rungs of kimi-k3-sliced8-tp8-ladder. L0 reproduces the shipped
    # kimi-k3-sliced8-tp8-ep; each later rung adds one serve knob. Only L4
    # presents a key that `tuned_params.py:1122` can match.
    L0_ship)  ;;
    L1_blk256)
      printf '%s\n' "K3_BLOCK_SIZE=256" ;;
    L2_tok4096)
      printf '%s\n' "MAX_NUM_BATCHED_TOKENS=4096" "K3_COMPILE_SIZES=[16,4096]" ;;
    L3_both)
      printf '%s\n' "MAX_NUM_BATCHED_TOKENS=4096" "K3_COMPILE_SIZES=[16,4096]" \
                     "K3_BLOCK_SIZE=256" ;;
    L4_hit)
      printf '%s\n' "MAX_NUM_BATCHED_TOKENS=4096" "K3_COMPILE_SIZES=[16,4096]" \
                     "K3_BLOCK_SIZE=256" "MAX_NUM_SEQS=8" "MAX_MODEL_LEN=32768" ;;
    # kimi-k3-tp32-ep-tunehit: pages_per_seq 40 (row :1169 selected) vs 41
    # (fallback), identical workload either way.
    podhit)   echo "MAX_MODEL_LEN=10240" ;;
    podmiss)  echo "MAX_MODEL_LEN=10496" ;;
    # Size the MLA v2 *mixed* fallback's KV block in tokens instead of one
    # page. At the shipped --block-size 16 the fallback walks a 16-token KV
    # tile per query tile; 256 and 768 are the two sizes lane B's LLO model
    # prices at 12.5 ms and 7.1 ms per 8192 prefill against the shipped
    # 163.7 ms ($AMPLIO_ARTIFACT_DIR/r2/b/KERNEL.md).
    mfix_off) ;;
    mfix256)  echo "TPU_MLA_MIXED_FALLBACK_BKV_TOKENS=256" ;;
    mfix768)  echo "TPU_MLA_MIXED_FALLBACK_BKV_TOKENS=768" ;;
    # Expert parallelism between chips (ep_size = chips) with tensor
    # parallelism between the two chiplets of a chip, instead of vLLM's flat
    # EP across every rank. Merged, default-off, and never measured on its own.
    # Multihost note: this must reach all 32 worker processes, not just the
    # driver. It does, because RayDistributedExecutorV2 copies the driver's
    # whole environment to every actor (`get_driver_env_vars`) -- which is why
    # exporting it here, per cell, works at all. `hier_verdict` below is the
    # proof, per cell, and does not take that on trust.
    hier)     echo "TPU_MOE_HIERARCHICAL_EP=1" ;;
    # W0 leg 3: the decode token bucket. tpu_runner.py:632 sets
    # num_tokens_paddings = compile_sizes, so the smallest compiled size IS the
    # decode batch, and 16 is a config line rather than a hardware floor.
    # [8,8192] is the shippable floor at --max-num-seqs 8; BK #325 measured
    # -11.8% device busy on the v7x-8 surrogate with both negative controls
    # held. hier8 is the composition cell: the two levers share a term
    # (rows at the gmm) and W1's P14 registers the pair at 76-87% of the sum.
    bucket16) echo "K3_COMPILE_SIZES=[16,8192]" ;;
    bucket8)  echo "K3_COMPILE_SIZES=[8,8192]" ;;
    hier8)    printf '%s\n' "K3_COMPILE_SIZES=[8,8192]" \
                             "TPU_MOE_HIERARCHICAL_EP=1" ;;
    # ---- round 4 lane B: gnovack/k3-dev's recipe, dummy vs real weights ----
    # Use with --config kimi-k3-tp32-theirs, which carries the rest of their
    # config. THE ONLY DIFFERENCE BETWEEN dummy AND real IS LOAD_FORMAT: same
    # branch, same flags, same server command, same client, same job, same
    # host. Everything below `real` is what their compile_flag_sweep.sbatch
    # exports, restated per arm so a cell's record is self-contained.
    dummy|real|dummyprof|realprof|dummyuniform)
      printf '%s\n' \
        "TPU_MOE_HIERARCHICAL_EP=1" \
        "TPU_LATENT_PROJ_INTRA_CHIP_TP=1" \
        "VLLM_USE_BREAKABLE_CUDAGRAPH=0" \
        "VLLM_TORCHTPU_KDA_FUSED=1" \
        "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0" \
        "ENABLE_PREFIX_CACHING=false" \
        "RAY_memory_usage_threshold=0.99" \
        "LIBTPU_INIT_ARGS=$K3_THEIRS_LIBTPU"
      case "$1" in
        dummy*) echo "LOAD_FORMAT=dummy" ;;
      esac
      case "$1" in
        # Dummy weights, but with expert selection replaced by a uniform draw
        # anchored to the real activations. Isolates the two halves of what
        # `dummy` changes: the ROUTING COLLAPSE (lane E: 30 distinct experts of
        # 896 against 805-892, because every float weight is <=1e-3 so top-k
        # becomes the argmax of a randomised e_score_correction_bias) from the
        # NUMERICS (mxfp4 expert weights zeroed). If this arm lands on the real
        # arm's timing, the collapse is the whole story and simulated routing
        # is the fix that makes a dummy rig quotable.
        # No kernel-path confound: fused EP already refuses K3 on
        # `e_score_correction_bias is not applied inside the kernel`, before it
        # ever reaches the simulation check (fused_moe_ep.py:560).
        *uniform) echo "VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random" ;;
      esac
      case "$1" in
        # The profiled cells are deliberately NOT part of the A,B,B,A: the
        # bench client's --profile costs ~+11% TTFT / -4% throughput, so a
        # profiled cell's serving numbers are not comparable to an unprofiled
        # one's. They are here for the per-category device diff, which both
        # arms pay equally, and for the router histogram that tests the Q3
        # mechanism directly instead of inferring it from a timing delta.
        *prof) printf '%s\n' "CAPTURE_PROFILE=1" \
                              "USE_PHASED_PROFILER=true" \
                              "TPU_MOE_EXPERT_LOAD_EVERY=8" \
                              "TPU_MOE_EXPERT_LOAD_DIR=$RESULTS_DIR/expert_load_$1" \
                              "K3_CONCURRENCY=1" \
                              "K3_ARM_REPS=1" ;;
      esac
      ;;
    *) echo "__UNKNOWN_ARM__" ;;
  esac
}

# The three SparseCore flags they credit for the TTFT win, on top of the same
# diagnostics-off trio their launcher always sets. Their note: the 32 MiB
# offload threshold is only a win WITH hierarchical all-reduce.
K3_THEIRS_LIBTPU="${K3_THEIRS_LIBTPU:-\
--xla_tpu_enable_diagnostic_coordinator=false \
--tpu_hbm_report_enable=false \
--enable_runtime_uptime_telemetry=false \
--xla_tpu_enable_sparse_core_hierarchical_all_reduce=true \
--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=33554432 \
--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=33554432}"

IFS=',' read -r -a arm_list <<< "$ARMS"
for arm in "${arm_list[@]}"; do
  if [ "$(arm_env "$arm")" = "__UNKNOWN_ARM__" ]; then
    echo "ERROR: unknown arm '$arm'"; exit 2
  fi
done

trap 'echo "--- matrix cleanup"; bash "$SCRIPT_DIR/cleanup_server.sh"' EXIT INT TERM

# Disk, on STDOUT, before the first stream and after every cell. BK #329 died
# in cell 4 of 4 with `No space left on device` after ~100 minutes, and its
# artifacts were four 0-byte tarballs -- so the only thing that survived was
# what it had echoed. A trend line here says which cell will die BEFORE it
# does, which is the difference between stopping with N good cells and losing
# the job mid-flight.
disk() { printf '[disk] %s %s\n' "$1" \
    "$(df -P /perf_eval_results / 2>/dev/null | awk 'NR>1{printf "%s:%s free(%s used) ", $6, $4, $5}')"; }
# WHERE the bytes go, which `df` cannot say. r3 left this as its one open disk
# question: #336's five cells burned 10.6/8.1/7.6/10.3/>=14.4 GB from ~51 GB
# and died, while #338's one-lifetime-per-container jobs burned 1.05 GB and
# 7.2 MB streaming the same weights -- so the growth belongs to the CELL LOOP,
# not to a server start or to the expert fetch, and "one du -sh between cells
# settles what is left" (r3/W0.md). This is that du.
du_breakdown() {
  local d
  for d in /perf_eval_results /tmp/ray /tmp/tpu_logs /root/.cache /var/lib/docker /root/k3-metadata; do
    [ -e "$d" ] && printf '[du] %s %s\n' "$1" "$(du -sh "$d" 2>/dev/null | head -1)"
  done
}

# Ray keeps one session directory per container and every restarted server
# appends 32 more actor logs to it; nothing in the loop ever reclaims them.
# copytruncate rather than unlink, because a live writer holds these fds.
PRUNE_RAY_LOGS="${PRUNE_RAY_LOGS:-1}"
prune_ray_logs() {
  [ "$PRUNE_RAY_LOGS" = "1" ] || return 0
  local n
  n=$(find /tmp/ray -type f \( -name '*.out' -o -name '*.err' -o -name '*.log' \) \
        -size +64M -print 2>/dev/null | wc -l)
  find /tmp/ray -type f \( -name '*.out' -o -name '*.err' -o -name '*.log' \) \
    -size +64M -exec truncate -s 0 {} + 2>/dev/null
  echo "[prune] truncated $n ray log file(s) >64M"
}

disk start
du_breakdown start

fail=0
cell=0
for arm in "${arm_list[@]}"; do
  cell=$((cell + 1))
  cell_id=$(printf 'cell%02d_%s' "$cell" "$arm")
  cell_dir="$RESULTS_DIR/$cell_id"
  cell_t0=$(date +%s)
  mkdir -p "$cell_dir"
  echo "--- $cell_id ($cell/${#arm_list[@]}) : $(date -u +%H:%M:%S)"

  bash "$SCRIPT_DIR/cleanup_server.sh"

  # Subshell so an arm's exports cannot leak into the next cell.
  (
    while IFS= read -r kv; do
      [ -n "$kv" ] && export "${kv?}"
    done < <(arm_env "$arm")
    # An arm may buy fewer repeats than the matrix default (K3_ARM_REPS from
    # arm_env): the profiled cells are not part of the drift-controlled ABBA,
    # so a second repeat of them buys nothing but wall clock and disk.
    export BENCHMARK_WARMUP_RUNS=$(( ${K3_ARM_REPS:-$REPS} - 1 ))
    # Ray prints one copy of a line the whole cluster emitted and appends
    # "[repeated Nx across cluster]". That is unusable as a per-rank control:
    # the batching window decides whether 32 ranks arrive as one message or
    # five. With dedup off every rank's line carries its own
    # `Worker_TP<n>_EP<n>` tag, so a control can count distinct RANKS.
    [ "$NO_LOG_DEDUP" = "1" ] && export RAY_DEDUP_LOGS=0
    echo "[arm] LIBTPU_INIT_ARGS='${LIBTPU_INIT_ARGS:-}' KV_CACHE_DTYPE='${KV_CACHE_DTYPE:-}'" \
      "TPU_MOE_HIERARCHICAL_EP='${TPU_MOE_HIERARCHICAL_EP:-}'" \
      | tee "$cell_dir/arm.txt"

    bash "$SCRIPT_DIR/run_benchmarks.sh" --config "$CONFIG_NAME" \
      --results-dir "$cell_dir" --keep-alive --port "$PORT" || exit 1

    model=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' \
      "$cell_dir/config.json")

    # Positive control. `run_in_docker.sh` forwards a hand-written allowlist of
    # env vars and LIBTPU_INIT_ARGS is not on it, which is why this script sets
    # the arm INSIDE the container -- but "I exported it" is not evidence that
    # the server got it. Read it back out of the live worker processes.
    for pid in $(pgrep -f 'VLLM::|vllm serve' 2>/dev/null | head -20); do
      tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
        | sed -n "s/^LIBTPU_INIT_ARGS=/pid $pid: /p"
    done > "$cell_dir/libtpu_readback.txt"
    echo "readback: $(wc -l < "$cell_dir/libtpu_readback.txt") worker(s) report LIBTPU_INIT_ARGS"

    # WHAT libtpu was actually given, not how many processes were asked. Three
    # separate things can make this differ from the arm:
    #   - LIBTPU_INIT_ARGS is not in the container passthrough allowlist, so it
    #     is set here and reaches workers only via Ray's driver-env copy;
    #   - env_override.py resolves a SparseCore offload default from
    #     TPU_ACCELERATOR_TYPE / GCE metadata, and where neither resolves it
    #     sets nothing, leaving libtpu's own 64 KiB default -- which offloads
    #     almost everything and is not what any of our baselines assumed;
    #   - an arm that overrides the variable replaces that default wholesale.
    # Print the distinct values and libtpu's own argv echo of the two flags
    # that decide it.
    sed 's/^pid [0-9]*: /[libtpu] env: /' "$cell_dir/libtpu_readback.txt" \
      | sort -u | head -4
    grep -aoE "argv\[[0-9]+\]: '--xla_tpu_(enable_sparse_core[a-z_]*|sparse_core[a-z_]*)=[^']*'" \
      "$cell_dir/server.log" 2>/dev/null | sed -E 's/argv\[[0-9]+\]/[libtpu] argv/' \
      | sort | uniq -c | head -8
    grep -aoiE '(sparse.?core|offload)[a-z_ ]*(min_size|threshold)[^,;]{0,60}' \
      "$cell_dir/server.log" 2>/dev/null | sort -u | sed 's/^/[libtpu] resolved /' | head -4

    # Per-RANK proof that the two env-only flags arrived. Neither is in
    # TpuPlatform.additional_env_vars nor carries a VLLM_ prefix. The V2 Ray
    # executor copies the driver's WHOLE environment (vllm's get_driver_env_vars
    # returns all of os.environ minus the worker-specific names), so they do
    # arrive on that path -- but the V1 executor filters by additional_env_vars
    # and would drop them silently, and "I exported it" is not evidence either
    # way. The intra-chip flag's own signal is the per-core column count.
    printf '[intra] arm=%s group_ready=%s sharded=%s replicated_fallback=%s\n' "$arm" \
      "$(grep -ac 'Intra-chip group ready' "$cell_dir/server.log" 2>/dev/null)" \
      "$(grep -ac 'sharded across .* cores per chip' "$cell_dir/server.log" 2>/dev/null)" \
      "$(grep -ac 'leaving the latent projections replicated\|leaving it replicated' "$cell_dir/server.log" 2>/dev/null)"
    grep -aoE 'Latent MoE projection \[[0-9, ]*\] sharded across [0-9]+ cores per chip \([0-9]+ columns each\)' \
      "$cell_dir/server.log" 2>/dev/null | sort | uniq -c | sed 's/^/[intra] /' | head -4

    # Echo the MLA tuning-table verdict to STDOUT. It is otherwise only in
    # server.log, which reaches nobody until the post-command artifact upload
    # -- so a cancelled job loses the one line that is the whole point of a
    # reachability run. The job log survives cancellation; server.log does not.
    grep -am1 'MLA v2 tuned params:' "$cell_dir/server.log" 2>/dev/null \
      | sed 's/^/[tuning] /' || echo "[tuning] no MLA v2 coverage line (warm compile cache: no lookups traced)"

    # How long the server took to answer /health, split from the benchmark.
    # Dummy-weight bring-up has never been timed anywhere; this is the number
    # that says how many more cells a dummy job can afford than a real one.
    printf '[boot] arm=%s ready=%ss\n' "$arm" \
      "$(grep -aoE 'Server ready on port [0-9]+ \(([0-9]+)s\)' "$cell_dir/server.log" 2>/dev/null \
         | grep -oE '[0-9]+s\)' | tr -d 's)' | tail -1)"

    # WHICH MLA rows the server actually asked for, and whether it got them.
    # `get_tuned_params` is silent on a hit and warns on a miss, so a table
    # that is being used and a table that does not exist look identical from
    # the outside; [mla-key] logs both, once per distinct key. pages_per_seq
    # is the field that decides it: 10240/16 = 640 selects their rows,
    # 10240/256 = 40 selects ours, and no key can select both.
    grep -ah '\[mla-key\]' "$cell_dir/server.log" 2>/dev/null \
      | sed -E 's/.*\[mla-key\]/[keys]/' | sort -u | head -20
    # r4 SHARED, mandatory for every hardware lane: get_tuned_params NEVER
    # returns None -- a miss silently falls back to kvpb 1 / qpb 16, which at
    # 8192 tokens is 512x32 blocks instead of 16x2. An arm that lost a row and
    # an arm that never had one look the same in the serving number.
    printf '[keys] arm=%s tuning_misses=%s\n' "$arm" \
      "$(grep -ac 'No tuned parameters found' "$cell_dir/server.log" 2>/dev/null)"
    grep -ah 'No tuned parameters found' "$cell_dir/server.log" 2>/dev/null \
      | grep -oE 'TuningKey\([^)]*\)' | sort -u | sed 's/^/[keys] MISS /' | head -12
    printf '[keys] arm=%s block_size=%s pages_per_seq=%s\n' "$arm" \
      "$(grep -aom1 "block_size=[0-9]*" "$cell_dir/server.log" 2>/dev/null | sed 's/.*=//')" \
      "$(grep -ah '\[mla-key\]' "$cell_dir/server.log" 2>/dev/null \
         | grep -oE 'pages_per_seq=[0-9]+' | sed 's/.*=//' | sort -un | tr '\n' ',')"

    # Hierarchical MoE verdict, per cell, to STDOUT (survives a cancel).
    # Three separate things have to be true and each fails silently on its own:
    #   reached  -- the env var got to the worker PROCESS, not just the driver
    #   split    -- the chip topology was readable, so EP is really per-chip
    #   loader   -- the Run:AI expert filter fetched the CHIP block (ep_size =
    #               chips), not the flat rank block; a mismatch here leaves
    #               half of every rank's expert slots uninitialised
    # Reported as tagged-ranks/lines, because neither alone is trustworthy: not
    # every forwarded worker line carries its `Worker_TP<n>_EP<n>` tag, and a
    # line emitted more than once per process would inflate a raw count. The
    # per-rank lines are the `logger.info` ones -- `moe_hierarchical.py:109`
    # uses `info_once`, which defaults to scope="local" and therefore samples
    # local rank 0 of each host (4 of 32 on the pod, 1 of 8 here) BY DESIGN.
    # With Ray's log dedup left ON (r4 lane B: turning it off is the leading
    # suspect for r3's ~9 GB/cell disk burn) the raw line count understates by
    # the size of the batching window, so add back the `[repeated Nx]` suffixes
    # the way their own slurm wrapper does.
    hier_ranks() {
      local hits suppressed
      hits=$(grep -Fc "$1" "$cell_dir/server.log" 2>/dev/null || echo 0)
      suppressed=$(grep -F "$1" "$cell_dir/server.log" 2>/dev/null | cut -c1-400 \
        | grep -oE 'repeated [0-9]+x' | grep -oE '[0-9]+' \
        | awk '{n+=$1} END {print n+0}')
      printf '%s/%s' "$(grep -F "$1" "$cell_dir/server.log" 2>/dev/null \
        | grep -oE 'Worker_TP[0-9]+_EP[0-9]+' | sort -u | wc -l)" \
        "$(( ${hits:-0} + ${suppressed:-0} ))"
    }
    printf '[hier] arm=%s split=%s filter=%s topology=%s fallback=%s loader=%s bank=%s (tagged_ranks/lines)\n' \
      "$arm" \
      "$(hier_ranks 'Hierarchical MoE parallelism: EP between chips')" \
      "$(hier_ranks 'EP weight filter realigned to chip-block ownership')" \
      "$(hier_ranks 'Chip topology:')" \
      "$(hier_ranks 'falling back to flat expert parallelism')" \
      "$(hier_ranks '[sharded-ep-load] ep_size=')" \
      "$(hier_ranks 'MoE expert bank on this rank:')"
    grep -hoE 'MoE expert bank on this rank: w13 \([0-9, ]+\) \(ep_size=[0-9]+, moe_tp_size=[0-9]+[^)]*\)' \
      "$cell_dir/server.log" 2>/dev/null | sort | uniq -c | sed 's/^/[hier] bank /'
    grep -hoE '\[sharded-ep-load\] ep_size=[0-9]+ ep_rank=[0-9]+ [^:]*: this rank keeps [0-9]+/[0-9]+' \
      "$cell_dir/server.log" 2>/dev/null \
      | sed -E 's/ep_rank=[0-9]+/ep_rank=*/' | sort | uniq -c | sed 's/^/[hier] loader /'
    grep -am1 -oE 'Chip topology: [0-9]+ chip\(s\) x [0-9]+ core\(s\)' \
      "$cell_dir/server.log" 2>/dev/null | sed 's/^/[hier] /'
    # The bucket arm's own positive control, on STDOUT: the job log survives a
    # cancel and server.log does not. compile_sizes IS the decode batch size
    # (tpu_runner.py:632), so an arm that did not take is an arm that measured
    # the control twice.
    # The bucket arm's own positive control, on STDOUT, because the job log
    # survives a cancel and server.log does not. Two lines: what was asked for
    # (compile_sizes, from the engine's own config echo) and what happened
    # (padded_total_num_scheduled_tokens, the decode batch the model actually
    # saw). compile_sizes IS the decode batch size -- tpu_runner.py:632 sets
    # num_tokens_paddings from it -- so an arm where these disagree, or where
    # either is missing, measured something other than the arm it names.
    printf '[bucket] arm=%s asked compile_sizes=%s got padded_tokens=%s\n' "$arm" \
      "$(grep -aom1 "compile_sizes': \[[^]]*\]" "$cell_dir/server.log" 2>/dev/null \
         | sed "s/.*: //" || true)" \
      "$(grep -aoh "padded_total_num_scheduled_tokens': [0-9]*" "$cell_dir/server.log" 2>/dev/null \
         | sed "s/.*: //" | sort -un | tr '\n' ',' || true)"
    python3 -c "import json,sys;c=json.load(open(sys.argv[1]));print('[bucket] '+' '.join(f'{k}={c.get(k)}' for k in ('async_scheduling','kv_cache_dtype','max_num_seqs','max_num_batched_tokens','max_model_len','concurrency_options','num_prompts')))" \
      "$cell_dir/config.json" 2>/dev/null || echo "[bucket] config.json unreadable"
    grep -am2 -oE 'num_gpu_blocks_override=[0-9]+[^;]*' "$cell_dir/server.log" 2>/dev/null \
      | sed 's/^/[hier] kv /'

    if [ "$PROBE" = "1" ]; then
      python3 "$SCRIPT_DIR/k3_completion_probe.py" --model "$model" \
        --base-url "http://localhost:$PORT" --out "$cell_dir/completions.json" \
        2>&1 | tee "$cell_dir/probe.log"
    fi

    if [ "$GSM8K_LIMIT" != "0" ] && \
       { [ -z "$GSM8K_CELLS" ] || [[ ",$GSM8K_CELLS," == *",$cell,"* ]]; }; then
      # /v1/completions, not /v1/chat/completions: K3 ships no chat template
      # and the hand-written one scores a 0.927 model at 0.0034.
      lm_eval --model local-completions \
        --model_args "model=${model},base_url=http://localhost:$PORT/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=8,timeout=5000,max_length=8192,max_gen_toks=256" \
        --tasks gsm8k --num_fewshot 5 --limit "$GSM8K_LIMIT" \
        --output_path "$cell_dir/gsm8k" 2>&1 | tee "$cell_dir/gsm8k.log"
      # On STDOUT, and never fatal to the cell: an accuracy check is the only
      # instrument that can see a miscompiled collective, and losing the cell's
      # serving numbers because lm_eval is missing would be the wrong trade.
      grep -aoE '\|(exact_match|flexible-extract|strict-match)\|[^|]*\|[0-9.]+' \
        "$cell_dir/gsm8k.log" 2>/dev/null | sed 's/^/[acc] /' | head -4 \
        || echo "[acc] $arm no gsm8k table parsed"
    fi
    true
  )
  cell_status=$?
  [ "$cell_status" -eq 0 ] || { echo "::error::$cell_id failed ($cell_status)"; fail=1; }
  # The cell's own numbers, on STDOUT, immediately. The results dir may never
  # be uploaded (BK #329) and the job log always survives.
  python3 - "$cell_dir" "$cell_id" <<'PYCELL' 2>/dev/null || echo "[cell] $cell_id no result json"
import json, os, re, sys
d, cid = sys.argv[1], sys.argv[2]
for fn in sorted(os.listdir(d)):
    if not re.match(r"^isl\d+_osl\d+_c\d+(\.warmup\d+)?\.json$", fn):
        continue
    try:
        j = json.load(open(os.path.join(d, fn)))
    except Exception:
        continue
    print(f"[cell] {cid} {fn} TPOT {j.get('median_tpot_ms'):.4f} "
          f"TTFT {j.get('median_ttft_ms'):.1f} ITL {j.get('median_itl_ms'):.4f} "
          f"tok/s {j.get('total_token_throughput'):.2f} completed {j.get('completed')}")
PYCELL
  printf '[wall] %s %ss\n' "$cell_id" "$(( $(date +%s) - cell_t0 ))"
  disk "after $cell_id"
  du_breakdown "after $cell_id"

  bash "$SCRIPT_DIR/cleanup_server.sh"
  # Keep the artifact bundle small: the interesting parts of a 190 KB
  # server.log are the config echo, the KV sizing and any error.
  if [ -f "$cell_dir/server.log" ]; then
    grep -nE 'KV cache|kv_cache|num_gpu_blocks|Maximum concurrency|init engine|ERROR|Traceback|tuned|TuningKey|mla-key|fallback|page_size|kv_packing|Using cache directory|Compiling FX graph|libtpu_init_args|moe_hierarchical|Chip topology|sharded-ep-load|Loading weights|load_format' \
      "$cell_dir/server.log" > "$cell_dir/server.grep.txt" 2>/dev/null
    gzip -f "$cell_dir/server.log"
  fi
  prune_ray_logs
  disk "after prune $cell_id"
done

echo "--- summary"
python3 "$SCRIPT_DIR/k3_knob_summary.py" --results-dir "$RESULTS_DIR" \
  | tee "$RESULTS_DIR/summary.txt"

# Cross-cell greedy agreement, when --probe captured completions. Two things
# it settles that no timing number can: whether two cells of the SAME arm are
# deterministic (they must be, or every delta in this job is noise), and
# whether two cells of DIFFERENT arms compute the same thing at all -- which
# is how a miscompiled collective, or a dummy cell mislabelled as real, shows
# itself. Free: the captures already exist.
if [ "$PROBE" = "1" ]; then
  cells=$(cd "$RESULTS_DIR" && ls -d cell*/ 2>/dev/null | tr -d / | paste -sd, -)
  [ -n "$cells" ] && python3 "$SCRIPT_DIR/k3_arm_verdict.py" agree "$RESULTS_DIR" "$cells"
fi
exit "$fail"
