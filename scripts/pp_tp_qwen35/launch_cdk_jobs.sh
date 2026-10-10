#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Helper script to launch Qwen3.5-397B-A17B-FP8 PP+TP CDK jobs on shriharim-cdk-vm.
# Always run as: sudo su - shriharim_google_com
# Usage:
#   bash scripts/pp_tp_qwen35/launch_cdk_jobs.sh <job_name> [--execute]
# By default, prints the exact cdk command (--dry-run).

set -euo pipefail

JOB_NAME="${1:-help}"
MODE="${2:---dry-run}"

PROJECT="cloud-tpu-inference-test"
ZONE="us-central1-c"
DOCKER_IMAGE="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-dev@sha256:58530632039b112295db7ba6c44f423684bdd1283dd58d57d53e72fd1f469851"
MODEL_ID="Qwen/Qwen3.5-397B-A17B-FP8"
REPO_DIR="/home/shriharim_google_com/vllm-torchtpu-pp-tp"
CDK_DIR="/home/shriharim_google_com/cloud-devkit"

COMMON_ENV="TPU_QWEN35_AUTO_DEFAULTS=0 TPU_EXPERT_PLACEMENT= TPU_SMALL_ALL_REDUCE=0 TPU_MOE_LOWTOK=0 TPU_MOE_SEQUENCE_PARALLEL=0 VLLM_DISABLE_COMPILE_CACHE=1 VLLM_LOGGING_LEVEL=INFO"
SP_EP_ENV="TPU_QWEN35_AUTO_DEFAULTS=0 TPU_EXPERT_PLACEMENT= TPU_SMALL_ALL_REDUCE=0 TPU_MOE_LOWTOK=0 TPU_MOE_SEQUENCE_PARALLEL=1 TPU_SP_MAX_ROWS=2048 USE_MOE_FUSED_EP_KERNEL=1 MOE_FUSED_EP_KERNEL_IMPL=v3 VLLM_DISABLE_COMPILE_CACHE=1 VLLM_LOGGING_LEVEL=INFO"
TWO_CHIP_ENV="${COMMON_ENV} VLLM_TPU_DEVICE_ORDER=0,2,1,3,4,6,5,7"

run_or_print() {
  echo "================================================================================"
  echo "[Command]"
  printf '%q ' "$@"
  echo ""
  echo "================================================================================"
  if [[ "${MODE}" == "--execute" ]]; then
    local out rc
    set +e
    out=$("$@" 2>&1)
    rc=$?
    set -e
    echo "${out}"
    if [[ "${rc}" -ne 0 ]]; then
      return "${rc}"
    fi
    local job_id
    job_id=$(echo "${out}" | grep -oE 'j-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}' | tail -n 1)
    if [[ -n "${job_id}" && "${KUEUE_BUMP_PRIORITY:-0}" == "1" ]]; then
      bump_kueue_priority "${job_id}"
    fi
  else
    echo "(Dry run only. Pass --execute to submit the CDK job.)"
  fi
}

# Team runbook for multi-node jobs: cdk-cluster-queue is BestEffortFIFO with
# LowerPriority preemption, so a 2-node job at the per-user fair-share priority
# (cdk-p1 = 400) starves behind 1-node backfill and is preempted by every new
# cdk-p0 job. The CDK webhook blocks JobSet/Job/Pod edits but not Kueue
# Workloads; patch the Workload to cdk-p0 (500) as soon as it exists. CDK's
# job-manager re-syncs it to cdk-p1 within seconds, but the window is enough
# for Kueue to reserve quota and admit; an admitted job is not un-admitted.
bump_kueue_priority() {
  local job_id="$1" w=""
  for _ in $(seq 1 24); do
    w=$(kubectl get workloads -n cloud-devkit-jobs --no-headers 2>/dev/null | awk -v j="${job_id}" '$1 ~ j {print $1}' | head -n 1)
    [[ -n "${w}" ]] && break
    sleep 5
  done
  if [[ -z "${w}" ]]; then
    echo "[kueue] no Workload for ${job_id} after 120s; skipping priority bump"
    return 0
  fi
  for _ in $(seq 1 6); do
    kubectl patch workload "${w}" -n cloud-devkit-jobs --type=merge \
      -p '{"spec":{"priority":500,"priorityClassRef":{"group":"kueue.x-k8s.io","kind":"WorkloadPriorityClass","name":"cdk-p0"}}}' >/dev/null 2>&1 || true
    sleep 3
    local st
    st=$(kubectl get workload "${w}" -n cloud-devkit-jobs -o jsonpath='{.spec.priority} admitted={.status.conditions[?(@.type=="Admitted")].status}' 2>/dev/null)
    echo "[kueue] ${w}: ${st}"
    [[ "${st}" == *"admitted=True"* ]] && return 0
  done
  return 0
}

cd "${CDK_DIR}"

case "${JOB_NAME}" in
  ut-1node)
    # 1. TPU unit test + 8-core PP4xTP2 / PP4xPCP2 wave & mesh verification
    run_or_print cdk job create \
      simple-run-command \
      DOCKER_IMAGE="${DOCKER_IMAGE}" \
      RUN_CMD="cp -r /workspace/vllm-torchtpu/src/vllm_torchtpu /root/torchtpu-vllm/src/ && PYTHONPATH=/workspace/vllm-torchtpu/src pytest /workspace/vllm-torchtpu/tests/distributed/test_ep_mesh.py /workspace/vllm-torchtpu/tests/distributed/test_pcp_collectives.py /workspace/vllm-torchtpu/tests/distributed/test_pp_wave.py /workspace/vllm-torchtpu/tests/platforms/test_pp_validation.py /workspace/vllm-torchtpu/tests/platforms/test_tpu_platform.py /workspace/vllm-torchtpu/tests/layers/adapter/test_small_all_reduce_op.py -v" \
      --map-dir "${REPO_DIR}:*:/workspace/vllm-torchtpu" \
      --tags "qwen35-pp-tp2-tpu-unit-tests,qwen35-pp-tp,agent-jobs"
    ;;

  pp4-tp2-1node)
    # 2a. Single-host PP4 x TP2 Pure TP (8 cores, 1 megachip / 2 cores per stage, no --ep) - GSM8K + prefill/serving sweep
    run_or_print cdk job create \
      vllm-torchtpu-pp-tp-two-node-mp-bench \
      NUM_NODES=1 \
      TPU_TOPOLOGY=2x2x1 \
      DOCKER_IMAGE="${DOCKER_IMAGE}" \
      MODEL_ID="${MODEL_ID}" \
      ENV_VARS="${COMMON_ENV} USE_MOE_FUSED_EP_KERNEL=0" \
      SERVER_ARGS="--pipeline-parallel-size 4 --tensor-parallel-size 2 --prefill-context-parallel-size 1 --no-async-scheduling --max-model-len 36864 --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --quantization fp8 --no-enable-prefix-caching" \
      BENCH_SWEEP="8192,1,1,8;8192,1,8,32;8192,1,16,64;8192,1,32,128;16384,1,1,8;16384,1,8,32;16384,1,16,64;32768,1,1,8;32768,1,8,32;1024,1024,1,8;1024,1024,8,32;1024,1024,32,128;8192,1024,1,8;8192,1024,8,32;8192,1024,16,64" \
      RUN_EVAL="true" \
      EVAL_TASK="gsm8k" \
      EVAL_ARGS="--num_fewshot 5" \
      --map-dir "${REPO_DIR}:*:/workspace/vllm-torchtpu" \
      --tags "qwen35-397b-fp8-pp4xtp2-pure-tp-1-host-gsm8k-sweep,qwen35-pp-tp,agent-jobs"
    ;;

  pp4-tp2-ep2-1node)
    # 2b. Single-host PP4 x TP2 + EP2 (8 cores, 1 megachip / 2 cores per stage, --enable-expert-parallel, SP=1 + Fused EP v3) - GSM8K + prefill/serving sweep
    run_or_print cdk job create \
      vllm-torchtpu-pp-tp-two-node-mp-bench \
      NUM_NODES=1 \
      TPU_TOPOLOGY=2x2x1 \
      DOCKER_IMAGE="${DOCKER_IMAGE}" \
      MODEL_ID="${MODEL_ID}" \
      ENV_VARS="${SP_EP_ENV}" \
      SERVER_ARGS="--pipeline-parallel-size 4 --tensor-parallel-size 2 --prefill-context-parallel-size 1 --enable-expert-parallel --no-async-scheduling --max-model-len 36864 --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --quantization fp8 --no-enable-prefix-caching" \
      BENCH_SWEEP="8192,1,1,8;8192,1,8,32;8192,1,16,64;8192,1,32,128;16384,1,1,8;16384,1,8,32;16384,1,16,64;32768,1,1,8;32768,1,8,32;1024,1024,1,8;1024,1024,8,32;1024,1024,32,128;8192,1024,1,8;8192,1024,8,32;8192,1024,16,64" \
      RUN_EVAL="true" \
      EVAL_TASK="gsm8k" \
      EVAL_ARGS="--num_fewshot 5" \
      --map-dir "${REPO_DIR}:*:/workspace/vllm-torchtpu" \
      --tags "qwen35-397b-fp8-pp4xtp2-ep2-sp-fused-v3-1-host-gsm8k-sweep,qwen35-pp-tp,agent-jobs"
    ;;

  pp8-1node)
    # 3. Single-host PP8 baseline (8 cores, 1 core per stage) - GSM8K + prefill/serving sweep
    run_or_print cdk job create \
      vllm-torchtpu-pp-tp-two-node-mp-bench \
      NUM_NODES=1 \
      TPU_TOPOLOGY=2x2x1 \
      DOCKER_IMAGE="${DOCKER_IMAGE}" \
      MODEL_ID="${MODEL_ID}" \
      ENV_VARS="${COMMON_ENV} USE_MOE_FUSED_EP_KERNEL=0" \
      SERVER_ARGS="--pipeline-parallel-size 8 --tensor-parallel-size 1 --prefill-context-parallel-size 1 --no-async-scheduling --max-model-len 36864 --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --quantization fp8 --no-enable-prefix-caching" \
      BENCH_SWEEP="8192,1,1,8;8192,1,8,32;8192,1,16,64;8192,1,32,128;16384,1,1,8;16384,1,8,32;16384,1,16,64;32768,1,1,8;32768,1,8,32;1024,1024,1,8;1024,1024,8,32;1024,1024,32,128;8192,1024,1,8;8192,1024,8,32;8192,1024,16,64" \
      RUN_EVAL="true" \
      EVAL_TASK="gsm8k" \
      EVAL_ARGS="--num_fewshot 5" \
      --map-dir "${REPO_DIR}:*:/workspace/vllm-torchtpu" \
      --tags "qwen35-397b-fp8-pp8-baseline-1-host-gsm8k-sweep,qwen35-pp-tp,agent-jobs"
    ;;

  pp4-pcp2-1node)
    # 4. Single-host PP4 x PCP2 baseline (8 cores, 2 chips per stage) - GSM8K + prefill/serving sweep
    run_or_print cdk job create \
      vllm-torchtpu-pp-tp-two-node-mp-bench \
      NUM_NODES=1 \
      TPU_TOPOLOGY=2x2x1 \
      DOCKER_IMAGE="${DOCKER_IMAGE}" \
      MODEL_ID="${MODEL_ID}" \
      ENV_VARS="${TWO_CHIP_ENV} USE_MOE_FUSED_EP_KERNEL=0" \
      SERVER_ARGS="--pipeline-parallel-size 4 --tensor-parallel-size 1 --prefill-context-parallel-size 2 --enable-expert-parallel --no-async-scheduling --max-model-len 36864 --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --quantization fp8 --no-enable-prefix-caching" \
      BENCH_SWEEP="8192,1,1,8;8192,1,8,32;8192,1,16,64;8192,1,32,128;16384,1,1,8;16384,1,8,32;16384,1,16,64;32768,1,1,8;32768,1,8,32;1024,1024,1,8;1024,1024,8,32;1024,1024,32,128;8192,1024,1,8;8192,1024,8,32;8192,1024,16,64" \
      RUN_EVAL="true" \
      EVAL_TASK="gsm8k" \
      EVAL_ARGS="--num_fewshot 5" \
      --map-dir "${REPO_DIR}:*:/workspace/vllm-torchtpu" \
      --tags "qwen35-397b-fp8-pp4xpcp2-baseline-1-host-gsm8k-sweep,qwen35-pp-tp,agent-jobs"
    ;;

  tp8-1node)
    # 5. Single-host TP8 EP baseline (8 cores, 1 stage) - GSM8K + prefill/serving sweep
    run_or_print cdk job create \
      vllm-torchtpu-pp-tp-two-node-mp-bench \
      NUM_NODES=1 \
      TPU_TOPOLOGY=2x2x1 \
      DOCKER_IMAGE="${DOCKER_IMAGE}" \
      MODEL_ID="${MODEL_ID}" \
      TORCH_TPU_WHEEL="NONE" \
      LIBTPU_WHEEL="NONE" \
      ENV_VARS="TPU_QWEN35_AUTO_DEFAULTS=0 TPU_EXPERT_PLACEMENT= USE_MOE_FUSED_EP_KERNEL=1 MOE_FUSED_EP_KERNEL_IMPL=v3 VLLM_DISABLE_COMPILE_CACHE=1 VLLM_LOGGING_LEVEL=INFO" \
      SERVER_ARGS="--tensor-parallel-size 8 --enable-expert-parallel --no-async-scheduling --max-model-len 36864 --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --quantization fp8 --no-enable-prefix-caching" \
      BENCH_SWEEP="8192,1,1,8;8192,1,8,32;8192,1,16,64;8192,1,32,128;16384,1,1,8;16384,1,8,32;16384,1,16,64;32768,1,1,8;32768,1,8,32;1024,1024,1,8;1024,1024,8,32;1024,1024,32,128;8192,1024,1,8;8192,1024,8,32;8192,1024,16,64" \
      RUN_EVAL="true" \
      EVAL_TASK="gsm8k" \
      EVAL_ARGS="--num_fewshot 5" \
      --map-dir "${REPO_DIR}:*:/workspace/vllm-torchtpu" \
      --tags "qwen35-397b-fp8-tp8-ep-baseline-1-host-gsm8k-sweep,qwen35-pp-tp,agent-jobs"
    ;;

  pp8-tp2-2node)
    # 6a. Multi-host PP8 x TP2 Pure TP (16 cores across 2 hosts, 1 megachip / 2 cores per stage, no --ep) - GSM8K + prefill/serving sweep
    run_or_print cdk job create \
      vllm-torchtpu-pp-tp-two-node-mp-bench \
      DOCKER_IMAGE="${DOCKER_IMAGE}" \
      MODEL_ID="${MODEL_ID}" \
      ENV_VARS="${COMMON_ENV} USE_MOE_FUSED_EP_KERNEL=0" \
      SERVER_ARGS="--pipeline-parallel-size 8 --tensor-parallel-size 2 --prefill-context-parallel-size 1 --no-async-scheduling --max-model-len 65536 --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --quantization fp8 --no-enable-prefix-caching" \
      BENCH_SWEEP="8192,1,1,8;8192,1,8,32;8192,1,16,64;8192,1,32,128;16384,1,1,8;16384,1,8,32;16384,1,16,64;32768,1,1,8;32768,1,8,32;64000,1,1,8;64000,1,4,16;1024,1024,1,8;1024,1024,8,32;1024,1024,32,128;8192,1024,1,8;8192,1024,8,32;8192,1024,16,64" \
      RUN_EVAL="true" \
      EVAL_TASK="gsm8k" \
      EVAL_ARGS="--num_fewshot 5" \
      --map-dir "${REPO_DIR}:*:/workspace/vllm-torchtpu" \
      --tags "qwen35-397b-fp8-pp8xtp2-pure-tp-2-host-2x2x2-gsm8k-sweep,qwen35-pp-tp,agent-jobs"
    ;;

  pp8-tp2-ep2-2node)
    # 6b. Multi-host PP8 x TP2 + EP2 (16 cores across 2 hosts, 1 megachip / 2 cores per stage, --enable-expert-parallel, SP=1 + Fused EP v3) - GSM8K + prefill/serving sweep
    run_or_print cdk job create \
      vllm-torchtpu-pp-tp-two-node-mp-bench \
      DOCKER_IMAGE="${DOCKER_IMAGE}" \
      MODEL_ID="${MODEL_ID}" \
      ENV_VARS="${SP_EP_ENV}" \
      SERVER_ARGS="--pipeline-parallel-size 8 --tensor-parallel-size 2 --prefill-context-parallel-size 1 --enable-expert-parallel --no-async-scheduling --max-model-len 65536 --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --quantization fp8 --no-enable-prefix-caching" \
      BENCH_SWEEP="8192,1,1,8;8192,1,8,32;8192,1,16,64;8192,1,32,128;16384,1,1,8;16384,1,8,32;16384,1,16,64;32768,1,1,8;32768,1,8,32;64000,1,1,8;64000,1,4,16;1024,1024,1,8;1024,1024,8,32;1024,1024,32,128;8192,1024,1,8;8192,1024,8,32;8192,1024,16,64" \
      RUN_EVAL="true" \
      EVAL_TASK="gsm8k" \
      EVAL_ARGS="--num_fewshot 5" \
      --map-dir "${REPO_DIR}:*:/workspace/vllm-torchtpu" \
      --tags "qwen35-397b-fp8-pp8xtp2-ep2-sp-fused-v3-2-host-2x2x2-gsm8k-sweep,qwen35-pp-tp,agent-jobs"
    ;;

  pp8-pcp2-2node)
    # 7. Multi-host PP8 x PCP2 baseline (16 cores across 2 hosts, 2 chips per stage) - GSM8K + prefill/serving sweep
    run_or_print cdk job create \
      vllm-torchtpu-pp-tp-two-node-mp-bench \
      DOCKER_IMAGE="${DOCKER_IMAGE}" \
      MODEL_ID="${MODEL_ID}" \
      ENV_VARS="${TWO_CHIP_ENV} USE_MOE_FUSED_EP_KERNEL=0" \
      SERVER_ARGS="--pipeline-parallel-size 8 --tensor-parallel-size 1 --prefill-context-parallel-size 2 --enable-expert-parallel --no-async-scheduling --max-model-len 65536 --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --quantization fp8 --no-enable-prefix-caching" \
      BENCH_SWEEP="8192,1,1,8;8192,1,8,32;8192,1,16,64;8192,1,32,128;16384,1,1,8;16384,1,8,32;16384,1,16,64;32768,1,1,8;32768,1,8,32;64000,1,1,8;64000,1,4,16;1024,1024,1,8;1024,1024,8,32;1024,1024,32,128;8192,1024,1,8;8192,1024,8,32;8192,1024,16,64" \
      RUN_EVAL="true" \
      EVAL_TASK="gsm8k" \
      EVAL_ARGS="--num_fewshot 5" \
      --map-dir "${REPO_DIR}:*:/workspace/vllm-torchtpu" \
      --tags "qwen35-397b-fp8-pp8xpcp2-baseline-2-host-2x2x2-gsm8k-sweep,qwen35-pp-tp,agent-jobs"
    ;;

  xprof-pp4-tp2-ep2-1node)
    # 8a. XProf hardware trace capture on 1-host PP4 x TP2 + EP2 (Regimes A, B1, B2, C)
    run_or_print cdk job create \
      vllm-torchtpu-pp-tp-two-node-mp-bench \
      NUM_NODES=1 \
      TPU_TOPOLOGY=2x2x1 \
      DOCKER_IMAGE="${DOCKER_IMAGE}" \
      MODEL_ID="${MODEL_ID}" \
      ENABLE_XPROF=1 \
      ENV_VARS="${SP_EP_ENV}" \
      SERVER_ARGS="--pipeline-parallel-size 4 --tensor-parallel-size 2 --prefill-context-parallel-size 1 --enable-expert-parallel --no-async-scheduling --max-model-len 36864 --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --quantization fp8 --no-enable-prefix-caching" \
      BENCH_SWEEP="8192,1,1,4;8192,1,16,32;16384,1,16,32;1024,128,32,32" \
      RUN_EVAL="false" \
      --map-dir "${REPO_DIR}:*:/workspace/vllm-torchtpu" \
      --tags "qwen35-397b-fp8-xprof-pp4xtp2-ep2-sp-fused-v3-1-host,qwen35-pp-tp,agent-jobs"
    ;;

  xprof-pp4-tp2-1node)
    # 8b. XProf hardware trace capture on 1-host PP4 x TP2 Pure-TP (Regimes A, B1, B2, C)
    run_or_print cdk job create \
      vllm-torchtpu-pp-tp-two-node-mp-bench \
      NUM_NODES=1 \
      TPU_TOPOLOGY=2x2x1 \
      DOCKER_IMAGE="${DOCKER_IMAGE}" \
      MODEL_ID="${MODEL_ID}" \
      ENABLE_XPROF=1 \
      ENV_VARS="${COMMON_ENV} USE_MOE_FUSED_EP_KERNEL=0" \
      SERVER_ARGS="--pipeline-parallel-size 4 --tensor-parallel-size 2 --prefill-context-parallel-size 1 --no-async-scheduling --max-model-len 36864 --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --quantization fp8 --no-enable-prefix-caching" \
      BENCH_SWEEP="8192,1,1,4;8192,1,16,32;16384,1,16,32;1024,128,32,32" \
      RUN_EVAL="false" \
      --map-dir "${REPO_DIR}:*:/workspace/vllm-torchtpu" \
      --tags "qwen35-397b-fp8-xprof-pp4xtp2-pure-tp-1-host,qwen35-pp-tp,agent-jobs"
    ;;

  xprof-pp8-1node)
    # 8c. XProf hardware trace capture on 1-host PP8 baseline (Regimes A, B1, B2, C)
    run_or_print cdk job create \
      vllm-torchtpu-pp-tp-two-node-mp-bench \
      NUM_NODES=1 \
      TPU_TOPOLOGY=2x2x1 \
      DOCKER_IMAGE="${DOCKER_IMAGE}" \
      MODEL_ID="${MODEL_ID}" \
      ENABLE_XPROF=1 \
      ENV_VARS="${COMMON_ENV} USE_MOE_FUSED_EP_KERNEL=0" \
      SERVER_ARGS="--pipeline-parallel-size 8 --tensor-parallel-size 1 --prefill-context-parallel-size 1 --no-async-scheduling --max-model-len 36864 --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --quantization fp8 --no-enable-prefix-caching" \
      BENCH_SWEEP="8192,1,1,4;8192,1,16,32;16384,1,16,32;1024,128,32,32" \
      RUN_EVAL="false" \
      --map-dir "${REPO_DIR}:*:/workspace/vllm-torchtpu" \
      --tags "qwen35-397b-fp8-xprof-pp8-baseline-1-host,qwen35-pp-tp,agent-jobs"
    ;;

  *)
    echo "Available jobs:"
    echo "  ut-1node                - TPU unit tests (2x2x1)"
    echo "  pp4-tp2-1node           - PP4 x TP2 Pure-TP on 1 host (2x2x1): GSM8K + prefill & serving sweep"
    echo "  pp4-tp2-ep2-1node       - PP4 x TP2 + EP2 on 1 host (2x2x1): GSM8K + prefill & serving sweep"
    echo "  pp8-1node               - PP8 baseline on 1 host (2x2x1): GSM8K + prefill & serving sweep"
    echo "  pp4-pcp2-1node          - PP4 x PCP2 baseline on 1 host (2x2x1): GSM8K + prefill & serving sweep"
    echo "  tp8-1node               - TP8 EP baseline on 1 host (2x2x1): GSM8K + prefill & serving sweep"
    echo "  pp8-tp2-2node           - PP8 x TP2 Pure-TP on 2 hosts (2x2x2): GSM8K + prefill & serving sweep"
    echo "  pp8-tp2-ep2-2node       - PP8 x TP2 + EP2 on 2 hosts (2x2x2): GSM8K + prefill & serving sweep"
    echo "  pp8-pcp2-2node          - PP8 x PCP2 baseline on 2 hosts (2x2x2): GSM8K + prefill & serving sweep"
    echo "  xprof-pp4-tp2-ep2-1node - XProf trace capture on PP4 x TP2 + EP2 (2x2x1)"
    echo "  xprof-pp4-tp2-1node     - XProf trace capture on PP4 x TP2 Pure-TP (2x2x1)"
    echo "  xprof-pp8-1node         - XProf trace capture on PP8 baseline (2x2x1)"
    ;;
esac
