# P/D Disaggregation on Kubernetes: Running and Testing It

This page is for developers who need to run prefill/decode (P/D) disaggregation
on GKE or add a test for it. It walks through the Kimi-K3 1P1D pair that runs
every night. For the serving flags, connectors and single-host quickstart, see
[Disaggregated Serving](../user_guide/disaggregated_serving.md).

## The pieces

| Piece | What it does | In the repo |
| :--- | :--- | :--- |
| Prefill engine | `vllm serve` with `"kv_role": "kv_producer"`; computes the prompt and hands its KV cache off | — |
| Decode engine | `vllm serve` with `"kv_role": "kv_consumer"`; pulls the KV cache and generates | — |
| KV connector | Moves KV blocks between the two. `TPURaidenConnector` handles multi-host engines and different P and D layouts (resharding); `TPUConnector` is the simple path. Both live in `vllm_torchtpu.distributed.kv_transfer.tpu_connector` | `src/vllm_torchtpu/distributed/kv_transfer/` |
| Raiden controller | Coordinates pool registration for the controller-based Raiden layouts in the user guide. The CI manifests below don't need it | `examples/disagg/run_raiden_controller.py` |
| Router | Sends each request to prefill with `max_tokens=1`, then to decode with the `kv_transfer_params` prefill returned | `examples/disagg/toy_proxy_server.py`, or vllm-router |

Every client goes through the router. The engines never see a request that
didn't go through those two hops.

## Running it with JobSet (the validated path)

CI runs every multi-pod P/D test as one [JobSet](https://jobset.sigs.k8s.io/),
submitted through Kueue by the Buildkite launcher. JobSet fits a test because a
test is a finite run:

- **One object for all roles.** Kueue admits prefill, decode and the client
  together or not at all, so no engine sits on chips while its peer waits.
- **Stable names.** With `enableDNSHostnames`, every pod is reachable as
  `<jobset>-<role>-<job index>-<pod index>.<jobset>`.
- **A defined end.** `successPolicy` on the client role ends the run. JobSet
  then tears down the engines, which would otherwise serve forever.
- **Readiness gating.** `dependsOn: Ready` holds the client until both
  engines answer `/health`, not just until their containers start.

### The Kimi-K3 1P1D pair

`.buildkite/kubernetes/manifests/workloads/kimi-k3-agentx-1p1d.yaml` has three
roles:

| Role | Hardware | Runs | Does |
| :--- | :--- | :--- | :--- |
| `prefill` | One 2x2x4 v7x slice: 4 hosts, 16 chips | `.buildkite/kubernetes/k3_1p1d.sh prefill` on each host | Serves Kimi-K3 at TP8 x DP4 x EP32, one DP replica per host. Host 0 serves the API; the other three join it headless |
| `decode` | Another 2x2x4 slice | `k3_1p1d.sh decode` | The same layout as `kv_consumer` |
| `client` | A CPU node, no chips | `.buildkite/kubernetes/k3_agentx_client_pod.sh` | Finds both heads by DNS, waits for `/health`, starts the proxy, then runs either the AgentX trace replay or the GSM8K eval through it, and uploads the results |

Both engines boot the serve config `kimi-k3-tp8-dp4-ep-agentx` (`K3_PD_CONFIG`),
under `scripts/vllm/benchmarking/configs/`. It holds each role's serve flags
and Raiden environment. The client is a pod of its own because the replay is
the measurement; on an engine host it would load the machine it is timing.

The full manifest, copied from the repo at f91671598 (Oct 14). The file in the
repo is the source of truth. The launcher substitutes `${WORKLOAD_NAME}`,
`${IMAGE}`, `${K3_ENGINE_IMAGE}` and `${K3_FLEET_CACHES}`; `common.sh`
defaults the last two.

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  annotations:
    tpu-ci.google.com/defaults: standard
    # The fleet's caches: `none` for the megakernel points, empty (kept) for
    # the rest (common.sh). A megakernel boot streams the TP32 pre-shard once,
    # and through the fleet's /cache/hf/hub the shards fill a 65Gi gcsfuse file
    # cache, which is host memory: 61-67 GiB on every engine host in
    # vllm-torchtpu-dev builds 562, 588 and 589, enough to get build 562's
    # prefill head evicted under a 1.5 TiB host KV store. Without the caches
    # the points also compile from scratch every boot, into the container's
    # disk. Quoted, so that empty is an empty string rather than null.
    tpu-ci.google.com/caches: "${K3_FLEET_CACHES}"
    alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
spec:
  # <jobset>-<role>-0-<index> under a headless service named for the JobSet,
  # which is how each host finds its role's hosts and prefill finds decode.
  network:
    enableDNSHostnames: true
  # Both engines serve until told to stop, so the client finishing is what
  # finishes the run; JobSet then tears the engines down.
  successPolicy:
    operator: All
    targetReplicatedJobs:
      - client
  replicatedJobs:
    # A group of its own per role: GKE's TPU webhook derives MEGASCALE_NUM_SLICES
    # from the group, and two roles sharing one would each wait in libtpu for a
    # second slice that never joins.
    - name: prefill
      groupName: prefill
      replicas: 1
      template:
        spec:
          parallelism: 4
          completions: 4
          # Indexed because enableDNSHostnames requires it.
          completionMode: Indexed
          # Over the fleet's default: two cold K3 boots stream 1.5 TB each and
          # compile before the client's first point.
          activeDeadlineSeconds: 28800
          backoffLimit: 0
          template:
            spec:
              nodeSelector:
                cloud.google.com/gke-tpu-topology: 2x2x4
                cloud.google.com/gke-tpu-accelerator: tpu7x
              # This commit's pair scripts and configs, for the engine image.
              initContainers:
                - name: lane-scripts
                  image: ${IMAGE}
                  command: ["/bin/sh", "-c", "cp -a /workspace/vllm-torchtpu/.buildkite/kubernetes /lane/kubernetes && cp -a /workspace/vllm-torchtpu/scripts/vllm/benchmarking/configs /lane/configs"]
                  volumeMounts:
                    - name: lane-scripts
                      mountPath: /lane
              containers:
                - name: workload
                  image: ${K3_ENGINE_IMAGE}
                  command: ["/bin/bash", "/workspace/vllm-torchtpu/.buildkite/kubernetes/k3_1p1d.sh", "prefill"]
                  env:
                    # Host i of this role is <prefix>i.<domain>.
                    - name: ROLE_HOST_PREFIX
                      value: ${WORKLOAD_NAME}-prefill-0-
                    - name: ROLE_HOST_DOMAIN
                      value: ${WORKLOAD_NAME}
                    - name: DECODE_HOST
                      value: ${WORKLOAD_NAME}-decode-0-0.${WORKLOAD_NAME}
                    - name: PREFILL_HOST
                      value: ${WORKLOAD_NAME}-prefill-0-0.${WORKLOAD_NAME}
                    - name: NUM_HOSTS
                      value: "4"
                    # Both roles and the client read this one config.
                    - name: K3_PD_CONFIG
                      value: kimi-k3-tp8-dp4-ep-agentx
                    # Serve only; the client pod drives the run.
                    - name: K3_PD_EXTERNAL_CLIENT
                      value: "1"
                    - name: K3_ENGINE_IMAGE
                      value: ${K3_ENGINE_IMAGE}
                    # The downward API, so it cannot come from the script.
                    - name: POD_IP
                      valueFrom:
                        fieldRef:
                          fieldPath: status.podIP
                    - name: TPU_ACCELERATOR_TYPE
                      value: tpu7x
                    # As the v7x-16 perf steps: the boot's temporary files are
                    # tens of GB, more than the container's disk.
                    - name: TMPDIR
                      value: /dev/shm
                    - name: VLLM_XLA_CACHE_PATH
                      value: /cache/jax/xla
                    - name: VLLM_CACHE_ROOT
                      value: /cache/jax/vllm
                  ports:
                    - containerPort: 8000
                  # Only the chips: memory comes from the shape's profile, and a
                  # cpu request can stop the pool scaling up at all.
                  resources:
                    limits:
                      google.com/tpu: "4"
                    requests:
                      google.com/tpu: "4"
                  volumeMounts:
                    # profile_cache must not persist, or a later run loads the
                    # kv=0 graph instead of compiling it.
                    - name: vllm-profile-cache
                      mountPath: /cache/jax/vllm/profile_cache
                    - name: lane-scripts
                      mountPath: /workspace/vllm-torchtpu/.buildkite/kubernetes
                      subPath: kubernetes
                    - name: lane-scripts
                      mountPath: /workspace/vllm-torchtpu/scripts/vllm/benchmarking/configs
                      subPath: configs
                    - name: k3-checkpoint
                      mountPath: /ckpt/k3
                      readOnly: true
                    # Inside the fleet's /cache/hf/hub when the pod has it,
                    # which the launcher mounts ahead of these.
                    - name: k3-presharded
                      mountPath: /cache/hf/hub/presharded/kimi-k3-tp32
                      readOnly: true
                    - name: k3-dspark
                      mountPath: /cache/hf/hub/models--RedHatAI--Kimi-K3-speculator.dspark
                      readOnly: true
              volumes:
                - name: vllm-profile-cache
                  emptyDir:
                    medium: Memory
                    sizeLimit: 16Gi
                - name: lane-scripts
                  emptyDir: {}
                # The checkpoint's files (gs://tpu-commons-ci/moonshootai/kimi/k3),
                # at the path the TP32 pre-shard was written from.
                - name: k3-checkpoint
                  csi:
                    driver: gcsfuse.csi.storage.gke.io
                    readOnly: true
                    volumeAttributes:
                      bucketName: tpu-commons-ci
                      mountOptions: "implicit-dirs,only-dir=moonshootai/kimi/k3"
                      skipCSIBucketAccessCheck: "true"
                # What the megakernel configs read from the model bucket the
                # fleet mounts at /cache/hf/hub (us-central1's, the only cluster
                # with 2x2x4 pools), each mounted on its own: the TP32 pre-shard,
                # with the DSpark kernel shards, and the DSpark speculator, of
                # which they read only the config. Their own mounts because
                # without the fleet's caches nothing else puts them at those
                # paths, and with them the pre-shard would fill the fleet
                # mount's file cache. With no fileCacheCapacity these have no
                # file cache; read_ahead_kb is the fleet mount's, for the
                # loader's large sequential reads.
                - name: k3-presharded
                  csi:
                    driver: gcsfuse.csi.storage.gke.io
                    readOnly: true
                    volumeAttributes:
                      bucketName: tpu-ci-models-60b4c3c7
                      mountOptions: "implicit-dirs,only-dir=presharded/kimi-k3-tp32,read_ahead_kb=131072"
                      skipCSIBucketAccessCheck: "true"
                - name: k3-dspark
                  csi:
                    driver: gcsfuse.csi.storage.gke.io
                    readOnly: true
                    volumeAttributes:
                      bucketName: tpu-ci-models-60b4c3c7
                      mountOptions: "implicit-dirs,only-dir=models--RedHatAI--Kimi-K3-speculator.dspark"
                      skipCSIBucketAccessCheck: "true"

    - name: decode
      groupName: decode
      replicas: 1
      template:
        spec:
          parallelism: 4
          completions: 4
          completionMode: Indexed
          activeDeadlineSeconds: 28800
          backoffLimit: 0
          template:
            spec:
              nodeSelector:
                cloud.google.com/gke-tpu-topology: 2x2x4
                cloud.google.com/gke-tpu-accelerator: tpu7x
              # This commit's pair scripts and configs, for the engine image.
              initContainers:
                - name: lane-scripts
                  image: ${IMAGE}
                  command: ["/bin/sh", "-c", "cp -a /workspace/vllm-torchtpu/.buildkite/kubernetes /lane/kubernetes && cp -a /workspace/vllm-torchtpu/scripts/vllm/benchmarking/configs /lane/configs"]
                  volumeMounts:
                    - name: lane-scripts
                      mountPath: /lane
              containers:
                - name: workload
                  image: ${K3_ENGINE_IMAGE}
                  command: ["/bin/bash", "/workspace/vllm-torchtpu/.buildkite/kubernetes/k3_1p1d.sh", "decode"]
                  env:
                    # Host i of this role is <prefix>i.<domain>.
                    - name: ROLE_HOST_PREFIX
                      value: ${WORKLOAD_NAME}-decode-0-
                    - name: ROLE_HOST_DOMAIN
                      value: ${WORKLOAD_NAME}
                    - name: DECODE_HOST
                      value: ${WORKLOAD_NAME}-decode-0-0.${WORKLOAD_NAME}
                    - name: PREFILL_HOST
                      value: ${WORKLOAD_NAME}-prefill-0-0.${WORKLOAD_NAME}
                    - name: NUM_HOSTS
                      value: "4"
                    # Both roles and the client read this one config.
                    - name: K3_PD_CONFIG
                      value: kimi-k3-tp8-dp4-ep-agentx
                    # Serve only; the client pod drives the run.
                    - name: K3_PD_EXTERNAL_CLIENT
                      value: "1"
                    - name: K3_ENGINE_IMAGE
                      value: ${K3_ENGINE_IMAGE}
                    - name: POD_IP
                      valueFrom:
                        fieldRef:
                          fieldPath: status.podIP
                    - name: TPU_ACCELERATOR_TYPE
                      value: tpu7x
                    - name: TMPDIR
                      value: /dev/shm
                    - name: VLLM_XLA_CACHE_PATH
                      value: /cache/jax/xla
                    - name: VLLM_CACHE_ROOT
                      value: /cache/jax/vllm
                  ports:
                    - containerPort: 8000
                  resources:
                    limits:
                      google.com/tpu: "4"
                    requests:
                      google.com/tpu: "4"
                  volumeMounts:
                    - name: vllm-profile-cache
                      mountPath: /cache/jax/vllm/profile_cache
                    - name: lane-scripts
                      mountPath: /workspace/vllm-torchtpu/.buildkite/kubernetes
                      subPath: kubernetes
                    - name: lane-scripts
                      mountPath: /workspace/vllm-torchtpu/scripts/vllm/benchmarking/configs
                      subPath: configs
                    - name: k3-checkpoint
                      mountPath: /ckpt/k3
                      readOnly: true
                    - name: k3-presharded
                      mountPath: /cache/hf/hub/presharded/kimi-k3-tp32
                      readOnly: true
                    - name: k3-dspark
                      mountPath: /cache/hf/hub/models--RedHatAI--Kimi-K3-speculator.dspark
                      readOnly: true
              volumes:
                - name: vllm-profile-cache
                  emptyDir:
                    medium: Memory
                    sizeLimit: 16Gi
                - name: lane-scripts
                  emptyDir: {}
                # The checkpoint's files (gs://tpu-commons-ci/moonshootai/kimi/k3),
                # at the path the TP32 pre-shard was written from.
                - name: k3-checkpoint
                  csi:
                    driver: gcsfuse.csi.storage.gke.io
                    readOnly: true
                    volumeAttributes:
                      bucketName: tpu-commons-ci
                      mountOptions: "implicit-dirs,only-dir=moonshootai/kimi/k3"
                      skipCSIBucketAccessCheck: "true"
                # The pre-shard and the speculator, as prefill's.
                - name: k3-presharded
                  csi:
                    driver: gcsfuse.csi.storage.gke.io
                    readOnly: true
                    volumeAttributes:
                      bucketName: tpu-ci-models-60b4c3c7
                      mountOptions: "implicit-dirs,only-dir=presharded/kimi-k3-tp32,read_ahead_kb=131072"
                      skipCSIBucketAccessCheck: "true"
                - name: k3-dspark
                  csi:
                    driver: gcsfuse.csi.storage.gke.io
                    readOnly: true
                    volumeAttributes:
                      bucketName: tpu-ci-models-60b4c3c7
                      mountOptions: "implicit-dirs,only-dir=models--RedHatAI--Kimi-K3-speculator.dspark"
                      skipCSIBucketAccessCheck: "true"

    # Chipless, so it is scheduled by its requests onto a worker-cpu node rather
    # than onto a slice, as qwen3.5-397b-1p1d.yaml's benchmark role is.
    - name: client
      # Both engines: the proxy fronts the pair, so both must answer /health.
      dependsOn:
        - name: prefill
          status: Ready
        - name: decode
          status: Ready
      replicas: 1
      template:
        spec:
          parallelism: 1
          completions: 1
          # Longer than the engines': it waits out both boots, then runs the
          # ladder.
          activeDeadlineSeconds: 32400
          completionMode: Indexed
          backoffLimit: 0
          template:
            spec:
              restartPolicy: Never
              nodeSelector:
                cloud.google.com/compute-class: worker-cpu
              serviceAccountName: tpu-workload
              containers:
                - name: workload
                  image: ${IMAGE}
                  command: ["/bin/bash", "/workspace/vllm-torchtpu/.buildkite/kubernetes/k3_agentx_client_pod.sh"]
                  env:
                    # Each engine head is <name>-<role>-0-0.<name>.
                    - name: WORKLOAD_NAME
                      value: ${WORKLOAD_NAME}
                    # The same config the engines booted; read here for the cases.
                    - name: K3_PD_CONFIG
                      value: kimi-k3-tp8-dp4-ep-agentx
                    # Physical chips the pair holds: 4 hosts x google.com/tpu 4
                    # per slice, twice. What is billed per chip-hour, and so what
                    # tokens per dollar is costed over - not the 64 ranks on them.
                    - name: AGENTX_CHIPS
                      value: "32"
                    # This pod's results are the step's only output.
                    - name: ARTIFACTS_DIR
                      value: perf_eval_results
                    # On a chipless pod the torchtpu plugin makes vLLM's CLI
                    # call torch.device("tpu"), and `import torch` auto-imports
                    # torch_tpu, whose native init aborts without libtpu.
                    - name: VLLM_PLUGINS
                      value: ""
                    - name: TORCH_DEVICE_BACKEND_AUTOLOAD
                      value: "0"
                    - name: PYTHONUNBUFFERED
                      value: "1"
                    - name: TMPDIR
                      value: /tmp
                  # Guaranteed (request == limit): it shares a worker-cpu node.
                  # The memory is aiperf tokenizing 100k-token traces; the disk
                  # is the corpus and the InferenceX checkout.
                  resources:
                    requests:
                      cpu: "12"
                      memory: "32Gi"
                      ephemeral-storage: "30Gi"
                    limits:
                      cpu: "12"
                      memory: "32Gi"
                      ephemeral-storage: "30Gi"
```

### Single-host engines

When each engine fits on one host, the shape is simpler. Drop the
`exclusive-topology` annotation and the `groupName`s, and use `parallelism: 1`,
`completions: 1` and the single-host topology (for v7x, `2x2x1`) on each
engine role. `qwen3.5-397b-1p1d.yaml` is the reference.

### Things that bit us

- **Give each multi-host engine its own `groupName`.** GKE derives
  `MEGASCALE_NUM_SLICES` from the JobSet group, so two multi-host engines in
  the default group each wait for a second slice and hang in libtpu init.
- **Use `exclusive-topology` on the node pool** so each multi-host role gets a
  slice of its own instead of both being split across the two.
- **Both engines ask for the same shape.** A Kueue workload is admitted against
  one queue, and a queue is one TPU shape. A 4-chip prefill with a 16-chip
  decode can't be a single workload today.
- **A pod knows itself from the downward API** (`status.podIP`) and its peers
  from the DNS names above. Don't use a Service selector; it would match any
  overlapping run.
- **Warm the Hugging Face modules cache in each pod** before starting Ray for
  remote-code models. Hosts racing on an empty cache fail to import the
  tokenizer.
- **Put `VLLM_CACHE_ROOT` and `VLLM_XLA_CACHE_PATH` on the shared cache mount.**
  The default sits on the node's boot disk, so every run recompiles.
- **Don't add a CPU request** to a TPU role. Set `activeDeadlineSeconds` when a
  cold compile can outlast the fleet default.

## How it runs in Buildkite

The nightly GSM8K eval of this pair, in `.buildkite/pipeline_perf_kube.yml`:

```yaml
      - label: ":kubernetes: :tpu: Eval: Kimi-K3 AgentX GSM8K 1P1D (TPU v7x 2x2x4 x2)"
        depends_on:
          - "build_docker"
        concurrency_group: "tpu/v7x-2x2x4"
        concurrency: 1
        if: 'build.env("NIGHTLY") == "1" || build.env("K3_AGENTX_EVAL_ONLY") == "1"'
        timeout_in_minutes: 1440
        agents:
          queue: "kube"
        plugins:
          - *kube_plugin
        env:
          AGENTX_EVAL_JOB: "1"
        commands:
          - .buildkite/kubernetes/run_multihost.sh --manifest kimi-k3-agentx-1p1d.yaml
```

The throughput step beside it runs the same manifest without
`AGENTX_EVAL_JOB`, so its client replays the AgentX traces instead. Both share
the `tpu/v7x-2x2x4` concurrency group, because one pair takes both 2x2x4
slices of the queue's nominal quota.

What happens when the step runs:

1. Buildkite hands the step to the `kube` queue. The agent stack starts a
   launcher pod in the manager cluster; it holds no chips.
2. `run_multihost.sh` finds `manifests/workloads/kimi-k3-agentx-1p1d.yaml`.
   `common.sh` resolves the image `build_docker` pushed (and `K3_ENGINE_IMAGE`
   and `K3_FLEET_CACHES`) and the step environment to forward, and the script
   calls the launcher.
3. The launcher substitutes the variables and merges the fleet defaults the
   annotations name. It works out the Kueue queue from the `nodeSelector` and
   chip count (the v7x 2x2x4 queue, 32 chips for both slices), sets the
   priority class from the build (post-merge, PR or nightly), and submits the
   JobSet.
4. Kueue admits the whole JobSet once both slices are free. The step log
   shows `waiting for quota` until then. MultiKueue hands the JobSet to the
   v7x worker cluster.
5. GKE brings up the eight TPU hosts and the pods start. Each engine's hosts
   find each other by DNS and form the engine; the client waits until both
   heads pass their readiness probes.
6. The launcher streams every pod's `workload` container into the step log as
   one stream with a role gutter (`prefill-0-1 | ...`, `decode-0-0 | ...`,
   `client-0-0 | ...`).
7. The client runs GSM8K through the proxy against the threshold, uploads the
   results as step artifacts, and exits 0. The JobSet succeeds, JobSet stops
   the engines, and the launcher exits 0: the step is green. If any pod fails,
   the JobSet fails and the step is red, with the container's termination
   reason in the log. A pod that loses its node to a repair or preemption is
   retried by the fleet defaults rather than failing the step.

To run it outside a nightly, start a `vllm-torchtpu-ci` build with
`K3_AGENTX_EVAL_ONLY=1`, which the step's `if:` admits. Add `CI_LANES=perf` to
skip the other lanes.

### Running a manifest without Buildkite

On a GKE cluster with JobSet, and Kueue if you use a queue, do the launcher's
part by hand: substitute the variables, turn `$$` back into `$`, then name and
label the JobSet. The K3 pair also needs what the fleet defaults supply (the
checkpoint buckets, the `worker-cpu` compute class, the `tpu-workload` service
account), so this is mostly useful for your own manifests.

```bash
export WORKLOAD_NAME=pd-test-$RANDOM IMAGE=<a vllm-torchtpu image> LOCAL_QUEUE=<your LocalQueue>
python3 - my-pd-workload.yaml <<'PY' | kubectl apply -f -
import os, string, sys, yaml
doc = yaml.safe_load(string.Template(open(sys.argv[1]).read()).substitute(os.environ))
doc["metadata"]["name"] = os.environ["WORKLOAD_NAME"]
doc["metadata"].setdefault("labels", {})["kueue.x-k8s.io/queue-name"] = os.environ["LOCAL_QUEUE"]
print(yaml.safe_dump(doc))
PY
kubectl get jobset "$WORKLOAD_NAME" -w
kubectl logs -f -l "jobset.sigs.k8s.io/jobset-name=$WORKLOAD_NAME,jobset.sigs.k8s.io/replicatedjob-name=client"
```

## How we test it

There are three layers, from cheapest to most expensive.

1. **CPU unit tests**, no TPU. They cover connector, metadata, resharding and
   router logic: `tests/distributed/test_tpu_connector.py`,
   `tests/distributed/test_raiden_*.py`, `tests/distributed/test_pd_reshard_oracle.py`
   and `tests/disagg/`. `test_vllm_router_pd.py` checks with stub engines that
   vllm-router speaks the same two-hop protocol as the toy proxy. These run in
   `Unit Tests - CPU`. The `*_tpu.py` files are marked `tpu_test`, need a chip,
   and run in the TPU unit-test steps instead.
2. **Single-pod correctness.** Both engines run on one TPU host, each with its
   own chips. The step runs
   `.buildkite/kubernetes/run.sh bash scripts/vllm/integration/run_<case>_disagg_correctness.sh`;
   examples are `qwen35-pcp4-dp4-disagg-correctness` and
   `Presubmit: disagg-single-host` in `.buildkite/pipeline_integration_kube.yml`.
3. **Multi-pod JobSet**, as above. Besides the AgentX pair:
   `kimi-k3-1p1d-4dp8tp.yaml` (the same K3 layout, where the prefill head runs
   the greedy checks, correctness stages and MMLU-Pro itself, so there is no
   client role), `qwen3.5-p8d8-disagg.yaml` (two 8-device hosts over the
   network) and `dsv4-pd-disagg.yaml` (nightly).

A correctness check verifies more than "it answered":

- **The launch used the intended layout.** For example, grep the server logs
  for the block size and the parallel layout each engine resolved.
- **Answers through the router are right.** Greedy prompts with known outputs,
  prefix-cache probes, or an eval against a threshold, such as GSM8K here.
- **The disaggregated path really ran.** Counter deltas from `/metrics`, such
  as prefix-cache hits or MTP acceptance when speculative decoding is on.
- **A failure says what was expected.** Use `exit 1` with the expected value,
  not a bare non-zero exit.

The same scripts run on VMs and in pods. The role comes from the script's
argument or `ROLE`, and peers from the pod DNS names, so a manifest only wires
those up.

### Adding a P/D test

1. Write the check as a script that runs by hand on TPU hosts first, under
   `scripts/vllm/integration/` or next to the K3 scripts in
   `.buildkite/kubernetes/`.
2. If both engines fit on one host, add a step to
   `pipeline_integration_kube.yml` that calls it through `run.sh` with the
   right `SHAPE`.
3. Otherwise, start from the nearest manifest in `manifests/workloads/` and
   add a step that calls `run_multihost.sh --manifest`.
4. Gate cases that take more than about an hour of TPU time on `NIGHTLY`.

## Honorable mention: LeaderWorkerSet and the wider OSS stack

For long-running serving rather than tests, the open-source default for
multi-host engines is [LeaderWorkerSet](https://lws.sigs.k8s.io/) (LWS). A
group (one leader and N workers) is one engine replica that scales, restarts
and rolls out as a unit, and you run one LWS for the prefill pool and one for
the decode pool. The K3 pair maps onto it directly: each engine role, with
host 0 serving the API and the others headless, is one LWS group. llm-d builds
its P/D deployments this way: Deployments for single-host engines, LWS for
multi-host ones. Its inference scheduler (the Gateway API Inference Extension)
does P/D-aware routing in place of a standalone proxy. Kueue has an LWS
integration too, so the queueing model carries over.

We test with JobSet because a test has to end and report a result, and JobSet
has that built in (`successPolicy`, a client role, one gang-admitted object).
The engines, connector, router protocol and correctness scripts don't depend
on which of the two controllers starts the pods.
