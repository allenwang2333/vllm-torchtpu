# P/D Disaggregation on Kubernetes: Running and Testing It

This page is for developers who need to run prefill/decode (P/D) disaggregation
on GKE or add a test for it. For the serving flags, connectors and single-host
quickstart, see [Disaggregated Serving](../user_guide/disaggregated_serving.md).

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

### A complete example

`example-1p1d-smoke.yaml` is a whole 1P1D test: Qwen3-0.6B on one 4-chip v7x
host per engine, with `TPUConnector`, and a CPU client that sends a short
benchmark through the toy proxy and fails the run if any request fails. It
follows the conventions of the files in
`.buildkite/kubernetes/manifests/workloads/`: the launcher fills in
`${WORKLOAD_NAME}` and `${IMAGE}`, and a literal `$` is written `$$`.

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  annotations:
    # CI only: the launcher merges in the fleet's caches, service account and
    # node-repair retries. Without the launcher it is an inert annotation.
    tpu-ci.google.com/defaults: standard
spec:
  network:
    enableDNSHostnames: true            # <jobset>-<role>-<job>-<pod>.<jobset>
  successPolicy:
    operator: All
    targetReplicatedJobs:
      - client                          # the client finishing ends the run
  # No failurePolicy: any child Job failing fails the whole JobSet.
  replicatedJobs:
    - name: prefill
      replicas: 1
      template:
        spec:
          parallelism: 1
          completions: 1
          completionMode: Indexed       # enableDNSHostnames requires it
          backoffLimit: 0
          activeDeadlineSeconds: 7200
          template:
            spec:
              restartPolicy: Never
              nodeSelector:             # with the chip count, this picks the Kueue queue
                cloud.google.com/gke-tpu-accelerator: tpu7x
                cloud.google.com/gke-tpu-topology: 2x2x1
              tolerations:
                - key: google.com/tpu
                  operator: Exists
                  effect: NoSchedule
              containers:
                - name: workload        # the launcher streams logs from this name
                  image: ${IMAGE}
                  command:
                    - vllm
                    - serve
                    - Qwen/Qwen3-0.6B
                    - --port=8000
                    - --tensor-parallel-size=4
                    - --max-model-len=2048
                    - --attention-backend=CUSTOM
                    - --kv-transfer-config={"kv_connector":"TPUConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer"}
                  env:
                    - name: VLLM_HOST_IP        # the address decode pulls KV from
                      valueFrom:
                        fieldRef:
                          fieldPath: status.podIP
                    - name: TPU_KV_TRANSFER_PORT
                      value: "9100"
                    - name: TPU_SIDE_CHANNEL_PORT
                      value: "9600"
                  ports:
                    - containerPort: 8000
                  readinessProbe:               # what the client's dependsOn waits on
                    httpGet:
                      path: /health
                      port: 8000
                    periodSeconds: 10
                  resources:                    # chips only; no CPU request
                    limits:
                      google.com/tpu: "4"
                    requests:
                      google.com/tpu: "4"
    - name: decode
      replicas: 1
      template:
        spec:
          parallelism: 1
          completions: 1
          completionMode: Indexed
          backoffLimit: 0
          activeDeadlineSeconds: 7200
          template:
            spec:
              restartPolicy: Never
              nodeSelector:
                cloud.google.com/gke-tpu-accelerator: tpu7x
                cloud.google.com/gke-tpu-topology: 2x2x1
              tolerations:
                - key: google.com/tpu
                  operator: Exists
                  effect: NoSchedule
              containers:
                - name: workload
                  image: ${IMAGE}
                  command:
                    - vllm
                    - serve
                    - Qwen/Qwen3-0.6B
                    - --port=8000
                    - --tensor-parallel-size=4
                    - --max-model-len=2048
                    - --attention-backend=CUSTOM
                    - --kv-transfer-config={"kv_connector":"TPUConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer"}
                  env:
                    - name: VLLM_HOST_IP
                      valueFrom:
                        fieldRef:
                          fieldPath: status.podIP
                    - name: TPU_KV_TRANSFER_PORT
                      value: "9100"
                    - name: TPU_SIDE_CHANNEL_PORT
                      value: "9600"
                  ports:
                    - containerPort: 8000
                  readinessProbe:
                    httpGet:
                      path: /health
                      port: 8000
                    periodSeconds: 10
                  resources:
                    limits:
                      google.com/tpu: "4"
                    requests:
                      google.com/tpu: "4"
    - name: client
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
          completionMode: Indexed
          backoffLimit: 0
          activeDeadlineSeconds: 3600
          template:
            spec:
              restartPolicy: Never
              containers:
                - name: workload
                  image: ${IMAGE}
                  env:
                    - name: PREFILL_HOST
                      value: ${WORKLOAD_NAME}-prefill-0-0.${WORKLOAD_NAME}
                    - name: DECODE_HOST
                      value: ${WORKLOAD_NAME}-decode-0-0.${WORKLOAD_NAME}
                    # A CPU pod: keep vLLM's CLI from loading the TPU platform.
                    - name: VLLM_PLUGINS
                      value: ""
                    - name: TORCH_DEVICE_BACKEND_AUTOLOAD
                      value: "0"
                  command: ["/bin/bash", "-c"]
                  args:
                    - |
                      set -euo pipefail
                      python /workspace/vllm-torchtpu/examples/disagg/toy_proxy_server.py \
                        --host 127.0.0.1 --port 10000 \
                        --prefiller-hosts "$$PREFILL_HOST" --prefiller-ports 8000 \
                        --decoder-hosts "$$DECODE_HOST" --decoder-ports 8000 &
                      until (exec 3<>/dev/tcp/127.0.0.1/10000) 2>/dev/null; do sleep 1; done
                      vllm bench serve \
                        --model Qwen/Qwen3-0.6B --host 127.0.0.1 --port 10000 \
                        --dataset-name random --random-input-len 512 --random-output-len 64 \
                        --ignore-eos --num-prompts 32 --max-concurrency 8 | tee /tmp/bench.txt
                      if ! grep -Eq '^Failed requests: +0 *$$' /tmp/bench.txt; then
                        echo "ERROR: requests failed through the P/D path" >&2
                        exit 1
                      fi
                  resources:
                    requests:
                      cpu: "4"
                      memory: 8Gi
                    limits:
                      cpu: "4"
                      memory: 8Gi
```

### Multi-host engines

When one engine needs a multi-host slice, the engine role becomes one Job with
a pod per host. Change these lines from the example above:

```yaml
metadata:
  annotations:
    # Two slices never share a node pool.
    alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
spec:
  replicatedJobs:
    - name: prefill
      groupName: prefill      # a group per role; see below
      template:
        spec:
          parallelism: 4      # the hosts of a 2x2x4 slice
          completions: 4
          template:
            spec:
              nodeSelector:
                cloud.google.com/gke-tpu-topology: 2x2x4
```

Host `i` of the prefill engine is `<jobset>-prefill-0-<i>.<jobset>`, and index
0 leads. `kimi-k3-agentx-1p1d.yaml` is the full multi-host version.

### Things that bit us

- **Give each multi-host engine its own `groupName`.** GKE derives
  `MEGASCALE_NUM_SLICES` from the JobSet group, so two multi-host engines in
  the default group each wait for a second slice and hang in libtpu init.
- **Both engines ask for the same shape.** A Kueue workload is admitted against
  one queue, and a queue is one TPU shape. A 4-chip prefill with a 16-chip
  decode can't be a single workload today.
- **A pod knows itself from the downward API** (`status.podIP`) and its peers
  from the DNS names above. Don't use a Service selector; it would match any
  overlapping run.
- **Warm the Hugging Face modules cache in each pod** before starting Ray for
  remote-code models. Hosts racing on an empty cache fail to import the
  tokenizer.
- **Put `VLLM_CACHE_ROOT` and `VLLM_XLA_CACHE_PATH` on the shared cache mount**
  for large models. The default sits on the node's boot disk, so every run
  recompiles.
- **Don't add a CPU request** to a TPU role. Set `activeDeadlineSeconds` when a
  cold compile can outlast the fleet default.

## How it runs in Buildkite

The step that runs the example, in `.buildkite/pipeline_integration_kube.yml`:

```yaml
  - label: ":kubernetes: Integration: example-1p1d-smoke"
    key: "integration_example_1p1d_smoke"
    depends_on: "build_docker"
    timeout_in_minutes: 180
    agents:
      queue: "kube"
    plugins:
      - *kube_plugin
    commands:
      - .buildkite/kubernetes/run_multihost.sh --manifest example-1p1d-smoke.yaml
```

What happens when it runs:

1. Buildkite hands the step to the `kube` queue. The agent stack starts a
   launcher pod in the manager cluster; it holds no chips.
2. `run_multihost.sh` finds `manifests/workloads/example-1p1d-smoke.yaml`.
   `common.sh` picks the image `build_docker` pushed and the step environment
   to forward, and the script calls the launcher.
3. The launcher substitutes `${WORKLOAD_NAME}` and `${IMAGE}` and merges the
   fleet defaults the annotation names. It works out the Kueue queue from the
   `nodeSelector` and chip count (here the v7x 2x2x1 queue, 8 chips in total),
   sets the priority class from the build (post-merge, PR or nightly), and
   submits the JobSet.
4. Kueue admits the whole JobSet once 8 chips are free, borrowing idle chips
   from other shapes if it can. The step log shows `waiting for quota` until
   then. MultiKueue hands the JobSet to the v7x worker cluster.
5. GKE adds two TPU nodes and the pods start. The client waits until both
   engines pass their readiness probes.
6. The launcher streams every pod's `workload` container into the step log as
   one stream with a role gutter: `prefill | ...`, `decode | ...`,
   `client | ...`.
7. The client exits 0, the JobSet succeeds, JobSet stops the engines, and the
   launcher exits 0: the step is green. If any pod fails, the JobSet fails and
   the step is red, with the container's termination reason in the log. A pod
   that loses its node to a repair or preemption is retried by the fleet
   defaults rather than failing the step.

### Running the same manifest without Buildkite

On any GKE cluster with JobSet, and Kueue if you use a queue, do the
launcher's part by hand: substitute the two names, turn `$$` back into `$`,
then name and label the JobSet.

```bash
export WORKLOAD_NAME=pd-smoke-$RANDOM IMAGE=<a vllm-torchtpu image> LOCAL_QUEUE=<your LocalQueue>
python3 - example-1p1d-smoke.yaml <<'EOF' | kubectl apply -f -
import os, string, sys, yaml
doc = yaml.safe_load(string.Template(open(sys.argv[1]).read()).substitute(os.environ))
doc["metadata"]["name"] = os.environ["WORKLOAD_NAME"]
doc["metadata"].setdefault("labels", {})["kueue.x-k8s.io/queue-name"] = os.environ["LOCAL_QUEUE"]
print(yaml.safe_dump(doc))
EOF
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
3. **Multi-pod JobSet**, as above. Examples: `qwen3.5-p8d8-disagg.yaml` (two
   8-device hosts over the network), `dsv4-pd-disagg.yaml` (nightly), and the
   Kimi-K3 1P1D runs on two 16-chip slices in the perf/eval lane.

The example's check (no failed requests) is the floor. The real correctness
scripts also check:

- **The launch used the intended layout.** For example, grep the server logs
  for the block size and the parallel layout each engine resolved.
- **Answers through the router are right.** Smoke prompts with known outputs,
  prefix-cache probes, or an eval such as GSM8K for the large models.
- **The disaggregated path really ran.** Counter deltas from `/metrics`, such
  as prefix-cache hits or MTP acceptance when speculative decoding is on.
- **A failure says what was expected.** Use `exit 1` with the expected value,
  not a bare non-zero exit.

The same script runs on VMs and in pods. The role comes from `ROLE` and peers
from the pod DNS names, so a manifest only wires those up.

### Adding a P/D test

1. Write `scripts/vllm/integration/run_<model>_<layout>_disagg_correctness.sh`
   so it runs by hand on a TPU host first.
2. If both engines fit on one host, add a step to
   `pipeline_integration_kube.yml` that calls it through `run.sh` with the
   right `SHAPE`.
3. Otherwise, start from the example above or the nearest manifest in
   `manifests/workloads/`, and add a step that calls
   `run_multihost.sh --manifest`.
4. Gate cases that take more than about an hour of TPU time on `NIGHTLY`.

## Honorable mention: LeaderWorkerSet and the wider OSS stack

For long-running serving rather than tests, the open-source default for
multi-host engines is [LeaderWorkerSet](https://lws.sigs.k8s.io/) (LWS). A
group (one leader and N workers) is one engine replica that scales, restarts
and rolls out as a unit, and you run one LWS for the prefill pool and one for
the decode pool. llm-d builds its P/D deployments this way: Deployments for
single-host engines, LWS for multi-host ones. Its inference scheduler (the
Gateway API Inference Extension) does P/D-aware routing in place of a
standalone proxy. Kueue has an LWS integration too, so the queueing model
carries over.

We test with JobSet because a test has to end and report a result, and JobSet
has that built in (`successPolicy`, a client role, one gang-admitted object).
The engines, connector, router protocol and correctness scripts don't depend
on which of the two controllers starts the pods.
