# P/D Disaggregation on Kubernetes: Running and Testing It

This page is for developers who need to run prefill/decode (P/D) disaggregation
on GKE or add a test for it. For the serving flags, connectors and single-host
quickstart, see [Disaggregated Serving](../user_guide/disaggregated_serving.md).

## The pieces

| Piece | What it does | In the repo |
| :--- | :--- | :--- |
| Prefill engine | `vllm serve` with `"kv_role": "kv_producer"`; computes the prompt and hands its KV cache off | — |
| Decode engine | `vllm serve` with `"kv_role": "kv_consumer"`; pulls the KV cache and generates | — |
| KV connector | Moves KV blocks between the two. Raiden handles multi-host engines and different P and D layouts (resharding); `TPUConnector` is the simple single-host path | `src/vllm_torchtpu/distributed/kv_transfer/` |
| Raiden controller | Registers both pools and coordinates transfers (Raiden only) | `examples/disagg/run_raiden_controller.py` |
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

A trimmed manifest (the real ones are in
`.buildkite/kubernetes/manifests/workloads/`):

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  annotations:
    tpu-ci.google.com/defaults: standard        # caches, retries, tolerations
    # Multi-host engines only: one slice per node pool.
    alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
spec:
  network:
    enableDNSHostnames: true
  successPolicy:
    operator: All
    targetReplicatedJobs: [client]
  replicatedJobs:
  - name: prefill
    groupName: prefill          # multi-host engines: one group per role
    template:
      spec:
        parallelism: 4          # hosts in the slice; 1 for a single-host engine
        completions: 4
        completionMode: Indexed
        template:
          spec:
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x
              cloud.google.com/gke-tpu-topology: 2x2x4
            containers:
            - name: workload
              command: ["bash", ".buildkite/kubernetes/disagg_engine.sh", "prefill"]
              readinessProbe: {httpGet: {path: /health, port: 8000}}
              resources: {limits: {google.com/tpu: "4"}, requests: {google.com/tpu: "4"}}
  - name: decode
    groupName: decode
    # ... same shape, "decode"
  - name: client                # holds no chips; runs the router and the checks
    dependsOn:
    - {name: prefill, status: Ready}
    - {name: decode, status: Ready}
    # ...
```

Things that bit us, so a new manifest doesn't repeat them:

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
- **Put `VLLM_CACHE_ROOT` and `VLLM_XLA_CACHE_PATH` on the shared cache mount.**
  The default sits on the node's boot disk, so every run recompiles.
- **Don't add a CPU request** to a TPU role. Set `activeDeadlineSeconds` when a
  cold compile can outlast the fleet default.

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
3. **Multi-pod JobSet.** The step runs
   `.buildkite/kubernetes/run_multihost.sh --manifest <name>.yaml`. Examples:
   `qwen3.5-p8d8-disagg.yaml` (two 8-device hosts over the network),
   `dsv4-pd-disagg.yaml` (nightly), and the Kimi-K3 1P1D runs on two 16-chip
   slices in the perf/eval lane.

A correctness script checks more than "it answered":

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
3. Otherwise, copy the nearest manifest in `manifests/workloads/` and add a
   step that calls `run_multihost.sh --manifest`.
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
