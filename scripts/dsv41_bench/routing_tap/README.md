# Routing tap for the DSv4.1 server-step harness

Records, from a real `run.py` decode start:

- every fused-EP MoE layer's global top-k expert ids, per steady decode step and per rank (the 40 target layers at 288
  rows, and the DSpark draft layers at 240 rows);
- every indexer-owning attention layer's CSA top-512 selection for the first 8 steady steps of each window;
- each worker's NUMA placement (all host tensors ≥ 1 MiB, Engram staging included: VMAs and pages per node, CPU
  affinity, allowed nodes).

Nothing under `src/` changes, and the harness files stay unmodified. `sitecustomize.py` in this directory loads
`routing_tap.py` when `DSV41_ROUTING_TAP_OUT` is set. The original `src/sitecustomize.py` is still run. The tap installs
post-import patches on `VllmDeepseekV4Mxfp4MoEMethod`, `DeepseekV41Attention` and `server_step_worker.ServerStepWorker`.
See the module docstring for how it handles the torch_tpu bridge: donated rings written back with `copy_`, ops built at
weight load, and no shape branch on the torch side.

```
dsv41-bench decode --out <run>-routing-tap --rounds 1 --profile \
  --env PYTHONPATH=/workspace/vllm-torchtpu/scripts/dsv41_bench/routing_tap:/workspace/vllm-torchtpu/src:/workspace/vllm-torchtpu \
  --env DSV41_ROUTING_TAP_OUT=$HOME/dsv41-work/runs/<run>-routing-tap/tap
python convert.py ~/dsv41-work/runs/<run>-routing-tap <out> --label round1    # and --label profile
```

The image's own PYTHONPATH (`/workspace/vllm-torchtpu/src:/workspace/vllm-torchtpu`) has to be repeated, because
`--env` replaces it. Dumps are written while the server is paused: `rank-<r>-round<n>.npz` (warmup = round0),
`rank-<r>-profile.npz`, and `rank-<r>-numa-round<n>.json`. The tap adds device work to every layer (a recomputed top-k per
MoE layer, ring writes), so a tapped start is **not a timing arm**.

Knobs: `DSV41_ROUTING_TAP_RING` (MoE steps kept, default 192), `DSV41_ROUTING_TAP_ROWS` (`experts:rows`, default
`384:288,128:240`), `DSV41_ROUTING_TAP_TOPK_RING` (8). Tests: `dsv41-docker --cpu python -m pytest
scripts/dsv41_bench/routing_tap -q`.

The ids are recomputed from the logits and the staged score bias with `lax.top_k`. They can differ from the kernel's
own selector only on exact f32 ties.
