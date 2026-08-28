# Running DeepSeek-V4 prefill/decode disaggregation

`launch_dsv4_dp4_dp4.sh` brings up a DSv4 prefill replica, a DSv4 decode
replica and a proxy in front of them, all on one 8-chip host, and moves KV
between them over Raiden. Prefill runs DP4 on chips 0-3 and decode runs DP4 on
chips 4-7. The assignment's step 1 is DP8 on both sides, which needs 16 chips
across two hosts; this is that configuration halved so it fits one host, with
the same sharding on both replicas and transfer parallelism 1.

## Before you start

You need an 8-chip TPU host with nothing else holding the chips, a
DeepSeek-V4-Flash snapshot on local disk, and a virtualenv with this repo's
dependencies. The script does not activate the venv for you, because it starts
the servers through `bash -lc` whose `PATH` may not carry it, so pass the
interpreter explicitly.

`MODEL_PATH` defaults to the DSv4-Flash snapshot under `/mnt/disk` on the dev
box. Override it anywhere else.

## Launch

```bash
PYTHON=/path/to/venv/bin/python ./examples/disagg/launch_dsv4_dp4_dp4.sh
```

The script prints a `RUN_DIR` and exits only after both replicas answer
`/health` with 200 and a smoke probe has gone through the proxy. Everything it
produces lands under `~/pd_disagg_runs/dsv4_pd_dp4_dp4_<timestamp>/`, and
`~/pd_disagg_runs/latest_dsv4_pd_dp4dp4` symlinks to the most recent run. Inside
you get `logs/prefill.log`, `logs/decode.log`, `logs/proxy.log`, the exact
commands used in `prefill_cmd.sh` and `decode_cmd.sh`, and the resolved settings
in `launch_params.txt`.

Startup takes several minutes, most of it compilation. The health poll waits up
to 30 minutes and tails the tracebacks into `logs/launch_err.log` if either
replica dies first.

## Send it a request

The proxy listens on port 8000 and speaks the OpenAI API:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "deepseek-ai/DeepSeek-V4-Flash",
       "prompt": "The capital of France is",
       "max_tokens": 32, "temperature": 0}'
```

Every request goes to prefill first, then its KV is pulled to decode over
Raiden, then decode generates. A reply with sensible text means the whole path
worked, since decode has no prefilled KV of its own to fall back on.

## Checking that Raiden really carried the KV

There is no log line announcing a successful pool manifest, so the signal is
the absence of failure. DSv4 admission is fail-closed: `build_dsv4_pool_manifest`
raises `ManifestError` rather than degrade, and that kills worker startup. If
both replicas reach 200 on `/health`, the manifest was accepted and every
attention and compressor-state group got a pool. If the geometry is wrong the
transfer itself fails loudly too, with `strided KV pull failed` in the decode
log.

The smoke probe the script runs at the end
(`scripts/vllm/integration/smoke_prefix_cache_correctness.py --quick-probe-only`)
is the cheapest correctness check. Run it again by hand against the proxy port
whenever you change something.

## Tearing down

The servers are started with `setsid`, so they survive the launching shell.
Re-running the script is the normal way to clean up: with `RESTART_EXISTING=1`,
the default, it kills whatever holds the managed ports and then sweeps leftover
`VLLM::` workers, which hold the chips without listening anywhere. Set
`RESTART_EXISTING=0` to make it refuse to start rather than kill anything.

To stop without relaunching, kill the pids in `prefill.pid`, `decode.pid` and
`proxy.pid`, then check `pgrep -u "$(id -u)" -f VLLM::` is empty before the next
run.

## Knobs worth knowing

`MAX_MODEL_LEN` is 9216 and `MAX_NUM_SEQS` is 8, both small enough to leave room
on one host. `GPU_MEMORY_UTILIZATION` is 0.92. Raising the first two or lowering
the last is the usual way to get more concurrency, at the risk of an OOM during
warmup.

Block size is deliberately not pinned. DSv4 pages hold compressed rows and the
platform picks the MLA page size for the architecture; overriding it
desynchronizes the pool geometry from the layout fingerprint and the transfer is
rejected.

`TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL` must stay 0, unlike the Qwen setups. The
unified pool gives pools an `aliased_raw` binding, and DSv4 admission requires
private typed tensors, so the manifest would be rejected. It also crashes
earlier, in `update_tpu_block_size_and_slot_config`, which asks the DSv4 SWA
backend for a page size it does not implement.

## Common failures

`PjRtClient is not initialized` means a previous attempt left workers holding
the chips. Re-run with the default `RESTART_EXISTING=1`, or clear them yourself
with `pkill -9 -u "$(id -u)" -f VLLM::` followed by
`rm -f /tmp/libtpu_lockfile`.

A model that loads but produces garbage usually means `VLLM_PLUGINS` lost
`torchtpu_layers`. Without it vLLM silently loads its CUDA DSv4 model, which
dies on `torch.cuda.Stream`.

`Port NNNN is busy` with `RESTART_EXISTING=0` lists the offending pids. The
script manages two ports per DP rank per role for the reshard store, a service
port and a dispatch port, so a stale run of either replica blocks a fresh one.
