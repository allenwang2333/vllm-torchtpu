# Two Shot serving campaign

The serving client runs measurements against an existing OpenAI-compatible K3
server. The launcher owns allocation, Ray workers, fresh server processes,
separate compile caches, startup failures and execution evidence. The client
never starts a server or initializes a TPU runtime.

The current candidate serving path requires synchronous scheduling and
`max_num_seqs=1`. Configurations permitting overlapping invocations use native
reduction, even when an operator shape has qualified. Keep the canonical ABBA
configuration identical across arms; if it is outside this support contract,
report that both arms executed native reduction and make no Two Shot serving
speedup claim. A separate serial configuration can investigate this qualification
gap, but it does not replace the canonical comparison.

## Fixed inputs

With the first baseline server ready, prepare the inputs once:

```bash
python scripts/collectives/serving_client.py prepare \
  --base-url http://127.0.0.1:8000 --model moonshotai/Kimi-K3 \
  --output /path/to/campaign/inputs.json
```

This downloads `test.jsonl` from the official OpenAI grade-school-math repository,
checks the full 1319-row test split, and selects its first 100 questions. Pass
`--dataset /path/to/test.jsonl` to use an already downloaded official copy.
The cache records the dataset SHA256, original questions, normalized gold answers,
chat messages and exact server-tokenized input IDs. GSM8K uses the server's chat
template with `thinking_effort=low` and an explicit final `####` numeric answer.

Performance inputs are deterministic varied text tokenized to exactly 1 or 8192
tokens. Each length has 10 measured prompts and eight separate warmup prompts.
All four starts must reuse this one immutable cache. `prepare` refuses to replace
an existing file. Tokenization and dataset preparation are outside timing.

## ABBA execution

Start fresh servers in this order, using identical model assets, runtime,
parallelism, memory limits, prefix-cache settings and serving flags:

| Slot | Arm | Accuracy |
| ---: | --- | --- |
| 0 | Baseline | 100 questions |
| 1 | Candidate | Same 100 questions |
| 2 | Candidate | Performance repeat |
| 3 | Baseline | Performance repeat |

Use distinct compile-cache directories for all four starts. For each start,
write a server metadata JSON receipt with these required string fields:

```json
{
  "arm": "baseline",
  "start_id": "baseline-0",
  "source_commit": "actual source commit",
  "compile_cache": "/path/to/campaign/cache-baseline-0"
}
```

Also retain the full launch configuration, model/tokenizer identities, environment,
server and worker logs, qualification manifest hash, and observed execution/bucket
evidence in the receipt or referenced artifacts. The client checks receipt arm
labels and distinct start/cache identities; the launcher must establish that the
recorded settings describe the actual server and that the expected path ran.

Run the client once against each new server:

```bash
python scripts/collectives/serving_client.py run \
  --base-url http://127.0.0.1:8000 --model moonshotai/Kimi-K3 \
  --inputs /path/to/campaign/inputs.json \
  --server-metadata /path/to/campaign/baseline-0-server.json \
  --worker-evidence-dir /path/to/campaign/baseline-0-worker-evidence \
  --slot 0 --accuracy --output /path/to/campaign/baseline-0
```

For slots 1, 2 and 3, change the slot, receipt and output directory. Include
`--accuracy` for slot 1 and omit it for slots 2 and 3. The output path must be fresh;
preserve failed starts and requests in their original directories.

Each run measures ISL `{1,8192}` × concurrency `{1,4}`, with exactly ten measured
requests per cell after two separate warmup waves. Performance completions use
OSL 1024, greedy sampling, seed zero and `ignore_eos=true`. GSM8K uses the same
sampling with EOS enabled, concurrency four and a maximum of 1024 output tokens.
Change `--quality-output-tokens` only before the campaign and keep it identical in
all four client invocations. Accuracy truncations and missing final answers are
reported; they are never silently retried or removed.

## Artifacts and comparison

The client flushes every request to `requests.jsonl`, including failures and
separately labeled warmups. Successful streams must finish, report token IDs and
usage, and agree with cached input lengths and observed output token counts.
Client-observed TTFT is time to the first token-bearing event. TPOT uses the first
and last token-bearing events divided by output tokens minus one. Multi-token
chunks are counted explicitly, and completely buffered performance streams fail.
Throughput counts successful output tokens over the measured request window;
failed requests remain in that window. Prometheus snapshots surround each cell,
with Unix timestamps around each metrics fetch.

`phases.jsonl` records start/end markers for every warmup wave, measured cell and
quality run. Each marker includes Unix time, monotonic time and clock sampling
uncertainty. Pass `--worker-evidence-dir` to correlate these boundaries with the
server's compiled callable invocation counters. Before each start and after each
end, the client atomically writes `snapshot-request.json` in that directory:

```json
{"request_id": "UUID", "phase": "slot-0/measured/isl-1/c-1/wave-None/start", "unix": 0.0}
```

The rank-zero observer must atomically write `snapshot-<UUID>.json` containing the
same `request_id`, a numeric `unix` timestamp and a `calls` object. The observer
also records compiled graph metadata and source/placement proof. The client polls
every 100 ms for up to 60 seconds; a missing or invalid acknowledgement fails the
run. These waits occur outside request timing. The summary records receipt paths
and SHA256 values and rechecks their contents during comparison. Other ranks'
placement receipts are gathered separately by the launcher. Counter differences
identify invoked compiled callables; interpreting their buckets and collective
implementation requires the corresponding compiler and placement artifacts.

After all four starts:

```bash
python scripts/collectives/serving_client.py summarize \
  --runs /path/to/campaign/baseline-0 /path/to/campaign/candidate-1 \
         /path/to/campaign/candidate-2 /path/to/campaign/baseline-3 \
  --output /path/to/campaign/serving-report.json
```

The JSON and Markdown report compare mean per-start median TTFT, mean TPOT and
throughput, preserving each start's spreads and both A/B pairing deltas. They do
not pool request samples across starts or claim confidence intervals from two
starts. Summary validation checks ABBA order, request artifact hashes, identical
input hashes/settings, and separate cold-start identities. Fixed-question accuracy
reports baseline/candidate counts plus paired answer and token disagreements.
Performance and paired accuracy completion are recorded separately. Server
execution proof and matching launch configurations remain separate review gates.
