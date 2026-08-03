# Reproduce compiled Kimi K3 GSM8K on TPU v7x-32

This branch reproduces the validated full-model run on one four-host TPU
v7x-32 slice: 16 chips, 32 logical devices, TP32 + EP32, 4,096-token context,
and compiled token buckets 16, 64, 256, and 512.

## Requirements

- A Slurm allocation with exactly four hosts and eight TPU devices per host.
- The repository's normal `.venv` installation on shared storage.
- TorchTPU/libtpu compatible with TPU v7x multi-host operation. The validated
  environment used libtpu 0.0.44.1.
- Access to `moonshotai/Kimi-K3` and about 1.6 TB for its Hugging Face cache.
- About 130 GiB of shared storage for the persistent compiled XLA cache.

The launcher defaults to the validated physical topology:

```text
TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
TPU_HOST_BOUNDS=1,1,4
TORCH_TPU_TOPOLOGY=2,2,4,2
```

Override those variables before launching if the physical slice differs.

## 1. Obtain a four-node allocation

Use the partition and account settings appropriate for your cluster. For
example:

```bash
salloc --nodes=4 --ntasks-per-node=1 --time=04:00:00 \
  --partition=<your-v7x-32-partition>
```

All commands below run from the repository root inside that allocation.

## 2. Verify startup with dummy weights

```bash
export KIMI_DUMMY_WEIGHTS=1
scripts/launch_kimi_k3_tp32_slurm.sh 2>&1 | tee logs/kimi-k3-dummy.log
```

The launcher owns the foreground terminal. In a second shell inside the same
allocation, wait for `GET http://<first-node-ip>:8000/v1/models` to return 200,
then send a small completion request. Stop the launcher after the smoke test.

Dummy and real runs intentionally use the same compilation shapes. Keep the
XLA cache to reuse compiled programs.

## 3. Start the real checkpoint

```bash
unset KIMI_DUMMY_WEIGHTS
scripts/launch_kimi_k3_tp32_slurm.sh 2>&1 | tee logs/kimi-k3-real.log
```

Cold startup downloads or reads 96 weight shards and compiles four shapes. In
the validated environment it took about 40 minutes. The server is ready when
`/v1/models` returns HTTP 200.

The important serving settings are embedded in the launcher:

- model revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`;
- TP32 + EP32 with native 3,072-wide experts;
- `max_model_len=4096`, `max_num_batched_tokens=512`, `max_num_seqs=4`;
- compile sizes `[16,64,256,512]` and `CUSTOM` attention;
- 1 GiB KV budget, 260 attention blocks, and compact recurrent-state slots.

Do not raise concurrency above four for this reproduction. A 16-request stress
test caused repeated-token degeneration at high KV occupancy.

## 4. Run the complete GSM8K test split

Set the first host's reachable IP and run:

```bash
HEAD_IP=$(srun --overlap -N1 -n1 -w \
  "$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)" \
  bash -lc "hostname -I | awk '{print \$1}'")

.venv/bin/python scripts/eval_kimi_k3_gsm8k.py \
  --base-url "http://${HEAD_IP}:8000/v1" \
  --output artifacts/kimi-k3-gsm8k.jsonl
```

The evaluator defaults reproduce the measured run: all 1,319 official test
prompts, deterministic sampling, concurrency four, and 512 maximum completion
tokens. It uses raw K3 XTML completions because the published tokenizer's
custom renderer does not expose a static template to vLLM.

For a quick validation first:

```bash
.venv/bin/python scripts/eval_kimi_k3_gsm8k.py \
  --base-url "http://${HEAD_IP}:8000/v1" --limit 100 \
  --output artifacts/kimi-k3-gsm8k-100.jsonl
```

## Validated result

The complete compiled run produced:

```text
HTTP 200:                1,319 / 1,319
correct:                 1,283 / 1,319 (97.27%)
benchmark time:          5,181.91 seconds (86.37 minutes)
median request latency:  14.29 seconds
finish reasons:          1,310 stop, 9 length
runtime recompilations:  0 observed
repetition failures:     0 observed
```

Exact timing and accuracy can vary with software versions, cache warmth, and
model-serving changes. With `VLLM_XLA_CHECK_RECOMPILATION=1`, the server should
continue to log `No new compiled graphs for case: decoding_step` throughout
the benchmark.
