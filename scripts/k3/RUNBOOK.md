# Kimi-K3 runbook

Run these commands from the repository root. Replace `<SERVER_IP>` with the IP
of the host serving the OpenAI-compatible endpoint.

## Non-Slurm cluster

This assumes the Ray cluster is already running. Start the server in a
dedicated terminal, replacing `<RAY_HEAD_IP>` with the Ray head IP:

```bash
RAY_ADDRESS="<RAY_HEAD_IP>:6379" PROF_DIR="$PWD/results/kimi-k3/profiles" scripts/k3/launch_kimi_k3_tp32_ep.sh
```

After the server is healthy, run the performance benchmark:

```bash
OUTPUT_DIR="$PWD/results/kimi-k3/benchmark" scripts/k3/bench_isl1_osl1k.sh "<SERVER_IP>"
```

Capture two post-warmup profiles:

```bash
OUTPUT_DIR="$PWD/results/kimi-k3/profile-requests" PROF_DIR="$PWD/results/kimi-k3/profiles" scripts/k3/profile_isl1_osl1k.sh "<SERVER_IP>"
```

Run the full GSM8K five-shot evaluation:

```bash
OUTPUT_DIR="$PWD/results/kimi-k3/gsm8k" scripts/k3/eval_gsm8k.sh "<SERVER_IP>"
```

## Slurm cluster

Submit the server wrapper, replacing `<SLURM_PARTITION>` with the partition to
use:

```bash
sbatch --partition="<SLURM_PARTITION>" --export=ALL,KIMI_K3_LOG_ROOT="$PWD/results/kimi-k3/server",PROF_DIR="$PWD/results/kimi-k3/profiles" scripts/k3/slurm_launch_k3_tp32_tp.sbatch
```

The Slurm log prints the server endpoint. After it is healthy, run the
performance benchmark from a host that can reach that endpoint:

```bash
OUTPUT_DIR="$PWD/results/kimi-k3/benchmark" scripts/k3/bench_isl1_osl1k.sh "<SERVER_IP>"
```

Capture two post-warmup profiles:

```bash
OUTPUT_DIR="$PWD/results/kimi-k3/profile-requests" PROF_DIR="$PWD/results/kimi-k3/profiles" scripts/k3/profile_isl1_osl1k.sh "<SERVER_IP>"
```

Run the full GSM8K five-shot evaluation:

```bash
OUTPUT_DIR="$PWD/results/kimi-k3/gsm8k" scripts/k3/eval_gsm8k.sh "<SERVER_IP>"
```
