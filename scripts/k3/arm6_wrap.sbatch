#!/usr/bin/env bash
# arm6 (tail-fused epilogue) with a pre-step that reclaims george's stale
# on-node XLA cache trees (the fresh "-tail" tree is kept). Queued on all
# four 4-node partitions; runs on whichever slice frees first.
#SBATCH --job-name=foa-arm6-tail
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-foa-arm6-tail-%j.out
set -euo pipefail
srun -N4 -n4 --ntasks-per-node=1 bash -c '
  ls -d /mnt/scratch/users/george/vllm-xla-cache/kimi-k3* 2>/dev/null | grep -v -- "-tail$" | xargs -r rm -rf
  df -h /mnt/scratch | tail -1
'
exec bash scripts/k3/slurm_launch_k3_tp32_tp.sbatch --load-format dummy
