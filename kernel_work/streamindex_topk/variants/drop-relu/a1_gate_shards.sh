#!/bin/bash
# Fan-out driver for the A1 (drop-the-ReLU) accuracy gate + the A3 M sweep.
#
# Five regimes over four shards. The work is float64 numpy on the host CPU --
# the TPU is not used by this gate, because A1's accuracy is a property of the
# arithmetic and the input distribution, not of the kernel. The pod is used for
# its host memory, its cores, and for a citable job id.
#
# Threads are capped per shard so four concurrent shards do not thrash the
# host's cores; nothing here is a timing measurement, so contention affects
# wall clock only.
set -u
SHARD="$1"
D=/workspace/kernel_work/streamindex_topk/variants/drop-relu
SWEEP=2048,2560,3072,4096,5120,6144,8192,12288,16384,24576,32768,49152,65536

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "===== shard $SHARD ====="
python3 -c "import numpy, jax, sys; print('numpy', numpy.__version__, 'jax', jax.__version__, 'py', sys.version)"

case "$SHARD" in
  0) REGIMES=decode_prod ; CHUNK=64 ;;
  1) REGIMES=mixed_prod  ; CHUNK=64 ;;
  2) REGIMES=mixed_32k   ; CHUNK=32 ;;
  3) REGIMES=mixed_64k,decode_64k ; CHUNK=16 ;;
  *) echo "bad shard"; exit 2 ;;
esac

echo "regimes=$REGIMES token_chunk=$CHUNK sweep=$SWEEP"
python3 "$D/a1_gate.py" --regimes "$REGIMES" --m-sweep "$SWEEP" \
  --token-chunk "$CHUNK" --out "/tmp/a1_gate_shard${SHARD}.json"
rc=$?
echo "shard $SHARD exit rc=$rc"
exit $rc
