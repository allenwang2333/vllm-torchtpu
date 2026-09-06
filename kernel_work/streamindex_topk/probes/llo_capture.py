# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""P1 + P2 -- the dump-side LLO/Mosaic capture driver for `streamindex_topk`.

One capture answers both questions, which is why they share a driver:

  P1  Does the `compute_scores` einsum reach the MXU as fp8, or does Mosaic
      upcast the operands to bf16 first? The JAX-level dtypes say fp8; that is
      not evidence, because the upcast would be inserted below JAX. The answer
      is in the lowered Mosaic MLIR and in the LLO MXU matmul count.
      Stake: `step2/combined.md` puts the mixed_prod floor at 48.4 us if the
      einsum is native fp8 and 120.6 us if it is upcast -- 18% vs 45% achieved.

  P2  Does Mosaic materialise the (B, bq_sz*32, bkv_sz) f32 einsum result as
      one VMEM buffer, or does it tile the contraction? Read the VMEM
      allocation off `get_llo_fit_summary`.
      Stake: candidates C4 and C12 only exist if the buffer is materialised.

      The discriminator is the VMEM total AT mixed_prod. A materialised s is
      1 * 128*32 * 2048 * 4 B = 32.0 MiB on its own, so the arm should read
      ~37 MiB; a tiled contraction should read ~6 MiB. At decode_prod the two
      hypotheses predict ~5.1 MiB either way, so THAT ARM IS A NON-
      DISCRIMINATING CONTROL for P2 -- it is captured because P1 needs it and
      because the roofline gate's regime table is stated at both.

Everything this driver produces is STATIC COMPILE-TIME ANALYSIS. The wall-clock
printed at the end is there only to prove the kernel ran; the shards share a
host and are contended, and `xprof-llo/SKILL.md` is explicit that a fanned-out
capture's wall-clock is not a performance number. Do not quote it.

Usage (one shard per regime):

    python3 /workspace/kernel_work/streamindex_topk/probes/llo_capture.py \
        --regime mixed_prod --out $CDK_OUTPUT_DIR/mixed_prod

The two dump flags must be in `LIBTPU_INIT_ARGS`, not `XLA_FLAGS`, and libtpu
reads that variable once at load time -- hence the `os.environ` block ABOVE
`import jax`. Do not move it.
"""

import argparse
import os
import sys

# --- must precede `import jax` -------------------------------------------
_ap = argparse.ArgumentParser()
_ap.add_argument("--regime", required=True)
_ap.add_argument("--out", required=True)
_ap.add_argument("--early-exit", action="store_true")
_args = _ap.parse_args()

_JF = os.path.join(_args.out, "jf")
_MOSAIC = os.path.join(_args.out, "mosaic")
os.makedirs(_JF, exist_ok=True)
os.makedirs(_MOSAIC, exist_ok=True)

os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={_JF} "
    f"--xla_jf_dump_llo_text=true "
    f"--xla_mosaic_dump_to={_MOSAIC}")
print(f"LIBTPU_INIT_ARGS={os.environ['LIBTPU_INIT_ARGS']}", flush=True)
# -------------------------------------------------------------------------

sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])

import time  # noqa: E402

import jax  # noqa: E402

from kernel_work.streamindex_topk.instruments import inputs as I  # noqa: E402
from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402


def main():
    print("jax", jax.__version__, "devices", jax.devices(), flush=True)
    regime = S.REGIMES_BY_NAME[_args.regime]
    print(f"regime={regime.name} phase={regime.phase} kv_len={regime.kv_len} "
          f"num_tokens={regime.num_tokens} max_num_seqs={regime.max_num_seqs} "
          f"num_bq={regime.num_bq} num_bkv={regime.num_bkv} "
          f"num_kv_pages_per_block={S.NUM_KV_PAGES_PER_BLOCK} "
          f"num_queries_per_block={S.NUM_QUERIES_PER_BLOCK}", flush=True)

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as impl

    d = I.build(regime)
    call = jax.jit(lambda q, w, c, s, p, cu, dist: impl.streamindex_topk(
        q, w, c, s, p, cu, dist,
        k=S.INDEX_TOPK,
        compression_ratio=S.COMPRESSION_RATIO,
        num_kv_pages_per_block=S.NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=S.NUM_QUERIES_PER_BLOCK,
        decode_req_batch_size=S.DECODE_REQ_BATCH_SIZE,
        enable_early_exit=_args.early_exit,
    ))
    args = (d["q"], d["indexer_weights"], d["cache_kv"], d["seq_lens"],
            d["page_indices"], d["cu_q_lens"], d["distribution"])

    for name, a in zip(("q", "indexer_weights", "cache_kv"), args):
        print(f"input {name}: shape={a.shape} dtype={a.dtype}", flush=True)

    t0 = time.perf_counter()
    out = call(*args)
    jax.block_until_ready(out)
    compile_and_run_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    for _ in range(5):
        out = call(*args)
    jax.block_until_ready(out)
    warm_s = (time.perf_counter() - t1) / 5

    print(f"out shape={out.shape} dtype={out.dtype}", flush=True)
    # NOT A PERFORMANCE NUMBER -- LLO-instrumented and host-contended. Printed
    # only as evidence the kernel executed.
    print(f"[not a perf number] compile+first {compile_and_run_s*1e3:.1f} ms, "
          f"warm {warm_s*1e3:.3f} ms", flush=True)

    for label, root in (("jf", _JF), ("mosaic", _MOSAIC)):
        n = sum(len(f) for _, _, f in os.walk(root))
        b = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(root) for f in fs)
        print(f"dump {label}: {n} files, {b/2**20:.1f} MiB at {root}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
