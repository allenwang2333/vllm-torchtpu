"""R1c -- dump-side LLO/Mosaic capture of the three `compute`-mode arms.

WHY THIS EXISTS
---------------
R1b (`probes/R1b.md`) measured Option A's stage-A inner loop to be 973.9 ns/block
cheaper than the incumbent's at `bkv_p = 9`, in a DMA-free `compute` arm, and
showed the byte reduction contributes ~nothing on the critical path because
94.3 % of the incumbent's fetch is already hidden.  So the whole of Option A's
value sits in that 973.9 ns, and it has exactly two candidate mechanisms:

  (T) the eliminated lane transpose.  The incumbent extracts its e8m0 scales as
      `flat[:, 128:129].T` -- a (bkv_sz,1) -> (1,bkv_sz) shuffle.  Option A's
      scale rows are already lane-major, so the same extraction is a reshape.

  (W) the halved payload width.  The incumbent materialises a 256-lane page
      record and reads lanes 0..127 out of it; Option A materialises 128 lanes.

The two imply opposite plans, which is why this is worth a job:

  T  =>  a variant captures the saving WITHOUT touching the cache format: no
         caller-side scatter, no writer change, no prefill exposure.  That
         variant strictly dominates Option A.
  W  =>  no such variant exists.  The saving is only reachable through the
         cache format, and Option A stays what R1b's numbers say it is.

R1b's own `a_concat` control tilts this toward (W): it builds the scale operand
a structurally very different way (72 materialised (1,128) rows concatenated
along lanes, vs one reshape) and lands within 0.6 % of `a_reshape` everywhere.
If the scale relayout dominated, two formulations that unlike should not agree
that closely.  **(W) is therefore the hypothesis this capture tries to BREAK,
not to confirm.**

WHAT DISCRIMINATES THEM, STATICALLY
-----------------------------------
  under (T): `inc` carries transpose/relayout/XLU ops that BOTH A arms lack,
             while the vector-load counts are comparable across arms.
  under (W): `inc` carries ~2x the vector-load traffic of both A arms, while
             the transpose counts do not separate `a_concat` from `a_reshape`
             in proportion to their (identical) measured time.

`a_concat` is the discriminating third column precisely because it shares A's
payload path and has a wildly different scale path.

>>> POST-HOC CORRECTION (added after reading the dump; see `R1c.md`) <<<
>>> ...and then RESOLVED by the fourth arm. Jump to the end of this block. <<<

The (W) test above is WRONG and the three arms cannot settle the question at
all. Two things this plan did not anticipate:

1. **Vector-load COUNT is not vector-load TRAFFIC.** The counts came back
   near-equal (9,296 vs 9,368) but the MASKS differ: `inc` loads with
   `sublanes [true,true,...]` against A's `[true,false,...]`, i.e. 1024 B vs
   512 B in the same instruction. `inc` reads 9.44 MB/block against A's
   4.76 MB, exactly the 2x this plan expected -- but with zero extra vector
   ops, so an opcode census sees nothing. (W) costs bytes and VMEM footprint,
   not instructions, and a static dump cannot price bytes.
2. **The three arms are collinear.** Every one pairs scale placement with
   record width: `inc` = {lane-128 scale, 256-lane record}, both A arms =
   {lane-major rows, 128-lane record}. `a_concat` varies only the reader-side
   FORMULATION of A's scale path, not the format. There is no arm with a
   256-lane record and lane-major scale rows -- and that missing fourth arm is
   the experiment that would settle it. It is a one-line addition to
   `LAYOUTS`/`CONSUMER_LAYOUT` in `probe_kpack_consumer.py` and a re-run of
   this driver.

What the capture DID establish: the whole instruction-count delta is on the
scale path (XLU bundle occupancy 46.24 % vs 1.05 %, 288 `vector<128x128xi8>`
transposes only in `inc`, `tpu.matmul` backward cone identical in all arms).

>>> RESOLVED. The fourth arm was built as `w_reshape` and captured in job
`j-a0566857-c6f2-4667-9eed`. It lands on `a_reshape` for XLU (1.04 % vs
1.05 %), transposes, spills, bundles and total ops, and on `inc` for VMEM read
bytes and allocated footprint. So (T) owns the instructions and (W) owns the
bytes, and the two questions were never the same question.

One thing section 7 of `R1c.md` is careful about and this driver cannot be:
a bundle-count share is NOT a share of R1b's measured 973.9 ns/block. Nothing
here is timed.

WHAT THIS IS NOT
----------------
**Everything this produces is static compile-time analysis** -- issue slots,
opcode counts and allocation sizes, not measured time.  It must be labelled as
such wherever it is quoted, and it must never reach `/log-experiment`.  The
measured numbers all live in R1b.  This job's wall-clock is meaningless: it runs
one iteration per arm purely to force compilation, with dump flags on.

CAPTURE NOTES
-------------
* `LIBTPU_INIT_ARGS` is read by libtpu **once, at load time**, so it is set here
  above `import jax` and above the probe import.  Setting these in `XLA_FLAGS`
  is accepted and does nothing.
* LLO text goes to `--xla_jf_dump_to`, NOT `--xla_dump_to`; the latter accepts
  `--xla_jf_dump_llo_text=true` alongside it and silently writes nothing.
* Everything is written under `$CDK_OUTPUT_DIR`, because `--map-dir` is
  upload-only and its tarball is built at job start.
* **jax/jaxlib are deliberately NOT upgraded to 0.11.0.**  The skill requires
  that for *trace-side* capture, where jaxlib 0.10.2 vs libtpu 0.0.44.1
  mismatches `PLUGIN_Profiler_Api` and the trace never starts.  This is a
  dump-side capture: no profiler is involved, only compiler flags writing files.
  Upgrading would risk changing the very lowering being censused, which must
  match the jax 0.10.2 compilation R1b timed.  If the dump directories come back
  empty this assumption is what to re-examine first.
* Each arm dumps to its own subdirectory: two dump flags pointed at one path
  interleave into an unreadable mess.
"""

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- everything above `import jax` -----------------------------------------

OUT_ROOT = os.environ.get("CDK_OUTPUT_DIR", "/tmp/llo")


def _dump_dirs(arm):
    base = os.path.join(OUT_ROOT, "llo", arm)
    return os.path.join(base, "jf"), os.path.join(base, "mosaic")


def _set_libtpu(arm, pass_regex):
    jf, mosaic = _dump_dirs(arm)
    os.makedirs(jf, exist_ok=True)
    os.makedirs(mosaic, exist_ok=True)
    flags = [
        f"--xla_jf_dump_to={jf}",
        "--xla_jf_dump_llo_text=true",
        f"--xla_mosaic_dump_to={mosaic}",
    ]
    if pass_regex:
        flags.append(f"--xla_jf_dump_llo_pass_label_regex={pass_regex}")
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(flags)
    return jf, mosaic


def _du(path):
    try:
        out = subprocess.run(["du", "-sh", path], capture_output=True,
                             text=True, timeout=300).stdout.split()
        n = subprocess.run(f"find {path} -type f | wc -l", shell=True,
                           capture_output=True, text=True,
                           timeout=300).stdout.strip()
        return f"{out[0] if out else '?'} / {n} files"
    except Exception as e:  # noqa: BLE001
        return f"<du failed: {e!r}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    help="consumer name: inc | a_reshape | a_concat | "
                         "w_reshape (the R1c 4.1 discriminator: inc's 256-lane "
                         "record with A's lane-major scale rows)")
    ap.add_argument("--bkv-p", type=int, default=9)
    ap.add_argument("--num-grid", type=int, default=8,
                    help="tiny on purpose: the dump is of the compiled program, "
                         "so the grid does not change the kernel body. This is "
                         "not a timing run.")
    ap.add_argument("--pages-per-seq", type=int, default=10)
    ap.add_argument("--mode", default="compute",
                    help="compute (the arm R1b's 973.9ns comes from); full is "
                         "available as a cross-check")
    ap.add_argument("--pass-regex", default=None)
    args = ap.parse_args()

    jf, mosaic = _set_libtpu(args.arm, args.pass_regex)
    print(f"[llo] arm={args.arm} mode={args.mode} bkv_p={args.bkv_p}", flush=True)
    print(f"[llo] LIBTPU_INIT_ARGS={os.environ['LIBTPU_INIT_ARGS']}", flush=True)

    # Import the probe only after LIBTPU_INIT_ARGS is set. It imports jax at
    # module level, and libtpu reads the variable once at load time.
    sys.path.insert(0, _HERE)
    import probe_kpack_consumer as P  # noqa: E402
    import jax  # noqa: E402
    import numpy as np  # noqa: E402

    print(f"[llo] jax {jax.__version__} devices={jax.devices()[:1]} "
          f"n={jax.device_count()}", flush=True)
    print(f"[llo] probe source_md5={P.source_md5()}", flush=True)
    if args.arm == "w_reshape":
        # w's scale row is (4,256), a different arithmetic from a's (2,4,128).
        # If it is permuted against the payload the arm compiles fine and
        # censuses the wrong program, so gate on it rather than trusting it.
        print(f"[llo] w slot-order check: {P.check_slot_order_w()}", flush=True)

    num_bkv = P.cdiv(P.CTX, args.bkv_p * P.PAGE_SIZE)
    num_pages = 4096

    f, src_shape, q_shape, w_shape, npi = P.build_arm(
        consumer=args.arm, mode=args.mode, bkv_p=args.bkv_p, num_bkv=num_bkv,
        num_grid=args.num_grid, pages_per_seq=args.pages_per_seq,
        num_pages=num_pages)

    keys = jax.random.split(jax.random.key(11), 2)
    page_indices = P.make_page_table(npi, num_pages)
    meta = P.jnp.full((P.SEQ_BATCH_SIZE, ), P.CTX, P.jnp.int32)

    def fp8_bytes(key, shape):
        raw = jax.random.bits(key, shape, dtype=P.jnp.uint8)
        exp = P.jnp.uint8(5) + ((raw >> P.jnp.uint8(3)) & P.jnp.uint8(3))
        return (raw & P.jnp.uint8(0x87)) | (exp << P.jnp.uint8(3))

    x = jax.block_until_ready(
        P.make_source(P.CONSUMER_LAYOUT[args.arm], num_pages))
    q = jax.block_until_ready(fp8_bytes(keys[0], q_shape))
    w = jax.block_until_ready(
        jax.random.uniform(keys[1], w_shape, P.jnp.float32, 0.25, 1.75))

    print(f"[llo] compiling+running once (num_bkv={num_bkv} "
          f"src={src_shape})", flush=True)
    out = np.asarray(jax.block_until_ready(f(page_indices, meta, x, q, w)))
    print(f"[llo] ran. checksum={float(out[0].sum()):.8e} "
          f"n_finite={float(out[1].sum()):.0f}", flush=True)
    del x, q, w

    print(f"[llo] jf     {jf}  {_du(jf)}", flush=True)
    print(f"[llo] mosaic {mosaic}  {_du(mosaic)}", flush=True)

    # Gate the capture here rather than discovering it in the reader: an empty
    # dump and a kernel with nothing to report look identical downstream.
    nj = len(os.listdir(jf)) if os.path.isdir(jf) else 0
    nm = len(os.listdir(mosaic)) if os.path.isdir(mosaic) else 0
    if nj == 0 or nm == 0:
        print(f"[llo] CAPTURE FAIL arm={args.arm} jf_entries={nj} "
              f"mosaic_entries={nm} -- flags did not take", flush=True)
        return 2
    print(f"[llo] CAPTURE OK arm={args.arm} jf_entries={nj} "
          f"mosaic_entries={nm}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
