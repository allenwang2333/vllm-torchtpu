"""Analytic roofline for streamindex_topk on GLM-5.2 / TPU7x.

Hardware constants are read back from pltpu.get_tpu_info() on job
j-1054e9b4-1e45-4de9-9009 (per TensorCore: num_cores=1, is_split_chip=True).
Shape derivations are in report.md. Run: python3 roofline.py
"""
import math

# ---- hardware, per TensorCore ----------------------------------------------
BW = 3.70e12   # mem_bw_bytes_per_second   measured (j-1054e9b4-1e45-4de9-9009)
FP8 = 2.30e15  # fp8_ops_per_second        measured (j-1054e9b4-1e45-4de9-9009)

# ---- model, from zai-org/GLM-5.2-FP8 config.json ----------------------------
H, DH, K = 32, 128, 2048
LROW_ACTUAL = 256  # align(packed_width=194,128); derivation + test:180-184
LROW_USEFUL = 129  # 128 fp8 value bytes + 1 e8m0 scale byte; test:310-322
IDX_LAYERS = 21    # 21 'full' entries in config.json indexer_types (of 78)

# MXU efficiency: the score einsum contracts D=128 and its M dim is the number
# of (query x head) rows in a tile. MIXED tiles 128 queries -> M=128*32, full.
# DECODE tiles 1 query -> M=32 of 128 sublanes. Whether seq_batch_size=4 packs
# 4 sequences into M (making it 128) is UNMODELLED -- see the register.
EFF_DECODE, EFF_MIXED = 0.25, 1.0


def cdiv(a, b):
    return -(-a // b)


def model(name, *, psz, pages_per_seq, T, R, D, seqlen, qlen=1, bkv_p=2):
    # n is the PADDED score width, not pages_per_seq*psz. streamindex_topk.py
    # :602-605 sets num_sublanes_total from align_to(pages_per_seq, bkv_p), so
    # an odd page count rounds up: at (A) 9 pages -> 10, n = 10240, not 9216.
    # This was 11.1% low here and in shapes.py/report.md. It matters because n
    # is what the -inf fill and the SparseCore read are sized by, and because
    # `p` and the second-pass branch below are decided against it.
    bkv_sz, P = psz * bkv_p, R - D
    n = cdiv(pages_per_seq, bkv_p) * bkv_p * psz
    nbkv = max(1, cdiv(seqlen, bkv_sz))
    nbq = max(1, cdiv(qlen, 128)) if P else 1

    kv_actual = (D * nbkv * bkv_sz + P * nbkv * bkv_sz * nbq) * LROW_ACTUAL
    kv_floor = (D + P) * seqlen * LROW_USEFUL

    inf_fill = T * n * 4                       # jnp.full(-inf) :745-749
    live = D + P * qlen
    pallas_wr = live * nbkv * bkv_sz * 4
    p = 1 if n <= 65536 else 2 ** math.ceil(math.log2(cdiv(n, 65536)))
    sc = live * min(n, nbkv * bkv_sz) * 4 + (live * p * K * 4 * 2 if p > 1 else 0)
    qw, out = T * H * DH + T * H * 2, T * K * 4

    mem_a = kv_actual + inf_fill + pallas_wr + sc + qw + out
    mem_f = kv_floor + qw + out                # scores stay in VMEM
    flops = (D + P * qlen) * seqlen * 2 * H * DH
    eff = EFF_MIXED if P else EFF_DECODE

    t_a = max(mem_a / BW, flops / (FP8 * eff))  # perfect overlap
    t_f = max(mem_f / BW, flops / FP8)
    return dict(name=name, n=n, kv=kv_actual, inf=inf_fill, wr=pallas_wr, sc=sc,
                mem_a=mem_a, mem_f=mem_f, flops=flops, t_a=t_a, t_f=t_f,
                t_mem=mem_a / BW, t_mxu=flops / (FP8 * eff),
                bound="mem" if mem_a / BW > flops / (FP8 * eff) else "mxu")


A = dict(psz=1024, pages_per_seq=9)     # (A) 8k1k  : block_size 1024 -> n=10240
B = dict(psz=512, pages_per_seq=2048)   # (B) AgentX: block_size 512  -> n=1048576

rows = [
    model("A-D64-8k", **A, T=64, R=64, D=64, seqlen=8192),
    model("A-D64-9k", **A, T=64, R=64, D=64, seqlen=9216),
    model("A-D32-8k", **A, T=32, R=32, D=32, seqlen=8192),
    model("A-D16-8k", **A, T=16, R=16, D=16, seqlen=8192),
    model("A-P1024-8k", **A, T=1024, R=1, D=0, seqlen=8192, qlen=1024),
    model("A-P512-8k", **A, T=512, R=1, D=0, seqlen=8192, qlen=512),
    model("B-D4-256k", **B, T=16, R=4, D=4, seqlen=262144),
    model("B-D4-512k", **B, T=16, R=4, D=4, seqlen=524288),
    model("B-D4-1M", **B, T=16, R=4, D=4, seqlen=1048576),
    model("B-D1-1M", **B, T=16, R=1, D=1, seqlen=1048576),
    model("B-D16-256k", **B, T=16, R=16, D=16, seqlen=262144),
    model("B-P1024-1M", **B, T=1024, R=1, D=0, seqlen=1048576, qlen=1024),
    model("B-P1024-256k", **B, T=1024, R=1, D=0, seqlen=262144, qlen=1024),
]

# Measured medians, us. instruments/bench.py, two 20-iteration blocks.
#   j-284a8842-3db6-44e0-ba49  (JobStatus Failed: 2 of 4 shards; the two bench
#                               shards below are the rc=0 ones)
#   j-59d0b337-9c4d-4dc0-aa7e  (JobStatus Succeeded, 4/4)
#   j-31ca6c31-a430-4463-8dbf  (JobStatus Succeeded, 4/4) -- supersedes the two
#       A-P rows above, which it re-ran WITH --decompose. The full-op medians
#       agree to 1.0%/1.1%, so the sc split is being attached to a reproduced
#       measurement, not a fresh one.
#
# DRIFT: the `spread` column below is block-to-block WITHIN one row, and it
# understates the noise. j-f90148cb-4f1b-4252-aece ran A-D16-8k twice in one
# job, separated by two other rows, and got 287.5 then 278.5 us -- 3.2% apart,
# against that row's own 1.9% and 1.3% block-to-block spreads. `built in` also
# fell 7.3s -> 4.3s between them, so a warm compile cache is the likely
# mechanism and the term is run-ORDER, not run-to-run randomness.
# Consequence: a difference under ~3% is not resolvable by comparing two rows
# in sequence, and no cross-JOB comparison here is worth anything at that size.
#
# The A-D16-8k > A-D32-8k inversion (278.7 vs 264.8) is 5.3%, just outside that
# band, and it has an exact mechanism -- so unlike a bare 5% gap it does not
# rest on the drift argument at all. sparsecore_topk.py:_pick_partition has a
# second loop that only ever RAISES p, with no check that raising it buys
# anything. At b=16, max_p = 32//16 = 2 and n//2 = 5120 clears both guards, so
# p goes to 2. But 16 rows at p=1 is already 16 of 32 subcores -- one wave --
# and 32 units is still one wave, so the wave count does not move. What p=2
# does buy is the `p > 1` second-pass merge: a second dispatch plus two XLA
# gathers. At b=32 and b=64, max_p collapses to 1 and no second pass happens.
# That is why the D16 row is slower than the D32 row rather than noise being
# read as a result, and it is the basis for the E-series candidate that
# suppresses the second loop (Δ time −9.0% at A-D16-8k, 0 elsewhere).
# (full, sparsecore_topk alone or None, job, block-to-block spread)
MEASURED = {
    "A-D64-8k":     (363.2, 189.5, "j-284a8842", 0.091),
    "A-D64-9k":     (365.1, 183.6, "j-31ca6c31", 0.010),
    "A-D32-8k":     (264.8, 175.6, "j-31ca6c31", 0.019),
    "A-D16-8k":     (278.7, 200.5, "j-31ca6c31", 0.019),
    "A-P1024-8k":   (841.6, 556.8, "j-31ca6c31", 0.008),
    "A-P512-8k":    (579.4, 352.1, "j-31ca6c31", 0.001),
    "B-D4-256k":    (1117.9, 795.9, "j-284a8842", 0.007),
    "B-D4-512k":    (1394.4, 832.1, "j-59d0b337", 0.002),
    "B-D4-1M":      (1826.7, 823.5, "j-284a8842", 0.003),
    "B-D1-1M":      (1481.9, 780.6, "j-59d0b337", 0.003),
    "B-D16-256k":   (2048.8, 1012.8, "j-59d0b337", 0.005),
    "B-P1024-256k": (50916.8, None, "j-59d0b337", 0.005),
    "B-P1024-1M":   (66958.7, None, "j-31ca6c31", 0.000),
}

MB = 1 << 20
h = (f"{'row':20} {'KV MB':>8} {'-inf MB':>8} {'SC MB':>8} {'tot MB':>9} "
     f"{'t_mem us':>9} {'t_mxu us':>9} {'bound':>6} {'t_act us':>9} "
     f"{'t_floor':>9} {'floor/act':>10}")
print(h)
print("-" * len(h))
for r in rows:
    print(f"{r['name']:20} {r['kv']/MB:>8.1f} {r['inf']/MB:>8.1f} {r['sc']/MB:>8.1f} "
          f"{r['mem_a']/MB:>9.1f} {r['t_mem']*1e6:>9.1f} {r['t_mxu']*1e6:>9.1f} "
          f"{r['bound']:>6} {r['t_a']*1e6:>9.1f} {r['t_f']*1e6:>9.1f} "
          f"{r['t_f']/r['t_a']:>10.0%}")

print("\nMEASURED vs MODELLED. The gate is meas/floor, not model/floor: the model"
      "\nis an input to the gate, the measurement is the gate.")
h2 = (f"{'row':20} {'meas us':>9} {'sc us':>8} {'sc share':>9} {'model us':>9} "
      f"{'meas/model':>11} {'floor us':>9} {'floor/meas':>11}  verdict")
print(h2)
print("-" * len(h2))
for r in rows:
    if r['name'] not in MEASURED:
        continue
    meas, sc_us, job, spread = MEASURED[r['name']]
    v = r['t_f'] / (meas * 1e-6)
    verdict = ("STOP - at floor" if v >= 0.80 else
               "one iteration" if v >= 0.50 else "STRUCTURAL - proceed")
    scs = f"{sc_us:8.1f}" if sc_us else " " * 8
    shr = f"{sc_us/meas:>8.0%}" if sc_us else " " * 8
    print(f"{r['name']:20} {meas:>9.1f} {scs} {shr} {r['t_a']*1e6:>9.1f} "
          f"{meas/(r['t_a']*1e6):>10.1f}x {r['t_f']*1e6:>9.1f} "
          f"{v:>10.1%}  {verdict}")

print("\nShare of modelled HBM traffic:")
for r in rows:
    a = r['mem_a']
    print(f"  {r['name']:20} KV {r['kv']/a:>5.1%}  -inf {r['inf']/a:>5.1%}  "
          f"pallas-wr {r['wr']/a:>5.1%}  SC {r['sc']/a:>5.1%}")

print("\nLever 1 -- fix ONLY the 256->129 row padding (KV x 129/256):")
for r in rows:
    new = max((r['mem_a'] - r['kv'] * (1 - 129 / 256)) / BW, r['t_mxu'])
    print(f"  {r['name']:20} {r['t_a']*1e6:>7.1f} -> {new*1e6:>7.1f} us   "
          f"delta {100*(new - r['t_a'])/r['t_a']:>+6.1f}% time")

print("\nLever 1+2 -- also keep scores out of HBM (-inf fill, pallas wr, SC read):")
for r in rows:
    rest = r['mem_a'] - r['kv'] - r['inf'] - r['wr'] - r['sc']
    new = max((r['kv'] * 129 / 256 + rest) / BW, r['t_mxu'])
    print(f"  {r['name']:20} {r['t_a']*1e6:>7.1f} -> {new*1e6:>7.1f} us   "
          f"delta {100*(new - r['t_a'])/r['t_a']:>+6.1f}% time   "
          f"(floor {r['t_f']*1e6:.1f})")
