"""owner_attention_phase on 16 CPU devices (interpret) under shard_map: ranks
0..7 own one request, ranks 8..15 are idle owners. Every active rank must
reproduce the single-device result; idle ranks must leave every cache alone.
Run: JAX_PLATFORMS=cpu python attn_phase_16dev.py [hca|csa]"""
import os
import sys

import jax

jax.config.update("jax_num_cpu_devices", 16)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_phase_harness as H  # noqa: E402


def main(kind):
    rng = np.random.default_rng(0)
    ls = H.build(rng, ("hca", "csa"), [1030], 2048)
    layer = 0 if kind == "hca" else 1
    inp = H.make_inputs(rng, ls, kind)
    ref_caches, ref_out = H.run_phase(ls, kind, layer, inp)
    fn, ops = H.build_call(ls, kind, layer, inp)
    mesh = Mesh(np.array(jax.devices()[:16]), ("x",))
    nreq = jnp.asarray([1] * 8 + [0] * 8, jnp.int32)

    def per_rank(nr, *rest):
        res = fn(nr, *rest)
        return tuple(x[None] for x in res)

    n_out = len(ls.arrays) + 1
    sm = jax.shard_map(per_rank, mesh=mesh, in_specs=(P("x"),) + (P(),) * (len(ops) - 1),
                       out_specs=(P("x"),) * n_out, check_vma=False)
    res = jax.jit(sm)(nreq, *ops[1:])
    ok = True
    for rank in range(16):
        for i in range(len(ls.arrays)):
            exp = ref_caches[i] if rank < 8 else ls.arrays[i]
            ok &= bool(np.array_equal(np.asarray(res[i][rank]), np.asarray(exp)))
        if rank < 8:
            ok &= bool(np.array_equal(np.asarray(res[-1][rank]), np.asarray(ref_out)))
    print("RESULT 16dev", kind, "all ranks as expected:", ok)
    return ok


if __name__ == "__main__":
    assert main(sys.argv[1] if len(sys.argv) > 1 else "hca")
