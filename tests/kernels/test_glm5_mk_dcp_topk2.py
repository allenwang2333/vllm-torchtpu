"""glm5_mk_dcp.topk2 vs an exact numpy reference (interpret mode, CPU).

Local top-K: winners = the K largest keys among positions < n_valid, ties -> lowest position,
emitted in ascending position order with their keys. Global: over R rank-major candidate rows, ties
-> lowest gathered position; rank `me` keeps its winning slots ascending.
Includes a positive control (a reversed tie order must fail the comparison)."""
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.glm5_mk_dcp import topk2

INT_MIN = np.int32(-(2 ** 31))


def np_topk(keys, n_valid, K, reverse_ties=False):
    pos = np.arange(keys.size)
    k = np.where(pos < n_valid, keys.reshape(-1), INT_MIN).astype(np.int64)
    valid = pos < n_valid
    order = sorted(np.nonzero(valid)[0], key=lambda p: (-k[p], -p if reverse_ties else p))
    win = np.sort(np.array(order[:K], np.int64))
    idx = np.full(K, -1, np.int32)
    kv = np.full(K, INT_MIN, np.int32)
    idx[:len(win)] = win
    kv[:len(win)] = k[win]
    return idx, kv


def run_local(keys, n_valid, K):
    ROWS = keys.shape[0]

    def kern(nv_ref, keys_in, idx_o, key_o, keys_s, pre, acc, cc):
        keys_s[...] = keys_in[...]
        topk2.local_topk(keys_s, nv_ref[0], K, idx_o, key_o, pre, acc, cc)

    f = pl.pallas_call(
        kern,
        out_shape=[jax.ShapeDtypeStruct((K // 128, 128), jnp.int32)] * 2,
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
        scratch_shapes=[pltpu.VMEM((ROWS, 128), jnp.int32), pltpu.VMEM((ROWS, 128), jnp.int32),
                        pltpu.VMEM((256, 768), jnp.float32), pltpu.VMEM((2, 128, 128), jnp.float32)],
        interpret=pltpu.InterpretParams())
    i, k = f(jnp.array([n_valid], jnp.int32), jnp.asarray(keys))
    return np.asarray(i).reshape(-1), np.asarray(k).reshape(-1)


def run_global(kall, me, K):
    NR = kall.shape[0]

    def kern(me_ref, kall_in, own_o, ks, pre, acc, cc, mk):
        ks[...] = kall_in[...]
        topk2.global_owned(ks, me_ref[0], K, own_o, pre, acc, cc, mk)

    f = pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((K // 128, 128), jnp.int32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[pltpu.VMEM((NR, 128), jnp.int32), pltpu.VMEM((128, 128), jnp.int32),
                        pltpu.VMEM((256, 768), jnp.float32), pltpu.VMEM((2, 128, 128), jnp.float32),
                        pltpu.VMEM((128, 128), jnp.int32)],
        interpret=pltpu.InterpretParams())
    return np.asarray(f(jnp.array([me], jnp.int32), jnp.asarray(kall))).reshape(-1)


def np_global(kall_flat, R, K, me, reverse_ties=False):
    idx, _ = np_topk(kall_flat, kall_flat.size, K, reverse_ties)
    mine = idx[(idx >= me * K) & (idx < (me + 1) * K)] - me * K
    out = np.full(K, -1, np.int32)
    out[:len(mine)] = np.sort(mine)
    return out


def test_local_and_global():
    rng = np.random.default_rng(0)
    K = 256
    ROWS = 128
    for trial, (nv, tie_mod) in enumerate([(100, 0), (256, 0), (5000, 0), (16384, 7), (9000, 3)]):
        keys = rng.integers(-2 ** 31 + 1, 2 ** 31 - 1, size=(ROWS, 128), dtype=np.int64)
        if tie_mod:                                     # many exact ties, incl. -0.0/+0.0 keys
            keys = (keys % tie_mod) - 1
        keys = keys.astype(np.int32)
        i, k = run_local(keys, nv, K)
        ie, ke = np_topk(keys, nv, K)
        assert np.array_equal(i, ie), (trial, nv)
        assert np.array_equal(k, ke), (trial, nv)
        if tie_mod:                                     # positive control: reversed ties differ
            ir, _ = np_topk(keys, nv, K, reverse_ties=True)
            assert not np.array_equal(i, ir), "tie control blind"
    # global over R=4 ranks of K candidates each (rank-major), many ties
    R = 4
    cand = (rng.integers(0, 5, size=(R * K // 128, 128)) - 2).astype(np.int32)
    cand[3, 7:] = INT_MIN                               # a rank with fewer candidates
    flat = cand.reshape(-1)
    differs = False
    for me in range(R):
        own = run_global(np.pad(cand, ((0, 128 - cand.shape[0]), (0, 0)), constant_values=INT_MIN),
                         me, K)
        exp = np_global(flat, R, K, me)
        assert np.array_equal(own, exp), me
        differs |= not np.array_equal(exp, np_global(flat, R, K, me, reverse_ties=True))
    assert differs, "global tie control blind"


if __name__ == "__main__":
    test_local_and_global()
    print("OK")
