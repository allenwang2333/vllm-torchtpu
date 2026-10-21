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
"""On device: kv_write.compress_store vs the stock Pallas compressor_forward.

The stock call projects with an identity matrix and no ape; the states it
stored are read back and handed to the megakernel step, so both start from
the same f32 states. Then every cache word is compared. Prints RESULT lines.
"""

import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_write_cases as K  # noqa: E402
from bench_attn_device import chain_time_pure  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import compressor_v1  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kv_write  # noqa: E402

SEQS = {"hca": [130, 259, 1030, 4100], "csa": [11, 40, 103, 4099], "idx": [11, 40, 103, 4099]}


def shared_case(case):
    """csa / idx serve their states from the KV array: put the state pages and
    the KV pages in one array (disjoint page ids); the RoPE companion array
    gets the same page count."""
    ns = case.state_host.shape[0]
    arr = jnp.concatenate([case.state_host, case.kv], axis=0)
    rope = None
    if case.rope is not None:
        rope = jnp.concatenate([jnp.zeros((ns,) + case.rope.shape[1:], case.rope.dtype),
                                case.rope], axis=0)
    return arr, case.kv_table + ns, rope


def stock(mode, states, seq_lens, state_table, state_host, arr, kv_table, rope, norm_w, cos_sin):
    """compressor_forward after its projection (identity weights, no ape)."""
    D, ratio, overlap, qb = kv_write._MODES[mode]
    sw = (1 + overlap) * D
    sbs = K.GEOM[mode][1]
    R = seq_lens.shape[0]
    pos = (seq_lens[:, None] - K.Q_LEN + jnp.arange(K.Q_LEN)[None, :]).reshape(-1)
    qsl = jnp.arange(R + 1, dtype=jnp.int32) * K.Q_LEN
    common = dict(
        hidden_states=states.reshape(-1, 2 * sw),
        wkv_wgate=jnp.eye(2 * sw, dtype=jnp.float32), ape=jnp.zeros((ratio, sw), jnp.float32),
        norm_weight=norm_w, cos_sin_cache=cos_sin, positions=pos.astype(jnp.int32),
        block_table=state_table.reshape(-1), query_start_loc=qsl,
        kv_block_table=kv_table.reshape(-1), distribution=jnp.array([0, R, R], jnp.int32),
        state_block_size=sbs, head_dim=D, compress_ratio=ratio, overlap=overlap,
        rms_eps=1e-6, quant_block=qb)
    if mode == "hca":
        return compressor_v1.compressor_forward(
            cache=csa_cache_layout.words_to_slabs(arr), rope_cache=None,
            state_cache=csa_cache_layout.words_to_slabs(state_host), **common)
    if mode == "csa":
        return compressor_v1.compressor_forward(cache=arr, rope_cache=rope, **common)
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import KVLayout
    return compressor_v1.compressor_forward(
        cache=csa_cache_layout.words_to_slabs(arr), rope_cache=None,
        record_layout=KVLayout.SEQ_ALONG_LANE, **common)


def read_states(mode, case, state_words):
    """The f32 states stored for each request's q_len rows."""
    D, ratio, overlap, _ = kv_write._MODES[mode]
    sw = (1 + overlap) * D
    sbs = K.GEOM[mode][1]
    L = state_words.shape[2]
    spt = 2 * sw // L
    w = np.asarray(state_words)
    out = np.zeros(case.states.shape, np.float32)
    for r, s in enumerate(np.asarray(case.seq_lens)):
        for t in range(K.Q_LEN):
            pos = s - K.Q_LEN + t
            page = int(case.state_table[r, pos // sbs])
            rows = w[page, (pos % sbs) * spt:(pos % sbs + 1) * spt]
            out[r, t] = rows.reshape(-1).view(np.float32)
    return jnp.asarray(out)


def main():
    print(jax.devices()[0], flush=True)
    for mode in sys.argv[1:] or ("hca", "csa", "idx"):
        case = K.make_case(np.random.default_rng(1), mode, SEQS[mode])
        sbs = K.GEOM[mode][1]
        if mode == "hca":
            arr, kv_table, rope, shared = case.kv, case.kv_table, None, False
        else:
            (arr, kv_table, rope), shared = shared_case(case), True
        sargs = (case.states, case.seq_lens, case.state_table, case.state_host, arr, kv_table,
                 rope, case.norm_w, case.cos_sin)
        sfn = functools.partial(stock, mode)
        out = jax.jit(sfn)(*sargs)
        if mode == "hca":
            s_kv = csa_cache_layout.slabs_to_words(out[0])
            s_state = csa_cache_layout.slabs_to_words(out[2])
        else:
            s_kv = out[0] if mode == "csa" else csa_cache_layout.slabs_to_words(out[0])
            s_state = s_kv
        states = read_states(mode, case, s_state)
        f = functools.partial(kv_write.compress_store, mode, state_block_size=sbs,
                              shared_state=shared)
        margs = (states, case.seq_lens, case.state_host, case.state_table, arr, kv_table,
                 case.norm_w, case.cos_sin) + ((rope,) if mode == "csa" else ())
        res = jax.jit(f)(*margs)
        m_kv = res[0] if shared else res[1]
        same = {"kv": bool(np.array_equal(np.asarray(m_kv), np.asarray(s_kv))),
                "kv_words_diff": int(np.sum(np.asarray(m_kv) != np.asarray(s_kv))),
                "kv_words_written": int(np.sum(np.asarray(s_kv) != np.asarray(arr)))}
        if not shared:
            same["state"] = bool(np.array_equal(np.asarray(res[0]), np.asarray(s_state)))
        if mode == "csa":
            same["rope"] = bool(np.array_equal(np.asarray(res[1]), np.asarray(out[1])))
        t_m = chain_time_pure(lambda *a: jax.tree.leaves(f(*a))[0], margs)
        t_s = chain_time_pure(lambda *a: jax.tree.leaves(sfn(*a))[0], sargs)
        print(f"RESULT kind=kv_write mode={mode} R={len(SEQS[mode])} {same} "
              f"mine_us={t_m[0]*1e6:.1f} [{t_m[1]*1e6:.1f},{t_m[2]*1e6:.1f}] "
              f"stock_us(incl. 56-row projection)={t_s[0]*1e6:.1f} [{t_s[1]*1e6:.1f},"
              f"{t_s[2]*1e6:.1f}]", flush=True)


if __name__ == "__main__":
    main()
