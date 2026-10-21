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
"""Compressor-store cases and the stock plain-JAX goldens they are checked against."""

import dataclasses

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import compressor_v1
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store.compress_store_ref import (
    ref_compress_norm_rope_store,
)
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store.project_and_save_state_ref import (
    ref_wkv_proj_and_save_state,
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kv_write

Q_LEN = 7
# mode -> (state host shape per page (rows, lanes), state block size, kv page shape)
GEOM = {
    "hca": ((256, 128), 32, (16, 128)),  # state on a CSA NoPE host; HCA page = 8 rows
    "csa": ((256, 128), 16, (256, 128)),
    "idx": ((64, 256), 32, (64, 256)),
}


def u8(words, lanes):
    """int32 [N, rows, lanes] -> the uint8 [N, rows, 4, lanes] slabs."""
    return csa_cache_layout.words_to_slabs(words)


@dataclasses.dataclass
class Case:
    mode: str
    states: jnp.ndarray  # f32 [R, 7, 2 SW]
    seq_lens: jnp.ndarray
    state_host: jnp.ndarray  # int32
    state_table: jnp.ndarray
    kv: jnp.ndarray  # int32
    kv_table: jnp.ndarray
    rope: jnp.ndarray | None
    norm_w: jnp.ndarray
    cos_sin: jnp.ndarray


def make_case(rng, mode, seq_lens):
    D, ratio, overlap, _ = kv_write._MODES[mode]
    sw = (1 + overlap) * D
    (srows, lanes), sbs, (krows, klanes) = GEOM[mode]
    seq_lens = np.asarray(seq_lens, np.int32)
    R = len(seq_lens)
    max_len = int(seq_lens.max())
    sp = -(-max_len // sbs)
    n_state = R * sp + 1
    state_host = rng.standard_normal((n_state, srows, lanes), dtype=np.float32)
    kv_tokens = {"hca": krows // 2, "csa": krows, "idx": klanes}[mode]
    kp = max(1, -(-(max_len // ratio + 1) // kv_tokens))
    n_kv = R * kp + 1
    kv = rng.integers(-(2**31), 2**31 - 1, size=(n_kv, krows, klanes)).astype(np.int32)
    return Case(
        mode=mode,
        states=jnp.asarray(rng.standard_normal((R, Q_LEN, 2 * sw), dtype=np.float32)),
        seq_lens=jnp.asarray(seq_lens),
        state_host=jnp.asarray(state_host.view(np.int32)),
        state_table=jnp.asarray(1 + rng.permutation(R * sp).reshape(R, sp).astype(np.int32)),
        kv=jnp.asarray(kv),
        kv_table=jnp.asarray(1 + rng.permutation(R * kp).reshape(R, kp).astype(np.int32)),
        rope=(jnp.asarray(rng.integers(-(2**31), 2**31 - 1, size=(n_kv, krows // 4, 128))
                          .astype(np.int32)) if mode == "csa" else None),
        norm_w=jnp.asarray(1.0 + 0.1 * rng.standard_normal(D, dtype=np.float32)),
        cos_sin=jnp.asarray(np.concatenate(
            [np.cos(np.arange(max_len + 8)[:, None] * 0.01 * np.arange(1, 33)),
             np.sin(np.arange(max_len + 8)[:, None] * 0.01 * np.arange(1, 33))], 1)
            .astype(np.float32)),
    )


def golden(case):
    """The stock goldens: ref_wkv_proj_and_save_state (identity projection, no
    ape, so the stored state is `states`), then ref_compress_norm_rope_store.
    Returns (state_host words, kv words, rope words or None, kv records [n, 512]
    bytes for the HAS layout of idx)."""
    mode = case.mode
    D, ratio, overlap, qb = kv_write._MODES[mode]
    sw = (1 + overlap) * D
    (_, lanes), sbs, _ = GEOM[mode]
    R = case.seq_lens.shape[0]
    pos = (case.seq_lens[:, None] - Q_LEN + jnp.arange(Q_LEN)[None, :]).reshape(-1)
    qsl = jnp.arange(R + 1, dtype=jnp.int32) * Q_LEN
    kv_u8 = u8(case.kv, None)
    if mode == "idx":  # golden writes HAS quads: a [N, T/4, 4, 256] scratch of the same pages
        kv_u8 = jnp.zeros((case.kv.shape[0], 64, 4, 256), jnp.uint8)
    state_u8 = u8(case.state_host, lanes)
    _, slots, kv_slots = compressor_v1.derive_metadata(
        positions=pos, block_table=case.state_table.reshape(-1), query_start_loc=qsl,
        kv_block_table=case.kv_table.reshape(-1), cache=kv_u8 if mode != "csa" else case.kv,
        compress_ratio=ratio, state_block_size=sbs, head_dim=D, overlap=overlap,
        cos_sin_cache=case.cos_sin, state_cache=state_u8)
    new_state = ref_wkv_proj_and_save_state(
        hidden_states=case.states.reshape(-1, 2 * sw), wkv_wgate=jnp.eye(2 * sw, dtype=jnp.float32),
        ape=jnp.zeros((ratio, sw), jnp.float32), positions=pos, slot_mapping=slots,
        cache=state_u8, state_block_size=sbs, head_dim=D, compress_ratio=ratio, overlap=overlap)
    req = jnp.repeat(jnp.arange(R), Q_LEN)
    cache_out, rope_out = ref_compress_norm_rope_store(
        cache=kv_u8 if mode != "csa" else case.kv,
        rope_cache=case.rope, positions=pos, slot_mapping=slots,
        block_table=case.state_table.reshape(-1), token_to_req_indices=req,
        kv_slot_mapping=kv_slots, rms_weight=case.norm_w, cos_sin_cache=case.cos_sin,
        block_table_stride=case.state_table.shape[1], state_block_size=sbs, head_dim=D,
        rope_head_dim=64, compress_ratio=ratio, overlap=overlap, rms_eps=1e-6,
        quant_block=qb, is_quantized=mode != "hca", has_rope=True,
        has_rope_cache=mode == "csa", state_cache=new_state)
    state_words = csa_cache_layout.slabs_to_words(new_state)
    if mode == "hca":
        return state_words, csa_cache_layout.slabs_to_words(cache_out), None
    if mode == "csa":
        return state_words, cache_out, rope_out
    return state_words, cache_out, None  # idx: HAS quads u8 [N, 64, 4, 256]
