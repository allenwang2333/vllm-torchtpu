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
"""The DSpark draft megakernel adapter on CPU: draft caches bound as the runner
binds them, the draft bucket of each target bucket, the operand assembly from
the proposer's padded draft block and the target step's verify rows, idle
steps, and the stand-in kernel's view of every operand."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
from vllm_torchtpu.kernels.deepseek_v4_megakernel import vllm_adapter as VA
from vllm_torchtpu.kernels.deepseek_v4_megakernel.draft import vllm_adapter as DA

from .runner_binding import runner_bound_model
from .test_vllm_adapter import cpu_sharded_jax_callable

H, K, CTX_Q, N_AUX, N_W = 128, 6, 7, 3, 70


class DraftEchoKernel:
    """hidden[r] = (ids, pos, seq_len, sum of the request's ctx_aux, sum of its
    ctx positions, ctx seq_len, block-table sum, rank, 0..)."""

    cache_inplace = True

    def __init__(self):
        self.built = []

    def build_draft_step(self, cfg, bucket, ctx_q):
        self.built.append((bucket, ctx_q))
        M, B, q, n_bt = bucket.own_rows, bucket.reqs, bucket.q, len(cfg.group_widths)
        assert cfg.model.compress_ratios == (1, 1, 1)
        assert all(lc.kind == KI.SWA for lc in cfg.layers)

        def fn(meta, ids, pos, seq, ctx_aux, ctx_pos, ctx_seq, *rest):
            assert ids.shape == (M,) and seq.shape == (B,)
            assert ctx_aux.shape == (B * ctx_q, N_AUX * H) and ctx_pos.shape == (
                B * ctx_q,
            )
            bts = rest[:n_bt]
            req = jnp.arange(M) // q
            per_req = lambda x: x.reshape(B, ctx_q, -1).sum((1, 2))  # noqa: E731
            cols = [ids, pos, seq[req], per_req(ctx_aux.astype(jnp.float32))[req],
                    per_req(ctx_pos)[req], ctx_seq[req],
                    sum(bt.sum(1) for bt in bts)[req],
                    jnp.full((M,), meta[KI.META_RANK])]  # fmt: skip
            h = jnp.zeros((M, H), jnp.float32)
            return (h.at[:, : len(cols)].set(jnp.stack(cols, 1).astype(jnp.float32)),)

        return fn


@pytest.fixture
def draft(monkeypatch):
    monkeypatch.setattr(VA, "sharded_jax_callable", cpu_sharded_jax_callable)
    targets, drafts, groups, kv = runner_bound_model(n_draft=3)
    spec = KI.ModelSpec(
        hidden_size=H, num_layers=3, compress_ratios=(1, 1, 1), dtype="float32",
        q=K, mode=VA.MODE_TP, axis_name="ep", num_ranks=1, aux_layers=(),
        num_hash_layers=0,
    )  # fmt: skip
    kernel = DraftEchoKernel()
    d = DA.DraftMegakernel(
        spec=spec, ctx_q=CTX_Q, n_aux=N_AUX, kernel=kernel,
        mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]), ("ep",)), rank=0,
        expert_block=0, weights=[torch.full((8,), float(i)) for i in range(N_W)],
        draft_layers=drafts, device="cpu", compile_fn=lambda f: f,
        direct_dispatch=False,
    )  # fmt: skip
    group_of = {n: g for g, grp in enumerate(groups) for n in grp.layer_names}
    caches = VA.bind_layer_caches(drafts, group_of)
    target_buckets = VA.parse_row_buckets("16,32", CTX_Q, 2)
    d.bind(caches, [5] * len(caches.group_ids), target_buckets)
    return d, kernel, drafts, target_buckets


def test_draft_buckets_hold_the_target_requests(draft):
    d, kernel, drafts, tbs = draft
    for tb, (b, ctx_q) in zip(tbs, kernel.built):
        assert (b.q, b.reqs, ctx_q) == (K, tb.reqs, CTX_Q)
        assert b.rows // KI.request_block_rows(K) == tb.rows // KI.request_block_rows(
            CTX_Q
        )
    # only the draft layers' SWA caches, each once
    assert {id(t) for t in d.caches.arrays} == {
        id(lyr.attn.swa_cache_layer.kv_cache) for lyr in drafts
    }


def test_draft_operands_from_the_proposer_block_and_verify_rows(draft):
    d, _, _, tbs = draft
    tb = tbs[0]  # T=16: 2 requests
    padded = 16  # the proposer's draft bucket (>= 2 * K)
    ids = torch.arange(padded, dtype=torch.int32) + 100
    pos = torch.arange(padded, dtype=torch.int32) + 50
    seq = torch.tensor([61, 72, 999, 999], dtype=torch.int32)  # stale past 2
    rows_t = 16  # the target step's padded rows
    aux = [torch.full((rows_t, H), float(a + 1)) for a in range(N_AUX)]
    cpos = torch.arange(rows_t, dtype=torch.int32) + 1000
    cseq = torch.tensor([60, 71, 999, 999], dtype=torch.int32)
    bt = [(torch.arange(4 * 5, dtype=torch.int32) + 1) for _ in d.caches.group_ids]
    h = d.forward(tb, ids, pos, seq, aux, cpos, cseq, torch.tensor(2), bt).numpy()
    assert h.shape == (padded, H)
    real = 2 * K
    np.testing.assert_array_equal(h[:real, 0], ids[:real].numpy())
    np.testing.assert_array_equal(h[:real, 1], pos[:real].numpy())
    np.testing.assert_array_equal(h[:real, 2], np.repeat([61, 72], K))
    aux_sum = CTX_Q * H * sum(range(1, N_AUX + 1))
    np.testing.assert_array_equal(h[:real, 3], aux_sum)
    want_cpos = [cpos[r * CTX_Q : (r + 1) * CTX_Q].sum().item() for r in range(2)]
    np.testing.assert_array_equal(h[:real, 4], np.repeat(want_cpos, K))
    np.testing.assert_array_equal(h[:real, 5], np.repeat([60, 71], K))
    want_bt = [
        len(bt) * (torch.arange(r * 5 + 1, r * 5 + 6).sum().item()) for r in range(2)
    ]
    np.testing.assert_array_equal(h[:real, 6], np.repeat(want_bt, K))
    np.testing.assert_array_equal(h[real:], 0)  # bucket rows past the requests
    # one request: the second request's rows and context are inactive
    h1 = d.forward(tb, ids, pos, seq, aux, cpos, cseq, torch.tensor(1), bt).numpy()
    np.testing.assert_array_equal(h1[K:], 0)


def test_idle_and_warmup_run_every_bucket(draft, monkeypatch):
    d, _, _, tbs = draft
    synced = []
    monkeypatch.setattr(VA, "_sync", lambda t: synced.append(t))
    d.warmup()
    assert len(synced) == len(tbs)
    d.idle(tbs[1])  # every request inactive: no error, no rows
    out = d._run(tbs[1], d._idle[tbs[1]])
    assert not out.any()


def test_draft_refuses_non_default_layout_operands(draft, monkeypatch):
    d, _, drafts, tbs = draft
    bad = d.caches.arrays[0]
    monkeypatch.setattr(VA, "default_layout", lambda t: t is not bad)
    with pytest.raises(VA.Unsupported, match="draft operand 0"):
        d.bind(d.caches, [5] * len(d.caches.group_ids), tbs)
