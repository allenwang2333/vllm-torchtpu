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
"""CPU tests for the megakernel's in-server diagnostics (diag/vllm_adapter.py):
the non-finite counter, the stock-vs-megakernel parity probe, and that with
both off the adapter does exactly what it did without them.

The kernel is a JAX function on CPU standing in for the torch_tpu export.
"""

from __future__ import annotations

import itertools
import re
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
from vllm_torchtpu.kernels.deepseek_v4_megakernel import vllm_adapter as VA
from vllm_torchtpu.kernels.deepseek_v4_megakernel.diag import vllm_adapter as SD

NS = types.SimpleNamespace
_op_ids = itertools.count()
H = 128
AUX = (2, 3)
Q = 3
NAN_ID = -7  # the fake kernel emits a NaN row for this input token


# ------------------------------------------------------------ pure counters


def test_nonfinite_counts_split_real_and_padding_rows():
    h = torch.zeros(6, 4, dtype=torch.bfloat16)
    lowest = torch.finfo(torch.bfloat16).min
    h[0, 0] = float("nan")
    h[1, :2] = float("inf")
    h[2, 3] = lowest  # finite, saturated
    h[4, :] = float("nan")  # padding
    h[5, 1] = -lowest
    aux = [torch.zeros(6, 4), torch.zeros(6, 4)]
    aux[1][3, 0] = float("nan")
    aux[0][5, 0] = float("-inf")
    rd = torch.tensor([0, 1, 1], dtype=torch.int32)  # one request of 3 rows
    got = dict(zip(SD._NF_FIELDS, SD.nonfinite_counts(h, aux, rd, 3).tolist()))
    assert got == dict(
        nan_real=1, inf_real=2, sat_real=1, bad_rows_real=2,
        nan_pad=4, inf_pad=0, sat_pad=1, bad_rows_pad=1,
        aux_bad_rows_real=0, aux_bad_rows_pad=2, steps_bad_real=1,
    )  # fmt: skip
    clean = SD.nonfinite_counts(torch.zeros(6, 4), [], rd, 3)
    assert clean.tolist() == [0] * len(SD._NF_FIELDS)


def test_nonfinite_stats_accumulate_drain_and_log_per_phase(capsys, monkeypatch):
    monkeypatch.setattr(SD, "_NF_DRAIN_STEPS", 2)
    st = SD.NonfiniteStats(3, 3, torch.device("cpu"), 3600.0, lambda f: f)
    h = torch.zeros(6, 4)
    h[1, 2] = float("nan")
    rd = torch.tensor([0, 2, 2], dtype=torch.int32)
    for _ in range(3):
        st.count(0, h, [], rd, 6)
    st.count_idle(0)
    st.count(1, torch.zeros(6, 4), [], rd, 6)
    assert st.totals[0]["nan_real"] == 2  # drained after two steps
    st.log(0)
    st.log(1)
    out = capsys.readouterr().err.splitlines()
    assert out[0] == (
        "[dsv4-megakernel rank 3] nonfinite pre-serving: steps busy 3 idle 1, "
        "steps_with_nonfinite_real 3; real rows 18: nan 3 inf 0 sat 0 bad_rows 3; "
        "pad rows: nan 0 inf 0 sat 0 bad_rows 0; aux bad_rows real 0 pad 0"
    )
    assert (
        "nonfinite served: steps busy 1 idle 0, steps_with_nonfinite_real 0;"
        in (out[1])
    )


# --------------------------------------------------------------- parity rows


def _head(vocab_start=1000, num_tokens=5, rows=8):
    w = torch.zeros(rows, H)
    for t in range(num_tokens):
        w[t, t] = 1.0  # logit of local token t = hidden[:, t]
    w[num_tokens:, :] = 100.0  # padding rows: must never win
    return SD.LocalHead(w, vocab_start, num_tokens)


def test_parity_rows_relative_error_and_shard_top1():
    head = _head()
    s = torch.zeros(3, H)
    s[:, 0] = 3.0
    s[:, 1] = torch.tensor([1.0, 2.0, 2.9])
    m = s.clone()
    m[1, 1] = 4.0  # row 1: top-1 flips from token 0 to token 1
    m[2, 5] = 0.3  # row 2: same top-1, small error
    aux_s = [s * 2]
    aux_m = [m * 2]
    f, i = SD.parity_rows(s, aux_s, m, aux_m, head.weight, head.vocab_start,
                          head.num_tokens)  # fmt: skip
    fr = dict(zip(SD._PF_FIELDS, f.tolist()))
    ir = dict(zip(SD._PI_FIELDS, i.tolist()))
    norm = [np.hypot(3, 1), np.hypot(3, 2), np.hypot(3, 2.9)]
    np.testing.assert_allclose(fr["rel_l2"], [0, 2 / norm[1], 0.3 / norm[2]],
                               rtol=1e-6)  # fmt: skip
    np.testing.assert_allclose(f[len(SD._PF_FIELDS)], fr["rel_l2"], rtol=1e-6)
    assert fr["max_abs_diff"] == pytest.approx([0, 2, 0.3])
    assert ir["s_tok"] == [1000, 1000, 1000]
    assert ir["m_tok"] == [1000, 1001, 1000]
    assert fr["s_at_m"] == pytest.approx([3, 2, 3])  # stock logit of mk's pick
    assert fr["m_at_s"] == pytest.approx([3, 3, 3])
    assert ir["bad_stock"] == [0, 0, 0] and ir["bad_mk"] == [0, 0, 0]
    m[0, 7] = float("nan")
    f, i = SD.parity_rows(s, aux_s, m, aux_m, head.weight, head.vocab_start,
                          head.num_tokens)  # fmt: skip
    assert i[3].tolist() == [1, 0, 0] and f[0, 0] == 0  # NaN left out, counted


def test_parity_selection_budgets_and_stride():
    cfg = SD.ParityConfig(_head(), served_budget=3, every=4, warmup_budget=2, tol=0.1)
    p = SD.ParityProbe(0, cfg, lambda f: f)
    assert [p.select(0) for _ in range(4)] == [True, True, False, False]
    got = [p.select(1) for _ in range(16)]
    assert [k for k, v in enumerate(got) if v] == [0, 4, 8]
    with pytest.raises(ValueError):
        SD.ParityProbe(0, SD.ParityConfig(_head(), 1, 0, 0, 0.1), lambda f: f)


# ------------------------------------------------------ adapter, end to end


def _fake_layers():
    """Two decoder layers (HCA, CSA) bound the way the runner binds them."""
    groups: dict[str, int] = {}

    def holder(name, t, gid):
        groups[name] = gid
        return NS(prefix=name, kv_cache=t)

    nope = torch.zeros(4, 8, 128, dtype=torch.int32)
    rope = torch.zeros(4, 2, 128, dtype=torch.int32)
    idx = torch.zeros(4, 2, 4, 256, dtype=torch.uint8)
    hca = torch.zeros(4, 4, 4, 128, dtype=torch.uint8)
    swa1 = torch.zeros(4, 8, 128, dtype=torch.int32)
    hca_attn = NS(
        compress_ratio=128, custom_prefix="l0", kv_cache=hca,
        swa_cache_layer=holder("l0.swa_cache", nope, 3),
        compressor=NS(state_cache=holder("l0.compressor.state_cache", swa1, 4)),
        indexer=None,
    )  # fmt: skip
    groups["l0"] = 1
    pair = (nope, rope)
    csa_attn = NS(
        compress_ratio=4, custom_prefix="l1", kv_cache=pair,
        swa_cache_layer=holder("l1.swa_cache", swa1, 3),
        compressor=NS(state_cache=holder("l1.compressor.state_cache", pair, 5)),
        indexer=NS(
            k_cache=holder("l1.indexer.k_cache", idx, 2),
            compressor=NS(state_cache=holder("l1.indexer.compressor.state_cache",
                                             idx, 6)),
        ),
    )  # fmt: skip
    groups["l1"] = 0
    return [NS(attn=hca_attn), NS(attn=csa_attn)], groups


class FakeKernel:
    """hidden[r, :2] = (id, position) of active rows, a NaN row where the id is
    NAN_ID; aux[a] = (a + 2) * hidden; every cache + 1 when a request is active."""

    cache_inplace = False

    def __init__(self):
        self.built = []

    def build_step(self, cfg, bucket):
        self.built.append((cfg, bucket))
        n_bt = len(cfg.group_widths)

        def fn(meta, ids, pos, seq, *rest):
            caches = rest[n_bt + 1 :]
            h = jnp.zeros((bucket.own_rows, H), jnp.float32)
            h = h.at[:, 0].set(ids.astype(jnp.float32))
            h = h.at[:, 1].set(pos.astype(jnp.float32))
            h = jnp.where((ids == NAN_ID)[:, None], jnp.nan, h)
            busy = jnp.any(seq > 0)
            outs = [h] + [h * (a + 2) for a in range(len(AUX))]
            return outs + [c + busy.astype(c.dtype) for c in caches]

        return fn


def _cpu_callable(name, fn, *, mesh, input_partition_specs, output_partition_specs,
                  donate_argnums=None):  # fmt: skip
    jitted = jax.jit(fn)

    def to_j(t):
        return jnp.asarray(t.numpy())

    def call(*tensors):
        return [torch.from_numpy(np.array(o)) for o in jitted(*map(to_j, tensors))]

    return call


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setattr(VA, "sharded_jax_callable", _cpu_callable)
    monkeypatch.setattr(VA.DSv4Megakernel, "direct_dispatch", False)
    syncs: list = []
    monkeypatch.setattr(VA, "_sync", lambda t: syncs.append(t))

    def build(rank=0, compile_fn=lambda f: f, **kw):
        layers, groups = _fake_layers()
        spec = KI.ModelSpec(hidden_size=H, num_layers=2, compress_ratios=(128, 4),
                            dtype="float32", q=Q, mode=VA.MODE_DP, axis_name="ep",
                            num_ranks=1, aux_layers=AUX)  # fmt: skip
        kernel = FakeKernel()
        mk = VA.DSv4Megakernel(
            spec=spec,
            buckets=VA.parse_row_buckets("16,32", Q, reqs=4),
            kernel=kernel,
            mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]), ("ep",)),
            rank=rank,
            expert_block=0,
            weights=[torch.ones(8)],
            decoder_layers=layers,
            device="cpu",
            compile_fn=compile_fn,
            op_namespace=f"dsv4_mk_diag_test{next(_op_ids)}",
            **kw,
        )
        bound = VA.bind_layer_caches(layers, groups)
        mk.bind(bound, [3] * len(bound.group_ids))
        return mk, kernel, syncs

    return build


def _metadata(mk, num_reqs, padded_reqs=4):
    seq = torch.full((padded_reqs,), 999, dtype=torch.int32)
    seq[:num_reqs] = 100
    return {
        name: NS(
            block_tables=torch.ones(padded_reqs * 3, dtype=torch.int32),
            seq_lens=seq,
            request_distribution=torch.tensor(
                [0, num_reqs, num_reqs], dtype=torch.int32
            ),  # fmt: skip
            sequence_layout_kind="all",
        )
        for name in mk.caches.group_layer_names
    }


def _decide(mk, rows, idle=False):
    local = None if idle else VA.LocalStep(VA.ELIGIBLE, VA.ELIGIBLE, rows)
    mk._local_real_rows = 0 if idle else rows // 4 * Q  # what classify() stores
    return mk.decide_dp(local, max_tokens=rows, max_chunks=1, any_prefill=False,
                        entries=[0, 0, rows])  # fmt: skip


def _parity_cfg(**kw):
    kw = dict(served_budget=2, every=1, warmup_budget=1, tol=0.1) | kw
    return SD.ParityConfig(_head(vocab_start=0, num_tokens=4), **kw)


def _never(*a):
    raise AssertionError("stock forward called")


def test_flags_off_runs_no_diagnostics_and_keeps_the_kernel_program(make, monkeypatch):
    """Off: no extra compiled graph, sync or stock call, every step owned by the
    kernel, its outputs served as returned; the kernel builds the same steps
    from the same config as with both diagnostics on."""
    compiled_off, compiled_on = [], []
    off, k_off, syncs = make(compile_fn=lambda f: compiled_off.append(f) or f)
    on, k_on, _ = make(compile_fn=lambda f: compiled_on.append(f) or f,
                       nonfinite_stats=True, parity=_parity_cfg())  # fmt: skip
    assert [(c.cache_shapes, c.group_widths, b) for c, b in k_off.built] == [
        (c.cache_shapes, c.group_widths, b) for c, b in k_on.built
    ]
    assert len(compiled_on) == len(compiled_off) + 2  # counter + comparison
    assert off._nonfinite is None and off._parity is None
    md = _metadata(off, 2)
    monkeypatch.setattr(VA, "_current_attn_metadata", lambda: md)
    ids = torch.arange(8, dtype=torch.int32) + 5
    before = [t.clone() for t in off.caches.arrays]
    for serving in (False, True):
        off.serving = serving
        d = _decide(off, 8)
        assert d.run and not d.parity and off.owns_step()
        hidden, aux = off.forward(_never, ids, ids, None, None)
        np.testing.assert_array_equal(hidden[:6, 0].numpy(), ids[:6].numpy())
        assert len(aux) == len(AUX)
    assert syncs == []
    assert all(torch.equal(t, b + 2) for t, b in zip(off.caches.arrays, before))
    off.mark_serving()  # a no-op beyond the phase flag


def test_nonfinite_stats_in_the_adapter(make, monkeypatch, capsys):
    mk, _, _ = make(nonfinite_stats=True)
    assert "self-check" not in capsys.readouterr().err  # bind_caches logs it
    md = _metadata(mk, 2)
    monkeypatch.setattr(VA, "_current_attn_metadata", lambda: md)
    ids = torch.arange(8, dtype=torch.int32)
    ids[4] = NAN_ID  # a real row (2 requests x 3 rows)
    for _ in range(2):
        _decide(mk, 8)
        mk.forward(_never, ids, ids, None, None)
    _decide(mk, 8, idle=True)
    mk.forward(_never, ids, ids, None, None)
    mk.mark_serving()
    _decide(mk, 8)
    mk.forward(_never, torch.zeros(8, dtype=torch.int32), ids, None, None)
    mk._nonfinite.log(mk.phase)
    err = capsys.readouterr().err.splitlines()
    pre = [x for x in err if "nonfinite pre-serving" in x]
    served = [x for x in err if "nonfinite served" in x]
    assert len(pre) == 1 and len(served) == 1
    assert (
        "steps busy 2 idle 1, steps_with_nonfinite_real 2; real rows 12: "
        f"nan {2 * H} inf 0 sat 0 bad_rows 2;" in pre[0]
    )
    assert "aux bad_rows real 2 pad 0" in pre[0]
    assert "steps busy 1 idle 0, steps_with_nonfinite_real 0;" in served[0]


def test_parity_runs_stock_then_kernel_and_serves_the_kernel(make, monkeypatch, capsys):
    mk, _, syncs = make(parity=_parity_cfg())
    md = _metadata(mk, 2)
    monkeypatch.setattr(VA, "_current_attn_metadata", lambda: md)
    ids = torch.arange(8, dtype=torch.int32) + 1
    events = []
    before = [t.clone() for t in mk.caches.arrays]

    def stock(input_ids, positions, embeds, inter):
        events.append(("stock", len(syncs)))
        h = torch.zeros(8, H)
        h[:, 0] = input_ids.float()
        h[:, 1] = positions.float()
        h[1, 2] = 0.5  # one row off by a known amount
        return h, [h * (a + 2) for a in range(len(AUX))]

    # pre-serving budget 1, then served budget 2
    picks = []
    for serving in (False, False, True, True, True):
        mk.serving = serving
        d = _decide(mk, 8)
        picks.append(d.parity)
        assert mk.owns_step() is not d.parity  # stock MoE setup on parity steps
        hidden, aux = mk.forward(stock if d.parity else _never, ids, ids, None, None)
        # the served output is the kernel's
        np.testing.assert_array_equal(hidden[:6, 0].numpy(), ids[:6].numpy())
        assert hidden[1, 2] == 0
    assert picks == [True, False, True, True, False]
    # the stock forward ran before a sync, the kernel after it
    assert [e[0] for e in events] == ["stock"] * 3
    assert all(syncs[n] is None for _, n in events)
    assert all(torch.equal(t, b + 5) for t, b in zip(mk.caches.arrays, before))
    err = capsys.readouterr().err
    steps = re.findall(r"parity (\S+) #(\d) step \d+ mk_q3_t16 ctx 0 rows 6 stock_mk: "
                       r"rel_l2 max (\S+) ", err)  # fmt: skip
    norm = np.sqrt(2**2 + 2**2 + 0.5**2)  # the stock row (id 2, position 2)
    assert [(p, n) for p, n, _ in steps] == [
        ("pre-serving", "1"),
        ("served", "1"),
        ("served", "2"),
    ]
    assert all(float(r) == pytest.approx(0.5 / norm, rel=1e-3) for *_, r in steps)
    assert err.count("shard top1 agree 6/6; over_tol 1 ") == 3  # 0.17 > tol 0.1
    assert "parity summary served stock_mk: calls 2 (with rows 2), rows 12" in err


def test_parity_on_an_idle_rank_pairs_the_stock_forward_and_logs_nothing(make, capsys):
    mk, _, syncs = make(rank=1, parity=_parity_cfg())
    calls = []

    def stock(*a):
        calls.append(1)
        return torch.zeros(8, H), [torch.zeros(8, H)] * len(AUX)

    mk.serving = True
    d = _decide(mk, 8, idle=True)
    assert d.run and d.idle and d.parity
    zeros = torch.zeros(8, dtype=torch.int32)
    mk.forward(stock, zeros, zeros, None, None)
    assert calls == [1] and syncs == [None, None]  # after each of the pair
    assert "parity served" not in capsys.readouterr().err


def test_parity_refuses_ab_and_warmup_compiles_the_diagnostics(make):
    with pytest.raises(VA.Unsupported, match="AB"):
        make(parity=_parity_cfg(), ab_enabled=True)
    names = []

    def compile_fn(f):
        names.append(f.__code__.co_name)
        return f

    mk, _, _ = make(compile_fn=compile_fn, nonfinite_stats=True, parity=_parity_cfg())
    warmed = []
    mk._nonfinite.warm = lambda h, a, rd: warmed.append(("nf", h.shape[0]))
    mk._parity.warm = lambda h, a: warmed.append(("parity", h.shape[0]))
    mk.warmup(NS(num_tokens_paddings=[8, 16, 32], num_reqs_max_model_len=4))
    # per bucket: the counter on the kernel's 12 own rows, the comparison at
    # every runner row count the bucket reaches
    assert warmed == [("nf", 12), ("parity", 8), ("parity", 16)] * 2
    assert {"update", "rows"} <= set(names)


@pytest.mark.parametrize("pair", [("mk", "mk"), ("stock", "stock"), ("mk", "stock")])
def test_parity_control_pairs(make, monkeypatch, capsys, pair):
    """Controls: the same path twice gives the comparison's floor (0 on CPU);
    the megakernel output is served and counted once."""
    mk, _, syncs = make(parity=_parity_cfg(pairs=(pair,)), nonfinite_stats=True)
    md = _metadata(mk, 2)
    monkeypatch.setattr(VA, "_current_attn_metadata", lambda: md)
    ids = torch.arange(8, dtype=torch.int32) + 1
    stock_calls = []

    def stock(input_ids, positions, embeds, inter):
        stock_calls.append(1)
        h = torch.zeros(8, H)
        h[:, 0] = input_ids.float()
        h[:, 1] = positions.float() + 1  # differs from the kernel's output
        return h, [h * (a + 2) for a in range(len(AUX))]

    before = [t.clone() for t in mk.caches.arrays]
    mk.serving = True
    assert _decide(mk, 8).parity
    hidden, _ = mk.forward(stock, ids, ids, None, None)
    np.testing.assert_array_equal(hidden[:6, 1].numpy(), ids[:6].numpy())  # kernel's
    assert len(stock_calls) == pair.count("stock")
    kernel_runs = max(pair.count("mk"), 1)
    assert all(
        torch.equal(t, b + kernel_runs) for t, b in zip(mk.caches.arrays, before)
    )
    assert syncs == [None, None]
    assert mk._nonfinite.busy[1] == 1
    err = capsys.readouterr().err
    tag = "_".join(pair)
    if pair[0] == pair[1]:
        assert f"rows 6 {tag}: rel_l2 max 0 mean 0;" in err
    else:
        assert f"rows 6 {tag}: rel_l2 max 0 " not in err
    with pytest.raises(ValueError):
        _parity_cfg(pairs=(("mk", "draft"),))


def test_parity_pairs_take_turns():
    cfg = _parity_cfg(served_budget=5, pairs=(("stock", "mk"), ("mk", "mk")))
    p = SD.ParityProbe(0, cfg, lambda f: f)
    got = []
    for _ in range(5):
        assert p.select(1)
        got.append(p.pair)
    assert got == [("stock", "mk"), ("mk", "mk")] * 2 + [("stock", "mk")]
