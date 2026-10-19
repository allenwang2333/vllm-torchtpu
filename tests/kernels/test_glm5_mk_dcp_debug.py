# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""CPU tests of the glm5_mk_dcp G0 debug instruments (kernels/glm5_mk_dcp/debug.py).

Run: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=16 pytest <this file>

- modes: plan flags, alt:k sequences, cycle advancing per new request;
- SPECDBG: on the runner's own padded spec metadata (SpeculativeDecodingManager) and the real
  rejection sampler (eager), the fixed stats equal the truth and the old formula does not; the
  position-0 E[p] vs realised acceptance control agrees;
- metadata identity: clean on consistent metadata, catches a one-position shift;
- KVDIFF / writer placement (TINY, interpret): the megakernel's barrier-free writer, fed staged rows,
  must produce exactly the caches main's DCP golden writer (update_sparse_mla_kv_cache_dcp_jax)
  produces from the same bytes (nope, rope) and the SAL record layout (idx); `compare_staging`
  reports 0 on both and catches a writer fed a shifted position (fault control).
"""

import types

import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.glm5_mk_dcp import debug as D


# ------------------------------------------------------------------------------- modes


def test_mode_plans():
    s = D.ModeSchedule("mk")
    p = s.plan()
    assert p.pure_mk and p.mk_writer and not p.run_stock
    s = D.ModeSchedule("stock")
    assert s.plan().pure_stock and not s.needs_probe
    sh = D.ModeSchedule("shadow")
    p = sh.plan()
    assert p.run_stock and p.run_mk and not p.mk_writer and p.ret == "S" and p.stats
    assert sh.needs_probe
    p = D.ModeSchedule("mix:MSM").plan()      # MK logits + drafter, stock KV
    assert p.run_stock and p.run_mk and not p.mk_writer and p.ret == "M" and p.draft == "M"
    p = D.ModeSchedule("mix:SMS").plan()      # stock logits + drafter, MK KV
    assert p.run_stock and p.mk_writer and p.ret == "S" and p.draft == "S"
    assert not D.ModeSchedule("mix:SMS").needs_probe
    with pytest.raises(ValueError):
        D.ModeSchedule("mix:SX")
    with pytest.raises(ValueError):
        D.ModeSchedule("bogus")


def test_alt_sequences():
    s = D.ModeSchedule("alt:2")
    seq = ["S" if s.plan().pure_stock else "M" for _ in range(6)]
    assert seq == ["M", "S", "M", "S", "M", "S"]          # == the old GLM5_MKD_ALT=1
    s = D.ModeSchedule("alt:4")
    seq = ["S" if s.plan().pure_stock else "M" for _ in range(8)]
    assert seq == ["M", "M", "M", "S", "M", "M", "M", "S"]


def test_cycle_advances_per_new_request():
    s = D.ModeSchedule("cycle:stock/mk,shadow")
    labels = []
    for req in ["a", "a", "b", "b", "b", "c", "d", "d"]:
        s.observe_requests([req])
        labels.append(s.plan().label)
    assert labels == ["0:stock", "0:stock", "1:mk", "1:mk", "1:mk", "2:shadow", "0:stock", "0:stock"]
    assert s.needs_probe


def test_env_default(monkeypatch):
    monkeypatch.delenv("GLM5_MKD_MODE", raising=False)
    monkeypatch.setenv("GLM5_MKD_ALT", "1")
    assert D.schedule_from_env().spec == "alt:2"
    monkeypatch.delenv("GLM5_MKD_ALT")
    assert D.schedule_from_env().spec == "mk"


# ------------------------------------------------------------------------------- SPECDBG


def _spec_md(num_draft, padded_num_reqs, input_ids, num_tokens_paddings=(16, 32, 64)):
    """The runner's own SpecDecodeMetadata for a step, plus the async draft extract."""
    from vllm_torchtpu.runner.speculative_decoding_manager import SpeculativeDecodingManager
    from vllm_torchtpu.runner.tpu_runner_async_output import extract_draft_token_ids

    runner = types.SimpleNamespace(
        input_ids_cpu=torch.as_tensor(input_ids, dtype=torch.int32),
        arange_np=np.arange(4096, dtype=np.int64),
        num_tokens_paddings=list(num_tokens_paddings),
        device=torch.device("cpu"),
    )
    mgr = SpeculativeDecodingManager.__new__(SpeculativeDecodingManager)
    mgr.runner = runner
    mgr._meta_cache_key = mgr._meta_cache = mgr._meta_cache_np = None
    nd = np.asarray(num_draft, dtype=np.int32)
    cu = np.cumsum(nd + 1).astype(np.int32)
    md = mgr.get_spec_decode_metadata(nd, cu, padded_num_reqs, defer_draft_token_ids=True)
    md.draft_token_ids = extract_draft_token_ids(
        torch.as_tensor(input_ids, dtype=torch.int64), md.final_logits_indices,
        md.target_logits_indices)
    return md


def test_valid_slots_exclude_async_padding():
    # CONC=1, K=3: one request, 4 logits rows, padded to a 16-token bucket. The async extract fills
    # every padded slot with request 0's first draft (a real token id), so `draft >= 0` counts 16.
    ids = np.array([5, 11, 12, 13] + [0] * 12)
    md = _spec_md([3], 1, ids)
    assert md.draft_token_ids.shape[0] == 16
    assert int((md.draft_token_ids >= 0).sum()) == 16          # what the old SPECDBG counted
    assert md.draft_token_ids[3:].eq(11).all()                  # padded slots = first draft
    v = D.valid_draft_slots(md.segment_ids, md.group_indices, md.draft_lengths)
    assert v.tolist() == [True] * 3 + [False] * 13


def _sampler():
    from vllm_torchtpu.layers.adapter.sample import rejection_sampler as RS

    return RS._random_rejection_sample_with_segment._torchdynamo_orig_callable


def test_specdbg_matches_sampler_truth_and_old_formula_does_not():
    torch.manual_seed(0)
    V, K = 64, 3
    fn = _sampler()
    dbg = D.SpecDbg(every=10 ** 9)
    p_true = torch.tensor([0.8, 0.9, 0.95])                      # p(draft) at slots 0..2
    old_acc = [0, 0]
    tot_acc = tot_req = 0
    nsteps = 3000
    for step in range(nsteps):
        ids = np.array([5, 11, 12, 13] + [0] * 12)
        md = _spec_md([K], 1, ids)
        n = md.draft_token_ids.shape[0]
        d = md.draft_token_ids.to(torch.int64)
        logits = torch.full((n, V), -1e4)
        for j in range(n):                                       # p(d_j) = p_true[min(j, 2)]
            pj = float(p_true[min(j, K - 1)])
            logits[j] = np.log((1 - pj) / (V - 1))
            logits[j, d[j]] = np.log(pj)
        temps = torch.ones(1, 1)
        top_k = torch.zeros(1, 1, dtype=torch.int32)
        top_p = torch.ones(1, 1)
        seg = md.segment_ids.clamp(max=0)
        accept_u = torch.rand(n)
        recover_u = torch.rand(n, V)
        sel = fn(md.draft_token_ids, logits, md.draft_lengths, torch.tensor([7]), md.segment_ids,
                 md.group_indices, K, temps[seg], top_k[seg], top_p[seg], accept_u, recover_u,
                 None, False, 0)
        dbg.on_logits("stock", md, logits, temps, top_k, top_p, 0, False)
        dbg.on_output("stock", md, sel, rank=0)
        v = int((sel[0] >= 0).sum())
        tot_acc += v - 1
        tot_req += 1
        old_acc[0] += int((sel >= 0).sum())                      # old: every row, padded included
        old_acc[1] += int(sel.shape[0])
    s = dbg._s("stock")
    assert s["accepted"] == tot_acc and s["reqs"] == tot_req and s["drafted"] == K * nsteps
    al_true = 1 + tot_acc / tot_req
    al_expect = 1 + 0.8 + 0.8 * 0.9 + 0.8 * 0.9 * 0.95
    assert abs(al_true - al_expect) < 0.08
    # positive control: E[p] over position-0 slots == realised position-0 acceptance
    assert abs(s["p0"] / s["n0"] - 0.8) < 1e-4
    assert abs(s["acc0"] / s["reqs"] - 0.8) < 0.03
    # mean p over real slots only = mean(p_true)
    assert abs(s["p"] / s["slots"] - float(p_true.mean())) < 1e-4
    assert s["slots"] == 3 * nsteps


def test_specdbg_padded_rows_dilute_old_formula():
    """With padded request rows (padded_num_reqs 8 at one real request) the old per-row mean is
    pulled to ~1 by the padded rows' bonus token; the fixed stat uses real rows only."""
    torch.manual_seed(1)
    V, K = 32, 3
    fn = _sampler()
    ids = np.array([5, 11, 12, 13] + [0] * 12)
    md = _spec_md([K], 8, ids)
    n = md.draft_token_ids.shape[0]
    logits = torch.full((n, V), -1e4)
    logits[torch.arange(n), md.draft_token_ids.to(torch.int64)] = 0.0   # p(draft) = 1
    P = md.draft_lengths.shape[0]
    temps, top_k, top_p = torch.ones(P, 1), torch.zeros(P, 1, dtype=torch.int32), torch.ones(P, 1)
    seg = md.segment_ids.clamp(max=P - 1)
    sel = fn(md.draft_token_ids, logits, md.draft_lengths, torch.full((P,), 7), md.segment_ids,
             md.group_indices, K, temps[seg], top_k[seg], top_p[seg], torch.rand(n),
             torch.rand(n, V), None, False, 0)
    old = float((sel >= 0).sum(-1).float().mean())
    dbg = D.SpecDbg(every=10 ** 9)
    dbg.on_logits("x", md, logits, temps, top_k, top_p, 0, False)
    dbg.on_output("x", md, sel, rank=0)
    s = dbg._s("x")
    assert 1 + s["accepted"] / s["reqs"] == 4.0                     # all 3 accepted + bonus
    assert old == pytest.approx((4 + 7 * 1) / 8)                    # 1.375: the dilution


# ---------------------------------------------------------------------- metadata identity


def test_metadata_identity():
    qsl = np.array([0, 4, 8, 8, 8])
    seq = np.array([1000, 701, 0, 0])
    pos = np.array([996, 997, 998, 999, 697, 698, 699, 700])
    assert D.metadata_identity(qsl, seq, pos)[:2] == (2, 0)
    pos2 = pos.copy()
    pos2[4:8] += 1                                                 # positions one ahead
    n, bad, ex = D.metadata_identity(qsl, seq, pos2)
    assert (n, bad) == (2, 1) and ex[0]["b"] == 1


# --------------------------------------------------------------- KVDIFF + writer placement


def _tiny_writer_case(shift_row=None):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.glm5_mk_dcp import config as C
    from vllm_torchtpu.kernels.glm5_mk_dcp import kernel as K
    from vllm_torchtpu.kernels.mla import kv_cache_utils as U

    cfg = C.TINY
    R, L, TMAX, pg, ILV = cfg.R, cfg.L, cfg.TMAX, cfg.page, cfg.ILV
    NB, MAXP = 16, 4
    rng = np.random.default_rng(7)
    seqs = [1003, 701]                                             # req 0 straddles an ILV chunk
    rows = [(b, seqs[b] - 4 + i) for b in range(2) for i in range(4)]
    pos = np.array([p for _, p in rows])
    req = np.array([b for b, _ in rows])
    scal = np.zeros(8, np.int32)
    scal[0], scal[1] = TMAX, 2
    rmeta = np.zeros((TMAX, 8), np.int32)
    for t, (b, p) in enumerate(rows):
        rmeta[t, :4] = (1, p, b, seqs[b])
    bt = np.zeros((TMAX, MAXP), np.int32)
    for b in range(2):
        bt[b] = (np.arange(MAXP) * 3 + 5 * b + 1) % NB
    # staged rows: nope words (any bytes, fp8 NaN-free), rope at the token's quarter (16 bits used)
    nrow = (rng.integers(0, 1 << 32, (L, TMAX, 128), dtype=np.uint64) & 0x37373737).astype(np.uint32)
    rrow = np.zeros((L, TMAX, 128), np.uint32)
    for t, p in enumerate(pos):
        q = p & 3
        rrow[:, t, 32 * q:32 * q + 32] = rng.integers(0, 1 << 16, (L, 32)) & 0x3737
    irec = np.zeros((L, TMAX, 256), np.int32)
    irec[..., :cfg.IREC] = rng.integers(0, 256, (L, TMAX, cfg.IREC)) & 0x37
    ck = K.cache_kinds(cfg, NB)

    def rand_cache(kind):
        _, shp, dt = ck[kind]
        nbytes = int(np.prod(shp)) * np.dtype(dt).itemsize
        return [(rng.integers(0, 256, nbytes, dtype=np.uint8) & 0x37).view(dt).reshape(shp)
                for _ in range(R)]

    caches0 = {(k, l): rand_cache(k) for k, (ls, _, _) in ck.items() for l in ls}
    c_names = [(k, l) for k, (ls, _, _) in ck.items() for l in ls]

    # --- golden: main's DCP writer (nope, rope) + the SAL record layout (idx) from the same bytes
    nope_spec = U.SparseMLAKVCacheSpec.create(U.KVCacheType.NOPE, U.KVCacheLayout.SPARSECORE, NB, pg, 512)
    rope_spec = U.SparseMLAKVCacheSpec.create(U.KVCacheType.ROPE, U.KVCacheLayout.SPARSECORE, NB, pg, 64)
    w = nrow.astype(np.uint64)
    nbytes = np.concatenate([(w >> (8 * s)) & 0xFF for s in range(4)], -1).astype(np.uint8)  # [L,T,512]
    rbytes = np.zeros((L, TMAX, 64), np.uint8)
    for t, p in enumerate(pos):
        q = p & 3
        rw = rrow[:, t, 32 * q:32 * q + 32].astype(np.uint32)
        rbytes[:, t, :32] = rw & 0xFF
        rbytes[:, t, 32:] = (rw >> 8) & 0xFF
    seq_lens = jnp.asarray(np.array(seqs, np.int32))
    qsl = jnp.asarray(np.array([0, 4, 8], np.int32))
    gold = {kl: [c.copy() for c in v] for kl, v in caches0.items()}
    for r in range(R):
        for l in range(L):
            kv_c = jax.lax.bitcast_convert_type(jnp.asarray(nbytes[l]), jnp.float8_e4m3fn)
            k_pe = jax.lax.bitcast_convert_type(jnp.asarray(rbytes[l]), jnp.float8_e4m3fn)
            n_new, r_new = U.update_sparse_mla_kv_cache_dcp_jax(
                jnp.asarray(gold[("nope", l)][r]), jnp.asarray(gold[("rope", l)][r]), kv_c, k_pe,
                seq_lens, jnp.asarray(bt[:2].reshape(-1)), qsl, jnp.int32(r),
                nope_spec=nope_spec, rope_spec=rope_spec, dcp_size=R, interleave_size=ILV)
            gold[("nope", l)][r] = np.asarray(n_new)
            gold[("rope", l)][r] = np.asarray(r_new)
        G = ILV * R
        for t, p in enumerate(pos):
            if (p % G) // ILV != r:
                continue
            lo = (p // G) * ILV + p % ILV
            page, slot = bt[req[t], lo // pg] % NB, lo % pg
            for l in ck["idx"][0]:
                col = irec[l, t, :cfg.IREC].astype(np.uint8)
                gold[("idx", l)][r][page, :, :, slot] = col.reshape(cfg.IREC // 4, 4)

    # --- the megakernel's barrier-free writer (TPU interpret mode, R CPU devices)
    rm = rmeta.copy()
    if shift_row is not None:
        rm[shift_row, 1] += 1
    writer, _ = K.build_writer(cfg, axis_name="d", nb=NB, max_pages=MAXP, bmax=TMAX, interpret=True)
    mesh = Mesh(np.array(jax.devices()[:R]), ("d",))

    def body(*ops):
        ops = [o[0] for o in ops]
        return tuple(o[None] for o in writer(*ops))

    n_in = 6 + len(c_names)
    fn = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("d"),) * n_in,
                               out_specs=(P("d"),) * len(c_names), check_vma=False))

    def rep(x):
        return np.broadcast_to(x[None], (R,) + x.shape).copy()

    ops = [rep(scal), rep(rm), rep(bt), rep(nrow), rep(rrow), rep(irec)] + [
        np.stack(caches0[kl]) for kl in c_names]
    outs = [np.asarray(o) for o in fn(*[jnp.asarray(o) for o in ops])]
    mk = {kl: [o[r] for r in range(R)] for kl, o in zip(c_names, outs)}
    return dict(cfg=cfg, R=R, NB=NB, pos=pos, req=req, bt=bt, nrow=nrow, rrow=rrow, irec=irec,
                gold=gold, mk=mk, caches0=caches0, ck=ck)


def _compare(case, which, r, snap_from=None):
    cfg = case["cfg"]
    caches = case[which]

    def cache_of(kind, l, src=caches):
        v = src.get((kind, l))
        if v is None:
            return None
        a = v[r]
        return torch.from_numpy(a.view(np.int32) if a.dtype == np.uint32 else a)

    c = D.owned_row_coords(torch.as_tensor(case["pos"]), torch.as_tensor(case["req"]),
                           torch.ones(len(case["pos"]), dtype=torch.bool), torch.as_tensor(case["bt"]),
                           case["NB"], cfg.R, cfg.ILV, cfg.page, r)
    snap = None
    if snap_from is not None:
        snap = D.snapshot(lambda k, l: cache_of(k, l, case[snap_from]), cfg.L,
                          list(case["ck"]["idx"][0]), c)
    return D.compare_staging(cache_of, cfg.L, list(case["ck"]["idx"][0]), c,
                             torch.from_numpy(case["nrow"].view(np.int32)),
                             torch.from_numpy(case["rrow"].view(np.int32)),
                             torch.from_numpy(case["irec"]), cfg.IDIM, cfg.IDIM + 1, neighbors=snap)


@pytest.fixture(scope="module")
def writer_case():
    import jax

    if len(jax.devices()) < 4:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count>=4")
    return _tiny_writer_case()


def test_writer_matches_main_golden_exactly(writer_case):
    """H3 placement oracle: the MK writer == main's DCP writer on identical bytes, whole caches."""
    case = writer_case
    for kl in case["gold"]:
        for r in range(case["R"]):
            g, m = case["gold"][kl][r], case["mk"][kl][r]
            assert np.array_equal(g, m), (kl, r, int((g != m).sum()))


def test_compare_staging_clean_on_golden_and_writer(writer_case):
    case = writer_case
    for which in ("gold", "mk"):
        for r in range(case["R"]):
            res = _compare(case, which, r, snap_from="caches0")
            assert all(res[k]["bad_rows"] == 0 for k in ("nope", "rope", "idx")), (which, r, res)
            assert res["clobber"] == {"rope": 0, "idx": 0}, (which, r, res)
    owned = sum(_compare(case, "gold", r)["nope"]["rows"] for r in range(case["R"]))
    assert owned == case["cfg"].L * len(case["pos"])               # every row owned exactly once


def test_compare_staging_catches_shifted_writer():
    """Fault control: a writer fed row 5's position + 1 writes it one slot over (and so into a
    neighbour's slot); compare_staging must flag it on the owning rank."""
    import jax

    if len(jax.devices()) < 4:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count>=4")
    case = _tiny_writer_case(shift_row=5)
    bad = sum(_compare(case, "mk", r)["nope"]["bad_rows"] for r in range(case["R"]))
    assert bad > 0
    assert any(not np.array_equal(case["gold"][kl][r], case["mk"][kl][r])
               for kl in case["gold"] for r in range(case["R"]))


# --------------------------------------------------------------- runner forward routing


def _routing_runner(plan):
    from unittest.mock import MagicMock

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    runner = MagicMock(spec=TPUModelRunner)
    runner.forward_model = TPUModelRunner.forward_model.__get__(runner)
    runner.model = MagicMock(return_value=torch.zeros(4, 8))
    mkd = MagicMock()
    mkd.return_value = torch.ones(4, 8)

    def aux(r, p, ids, pos, stock_out):
        assert torch.equal(stock_out, torch.zeros(4, 8))
        r._mkd_draft_hidden = torch.full((4, 8), 3.0) if p.draft != p.ret else None
        return torch.full((4, 8), 2.0) if p.ret == "M" else stock_out

    mkd.aux = MagicMock(side_effect=aux)
    runner.glm5_mkd = mkd
    runner._mkd_plan = plan
    runner._mkd_step = plan is not None and not plan.pure_stock
    return runner


def test_forward_routing():
    ids, pos = torch.zeros(4, dtype=torch.int32), torch.zeros(4)
    r = _routing_runner(None)
    out, _ = r.forward_model(ids, pos)
    assert torch.equal(out, torch.zeros(4, 8)) and r._fwd_was_mkd == "stock"
    r = _routing_runner(D.MK)
    out, _ = r.forward_model(ids, pos)
    assert torch.equal(out, torch.ones(4, 8)) and not r.model.called and r._fwd_was_mkd == "mk"
    r = _routing_runner(D.ModeSchedule("shadow").plan())
    out, _ = r.forward_model(ids, pos)
    assert r.model.called and r.glm5_mkd.aux.called and torch.equal(out, torch.zeros(4, 8))
    r = _routing_runner(D.ModeSchedule("mix:MSS").plan())
    out, _ = r.forward_model(ids, pos)
    assert torch.equal(out, torch.full((4, 8), 2.0)) and r._mkd_draft_hidden is not None
    s = D.ModeSchedule("alt:2")
    s.plan()
    r = _routing_runner(s.plan())                                  # alt:2's stock step
    out, _ = r.forward_model(ids, pos)
    assert torch.equal(out, torch.zeros(4, 8)) and not r.glm5_mkd.aux.called
    assert r._fwd_was_mkd == "alt:2/stock"


def test_arm_flags():
    s = D.ModeSchedule("cycle:mk-sd/mk+kv+id/alt:4-sd/shadow+kv")
    p = s.plan()
    assert p.pure_mk and p.specdbg is False and p.kvdiff is None and p.label == "0:mk-sd"
    s.observe_requests(["a"])
    s.observe_requests(["b"])
    p = s.plan()
    assert p.pure_mk and p.kvdiff is True and p.idcheck is True
    s.observe_requests(["c"])
    p = s.plan()
    assert p.pure_mk and p.specdbg is False and p.label == "2:alt:4-sd/mk"
    assert s.any_flag("kvdiff") and s.needs_probe
    assert not D.ModeSchedule("mk").any_flag("kvdiff")
    with pytest.raises(ValueError):
        D.ModeSchedule("mk+zz+kv")
