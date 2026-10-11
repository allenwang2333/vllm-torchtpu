# SPDX-License-Identifier: Apache-2.0
"""TD8.72: CPU coverage for the Eagle3/MTP proposer host logic.

`tests/conftest.py` defaults `--use-tpu` to True, so every `device`-fixture
test in `test_eagle3.py` is TPU-only, and the `torch.compile(backend="tpu")`
helpers abort the process on a host with no TPU. These tests pin CPU
explicitly and run the four compiled-region bodies eagerly through
`inspect.unwrap`, so the numerics under test are the real ones.
"""

import contextlib
import functools
import inspect
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.spec_decode import eagle3 as e3
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer
from vllm_torchtpu.spec_decode.utils import DraftChunkInputs

CPU = torch.device("cpu")
_COMPILED = (
    "_draft_propose_token",
    "_draft_gather_carries",
    "_draft_combine_hidden_states",
    "_draft_seed_input_ids",
)


def _runner(**overrides):
    attrs = dict(
        device=CPU,
        max_num_reqs=4,
        num_reqs_max_model_len=4,
        num_reqs_most_model_len=4,
        num_tokens_paddings=[4, 8, 16],
        num_reqs_paddings=[2, 8],
        uses_mrope=False,
        mesh=None,
        enforce_eager=False,
        requests={},
        input_batch=SimpleNamespace(num_reqs=2, req_ids=["r0", "r1", "r2"]),
        _hidden_states_dtype=torch.float32,
        _pcp_enabled=False,
        _mtp_enabled=False,
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda n: None,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _proposer(method="eagle3", *, draft_tp=1, target_tp=1, K=2, **runner_kw):
    spec = SimpleNamespace(
        draft_tensor_parallel_size=draft_tp,
        method=method,
        draft_model_config=None,
        num_speculative_tokens=K,
    )
    vllm_config = SimpleNamespace(
        speculative_config=spec,
        parallel_config=SimpleNamespace(tensor_parallel_size=target_tp),
    )
    proposer = Eagle3Proposer(runner=_runner(**runner_kw), vllm_config=vllm_config)
    proposer.draft_vllm_config = SimpleNamespace(name="draft-config")
    return proposer


def _eager(proposer):
    """Bind the compiled-region bodies eagerly (backend="tpu" needs a TPU)."""
    for name in _COMPILED:
        raw = inspect.unwrap(getattr(Eagle3Proposer, name))
        setattr(proposer, name, functools.partial(raw, proposer))


def _null_contexts(monkeypatch):
    def null(*_args, **_kwargs):
        return contextlib.nullcontext()

    for name in (
        "set_model_tag",
        "set_forward_context",
        "set_vllm_model_wrapper_context",
        "set_current_vllm_config",
    ):
        monkeypatch.setattr(e3, name, null)


# --------------------------------------------------------------------------
# Module helpers and compiled-region bodies
# --------------------------------------------------------------------------


def test_pad_helpers():
    t1 = torch.tensor([1, 2])
    assert e3._maybe_pad_dim0(t1, 2) is t1
    assert e3._maybe_pad_dim0(t1, 4).tolist() == [1, 2, 0, 0]
    t2 = torch.ones(1, 2)
    assert e3._maybe_pad_dim0(t2, 3).tolist() == [[1, 1], [0, 0], [0, 0]]
    with pytest.raises(ValueError, match="unsupported ndim 3"):
        e3._maybe_pad_dim0(torch.ones(1, 1, 1), 2)
    assert e3._maybe_pad_dim1(t2, 2) is t2
    assert e3._maybe_pad_dim1(t2, 4).tolist() == [[1, 1, 0, 0]]


def test_propose_token_body_with_and_without_vocab_map():
    proposer = _proposer()
    _eager(proposer)
    hidden = torch.tensor([[0.0, 3.0, 1.0], [5.0, 0.0, 0.0]])
    proposer.draft_model = SimpleNamespace(compute_logits=lambda h: h)
    out = proposer._draft_propose_token(hidden)
    assert out.dtype == torch.int32 and out.tolist() == [1, 0]

    seen = []
    proposer.draft_model = SimpleNamespace(
        draft_id_to_target_id=torch.tensor([10, 20, 30]),
        lm_head="head",
        logits_processor=lambda head, h: (seen.append(head), h)[1],
    )
    out = proposer._draft_propose_token(hidden)
    assert seen == ["head"]
    # draft ids [1, 0] + d2t[[1, 0]] = [21, 10]
    assert out.dtype == torch.int32 and out.tolist() == [21, 10]


@pytest.mark.parametrize("mrope", [False, True])
def test_gather_carries_body(mrope):
    proposer = _proposer()
    _eager(proposer)
    hidden = torch.arange(12.0).view(4, 3)
    last_hidden = -hidden
    positions = torch.arange(12).view(3, 4) if mrope else torch.tensor([10, 11, 12, 13])
    idx = torch.tensor([3, 1])
    h, pos, last = proposer._draft_gather_carries(hidden, positions, last_hidden, idx)
    torch.testing.assert_close(h, hidden[idx])
    torch.testing.assert_close(last, last_hidden[idx])
    torch.testing.assert_close(pos, positions[:, idx] if mrope else positions[idx])


def test_combine_and_seed_bodies():
    proposer = _proposer()
    _eager(proposer)
    proposer.draft_model = SimpleNamespace(combine_hidden_states=lambda x: x * 2)
    aux = [torch.full((2, 1), float(i)) for i in range(3)]
    out = proposer._draft_combine_hidden_states(*aux)
    torch.testing.assert_close(out, torch.tensor([[0.0, 2.0, 4.0]] * 2))

    ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    seeded = proposer._draft_seed_input_ids(
        ids, torch.tensor([1, 3]), torch.tensor([90, 91], dtype=torch.int64)
    )
    assert seeded.dtype == torch.int32
    assert seeded.tolist() == [1, 90, 3, 91]
    single = proposer._draft_seed_input_ids(
        torch.tensor([7], dtype=torch.int32), torch.tensor([0]), torch.tensor([5])
    )
    assert single.tolist() == [5]


def test_unwrap_model_out():
    a, b = torch.zeros(1), torch.ones(1)
    assert Eagle3Proposer._unwrap_model_out(a) == (a, a)
    assert Eagle3Proposer._unwrap_model_out([a]) == (a, a)
    assert Eagle3Proposer._unwrap_model_out((a, b)) == (a, b)
    assert Eagle3Proposer._unwrap_model_out([(a, b)]) == (a, b)
    with pytest.raises(RuntimeError, match="3-element tuple"):
        Eagle3Proposer._unwrap_model_out((a, b, a))


# --------------------------------------------------------------------------
# load_model and its validators
# --------------------------------------------------------------------------


def _load_harness(monkeypatch, proposer, *, draft_model=None):
    registries = iter([{"t.attn": 1}, {"t.attn": 1, "d.attn": 2}])
    monkeypatch.setattr(
        e3, "get_layers_from_vllm_config", lambda *_a, **_k: next(registries)
    )
    shared = []
    monkeypatch.setattr(
        e3, "maybe_share_embeddings", lambda *a: shared.append(("embed", a[2]))
    )
    monkeypatch.setattr(
        e3, "maybe_share_lm_head", lambda *a, **k: shared.append(("head", k))
    )
    model = draft_model if draft_model is not None else SimpleNamespace()

    def load():
        proposer.draft_model = model

    proposer._load_draft_model = load
    return shared


_AUX_MODULE = "vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils"


def test_load_model_eagle3_registers_aux_layers(monkeypatch):
    proposer = _proposer("eagle3")
    shared = _load_harness(monkeypatch, proposer)
    calls = []
    fake = types.ModuleType(_AUX_MODULE)
    fake.set_eagle3_aux_hidden_state_layers = lambda target, spec: calls.append(
        (target, spec)
    )
    monkeypatch.setitem(sys.modules, _AUX_MODULE, fake)
    proposer.load_model("target")
    assert proposer._draft_attn_layer_names == {"d.attn"}
    assert calls == [("target", proposer.speculative_config)]
    assert shared[0] == ("embed", True)
    assert shared[1][1]["materialize_if_mismatched"] is False
    assert proposer._share_mtp_indices is False
    assert proposer._mtp_fused_loop is None


def test_load_model_eagle3_without_aux_helper_raises(monkeypatch):
    proposer = _proposer("eagle3")
    _load_harness(monkeypatch, proposer)
    monkeypatch.setitem(sys.modules, _AUX_MODULE, None)
    with pytest.raises(ImportError, match="eagle3_utils"):
        proposer.load_model("target")


def test_load_model_other_method_validates_pcp_draft(monkeypatch):
    proposer = _proposer("draft_model")
    proposer.runner._pcp_enabled = True
    proposer.runner._mtp_enabled = True
    _load_harness(monkeypatch, proposer)
    validated = []
    proposer._validate_pcp_draft_model = validated.append
    proposer.load_model("target")
    assert validated == [{"d.attn"}]


@pytest.mark.parametrize("kv_shared", [False, True])
def test_load_model_mtp_index_share(monkeypatch, kv_shared):
    proposer = _proposer("mtp")
    _load_harness(monkeypatch, proposer)
    order = []

    def gemma(target, names):
        order.append(("gemma", names))
        proposer._draft_kv_shared = kv_shared

    def share():
        order.append("share")
        proposer._share_mtp_indices = True

    proposer._setup_gemma4_kv_sharing = gemma
    proposer._init_mtp_index_sharing = share
    proposer._build_mtp_loop_model = lambda: order.append("loop")
    proposer._build_mtp_fused_loop = lambda: order.append("fused")
    if kv_shared:
        with pytest.raises(RuntimeError, match="IndexShare and cross-model"):
            proposer.load_model("target")
    else:
        proposer.load_model("target")
    assert order == [("gemma", {"t.attn"}), "share", "loop", "fused"]


def _index_share_proposer(mtp_model):
    proposer = _proposer("mtp")
    proposer.speculative_config.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(index_share_for_mtp_iteration=True)
    )
    proposer.draft_model = SimpleNamespace(model=mtp_model)
    return proposer


def test_index_sharing_is_off_for_non_mtp():
    proposer = _proposer("eagle3")
    proposer._share_mtp_indices = True
    proposer._init_mtp_index_sharing()
    assert proposer._share_mtp_indices is False


def test_index_sharing_needs_set_skip_topk():
    proposer = _index_share_proposer(SimpleNamespace(layers={}))
    proposer._init_mtp_index_sharing()
    assert proposer._share_mtp_indices is False


def test_index_sharing_needs_a_topk_buffer():
    layer = SimpleNamespace(mtp_block=SimpleNamespace(self_attn=None))
    proposer = _index_share_proposer(
        SimpleNamespace(set_skip_topk=lambda s: None, layers={"0": layer})
    )
    proposer._init_mtp_index_sharing()
    assert proposer._share_mtp_indices is False
    assert proposer._mtp_topk_buffers == []


def _fused_loop_proposer(monkeypatch, *, K=3, dims=True):
    monkeypatch.setitem(vars(envs), "VLLM_TPU_FUSED_DRAFT_LOOP", True)
    _null_contexts(monkeypatch)
    proposer = _proposer("mtp", K=K)
    proposer.draft_model = SimpleNamespace(
        _dynamic_arg_dims={"input_ids": 0} if dims else None
    )
    return proposer


def test_build_mtp_fused_loop_builds_program(monkeypatch):
    import vllm_torchtpu.spec_decode.mtp_fused_loop as mfl

    proposer = _fused_loop_proposer(monkeypatch)
    built = []

    def build(**kwargs):
        built.append(kwargs)
        return "program"

    monkeypatch.setattr(mfl, "build_mtp_fused_loop", build)
    proposer._mtp_fused_loop_warm = True
    proposer._build_mtp_fused_loop()
    assert proposer._mtp_fused_loop == "program"
    assert proposer._mtp_fused_loop_warm is False
    (kwargs,) = built
    assert kwargs["prefix"] == "mtp_fused_loop_k3"
    assert kwargs["num_steps"] == 2
    assert kwargs["mtp"] is proposer.draft_model
    assert kwargs["vllm_config"] is proposer.draft_vllm_config
    assert kwargs["dynamic_arg_dims"] == {"input_ids": 0}


def test_build_mtp_fused_loop_falls_back_on_error(monkeypatch):
    import vllm_torchtpu.spec_decode.mtp_fused_loop as mfl

    proposer = _fused_loop_proposer(monkeypatch)

    def boom(**_kwargs):
        raise RuntimeError("no program")

    monkeypatch.setattr(mfl, "build_mtp_fused_loop", boom)
    proposer._mtp_fused_loop = "stale"
    proposer._build_mtp_fused_loop()
    assert proposer._mtp_fused_loop is None


@pytest.mark.parametrize("case", ["no-dims", "single-step"])
def test_build_mtp_fused_loop_skips(monkeypatch, case):
    import vllm_torchtpu.spec_decode.mtp_fused_loop as mfl

    proposer = _fused_loop_proposer(
        monkeypatch, K=1 if case == "single-step" else 3, dims=case != "no-dims"
    )
    monkeypatch.setattr(
        mfl, "build_mtp_fused_loop", lambda **_k: pytest.fail("must not build")
    )
    proposer._build_mtp_fused_loop()
    assert proposer._mtp_fused_loop is None


def _md(seq_lens):
    return AttentionMetadata(input_positions=torch.zeros(1), seq_lens=seq_lens)


def test_run_mtp_fused_loop_rejects_split_seq_lens(monkeypatch):
    proposer = _proposer("mtp")
    proposer._build_draft_attn_metadata = lambda **_k: {
        "a": _md(torch.zeros(2)),
        "b": _md(torch.zeros(2)),
    }
    assert (
        proposer._run_mtp_fused_loop(
            None, torch.zeros(2), torch.zeros(2), torch.zeros(2), None, None, None, None
        )
        is None
    )


def test_run_mtp_fused_loop_warms_then_runs(monkeypatch):
    _null_contexts(monkeypatch)
    proposer = _proposer("mtp")
    seq_lens = torch.tensor([5, 7], dtype=torch.int32)
    md = _md(seq_lens)
    built = []

    def build(**kwargs):
        built.append(kwargs)
        return {"a": md, "b": md}

    proposer._build_draft_attn_metadata = build
    chunk = object()
    proposer._draft_md_cache[id(chunk)] = "stale"
    calls = []

    def program(input_ids, positions, hidden, sl):
        calls.append(sl.clone())
        sl.add_(2)  # the program advances seq_lens in place
        return (input_ids + 1, input_ids + 2)

    proposer._mtp_fused_loop = program
    ids = torch.tensor([3, 4], dtype=torch.int32)
    out = proposer._run_mtp_fused_loop(
        chunk, ids, torch.zeros(2), torch.zeros(2, 1), None, "rej", "qsl", "rd"
    )
    assert [t.tolist() for t in out] == [[4, 5], [5, 6]]
    # Both calls start from the step-1 seq_lens: the warm-up's advance is undone.
    assert [c.tolist() for c in calls] == [[5, 7], [5, 7]]
    assert seq_lens.tolist() == [7, 9]
    assert proposer._mtp_fused_loop_warm is True
    assert id(chunk) not in proposer._draft_md_cache
    assert built[0]["step_idx"] == 1 and built[0]["seq_lens_delta"] == 1
    assert built[0]["num_rejected_dev"] == "rej"
    # Already warm: a second call runs the program once.
    calls.clear()
    proposer._run_mtp_fused_loop(
        chunk, ids, torch.zeros(2), torch.zeros(2, 1), None, None, None, None
    )
    assert len(calls) == 1


def test_kv_sharing_keeps_default_prefix_without_layer_names():
    proposer = _proposer("mtp")
    impl = SimpleNamespace()
    attn = SimpleNamespace(impl=impl)
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(is_kv_shared_layer=True, attn=attn)
    )
    proposer.draft_model = SimpleNamespace(
        model=SimpleNamespace(layers=[layer]),
        config=SimpleNamespace(get_text_config=lambda: SimpleNamespace()),
    )
    target = SimpleNamespace(
        config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(
                layer_types=["full_attention", "full_attention"]
            )
        )
    )
    proposer._validate_kv_sharing_supported = lambda: None
    proposer._setup_gemma4_kv_sharing(target, {"lm_head.attn", "embed.attn"})
    assert attn.kv_sharing_target_layer_name == "model.layers.1.self_attn.attn"
    assert impl.kv_sharing_target_layer_name == "model.layers.1.self_attn.attn"
    assert proposer._draft_kv_shared is True


def test_split_dsa_layers_without_dsa_build(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.deepseek_v2", None)
    assert Eagle3Proposer._split_dsa_draft_layers({"x": object()}, {"x"}) == ([], [])


# --------------------------------------------------------------------------
# propose
# --------------------------------------------------------------------------

HID = 4
VOCAB = 64


class _Draft:
    """Identity combine; one-hot logits recover round(hidden[:, 0])."""

    def __init__(self, uses_aux):
        self.model = SimpleNamespace(use_aux_hidden_state=uses_aux)
        self.combined = []

    def combine_hidden_states(self, x):
        self.combined.append(tuple(x.shape))
        return x[..., :HID]

    def compute_logits(self, hidden):
        tokens = hidden[:, 0].round().clamp(0, VOCAB - 1).to(torch.int64)
        return torch.nn.functional.one_hot(tokens, VOCAB).to(torch.float32)


def _chunk(start, num_reqs, *, use_max_model_len=True, hidden_states=None):
    return DraftChunkInputs(
        input_ids=torch.zeros(4, dtype=torch.int32),
        position_ids=torch.arange(4, dtype=torch.int32),
        query_start_loc_np=np.arange(num_reqs + 1, dtype=np.int32),
        attn_ctx=SimpleNamespace(use_max_model_len=use_max_model_len),
        start_index=start,
        num_reqs=num_reqs,
        aux_hidden_states=[torch.zeros(4, HID) for _ in range(3)],
        hidden_states=hidden_states,
    )


def _propose_harness(method="eagle3", *, K=2, sizes=(2,), uses_aux=True, rejected):
    proposer = _proposer(method, K=K)
    proposer.runner.input_batch.num_reqs = sum(sizes)
    _eager(proposer)
    proposer.draft_model = _Draft(uses_aux)
    chunks, start = [], 0
    for ci, nr in enumerate(sizes):
        chunks.append(
            _chunk(
                start,
                nr,
                use_max_model_len=ci % 2 == 0,
                hidden_states=torch.zeros(4, HID),
            )
        )
        start += nr
    proposer.draft_chunks = chunks
    forwards = []

    def prepare(chunk, *_a, **_k):
        nr = chunk.num_reqs
        lti = torch.tensor(list(range(nr)) + [nr - 1] * (4 - nr))
        return (torch.zeros(4, dtype=torch.int32), torch.arange(4), lti, rejected(nr))

    def forward(*, chunk, input_ids, step_idx, **kwargs):
        forwards.append(dict(kwargs, step_idx=step_idx, chunk=chunk))
        last = torch.zeros(input_ids.shape[0], HID)
        if step_idx == 0:
            base = torch.arange(chunk.num_reqs) + chunk.start_index + 1
            last[: chunk.num_reqs, 0] = 10.0 * base
        else:
            last[:, 0] = (input_ids + 1).to(torch.float32)
        return last, last

    proposer._prepare_draft_inputs = prepare
    proposer._forward_draft = forward
    return proposer, forwards


def _expected(sizes, K):
    return [[10 * (i + 1) + s for s in range(K)] for i in range(sum(sizes))]


def test_propose_without_chunks_raises():
    proposer = _proposer()
    with pytest.raises(RuntimeError, match="no draft chunks"):
        proposer.propose(None, [], None, None)


def test_propose_aux_sync_rejections_return_device():
    proposer, forwards = _propose_harness(
        sizes=(2, 1), rejected=lambda nr: np.arange(1, nr + 1, dtype=np.int64)
    )
    out = proposer.propose([[0]] * 3, [], None, None, return_device=True)
    assert isinstance(out, torch.Tensor)
    assert out.tolist() == _expected((2, 1), 2)
    assert proposer.draft_model.combined == [(4, 3 * HID)] * 2
    loop = [f for f in forwards if f["step_idx"] == 1]
    assert loop[0]["num_rejected_dev"].tolist() == [1, 2, 0, 0]
    assert loop[1]["num_rejected_dev"].tolist() == [1, 0, 0, 0]
    assert loop[0]["loop_query_start_loc"].tolist() == [0, 1, 2, 2, 2]
    assert loop[1]["loop_request_distribution"].tolist() == [1, 1, 1]
    assert all(f["seq_lens_delta"] == 1 for f in loop)


def test_propose_eagle3_plain_hidden_no_rejections():
    proposer, forwards = _propose_harness(
        sizes=(2,), uses_aux=False, rejected=lambda nr: np.zeros(nr, np.int32)
    )
    out = proposer.propose([[0]] * 2, [], None, None)
    assert out == _expected((2,), 2)
    # Plain hidden runs through combine_hidden_states for eagle3.
    assert proposer.draft_model.combined == [(4, HID)]
    assert forwards[1]["num_rejected_dev"] is None


def test_propose_async_kv_shared_keeps_positions():
    proposer, forwards = _propose_harness(
        "mtp", K=3, sizes=(2,), rejected=lambda nr: torch.ones(nr, dtype=torch.int32)
    )
    proposer._draft_kv_shared = True
    out = proposer.propose(
        None, [], None, None, next_tokens_per_chunk=[torch.zeros(2, 4)]
    )
    assert out == _expected((2,), 3)
    loop = [f for f in forwards if f["step_idx"] > 0]
    assert [f["seq_lens_delta"] for f in loop] == [0, 0]
    assert loop[0]["num_rejected_dev"].tolist() == [1, 1, 0, 0]
    # Positions stay at the step-0 carry when the draft shares the target KV.
    assert all(f["positions"].tolist() == [0, 1, 1, 1] for f in loop)


def test_propose_fused_loop_with_per_step_fallback():
    proposer, forwards = _propose_harness(
        "mtp", K=3, sizes=(2, 1), rejected=lambda nr: None
    )
    proposer._mtp_fused_loop = object()
    fused_calls = []

    def run_fused(**kwargs):
        fused_calls.append(kwargs)
        if kwargs["chunk"].start_index == 0:
            ids = kwargs["input_ids"]
            return (ids + 1, ids + 2)
        return None  # second chunk falls back to the per-step loop

    proposer._run_mtp_fused_loop = run_fused
    out = proposer.propose([[0]] * 3, [], None, None)
    assert out == _expected((2, 1), 3)
    assert fused_calls[0]["positions"].tolist() == [1, 2, 2, 2]
    loop = [f for f in forwards if f["step_idx"] > 0]
    assert [f["chunk"].start_index for f in loop] == [2, 2]


@pytest.mark.parametrize("step_chunks,padded", [(3, [2]), (1, [])])
def test_propose_pads_lockstep_chunks(step_chunks, padded):
    proposer, _ = _propose_harness(sizes=(2,), K=1, rejected=lambda nr: None)
    proposer.runner._dp_lockstep_enabled = lambda: True
    proposer.runner._dp_step_num_chunks = step_chunks
    dummy = []
    proposer.run_dp_dummy_draft = dummy.append
    assert proposer.propose([[0]] * 2, [], None, None) == _expected((2,), 1)
    assert dummy == padded


# --------------------------------------------------------------------------
# _prepare_draft_inputs (real, on CPU)
# --------------------------------------------------------------------------


def _prep_setup(*, draft_lengths=None):
    proposer = _proposer()
    proposer.runner.requests = {
        "r1": SimpleNamespace(num_computed_tokens=3, get_token_id=lambda i: 500 + i)
    }
    chunk = DraftChunkInputs(
        input_ids=torch.arange(8, dtype=torch.int32),
        position_ids=torch.arange(8, dtype=torch.int32),
        query_start_loc_np=np.array([0, 3, 5, 6], dtype=np.int32),
        attn_ctx=SimpleNamespace(
            query_start_loc=torch.tensor([0, 3, 5, 6], dtype=torch.int32)
        ),
        start_index=0,
        num_reqs=3,
        aux_hidden_states=[],
        draft_lengths=draft_lengths,
    )
    sched = SimpleNamespace(
        num_scheduled_tokens={"r1": 2},
        scheduled_spec_decode_tokens={"r0": [1, 1], "r1": [1], "r2": [1, 1]},
    )
    _eager(proposer)
    return proposer, chunk, sched


def test_prepare_draft_inputs_sync_with_discard_and_missing_samples():
    proposer, chunk, sched = _prep_setup()
    ids, pos, gather, rej = proposer._prepare_draft_inputs(
        chunk,
        sampled_token_ids=[[101, 102], [0]],
        discard_sampled_tokens_req_indices=[1],
        num_rejected_tokens_np=np.array([5, 0, 0]),
        scheduler_output=sched,
    )
    # rejected clamps to len-1 = [2, 1, 0]; seed slots qsl_end-1-rej = [0, 4, 5].
    assert rej.tolist() == [2, 0, 0] and rej.dtype == np.int32
    assert gather.tolist() == [0, 4, 5, 5]
    # shift [0..7] -> [1..7, 7]; seeds r0=102, r1 (discard)=get(3+2)=505, r2=0.
    assert ids.tolist() == [102, 2, 3, 4, 505, 0, 7, 7]
    assert pos is chunk.position_ids


@pytest.mark.parametrize("with_lengths", [False, True])
def test_prepare_draft_inputs_async(with_lengths):
    lengths = torch.tensor([2, 2, 2], dtype=torch.int32) if with_lengths else None
    proposer, chunk, sched = _prep_setup(draft_lengths=lengths)
    next_tokens = torch.tensor([[7, 8, -1], [9, -1, -1], [11, 12, 13]])
    ids, _, gather, rej = proposer._prepare_draft_inputs(
        chunk,
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[] if with_lengths else [1],
        num_rejected_tokens_np=None,
        scheduler_output=sched,
        next_tokens_device=next_tokens,
    )
    # num_valid [2, 1, 3]; seeds [8, 9, 13]; accepted min([1, 0, 2], drafts).
    if with_lengths:  # drafts [2, 2, 2] -> rejected [1, 2, 0]
        assert rej.tolist() == [1, 2, 0]
        assert gather.tolist() == [1, 2, 5, 0]
        assert ids.tolist() == [1, 8, 9, 4, 5, 13, 7, 7]
    else:  # drafts from the scheduler [2, 1, 2] -> rejected [1, 1, 0]
        assert rej.tolist() == [1, 1, 0]
        assert gather.tolist() == [1, 3, 5, 0]
        # r1 is a partial prefill: seeded from host request state (505).
        assert ids.tolist() == [1, 8, 3, 505, 5, 13, 7, 7]


@pytest.mark.parametrize("discard", [[], [1]])
def test_prepare_draft_inputs_device_seed(discard):
    proposer, chunk, sched = _prep_setup()
    seed = torch.tensor([100, 200, 300, 999])
    ids, _, gather, rej = proposer._prepare_draft_inputs(
        chunk,
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=discard,
        num_rejected_tokens_np=None,
        scheduler_output=sched,
        device_seed=seed,
    )
    middle = 505 if discard else 200
    assert ids.tolist() == [1, 2, 100, 4, middle, 300, 7, 7]
    assert gather.tolist() == [2, 4, 5, 0]
    assert rej.tolist() == [0, 0, 0]
    assert seed.tolist() == [100, 200, 300, 999]  # caller's tensor untouched


# --------------------------------------------------------------------------
# _build_draft_attn_metadata
# --------------------------------------------------------------------------


def _md_runner(proposer, *, mamba=False, layers=("d.0", "d.1", "t.0"), share=True):
    builds = []

    def build(**kwargs):
        ctx = proposer.runner._attn_metadata_builder_ctx
        builds.append(dict(kwargs, ctx=ctx))
        md = _md(ctx.seq_lens)
        other = md if share else _md(ctx.seq_lens)
        result = {"d.0": md, "d.1": other, "t.0": md}
        return {k: v for k, v in result.items() if k in layers}, None

    proposer.runner = _runner(
        _has_mamba_state=mamba,
        _unified_kv_layout=False,
        _attn_metadata_builder_ctx="saved",
        empty_slot_mappings="slots",
        _build_attention_metadata=build,
        _build_mamba_state_indices=lambda start, nr, knr: ("mamba", start, nr, knr),
        num_reqs_max_model_len=4,
        num_reqs_most_model_len=6,
    )
    proposer._draft_attn_layer_names = {"d.0", "d.1"}
    return builds


def _md_chunk(*, use_max_model_len=True, override=None):
    return SimpleNamespace(
        num_reqs=2,
        start_index=1,
        query_start_loc_np=np.array([0, 3, 5], dtype=np.int32),
        attn_ctx=SimpleNamespace(
            use_max_model_len=use_max_model_len,
            # Padded to kernel_num_reqs: 4 (max-model-len) or 6 (most).
            seq_lens=torch.tensor(
                [5, 9] + [0] * (2 if use_max_model_len else 4), dtype=torch.int32
            ),
            query_start_loc="QSL",
            request_distribution="RD",
            position_ids_override=override,
            sequence_layout_descriptor="desc",
        ),
    )


def test_build_draft_attn_metadata_step0_with_mamba():
    proposer = _proposer()
    builds = _md_runner(proposer, mamba=True)
    chunk = _md_chunk()
    md = proposer._build_draft_attn_metadata(chunk, 0, 0, None)
    assert set(md) == {"d.0", "d.1"}
    (b,) = builds
    assert b["num_tokens"] == 5 and b["num_reqs"] == 4 and b["max_query_len"] == 1
    assert b["slot_mappings"] == "slots"
    ctx = b["ctx"]
    assert ctx.seq_lens is chunk.attn_ctx.seq_lens
    assert ctx.query_start_loc == "QSL" and ctx.request_distribution == "RD"
    assert ctx.mamba_state_indices == ("mamba", 1, 2, 4)
    assert ctx.sequence_layout_descriptor == "desc"
    assert proposer.runner._attn_metadata_builder_ctx == "saved"
    assert not proposer._draft_md_cache  # mamba state disables the cache


def test_build_draft_attn_metadata_loop_fallbacks_and_kv_shared():
    proposer = _proposer()
    builds = _md_runner(proposer)
    proposer._draft_kv_shared = True
    chunk = _md_chunk(use_max_model_len=False)
    md = proposer._build_draft_attn_metadata(chunk, 1, 1, None, num_tokens_padded=8)
    assert set(md) == {"d.0", "d.1", "t.0"}  # KV-sharing keeps the whole group
    (b,) = builds
    assert b["num_tokens"] == 8 and b["num_reqs"] == 6
    ctx = b["ctx"]
    assert ctx.query_start_loc.tolist() == [0, 1, 2, 2, 2, 2, 2]
    assert ctx.request_distribution.tolist() == [2, 2, 2]
    assert ctx.seq_lens.tolist() == [6, 10, 1, 1, 1, 1]
    assert id(chunk) in proposer._draft_md_cache


@pytest.mark.parametrize("share", [True, False])
def test_build_draft_attn_metadata_cached_steps(share):
    proposer = _proposer()
    builds = _md_runner(proposer, share=share)
    chunk = _md_chunk()
    rej = torch.tensor([1, 0, 0, 0], dtype=torch.int32)
    qsl, rd = torch.arange(5), torch.tensor([2, 2, 2])
    md1 = proposer._build_draft_attn_metadata(
        chunk,
        1,
        1,
        None,
        num_rejected_dev=rej,
        loop_query_start_loc=qsl,
        loop_request_distribution=rd,
    )
    md2 = proposer._build_draft_attn_metadata(chunk, 2, 2, None)
    assert len(builds) == 1
    assert builds[0]["ctx"].query_start_loc is qsl
    assert builds[0]["ctx"].request_distribution is rd
    assert md1["d.0"].seq_lens.tolist() == [5, 10, 1, 1]
    assert md2["d.0"].seq_lens.tolist() == [6, 11, 2, 2]
    assert md2["d.0"] is not md1["d.0"]
    assert (md2["d.0"] is md2["d.1"]) is share


def test_build_draft_attn_metadata_step2_without_cache_rebuilds():
    proposer = _proposer()
    builds = _md_runner(proposer)
    chunk = _md_chunk()
    md = proposer._build_draft_attn_metadata(
        chunk, 2, 2, np.array([1, 1], dtype=np.int32)
    )
    assert len(builds) == 1
    assert md["d.0"].seq_lens.tolist() == [6, 10, 2, 2]


def test_build_draft_attn_metadata_override_is_not_cached():
    proposer = _proposer()
    _md_runner(proposer)
    chunk = _md_chunk(override=torch.zeros(4))
    proposer._build_draft_attn_metadata(chunk, 1, 1, None)
    assert not proposer._draft_md_cache


def test_build_draft_attn_metadata_missing_layer_raises():
    proposer = _proposer()
    _md_runner(proposer, layers=("d.0", "t.0"))
    with pytest.raises(RuntimeError, match=r"\['d.1'\] have no attention metadata"):
        proposer._build_draft_attn_metadata(_md_chunk(), 0, 0, None)
    assert proposer.runner._attn_metadata_builder_ctx == "saved"


# --------------------------------------------------------------------------
# _forward_draft, precompile and the EP-DP dummy draft
# --------------------------------------------------------------------------


def test_forward_draft_eagle3_passes_no_step_index(monkeypatch):
    _null_contexts(monkeypatch)
    proposer = _proposer("eagle3")
    proposer._build_draft_attn_metadata = lambda **_k: {}
    calls = []

    def draft(**kwargs):
        calls.append(kwargs)
        return kwargs["hidden_states"] * 2

    proposer.draft_model = draft
    hidden = torch.ones(2, 3)
    last, carry = proposer._forward_draft(
        chunk=None,
        input_ids=torch.zeros(2),
        positions=torch.zeros(2),
        target_hidden_states=hidden,
        step_idx=1,
        seq_lens_delta=1,
        num_rejected_np=None,
    )
    assert set(calls[0]) == {"input_ids", "positions", "hidden_states"}
    torch.testing.assert_close(last, hidden * 2)
    assert carry is last


@pytest.mark.parametrize("eager_mode", [False, True])
def test_precompile_dispatch(eager_mode):
    proposer = _proposer(enforce_eager=eager_mode)
    order = []
    proposer._precompile_combine_hidden_states = lambda: order.append("combine")
    proposer._precompile_draft_seed = lambda: order.append("seed")
    proposer._precompile_compute_logits = lambda: order.append("logits")
    proposer.precompile()
    assert order == ([] if eager_mode else ["combine", "seed", "logits"])


def test_draft_hidden_size_falls_back_to_text_config():
    proposer = _proposer()
    proposer.draft_model = SimpleNamespace(config=SimpleNamespace())
    proposer.speculative_config.draft_model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(hidden_size=12)
    )
    assert proposer._draft_hidden_size() == 12


def _precompile_harness(monkeypatch, proposer):
    synced = []
    monkeypatch.setattr(e3, "synchronize_tensors", synced.append)
    timed = []

    @contextlib.contextmanager
    def precompile_timed(label):
        timed.append(label)
        yield

    proposer.runner._precompile_timed = precompile_timed
    return synced, timed


@pytest.mark.parametrize("method,uses_aux", [("mtp", True), ("eagle3", False)])
def test_precompile_combine_skips(monkeypatch, method, uses_aux):
    proposer = _proposer(method)
    proposer.draft_model = SimpleNamespace(
        model=SimpleNamespace(use_aux_hidden_state=uses_aux)
    )
    synced, timed = _precompile_harness(monkeypatch, proposer)
    proposer._precompile_combine_hidden_states()
    assert synced == [] and timed == []


def test_precompile_combine_warms_every_token_bucket(monkeypatch):
    proposer = _proposer()
    _eager(proposer)
    proposer.draft_model = SimpleNamespace(
        model=SimpleNamespace(use_aux_hidden_state=True, fc_input_size=6),
        combine_hidden_states=lambda x: x.sum(-1),
    )
    synced, timed = _precompile_harness(monkeypatch, proposer)
    proposer._precompile_combine_hidden_states()
    assert timed == ["drafter combine_hidden_states"]
    assert [tuple(t.shape) for t in synced] == [(4,), (8,), (16,)]


def test_precompile_draft_seed_and_logits_buckets(monkeypatch):
    proposer = _proposer()
    _eager(proposer)
    proposer.draft_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=3), compute_logits=lambda h: h
    )
    synced, timed = _precompile_harness(monkeypatch, proposer)
    seeds = []
    raw_seed = proposer._draft_seed_input_ids

    def seed(ids, lti, nti):
        seeds.append((ids.shape[0], lti.shape[0]))
        return raw_seed(ids, lti, nti)

    proposer._draft_seed_input_ids = seed
    proposer._precompile_draft_seed()
    # idx sizes = num_reqs_paddings {2, 8} | token buckets <= pad(4)=4 -> [2, 4, 8].
    assert seeds == [(4, 2), (4, 4), (8, 2), (8, 4), (8, 8), (16, 2), (16, 4), (16, 8)]
    synced.clear()
    proposer._precompile_compute_logits()
    assert timed == ["drafter seed_input_ids", "drafter compute_logits"]
    assert sorted(t.shape[0] for t in synced) == [2, 4, 8]


def test_dummy_draft_noop_without_lockstep_or_work():
    proposer = _proposer(draft_tp=2, target_tp=2)
    proposer._forward_draft = lambda **_k: pytest.fail("must not run")
    proposer.run_dp_dummy_draft(2)  # no lockstep
    proposer.runner._dp_lockstep_enabled = lambda: True
    proposer._draft_md_cache[1] = "stale"
    proposer.run_dp_dummy_draft(0)  # no chunks to pad
    assert not proposer._draft_md_cache


def test_dummy_draft_eagle3_plain_combine(monkeypatch):
    monkeypatch.setattr(e3, "synchronize_tensors", lambda *a: None)
    proposer = _proposer(draft_tp=2, target_tp=2, K=2)
    proposer.runner._dp_lockstep_enabled = lambda: True
    proposer.runner._dp_target_bucket = 8
    draft = _Draft(uses_aux=False)
    draft.config = SimpleNamespace(hidden_size=HID)
    proposer.draft_model = draft
    proposer._forward_draft = lambda **kw: (
        torch.zeros(kw["num_tokens_padded"], HID),
        None,
    )
    proposer._draft_propose_token = lambda h: torch.zeros(h.shape[0], dtype=torch.int32)
    proposer.run_dp_dummy_draft(2)
    assert draft.combined == [(8, HID), (8, HID)]


# --------------------------------------------------------------------------
# TD8.105: Full 100% statement & branch coverage for eagle3.py
# --------------------------------------------------------------------------


def test_static_shape_args_unmarks_dynamo_dynamic_indices():
    @e3._static_shape_args
    def fn(a, b, *, c=None, d=None):
        return a, b, c, d

    t_marked = torch.tensor([1, 2, 3])
    t_marked._dynamo_dynamic_indices = {0}
    t_plain = torch.tensor([4, 5])
    kw_marked = torch.tensor([6, 7])
    kw_marked._dynamo_dynamic_indices = {0}
    ra, rb, rc, rd = fn(t_marked, t_plain, c=kw_marked, d="not_a_tensor")
    assert ra is not t_marked and torch.equal(ra, t_marked)
    assert getattr(ra, "_dynamo_dynamic_indices", None) is None
    assert rb is t_plain
    assert rc is not kw_marked and torch.equal(rc, kw_marked)
    assert rd == "not_a_tensor"


def test_collect_topk_indices_buffers_deduplicates():
    buf1 = torch.zeros(4, 2, dtype=torch.int32)
    buf2 = torch.zeros(4, 2, dtype=torch.int32)
    l0 = SimpleNamespace(
        mtp_block=SimpleNamespace(
            self_attn=SimpleNamespace(
                mla_attn=SimpleNamespace(topk_indices_buffer=buf1)
            )
        )
    )
    l1 = SimpleNamespace(
        mtp_block=SimpleNamespace(
            self_attn=SimpleNamespace(
                mla_attn=SimpleNamespace(topk_indices_buffer=buf1)
            )
        )
    )
    l2 = SimpleNamespace(
        mtp_block=SimpleNamespace(
            self_attn=SimpleNamespace(
                mla_attn=SimpleNamespace(topk_indices_buffer=buf2)
            )
        )
    )
    l3 = SimpleNamespace(
        mtp_block=SimpleNamespace(
            self_attn=SimpleNamespace(
                mla_attn=SimpleNamespace(topk_indices_buffer=None)
            )
        )
    )
    mtp_model = SimpleNamespace(layers={"0": l0, "1": l1, "2": l2, "3": l3})
    res = e3._collect_topk_indices_buffers(mtp_model)
    assert len(res) == 2 and res[0] is buf1 and res[1] is buf2


def test_proposer_init_default_and_invalid_draft_tp():
    p = _proposer("eagle3", draft_tp=None, target_tp=4)
    assert p.speculative_config.draft_tensor_parallel_size == 4
    assert p._draft_replicated is False
    assert p._draft_tp_matches_target is True
    with pytest.raises(ValueError, match="unsupported on TPU"):
        _proposer("eagle3", draft_tp=2, target_tp=4)


def test_validate_mtp_layer_count_multi_layer(monkeypatch):
    p = _proposer("mtp")
    p.draft_model = SimpleNamespace(model=SimpleNamespace(num_mtp_layers=2))
    monkeypatch.setattr(e3, "iter_mtp_shared_heads", lambda _m: iter([]))
    p._validate_mtp_layer_count()
    monkeypatch.setattr(e3, "iter_mtp_shared_heads", lambda _m: iter(["h0", "h1"]))
    with pytest.raises(NotImplementedError, match="num_nextn_predict_layers=2"):
        p._validate_mtp_layer_count()


def test_init_mtp_index_sharing_disabled_and_buffer_sizing():
    p = _proposer("mtp")
    p.speculative_config.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(index_share_for_mtp_iteration=False)
    )
    p._init_mtp_index_sharing()
    assert p._share_mtp_indices is False

    small_buf = torch.zeros((2, 4), dtype=torch.int32)
    l0 = SimpleNamespace(
        mtp_block=SimpleNamespace(
            self_attn=SimpleNamespace(
                mla_attn=SimpleNamespace(topk_indices_buffer=small_buf)
            )
        )
    )
    p2 = _index_share_proposer(
        SimpleNamespace(set_skip_topk=lambda _s: None, layers={"0": l0})
    )
    p2.vllm_config.scheduler_config = SimpleNamespace(max_num_batched_tokens=2)
    with pytest.raises(RuntimeError, match="topk_indices_buffer holds 2 rows"):
        p2._init_mtp_index_sharing()

    good_buf = torch.zeros((8, 4), dtype=torch.int32)
    l1 = SimpleNamespace(
        mtp_block=SimpleNamespace(
            self_attn=SimpleNamespace(
                mla_attn=SimpleNamespace(topk_indices_buffer=good_buf)
            )
        )
    )
    p3 = _index_share_proposer(
        SimpleNamespace(set_skip_topk=lambda _s: None, layers={"0": l1})
    )
    p3._init_mtp_index_sharing()
    assert p3._share_mtp_indices is True
    assert torch.all(good_buf == -1)


def test_build_mtp_loop_model_and_step_helpers(monkeypatch):
    import vllm_torchtpu.spec_decode.mtp_loop_model as mlm

    _null_contexts(monkeypatch)
    p = _proposer("mtp")
    p._share_mtp_indices = True
    p._mtp_topk_buffers = [torch.zeros((4, 2), dtype=torch.int32)]
    p.draft_model = SimpleNamespace(_dynamic_arg_dims={"bad": 99})
    p._build_mtp_loop_model()
    assert p._share_mtp_indices is False
    assert p._mtp_topk_buffers == []
    assert p._mtp_loop_model is None

    monkeypatch.setattr(mlm, "MtpLoopStepModel", lambda **kw: ("loop_prog", kw))
    for dims in (None, dict(mlm.MTP_DYNAMIC_ARG_DIMS)):
        p._share_mtp_indices = True
        p._mtp_topk_buffers = [torch.zeros((4, 2), dtype=torch.int32)]
        p.draft_model = SimpleNamespace(
            _dynamic_arg_dims=dims,
            model=SimpleNamespace(set_skip_topk=lambda s: None),
        )
        p._build_mtp_loop_model()
        assert p._mtp_loop_model[0] == "loop_prog"
        assert p._mtp_loop_model_warm is False

    skips = []
    compacted = []
    buf = torch.arange(8, dtype=torch.int32).view(4, 2)
    p._mtp_topk_buffers = [buf]
    p.draft_model = SimpleNamespace(
        model=SimpleNamespace(
            set_skip_topk=skips.append,
            compact_topk_indices=lambda g: compacted.append(g.tolist()),
        )
    )
    assert p._draft_model_for_step(0) is p.draft_model
    assert p._draft_model_for_step(1) == p._mtp_loop_model
    assert skips == [False, True]
    p._share_mtp_indices = False
    p._mtp_set_skip_topk(True)
    assert skips == [False, True]
    p._share_mtp_indices = True
    saved = p._mtp_compact_and_save(torch.tensor([0, 2]))
    assert compacted == [[0, 2]]
    assert saved[0].tolist() == [[0, 1], [2, 3]]
    buf.zero_()
    p._mtp_restore_topk([torch.tensor([[9, 8], [7, 6]], dtype=torch.int32)])
    assert buf[:2].tolist() == [[9, 8], [7, 6]]


def test_validate_and_setup_gemma4_kv_sharing_branches(monkeypatch):
    p = _proposer("mtp")
    monkeypatch.setitem(vars(envs), "VLLM_TPU_BLOCK_MAJOR_KV", True)
    with pytest.raises(NotImplementedError, match="VLLM_TPU_BLOCK_MAJOR_KV"):
        p._validate_kv_sharing_supported()

    monkeypatch.setitem(vars(envs), "VLLM_TPU_BLOCK_MAJOR_KV", False)
    monkeypatch.setattr(
        e3.PcpStaticSupportValidator,
        "from_vllm_config",
        classmethod(lambda cls, _cfg: SimpleNamespace(enabled=True)),
    )
    with pytest.raises(NotImplementedError, match="PCP streaming attention"):
        p._validate_kv_sharing_supported()

    monkeypatch.setattr(
        e3.PcpStaticSupportValidator,
        "from_vllm_config",
        classmethod(lambda cls, _cfg: SimpleNamespace(enabled=False)),
    )
    p._validate_kv_sharing_supported()

    p.draft_model = SimpleNamespace()
    p._setup_gemma4_kv_sharing("target", set())
    assert p._draft_kv_shared is False

    p.draft_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=SimpleNamespace(is_kv_shared_layer=False))
            ]
        )
    )
    p._setup_gemma4_kv_sharing("target", set())
    assert p._draft_kv_shared is False

    target = SimpleNamespace(
        config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(layer_types=["full_attention"])
        )
    )
    p.draft_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(is_kv_shared_layer=True, attn=None)
                )
            ]
        ),
        config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(layer_types=["full_attention"])
        ),
    )
    with pytest.raises(RuntimeError, match="exposes no self_attn.attn"):
        p._setup_gemma4_kv_sharing(
            target, {"language_model.model.layers.0.self_attn.attn"}
        )

    p.draft_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        is_kv_shared_layer=True,
                        attn=SimpleNamespace(impl=SimpleNamespace()),
                    )
                )
            ]
        ),
        config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(layer_types=["sliding_attention"])
        ),
    )
    with pytest.raises(
        RuntimeError, match="No target layer of type 'sliding_attention'"
    ):
        p._setup_gemma4_kv_sharing(
            target, {"language_model.model.layers.0.self_attn.attn"}
        )

    p.draft_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        is_kv_shared_layer=True,
                        attn=SimpleNamespace(impl=None),
                    )
                )
            ]
        ),
        config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(layer_types=["full_attention"])
        ),
    )
    with pytest.raises(RuntimeError, match="has no `impl`"):
        p._setup_gemma4_kv_sharing(
            target, {"language_model.model.layers.0.self_attn.attn"}
        )


def test_load_draft_model_and_pcp_dsa_validators(monkeypatch):
    _null_contexts(monkeypatch)
    loaded = []
    monkeypatch.setattr(
        e3,
        "get_model_loader",
        lambda _cfg: SimpleNamespace(
            load_model=lambda **kw: (loaded.append(kw), "loaded_draft")[1]
        ),
    )
    monkeypatch.setattr(e3, "_force_draft_tp1", contextlib.nullcontext)
    for draft_tp, target_tp in ((1, 1), (2, 2)):
        p = _proposer("eagle3", draft_tp=draft_tp, target_tp=target_tp)
        p.vllm_config.load_config = "load_cfg"
        p.vllm_config.compilation_config = SimpleNamespace(inductor_compile_config={})
        p.speculative_config.draft_model_config = SimpleNamespace(runner_type="other")
        p._load_draft_model()
        assert p.draft_model == "loaded_draft"
        assert (
            p.draft_vllm_config.compilation_config.inductor_compile_config[
                "_vllm_model_tag"
            ]
            == "eagle_head"
        )
        assert loaded[-1]["model_config"].runner_type == "draft"

    # _validate_pcp_draft_model
    p_pcp = _proposer("mtp")
    full_spec = e3.FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    p_pcp.runner.get_kv_cache_spec = lambda: {"d.0": full_spec}
    p_pcp._validate_pcp_draft_model({"d.0"})
    p_pcp.runner.get_kv_cache_spec = lambda: {"d.0": "not_full_spec"}
    with pytest.raises(NotImplementedError, match="requires FullAttentionSpec"):
        p_pcp._validate_pcp_draft_model({"d.0"})

    # _split_dsa_draft_layers and _validate_dsa_draft
    class FakeMLA:
        pass

    class FakeIndexer:
        pass

    fake_mla_mod = types.ModuleType(
        "vllm.model_executor.layers.attention.mla_attention"
    )
    fake_mla_mod.MLAAttention = FakeMLA
    fake_ds_mod = types.ModuleType("vllm.model_executor.models.deepseek_v2")
    fake_ds_mod.DeepseekV32IndexerCache = FakeIndexer
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.attention.mla_attention", fake_mla_mod
    )
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.deepseek_v2", fake_ds_mod
    )

    # No indexer -> returns early
    p_dsa = _proposer("mtp", draft_tp=2, target_tp=2)
    p_dsa._validate_dsa_draft({"d.mla": FakeMLA()}, {"d.mla"})

    # Indexer without MLA -> raises RuntimeError
    with pytest.raises(RuntimeError, match="no MLA attention layer"):
        p_dsa._validate_dsa_draft({"d.idx": FakeIndexer()}, {"d.idx"})

    # Replicated DSA draft when target_tp > 1 -> raises NotImplementedError
    p_rep = _proposer("mtp", draft_tp=1, target_tp=2)
    with pytest.raises(NotImplementedError, match="cannot run replicated"):
        p_rep._validate_dsa_draft(
            {"d.idx": FakeIndexer(), "d.mla": FakeMLA()}, {"d.idx", "d.mla"}
        )

    # Unregistered KV cache spec -> raises RuntimeError
    p_dsa.runner.get_kv_cache_spec = lambda: {"d.mla": full_spec}
    with pytest.raises(RuntimeError, match="got no KVCacheSpec"):
        p_dsa._validate_dsa_draft(
            {"d.idx": FakeIndexer(), "d.mla": FakeMLA()}, {"d.idx", "d.mla"}
        )

    # All registered -> succeeds
    p_dsa.runner.get_kv_cache_spec = lambda: {"d.mla": full_spec, "d.idx": full_spec}
    p_dsa._validate_dsa_draft(
        {"d.idx": FakeIndexer(), "d.mla": FakeMLA()}, {"d.idx", "d.mla"}
    )


def test_propose_zero_reqs_and_index_share_and_forward_loop_model(monkeypatch):
    _null_contexts(monkeypatch)
    p = _proposer("mtp")
    p.runner.input_batch.num_reqs = 0
    assert p.propose(None, [], None, None) == []

    # propose with _share_mtp_indices=True
    proposer, _ = _propose_harness(
        "mtp", K=2, sizes=(2,), uses_aux=False, rejected=lambda nr: None
    )
    proposer._share_mtp_indices = True
    compacted = []
    restored = []
    proposer._mtp_compact_and_save = lambda idx: (
        compacted.append(idx.tolist()),
        ["saved_rows"],
    )[1]
    proposer._mtp_restore_topk = restored.append
    out = proposer.propose([[0]] * 2, [], None, None)
    assert out == _expected((2,), 2)
    assert len(compacted) == 1
    assert restored == [["saved_rows"]]

    # _forward_draft with method="mtp" and _mtp_loop_model warm-up
    p_fwd = _proposer("mtp")
    p_fwd._build_draft_attn_metadata = lambda **_k: {}
    loop_calls = []

    def loop_model(**kw):
        loop_calls.append(kw)
        return kw["hidden_states"] + 1

    p_fwd.draft_model = SimpleNamespace(
        model=SimpleNamespace(set_skip_topk=lambda _s: None)
    )
    p_fwd._mtp_loop_model = loop_model
    p_fwd._mtp_loop_model_warm = False
    last, carry = p_fwd._forward_draft(
        chunk=None,
        input_ids=torch.zeros(2),
        positions=torch.zeros(2),
        target_hidden_states=torch.ones(2, 3),
        step_idx=1,
        seq_lens_delta=1,
        num_rejected_np=None,
    )
    assert len(loop_calls) == 2  # first warms, second runs
    assert loop_calls[0]["spec_step_idx"] == 1
    assert p_fwd._mtp_loop_model_warm is True
    torch.testing.assert_close(last, torch.full((2, 3), 2.0))
    assert carry is last

    # Second call when already warm runs once
    loop_calls.clear()
    p_fwd._forward_draft(
        chunk=None,
        input_ids=torch.zeros(2),
        positions=torch.zeros(2),
        target_hidden_states=torch.ones(2, 3),
        step_idx=1,
        seq_lens_delta=1,
        num_rejected_np=None,
    )
    assert len(loop_calls) == 1


def test_draft_has_moe_and_dummy_draft_aux_and_mtp(monkeypatch):
    monkeypatch.setattr(e3, "synchronize_tensors", lambda *a: None)
    p = _proposer("eagle3")
    assert p._draft_has_moe() is False  # draft_model is None -> False (not cached)

    import vllm.model_executor.models.interfaces as interfaces

    monkeypatch.setattr(interfaces, "is_mixture_of_experts", lambda _m: True)
    p.draft_model = SimpleNamespace()
    assert p._draft_has_moe() is True
    assert p._draft_has_moe() is True  # second call hits _draft_has_moe_cache

    # run_dp_dummy_draft with uses_aux=True
    p_aux = _proposer("eagle3", draft_tp=2, target_tp=2, K=2)
    p_aux.runner._dp_lockstep_enabled = lambda: True
    p_aux.runner._dp_target_bucket = 8
    draft_aux = _Draft(uses_aux=True)
    draft_aux.model.fc_input_size = 3 * HID
    draft_aux.config = SimpleNamespace(hidden_size=HID)
    p_aux.draft_model = draft_aux
    _eager(p_aux)
    p_aux._forward_draft = lambda **kw: (
        torch.zeros(kw["num_tokens_padded"], HID),
        None,
    )
    p_aux._draft_propose_token = lambda h: torch.zeros(h.shape[0], dtype=torch.int32)
    p_aux.run_dp_dummy_draft(1)
    assert draft_aux.combined == [(8, 3 * HID)]

    # run_dp_dummy_draft with method="mtp" (neither uses_aux nor combines_plain) +
    # uses_mrope=True
    p_mtp = _proposer("mtp", draft_tp=2, target_tp=2, K=2, uses_mrope=True)
    p_mtp.runner._dp_lockstep_enabled = lambda: True
    p_mtp.runner._dp_target_bucket = 8
    p_mtp.draft_model = SimpleNamespace(config=SimpleNamespace(hidden_size=HID))
    fwd_shapes = []
    p_mtp._forward_draft = lambda **kw: (
        fwd_shapes.append(tuple(kw["positions"].shape)),
        (torch.zeros(kw["num_tokens_padded"], HID), None),
    )[1]
    p_mtp._draft_propose_token = lambda h: torch.zeros(h.shape[0], dtype=torch.int32)
    p_mtp.run_dp_dummy_draft(1)
    assert fwd_shapes == [(3, 8), (3, 4)]


def test_td8274_prepare_draft_inputs_async_1d_flat_int32_gather(monkeypatch):
    select_dtypes = []
    orig_index_select = torch.Tensor.index_select

    def spy_index_select(self, dim, index):
        select_dtypes.append((self.ndim, dim, index.dtype))
        return orig_index_select(self, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", spy_index_select)
    for with_lengths in (False, True):
        lengths = torch.tensor([2, 2, 2], dtype=torch.int64) if with_lengths else None
        proposer, chunk, sched = _prep_setup(draft_lengths=lengths)
        next_tokens = torch.tensor(
            [[7, 8, -1], [9, -1, -1], [11, 12, 13]], dtype=torch.int32
        )
        select_dtypes.clear()
        ids, _, gather, rej = proposer._prepare_draft_inputs(
            chunk,
            sampled_token_ids=None,
            discard_sampled_tokens_req_indices=[],
            num_rejected_tokens_np=None,
            scheduler_output=sched,
            next_tokens_device=next_tokens,
        )
        assert (1, 0, torch.int32) in select_dtypes
        assert ids.dtype == torch.int32
        assert rej.dtype == torch.int32
        assert gather.dtype == torch.int32
        if with_lengths:
            assert rej.tolist() == [1, 2, 0]
            assert gather.tolist() == [1, 2, 5, 0]
            assert ids.tolist() == [1, 8, 9, 4, 5, 13, 7, 7]
        else:
            assert rej.tolist() == [1, 1, 0]
            assert gather.tolist() == [1, 3, 5, 0]
            assert ids.tolist() == [1, 8, 3, 9, 5, 13, 7, 7]


def test_td8279_draft_propose_token_and_gather_carries_int32_index_select(monkeypatch):
    select_calls = []
    orig_index_select = torch.index_select
    orig_tensor_index_select = torch.Tensor.index_select

    def spy_torch_index_select(t, dim, index):
        select_calls.append((t.ndim, dim, index.dtype))
        return orig_index_select(t, dim, index)

    def spy_tensor_index_select(self, dim, index):
        select_calls.append((self.ndim, dim, index.dtype))
        return orig_tensor_index_select(self, dim, index)

    monkeypatch.setattr(torch, "index_select", spy_torch_index_select)
    monkeypatch.setattr(torch.Tensor, "index_select", spy_tensor_index_select)

    p = _proposer("eagle3")
    _eager(p)
    logits = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.1, 0.2]], dtype=torch.float32)
    p.draft_model = SimpleNamespace(
        draft_id_to_target_id=torch.tensor([10, 20, 30], dtype=torch.int32),
        lm_head="head",
        logits_processor=lambda _h, _x: logits,
    )
    select_calls.clear()
    out_tok = p._draft_propose_token(torch.zeros((2, 4)))
    assert out_tok.dtype == torch.int32
    assert out_tok.tolist() == [21, 10]
    assert select_calls == [(1, 0, torch.int32)]

    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    last_hidden = (hidden + 100).clone()
    pos_1d = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    pos_2d = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    indices_i64 = torch.tensor([1, 3], dtype=torch.int64)

    select_calls.clear()
    h1, p1, lh1 = p._draft_gather_carries(hidden, pos_1d, last_hidden, indices_i64)
    assert p1.tolist() == [20, 40]
    assert torch.equal(h1, hidden[[1, 3]])
    assert torch.equal(lh1, last_hidden[[1, 3]])
    assert all(dt == torch.int32 for _, _, dt in select_calls)

    select_calls.clear()
    _, p2, _ = p._draft_gather_carries(hidden, pos_2d, last_hidden, indices_i64)
    assert torch.equal(p2, pos_2d[:, [1, 3]])
    assert all(dt == torch.int32 for _, _, dt in select_calls)


def test_td8297_last_token_indices_and_gather_indices_int32(monkeypatch):
    proposer, chunk, sched = _prep_setup()
    # 1. Pure sync path: gather_indices is torch.int32
    ids_sync, _, gather_sync, _ = proposer._prepare_draft_inputs(
        chunk,
        sampled_token_ids=[[11], [22], [33]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=np.array([0, 1, 0], dtype=np.int32),
        scheduler_output=sched,
    )
    assert gather_sync.dtype == torch.int32
    assert ids_sync.dtype == torch.int32

    # 2. device_seed bootstrap path: gather_indices is torch.int32
    ids_boot, _, gather_boot, _ = proposer._prepare_draft_inputs(
        chunk,
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=sched,
        device_seed=torch.tensor([11, 22, 33], dtype=torch.int32),
    )
    assert gather_boot.dtype == torch.int32
    assert ids_boot.dtype == torch.int32

    # 3. _draft_seed_input_ids accepts both torch.int32 and torch.int64 indices
    base_ids = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    nti = torch.tensor([99, 88], dtype=torch.int32)
    out_i32 = proposer._draft_seed_input_ids(
        base_ids, torch.tensor([1, 3], dtype=torch.int32), nti
    )
    out_i64 = proposer._draft_seed_input_ids(
        base_ids, torch.tensor([1, 3], dtype=torch.int64), nti
    )
    assert torch.equal(out_i32, out_i64)

    # 4. _draft_gather_carries accepts torch.int32 without casting
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    pos_1d = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    h_i32, p_i32, lh_i32 = proposer._draft_gather_carries(
        hidden, pos_1d, hidden + 1, torch.tensor([0, 2], dtype=torch.int32)
    )
    assert p_i32.tolist() == [10, 30]
    assert torch.equal(h_i32, hidden[[0, 2]])
    assert torch.equal(lh_i32, (hidden + 1)[[0, 2]])

    # 5. _precompile_draft_seed passes torch.int32 lti
    p_pre = _proposer()
    _eager(p_pre)
    _precompile_harness(monkeypatch, p_pre)
    seen_lti_dtypes = []
    raw_seed = p_pre._draft_seed_input_ids

    def spy_seed(ids, lti, nti):
        seen_lti_dtypes.append(lti.dtype)
        return raw_seed(ids, lti, nti)

    p_pre._draft_seed_input_ids = spy_seed
    p_pre._precompile_draft_seed()
    assert seen_lti_dtypes and all(dt == torch.int32 for dt in seen_lti_dtypes)
