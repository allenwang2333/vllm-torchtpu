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

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import torch._dynamo

from vllm_torchtpu.spec_decode import utils as spec_decode_utils
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.eagle3 import DraftChunkInputs


_COMPILED_METHODS = (
    "_tpu_precompute_and_update_kv_cache",
    "_dflash_forward_and_sample",
    "_tpu_build_dflash_inputs_next_tokens",
    "_tpu_build_dflash_inputs_seed",
)


@pytest.fixture(autouse=True)
def _eager_on_cpu(request, monkeypatch):
    """The proposer's `@torch.compile(backend="tpu")` graphs only trace on
    TPU tensors; run them eagerly when the `device` fixture resolves to CPU
    so the input builders stay unit-testable without a chip."""
    if "device" in request.fixturenames and request.getfixturevalue(
            "device").type == "cpu":
        for name in _COMPILED_METHODS:
            monkeypatch.setattr(
                DFlashProposer, name,
                torch._dynamo.disable(getattr(DFlashProposer, name)))


def _make_proposer(draft_tp: int | None = 1,
                   target_tp: int = 1,
                   target_layer_ids: list[int] = (1, 2)) -> DFlashProposer:
    hf_config_mock = mock.MagicMock()
    # A non-zero mask id so mask slots are distinguishable from zero-fill.
    hf_config_mock.to_dict.return_value = {
        "dflash_config": {
            "mask_token_id": 7,
            "target_layer_ids": list(target_layer_ids),
        }
    }

    speculative_config = SimpleNamespace(
        num_speculative_tokens=4,
        draft_tensor_parallel_size=draft_tp,
        draft_model_config=SimpleNamespace(hf_config=hf_config_mock))
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        model_config=SimpleNamespace(max_model_len=2048),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "model.layers.0.self_attn": mock.MagicMock(),
                "model.layers.1.self_attn": mock.MagicMock(),
            },
            fast_moe_cold_start=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=target_tp,
            is_moe_model=False,
            use_sequence_parallel_moe=False,
        ),
    )
    runner = mock.MagicMock()
    runner.device = torch.device("cpu")
    return DFlashProposer(runner=runner, vllm_config=vllm_config)


def _make_vocab_models():
    target_embed = torch.nn.Embedding(3, 2)
    target_embed.org_vocab_size = 3
    draft_embed = torch.nn.Embedding(3, 2)
    draft_embed.org_vocab_size = 3
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target_embed),
        lm_head=torch.nn.Linear(2, 3, bias=False),
        set_aux_hidden_state_layers=lambda _: None,
    )
    draft = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=draft_embed, layers=[]),
        lm_head=torch.nn.Linear(2, 3, bias=False),
        get_draft_attn_causal=lambda: [],
        logits_processor=SimpleNamespace(
            _gather_logits=lambda logits: logits.clone()),
        has_own_embed_tokens=False,
        has_own_lm_head=False,
    )
    return draft, target


def _fake_gather_sharded_weight(module):
    weight = module.weight.detach().clone()
    return weight, weight.shape[0], weight.shape[1], weight.device


@pytest.mark.parametrize("draft_tp,target_tp", [(1, 1), (1, 8), (8, 8)])
def test_load_model_vocab_weights_follow_tp_layout(draft_tp, target_tp,
                                                   monkeypatch):
    proposer = _make_proposer(draft_tp=draft_tp, target_tp=target_tp)
    draft, target = _make_vocab_models()

    def load_draft_model():
        proposer.draft_model = draft

    proposer._load_draft_model = load_draft_model
    monkeypatch.setattr(spec_decode_utils, "gather_sharded_weight",
                        _fake_gather_sharded_weight)

    with mock.patch(
            "vllm_torchtpu.spec_decode.dflash.get_layers_from_vllm_config",
            return_value={}):
        proposer.load_model(target)

    if draft_tp == target_tp:
        assert draft.model.embed_tokens is target.model.embed_tokens
        assert draft.lm_head is target.lm_head
    else:
        assert draft.model.embed_tokens is not target.model.embed_tokens
        assert torch.equal(draft.model.embed_tokens.weight,
                           target.model.embed_tokens.weight)
        assert draft.lm_head is not target.lm_head
        assert torch.equal(draft.lm_head.weight, target.lm_head.weight)


def test_propose_empty_batch():
    proposer = _make_proposer(draft_tp=1)
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=0),
                                      device=torch.device("cpu"))
    assert proposer.propose([], [], None, None) == []
    # The async caller expects a tensor, even an empty one.
    out = proposer.propose([], [], None, None, return_device=True)
    assert isinstance(out, torch.Tensor) and out.shape == (0, 4)


def test_tpu_precompute_context_kv(device):
    # Setup dimension sizes
    L = 2  # layers
    nkv = 2  # KV heads
    hd = 8  # head dim
    target_hidden_size = 32
    num_ctx = 3  # token count

    proposer = _make_proposer(draft_tp=1)

    # Create random hidden states and positions
    hidden_states = torch.randn((num_ctx, target_hidden_size),
                                dtype=torch.float32,
                                device=device)
    positions = torch.tensor([10, 11, 12], dtype=torch.int32, device=device)

    # Create draft model mock
    draft_model = mock.MagicMock()
    del draft_model.combine_hidden_states
    del draft_model.fc
    self_model = mock.MagicMock()

    # Initialize fused weight matrix
    fused_weight = torch.randn((L * 2 * nkv * hd, target_hidden_size),
                               dtype=torch.float32,
                               device=device)
    self_model._fused_kv_weight = fused_weight
    del self_model.fc
    del self_model.hidden_norm
    self_model._fused_kv_bias = None

    # Split the fused weight to mock individual layers eagerly
    # all_kv shape: [num_ctx, L, 2, nkv, hd]
    # Weight shape: [L, 2, nkv, hd, target_hidden_size]
    weight_structured = fused_weight.view(L, 2, nkv, hd, target_hidden_size)

    # Define layers and their proj weights
    layers = []
    for i in range(L):
        layer = mock.MagicMock()
        layer.self_attn.head_dim = hd
        layer.self_attn.num_kv_heads = nkv
        del layer.self_attn.k_norm

        # Extract individual weights for assertion comparison
        k_w = weight_structured[i, 0].reshape(nkv * hd, target_hidden_size)
        v_w = weight_structured[i, 1].reshape(nkv * hd, target_hidden_size)

        layer.self_attn.k_proj = torch.nn.Linear(target_hidden_size,
                                                 nkv * hd,
                                                 bias=False).to(device)
        layer.self_attn.k_proj.weight.data.copy_(k_w)

        layer.self_attn.v_proj = torch.nn.Linear(target_hidden_size,
                                                 nkv * hd,
                                                 bias=False).to(device)
        layer.self_attn.v_proj.weight.data.copy_(v_w)

        layers.append(layer)

    self_model.layers = layers

    # Mock rotary_emb: adds positions to keys to verify they map 1-to-1
    def mock_rotary_emb(positions_rep, dummy_q, keys_flat):
        # positions_rep has shape [L * num_ctx]
        # keys_flat has shape [L * num_ctx, nkv, hd]
        roped_k = keys_flat + positions_rep.view(-1, 1, 1).to(keys_flat.dtype)
        return dummy_q, roped_k

    self_model.layers[0].self_attn.rotary_emb = mock_rotary_emb

    draft_model.model = self_model
    proposer.draft_model = draft_model

    # Initialize mock forward implementations on the layers to record outputs
    for layer in layers:
        layer.self_attn.attn.impl = mock.MagicMock()
        layer.self_attn.attn.impl.forward = mock.MagicMock(
            side_effect=lambda **kwargs: kwargs["key"])

    # 1. Run compiled/batched math eagerly inside test suite
    fn_eager = torch._dynamo.disable(
        DFlashProposer._tpu_precompute_and_update_kv_cache)
    dummy_md = mock.MagicMock()
    fn_eager(proposer, hidden_states, positions, tuple([dummy_md] * L))

    # Extract roped keys and values from the mocked layer forward calls
    roped_k_list = []
    all_v_list = []
    for layer in layers:
        call_kwargs = layer.self_attn.attn.impl.forward.call_args[1]
        roped_k_list.append(call_kwargs["key"])
        all_v_list.append(call_kwargs["value"])
    roped_k_all = torch.stack(roped_k_list, dim=0)
    all_v = torch.stack(all_v_list, dim=0)

    # 2. Compute expected values layer-by-layer eagerly
    expected_ks = []
    expected_vs = []

    for i in range(L):
        layer = self_model.layers[i]
        # Project target hidden to layer keys and values
        k_proj_out = layer.self_attn.k_proj(hidden_states).view(
            num_ctx, nkv, hd)
        v_proj_out = layer.self_attn.v_proj(hidden_states).view(
            num_ctx, nkv, hd)

        # Apply RoPE layer-by-layer
        dummy_q_i = torch.zeros_like(k_proj_out)
        _, k_roped_i = mock_rotary_emb(positions, dummy_q_i, k_proj_out)

        expected_ks.append(k_roped_i)
        expected_vs.append(v_proj_out)

    expected_k_all = torch.stack(expected_ks, dim=0)
    expected_v_all = torch.stack(expected_vs, dim=0)

    # 3. Assert exact equality
    assert torch.allclose(roped_k_all, expected_k_all, atol=1e-5)
    assert torch.allclose(all_v, expected_v_all, atol=1e-5)


def _make_chunk(num_reqs,
                start_index,
                device,
                query_start_loc_np,
                position_ids,
                aux_hidden_states=None,
                seq_lens=None,
                draft_lengths=None):
    query_start_loc = torch.from_numpy(query_start_loc_np).to(
        device) if query_start_loc_np is not None else None
    if draft_lengths is not None:
        draft_lengths = torch.tensor(draft_lengths,
                                     dtype=torch.int32,
                                     device=device)

    return DraftChunkInputs(
        input_ids=None,
        position_ids=position_ids,
        query_start_loc_np=query_start_loc_np,
        attn_ctx=SimpleNamespace(seq_lens=seq_lens,
                                 query_start_loc=query_start_loc,
                                 use_max_model_len=True,
                                 position_ids_override=None),
        start_index=start_index,
        num_reqs=num_reqs,
        aux_hidden_states=aux_hidden_states,
        draft_lengths=draft_lengths,
        attn_metadata={
            "dummy_layer":
            SimpleNamespace(block_tables=torch.ones(10,
                                                    dtype=torch.int32,
                                                    device=device),
                            seq_lens=seq_lens)
        },
    )


def _input_proposer(device, K=3, draft_tp=1, target_tp=1, max_chunk_reqs=0):
    proposer = _make_proposer(draft_tp=draft_tp, target_tp=target_tp)
    proposer.speculative_config.num_speculative_tokens = K
    proposer.runner = SimpleNamespace(
        device=device,
        num_tokens_paddings=[8, 16, 32],
        _dp_lockstep_enabled=lambda: max_chunk_reqs > 0,
        _dp_step_max_reqs=max_chunk_reqs,
    )
    return proposer


def _t(values, device):
    return torch.tensor(values, dtype=torch.int32, device=device)


def test_prepare_dflash_inputs_verify_window_anchors_after_accepted(device):
    """A verified request anchors at start + num_accepted, so a rejection
    pulls the next window back. The prompt-chunk anchor (last + 1) would
    give 14 for request 0; only the verify path gives 12."""
    proposer = _input_proposer(device)
    chunk = _make_chunk(num_reqs=2,
                        start_index=0,
                        device=device,
                        query_start_loc_np=np.array([0, 4, 8], dtype=np.int32),
                        position_ids=_t([10, 11, 12, 13, 40, 41, 42, 43],
                                        device),
                        draft_lengths=[3, 3])
    next_tokens = _t([[101, 102, -1, -1], [201, 202, 203, 204]], device)

    input_ids, positions, seq_lens = proposer._prepare_dflash_inputs(
        chunk, next_tokens_device=next_tokens)

    assert torch.equal(input_ids, _t([102, 7, 7, 7, 204, 7, 7, 7], device))
    assert torch.equal(positions, _t([12, 13, 14, 15, 44, 45, 46, 47],
                                     device))
    assert torch.equal(seq_lens, _t([16, 48], device))


def test_prepare_dflash_inputs_routes_prompt_chunk_in_a_spec_step(device):
    """A request that finished prefill while others verified has
    draft_lengths == 0 and must anchor one past its last token (45), not
    at start + num_accepted (41)."""
    proposer = _input_proposer(device)
    chunk = _make_chunk(num_reqs=2,
                        start_index=0,
                        device=device,
                        query_start_loc_np=np.array([0, 4, 9], dtype=np.int32),
                        position_ids=_t([10, 11, 12, 13, 40, 41, 42, 43, 44],
                                        device),
                        draft_lengths=[3, 0])
    next_tokens = _t([[101, 102, -1, -1], [201, -1, -1, -1]], device)

    input_ids, positions, _ = proposer._prepare_dflash_inputs(
        chunk, next_tokens_device=next_tokens)

    assert torch.equal(input_ids, _t([102, 7, 7, 7, 201, 7, 7, 7], device))
    assert torch.equal(positions, _t([12, 13, 14, 15, 45, 46, 47, 48],
                                     device))


def test_prepare_dflash_inputs_without_spec_metadata_uses_prompt_anchor(
        device):
    """No spec metadata means nothing was verified: every request anchors
    after its last token, and the zero draft_lengths that route it is
    cached at the target chunk's padded request count."""
    proposer = _input_proposer(device)
    chunk = _make_chunk(num_reqs=2,
                        start_index=0,
                        device=device,
                        query_start_loc_np=np.array([0, 3, 5], dtype=np.int32),
                        position_ids=_t([10, 11, 12, 40, 41, 0, 0, 0], device))
    next_tokens = _t([[101, -1, -1, -1], [201, -1, -1, -1]], device)

    _, positions, _ = proposer._prepare_dflash_inputs(
        chunk, next_tokens_device=next_tokens)

    assert torch.equal(positions, _t([13, 14, 15, 16, 42, 43, 44, 45],
                                     device))
    assert list(proposer._zero_draft_lengths_cache) == [2]
    assert proposer._zero_draft_lengths_cache[2].shape == (2, )


def test_prepare_dflash_inputs_seed_anchors_after_last_prompt_token(device):
    """The post-prefill seed window starts one past the chunk's LAST prompt
    token (13, 42), not at the chunk's first position (10, 40)."""
    proposer = _input_proposer(device)
    chunk = _make_chunk(num_reqs=2,
                        start_index=0,
                        device=device,
                        query_start_loc_np=np.array([0, 3, 5], dtype=np.int32),
                        position_ids=_t([10, 11, 12, 40, 41, 0, 0, 0], device))

    input_ids, positions, seq_lens = proposer._prepare_dflash_inputs(
        chunk, device_seed=_t([5, 9], device))

    assert torch.equal(input_ids, _t([5, 7, 7, 7, 9, 7, 7, 7], device))
    assert torch.equal(positions, _t([13, 14, 15, 16, 42, 43, 44, 45],
                                     device))
    assert torch.equal(seq_lens, _t([17, 46], device))


@pytest.mark.parametrize("path", ["next_tokens", "seed"])
def test_prepare_dflash_inputs_padding_rows_and_context_limit(device, path):
    """One live request padded to a 2-row bucket at the context limit.

    Padding rows carry the mask id (a -1 base token would index the
    embedding table negatively), anchor at position 0, and keep a nonzero
    seq_len so their null-block writes stay off the kernel's kv_len == 0
    path. The
    live row's positions clamp to max_model_len - 1 and its seq_len to
    max_model_len.
    """
    proposer = _input_proposer(device)
    max_len = proposer._max_model_len
    chunk = _make_chunk(num_reqs=1,
                        start_index=0,
                        device=device,
                        query_start_loc_np=np.array([0, 4], dtype=np.int32),
                        position_ids=_t(
                            [max_len - 4, max_len - 3, max_len - 2, max_len - 1],
                            device),
                        draft_lengths=[3])
    if path == "next_tokens":
        kwargs = dict(next_tokens_device=_t([[101, -1, -1, -1]], device))
        base = max_len - 3  # start + 1 accepted
    else:
        kwargs = dict(device_seed=_t([101], device))
        base = max_len  # last + 1

    input_ids, positions, seq_lens = proposer._prepare_dflash_inputs(
        chunk, **kwargs)

    assert input_ids.shape == (8, )
    assert input_ids[0] == 101
    assert torch.equal(input_ids[1:], torch.full((7, ), 7, dtype=torch.int32,
                                                 device=device))
    expected = [min(base + i, max_len - 1) for i in range(4)] + [0, 1, 2, 3]
    assert torch.equal(positions, _t(expected, device))
    assert seq_lens[0] == min(base + 4, max_len)
    assert seq_lens[1] == 4


def test_prepare_dflash_inputs_dp_lockstep_pads_to_coordinated_bound(device):
    proposer = _input_proposer(device,
                               draft_tp=8,
                               target_tp=8,
                               max_chunk_reqs=4)
    chunk = _make_chunk(num_reqs=1,
                        start_index=0,
                        device=device,
                        query_start_loc_np=np.array([0, 3], dtype=np.int32),
                        position_ids=_t([10, 11, 12, 0], device),
                        draft_lengths=[3])

    input_ids, positions, _ = proposer._prepare_dflash_inputs(
        chunk, next_tokens_device=_t([[101, 102, 103, -1]], device))

    # Coordinated reqs = 4 -> 4 * 4 = 16 tokens padded; rows 1..3 are
    # padding.
    assert input_ids.shape == (16, )
    assert torch.equal(input_ids[4:], torch.full((12, ), 7, dtype=torch.int32,
                                                 device=device))
    assert torch.equal(positions[4:], _t([0, 1, 2, 3] * 3, device))


def test_update_draft_kv_cache_from_target_unit(device):
    proposer = _make_proposer(draft_tp=1)
    proposer._tpu_precompute_and_update_kv_cache = mock.MagicMock()
    proposer.runner = SimpleNamespace(device=device, mesh=None)

    draft_model = mock.MagicMock()
    layer_mock1 = mock.MagicMock()
    layer_mock2 = mock.MagicMock()
    # Draft layers resolve their metadata by name; an unnamed layer is an
    # error, not a cue to borrow another group's block table.
    layer_mock1.self_attn.attn.layer_name = "dummy_layer"
    layer_mock2.self_attn.attn.layer_name = "dummy_layer"
    draft_model.model.layers = [layer_mock1, layer_mock2]
    proposer.draft_model = draft_model

    aux = [torch.zeros((3, 32), dtype=torch.float32, device=device)]
    chunk = _make_chunk(num_reqs=1,
                        start_index=0,
                        device=device,
                        query_start_loc_np=np.array([0, 3]),
                        position_ids=_t([7, 8, 9, 0, 0], device),
                        aux_hidden_states=aux)

    proposer._update_draft_kv_cache_from_target(chunk)

    hidden, positions, md_tuple = (
        proposer._tpu_precompute_and_update_kv_cache.call_args.args)
    assert hidden is aux
    # Positions are cut to the aux token count, in layer order.
    assert torch.equal(positions, _t([7, 8, 9], device))
    assert md_tuple == (chunk.attn_metadata["dummy_layer"], ) * 2


def test_update_draft_kv_cache_from_target_rejects_missing_aux(device):
    """The runner stores [] (never None) when the target yielded no aux
    states; DFlash cannot fill its context KV without them."""
    proposer = _make_proposer(draft_tp=1)
    chunk = _make_chunk(num_reqs=1,
                        start_index=0,
                        device=device,
                        query_start_loc_np=np.array([0, 3]),
                        position_ids=torch.zeros(3),
                        aux_hidden_states=[])

    with pytest.raises(RuntimeError, match="aux hidden states"):
        proposer._update_draft_kv_cache_from_target(chunk)


def _propose_runner(device, num_reqs, build_metadata):
    return SimpleNamespace(
        device=device,
        input_batch=SimpleNamespace(num_reqs=num_reqs),
        _attn_metadata_builder_ctx=None,
        build_attention_metadata_for_layers=build_metadata,
        mesh=None,
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda num_tokens: None,
        num_tokens_paddings=[8, 16, 32],
    )


def test_propose_unit(device):
    proposer = _make_proposer(draft_tp=1)
    K = 3
    proposer.speculative_config.num_speculative_tokens = K
    proposer._draft_attn_layer_names = {"dummy_layer"}
    proposer.runner = _propose_runner(device, 2,
                                      mock.MagicMock(return_value={}))

    chunk = _make_chunk(num_reqs=2,
                        start_index=0,
                        device=device,
                        query_start_loc_np=np.array([0, 2, 4]),
                        position_ids=torch.zeros(4))
    proposer.draft_chunks = [chunk]

    proposer._prepare_dflash_inputs = mock.MagicMock(
        return_value=(torch.zeros(8, dtype=torch.int32, device=device),
                      torch.zeros(8, dtype=torch.int32, device=device),
                      torch.zeros(2, dtype=torch.int32, device=device)))
    proposer._update_draft_kv_cache_from_target = mock.MagicMock()
    expected_tokens = torch.tensor([[500, 501, 502], [600, 601, 602]],
                                   device=device)
    proposer._dflash_forward_and_sample = mock.MagicMock(
        return_value=(expected_tokens, torch.zeros(8, 32, device=device)))

    draft_tokens = proposer.propose(
        sampled_token_ids=[[10], [20]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=np.array([0, 0]),
        scheduler_output=None,
        next_tokens_per_chunk=[
            torch.zeros((2, 4), dtype=torch.int32, device=device)
        ],
    )

    assert draft_tokens == [[500, 501, 502], [600, 601, 602]]
    proposer._prepare_dflash_inputs.assert_called_once()
    proposer._update_draft_kv_cache_from_target.assert_called_once()
    proposer._dflash_forward_and_sample.assert_called_once()


def test_propose_slices_seed_per_chunk_and_stages_real_num_reqs(device):
    """Two chunks: each sees only its own slice of the batch-wide seed, and
    the draft metadata context carries the chunk's REAL request count (the
    staging walk copies exactly that many block-table rows; a padded count
    would copy departed requests' freed blocks into the padding rows)."""
    proposer = _make_proposer(draft_tp=1)
    K = 3
    proposer.speculative_config.num_speculative_tokens = K
    proposer._draft_attn_layer_names = {"dummy_layer"}

    staged = []

    def build_metadata(layer_names, padded_num_reqs):
        ctx = proposer.runner._attn_metadata_builder_ctx
        staged.append((ctx.num_reqs, ctx.start_index, padded_num_reqs,
                       ctx.seq_lens))
        return {}

    proposer.runner = _propose_runner(device, 3, build_metadata)
    chunks = [
        _make_chunk(num_reqs=2,
                    start_index=0,
                    device=device,
                    query_start_loc_np=np.array([0, 3, 5], dtype=np.int32),
                    position_ids=_t([10, 11, 12, 40, 41, 0, 0, 0], device)),
        _make_chunk(num_reqs=1,
                    start_index=2,
                    device=device,
                    query_start_loc_np=np.array([0, 2], dtype=np.int32),
                    position_ids=_t([70, 71, 0, 0], device)),
    ]
    proposer.draft_chunks = chunks
    proposer._update_draft_kv_cache_from_target = mock.MagicMock()
    proposer._dflash_forward_and_sample = mock.MagicMock(
        side_effect=lambda input_ids, positions, block_size: (torch.zeros(
            (input_ids.shape[0] // block_size, K),
            dtype=torch.int32,
            device=device), None))

    out = proposer.propose(None, [],
                           None,
                           None,
                           return_device=True,
                           device_seed=_t([5, 6, 9], device))

    assert out.shape == (3, K)
    # Chunk 2 anchors on its own seed (9), not on chunk 1's tokens.
    seeds = [
        call.args[0][::K + 1]
        for call in proposer._dflash_forward_and_sample.call_args_list
    ]
    assert torch.equal(seeds[0], _t([5, 6], device))
    assert torch.equal(seeds[1][:1], _t([9], device))
    assert [(n, s, p) for n, s, p, _ in staged] == [(2, 0, 2), (1, 2, 2)]
    # The staged seq_lens are the per-request draft lengths just built.
    assert torch.equal(staged[0][3], _t([17, 46], device))
    assert torch.equal(staged[1][3], _t([76, 4], device))
    # The builder context is restored after each build.
    assert proposer.runner._attn_metadata_builder_ctx is None


def test_build_draft_layer_metadata_rejects_an_unresolved_layer():
    """An unnamed draft layer must raise, not borrow another group's metadata.

    `chunk.attn_metadata` is the target's full per-layer dict, so falling back
    to an arbitrary entry can hand a draft layer a mamba group's block table —
    a different block size indexing different blocks. That corrupts the draft
    KV cache silently instead of failing.
    """
    proposer = _make_proposer(draft_tp=1)
    named = mock.MagicMock()
    named.self_attn.attn.layer_name = "dummy_layer"
    unnamed = mock.MagicMock()
    unnamed.self_attn.attn.layer_name = "not_in_the_metadata"
    draft_model = mock.MagicMock()
    draft_model.model.layers = [named, unnamed]
    proposer.draft_model = draft_model

    md = {"dummy_layer": SimpleNamespace(block_tables=None)}

    with pytest.raises(RuntimeError, match="not_in_the_metadata"):
        proposer._build_draft_layer_metadata(md)

    # The resolvable layer on its own comes back in layer order.
    draft_model.model.layers = [named]
    assert proposer._build_draft_layer_metadata(md) == (md["dummy_layer"], )


def _lockstep_proposer(draft_tp, target_tp):
    proposer = _make_proposer(draft_tp=draft_tp, target_tp=target_tp)
    proposer.draft_model = mock.MagicMock()
    proposer.runner = SimpleNamespace(
        _dp_lockstep_enabled=lambda: True,
        _dp_target_bucket=32,
        _dp_step_max_reqs=4,
        num_reqs_max_model_len=8,
        max_num_blocks_per_req=2,
        num_tokens_paddings=[8, 16, 32],
    )
    proposer._dummy_precompute_and_update_kv_cache = mock.MagicMock()
    proposer._dummy_draft_forward = mock.MagicMock()
    return proposer


def test_dp_lockstep_run_dummy_draft_replays_without_host_sync():
    """A sharded draft replays KV-updates then forwards, phase by phase,
    at the coordinated shapes -- and never waits on the host: these run
    inside propose() on the decode path."""
    proposer = _lockstep_proposer(draft_tp=8, target_tp=8)

    proposer.run_dp_dummy_draft(num_chunks=2)

    kv_calls = proposer._dummy_precompute_and_update_kv_cache.call_args_list
    fwd_calls = proposer._dummy_draft_forward.call_args_list
    assert len(kv_calls) == 2 and len(fwd_calls) == 2
    for call in kv_calls:
        assert call.kwargs["num_tokens"] == 32
        assert call.kwargs["sync"] is False
    for call in fwd_calls:
        # 4 coordinated reqs * block 5 = 20 -> bucket 32.
        assert call.kwargs["num_tokens"] == 32
        assert call.kwargs["sync"] is False


@pytest.mark.parametrize("draft_tp,target_tp,num_chunks,loaded",
                         [(8, 8, 0, True), (8, 8, 2, False)])
def test_dp_lockstep_run_dummy_draft_skips(draft_tp, target_tp, num_chunks,
                                           loaded):
    """No replay for zero padding chunks or an unloaded draft."""
    proposer = _lockstep_proposer(draft_tp=draft_tp, target_tp=target_tp)
    if not loaded:
        proposer.draft_model = None

    proposer.run_dp_dummy_draft(num_chunks=num_chunks)

    proposer._dummy_precompute_and_update_kv_cache.assert_not_called()
    proposer._dummy_draft_forward.assert_not_called()


def test_precompile_syncs_and_warms_the_runtime_kv_update_shape(device):
    """precompile blocks on each dummy (sync=True) and feeds the KV update
    the shape propose() will: one tensor per tagged target layer, at both
    chunk geometries the target uses."""
    import contextlib

    proposer = _make_proposer(draft_tp=1)
    K = 3
    proposer.speculative_config.num_speculative_tokens = K
    proposer._draft_attn_layer_names = {"dummy_layer"}
    draft_model = mock.MagicMock()
    draft_model.model.fc.weight = torch.zeros((16, 64))
    draft_model.model.embed_tokens.weight = torch.zeros((4, 16),
                                                        dtype=torch.bfloat16)
    draft_model.model.layers = [mock.MagicMock(), mock.MagicMock()]
    proposer.draft_model = draft_model
    proposer.runner = SimpleNamespace(
        device=device,
        mesh=None,
        num_tokens_paddings=[8, 16],
        num_reqs_max_model_len=2,
        num_reqs_most_model_len=4,
        most_model_len=64,
        max_num_blocks_per_req=3,
        num_blocks_per_most_len_req=1,
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )
    proposer._tpu_precompute_and_update_kv_cache = mock.MagicMock()
    proposer._dummy_draft_forward = mock.MagicMock()

    with mock.patch("vllm_torchtpu.spec_decode.dflash.synchronize_tensors"
                    ) as sync:
        proposer.precompile()

    kv_calls = proposer._tpu_precompute_and_update_kv_cache.call_args_list
    # (8, 16) tokens x (max-model-len, most-model-len) geometries.
    assert len(kv_calls) == 4
    assert sync.call_count == 4
    hidden, positions, md_tuple = kv_calls[0].args
    # Two tagged target layers -> two aux tensors of fc_in / 2 = 32.
    assert [t.shape for t in hidden] == [(8, 32), (8, 32)]
    assert all(t.dtype == torch.bfloat16 for t in hidden)
    assert positions.shape == (8, )
    assert len(md_tuple) == 2
    tables = [call.args[2][0].block_tables.shape[0] for call in kv_calls]
    # num_reqs * num_blocks: 2*3 (max) then 4*1 (most), per token bucket.
    assert tables == [6, 4, 6, 4]
    # Draft forwards warm at both geometries and never below one block.
    fwd_kwargs = [c.kwargs for c in proposer._dummy_draft_forward.call_args_list]
    assert [(c["num_tokens"], c["use_max_model_len"], c["sync"])
            for c in fwd_kwargs] == [(8, True, True), (8, False, True),
                                     (16, True, True), (16, False, True)]


def test_precompile_skips_buckets_below_one_block(device):
    import contextlib

    proposer = _make_proposer(draft_tp=1)
    proposer.speculative_config.num_speculative_tokens = 15  # block 16
    proposer._draft_attn_layer_names = {"dummy_layer"}
    proposer.draft_model = mock.MagicMock()
    proposer.runner = SimpleNamespace(
        device=device,
        mesh=None,
        num_tokens_paddings=[8, 16, 32],
        num_reqs_max_model_len=1,
        num_reqs_most_model_len=None,
        most_model_len=None,
        max_num_blocks_per_req=1,
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )
    proposer._dummy_precompute_and_update_kv_cache = mock.MagicMock()
    proposer._dummy_draft_forward = mock.MagicMock()

    proposer.precompile()

    assert [c.kwargs["num_tokens"]
            for c in proposer._dummy_draft_forward.call_args_list] == [16]


def test_dummy_forward_inputs_are_cached_per_bucket(device):
    proposer = _make_proposer(draft_tp=1)
    proposer.speculative_config.num_speculative_tokens = 3
    proposer.runner = SimpleNamespace(device=device)

    first = proposer._dummy_forward_inputs(16)
    again = proposer._dummy_forward_inputs(16)
    other = proposer._dummy_forward_inputs(8)

    assert first is again and first is not other
    input_ids, positions, query_start_loc, seq_lens, dist, num_reqs = first
    assert num_reqs == 4
    assert input_ids.shape == positions.shape == (16, )
    assert torch.equal(query_start_loc, _t([0, 4, 8, 12, 16], device))
    assert torch.equal(seq_lens, _t([4, 4, 4, 4], device))
    assert torch.equal(dist, _t([0, 0, 4], device))


def test_missing_target_layer_ids_is_rejected_at_construction():
    with pytest.raises(ValueError, match="target_layer_ids"):
        _make_proposer(draft_tp=1, target_layer_ids=[])
