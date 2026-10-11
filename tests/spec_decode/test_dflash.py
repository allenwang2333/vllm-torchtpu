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


@pytest.fixture
def device(monkeypatch):
    try:
        from vllm_torchtpu.tpu_info import get_num_chips

        if get_num_chips() == 0:
            raise RuntimeError("No attached TPU chips")
        import torch_tpu  # noqa: F401

        return torch.device("tpu")
    except Exception:
        monkeypatch.setattr(
            DFlashProposer,
            "_tpu_build_dflash_inputs_next_tokens",
            torch._dynamo.disable(DFlashProposer._tpu_build_dflash_inputs_next_tokens),
        )
        monkeypatch.setattr(
            DFlashProposer,
            "_tpu_build_dflash_inputs_seed",
            torch._dynamo.disable(DFlashProposer._tpu_build_dflash_inputs_seed),
        )
        return torch.device("cpu")


def _make_proposer(draft_tp: int | None = 1, target_tp: int = 1) -> DFlashProposer:
    hf_config_mock = mock.MagicMock()
    hf_config_mock.to_dict.return_value = {
        "dflash_config": {
            "mask_token_id": 0,
            "target_layer_ids": [1, 2],
        }
    }

    speculative_config = SimpleNamespace(
        num_speculative_tokens=4,
        draft_tensor_parallel_size=draft_tp,
        draft_model_config=SimpleNamespace(hf_config=hf_config_mock),
    )
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
        logits_processor=SimpleNamespace(_gather_logits=lambda logits: logits.clone()),
        has_own_embed_tokens=False,
        has_own_lm_head=False,
    )
    return draft, target


def _fake_gather_sharded_weight(module):
    weight = module.weight.detach().clone()
    return weight, weight.shape[0], weight.shape[1], weight.device


@pytest.mark.parametrize("draft_tp,target_tp", [(1, 1), (1, 8), (8, 8)])
def test_load_model_vocab_weights_follow_tp_layout(draft_tp, target_tp, monkeypatch):
    proposer = _make_proposer(draft_tp=draft_tp, target_tp=target_tp)
    draft, target = _make_vocab_models()

    def load_draft_model():
        proposer.draft_model = draft

    proposer._load_draft_model = load_draft_model
    monkeypatch.setattr(
        spec_decode_utils, "gather_sharded_weight", _fake_gather_sharded_weight
    )

    with mock.patch(
        "vllm_torchtpu.spec_decode.dflash.get_layers_from_vllm_config", return_value={}
    ):
        proposer.load_model(target)

    if draft_tp == target_tp:
        assert draft.model.embed_tokens is target.model.embed_tokens
        assert draft.lm_head is target.lm_head
    else:
        assert draft.model.embed_tokens is not target.model.embed_tokens
        assert torch.equal(
            draft.model.embed_tokens.weight, target.model.embed_tokens.weight
        )
        assert draft.lm_head is not target.lm_head
        assert torch.equal(draft.lm_head.weight, target.lm_head.weight)


def test_propose_empty_batch():
    proposer = _make_proposer(draft_tp=1)
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=0))
    assert proposer.propose([], [], None, None) == []


def test_precompile_skips_sub_block_draft_forward():
    proposer = _make_proposer(draft_tp=8, target_tp=8)
    proposer.speculative_config.num_speculative_tokens = 7
    proposer.draft_model = SimpleNamespace(
        model=SimpleNamespace(
            context_proj=SimpleNamespace(weight=torch.empty((1, 32))),
            embed_tokens=SimpleNamespace(weight=torch.empty((1, 1))),
        )
    )
    proposer.runner = SimpleNamespace(
        num_reqs_max_model_len=8,
        num_reqs_most_model_len=None,
        most_model_len=None,
        max_num_blocks_per_req=4,
        num_tokens_paddings=[1, 8, 16, 512],
        _precompile_timed=mock.MagicMock(
            return_value=mock.MagicMock(
                __enter__=mock.MagicMock(return_value=None),
                __exit__=mock.MagicMock(return_value=False),
            )
        ),
    )
    proposer._dummy_precompute_and_update_kv_cache = mock.MagicMock()
    proposer._dummy_draft_forward = mock.MagicMock()

    proposer.precompile()

    assert [
        call.kwargs["num_tokens"]
        for call in proposer._dummy_precompute_and_update_kv_cache.call_args_list
    ] == [1, 8, 16, 512]
    # 1 is skipped (sub-block); 512 stays: _get_padded_len rounds the
    # 8-req * 8-token upper bound (64) up to the next runner bucket.
    assert [
        call.kwargs["num_tokens"]
        for call in proposer._dummy_draft_forward.call_args_list
    ] == [8, 16, 512]


def test_draft_buckets_fit_dense_draft_blocks():
    # DSpark dense layout: K=5 draft rows per request vs K+1=6 verify rows.
    proposer = _make_proposer(draft_tp=8, target_tp=8)
    proposer.speculative_config.num_speculative_tokens = 5
    proposer._query_block_size = lambda K: K
    proposer.draft_model = SimpleNamespace(
        model=SimpleNamespace(
            context_proj=SimpleNamespace(weight=torch.empty((1, 32))),
            embed_tokens=SimpleNamespace(weight=torch.empty((1, 1))),
        )
    )
    paddings = [4, 8, 16, 24, 32, 48, 64, 96, 144, 192, 288, 384, 576, 2048]
    proposer.runner = SimpleNamespace(
        num_reqs_max_model_len=48,
        num_reqs_most_model_len=None,
        most_model_len=None,
        max_num_blocks_per_req=4,
        num_tokens_paddings=paddings,
        _precompile_timed=mock.MagicMock(
            return_value=mock.MagicMock(
                __enter__=mock.MagicMock(return_value=None),
                __exit__=mock.MagicMock(return_value=False),
            )
        ),
    )
    # Verify buckets 48..576 carry 8..96 requests -> 40..480 draft rows.
    assert proposer._draft_tokens_paddings() == sorted(
        set(paddings) | {40, 80, 120, 160, 240, 320, 480}
    )
    assert proposer._draft_tokens_paddings() is proposer._draft_tokens_paddings()
    assert proposer._get_padded_len(48 * 5) == 240
    assert proposer._get_padded_len(47 * 5) == 240
    assert proposer._get_padded_len(49 * 5) == 288
    assert proposer._get_padded_len(4096) == 4096

    proposer._dummy_precompute_and_update_kv_cache = mock.MagicMock()
    proposer._dummy_draft_forward = mock.MagicMock()
    proposer.precompile()
    assert [
        call.kwargs["num_tokens"]
        for call in proposer._dummy_precompute_and_update_kv_cache.call_args_list
    ] == paddings
    assert [
        call.kwargs["num_tokens"]
        for call in proposer._dummy_draft_forward.call_args_list
    ] == [8, 16, 24, 32, 40, 48, 64, 80, 96, 120, 144, 160, 192, 240]

    # A changed runner bucket list (e.g. pipeline chunk buckets) re-derives.
    proposer.runner.num_tokens_paddings = [8, 576]
    assert proposer._draft_tokens_paddings() == [8, 480, 576]


@pytest.mark.parametrize("num_tokens", [24, 240, 288])
def test_dummy_draft_forward_uses_real_block_layout(num_tokens):
    proposer = _make_proposer(draft_tp=8, target_tp=8)
    proposer.speculative_config.num_speculative_tokens = 5
    proposer._query_block_size = lambda K: K
    proposer.runner = SimpleNamespace(
        device=torch.device("cpu"),
        mesh=None,
        _dp_num_tokens_across_dp=lambda n: None,
    )
    proposer._build_attn_metadata_for_draft = mock.MagicMock(return_value={})
    proposer._dflash_forward_and_sample = mock.MagicMock(
        return_value=(torch.zeros(1), None)
    )
    with (
        mock.patch("vllm_torchtpu.spec_decode.dflash.set_forward_context"),
        mock.patch("vllm_torchtpu.spec_decode.dflash.set_vllm_model_wrapper_context"),
        mock.patch("vllm_torchtpu.spec_decode.dflash.synchronize_tensors"),
    ):
        proposer._dummy_draft_forward(
            num_tokens=num_tokens, use_max_model_len=True, sync=False
        )

    kwargs = proposer._build_attn_metadata_for_draft.call_args.kwargs
    num_reqs = num_tokens // 5
    assert kwargs["num_reqs"] == kwargs["padded_num_reqs"] == num_reqs
    assert kwargs["query_start_loc"].tolist() == [5 * r for r in range(num_reqs + 1)]
    assert kwargs["request_distribution"].tolist() == [0, 0, num_reqs]
    assert kwargs["seq_lens"].tolist() == [5] * num_reqs
    # Same static tensors as the real draft forward's metadata.
    assert kwargs["query_start_loc"] is proposer._get_static_attn_tensors(num_reqs)[0]
    input_ids, positions, block = proposer._dflash_forward_and_sample.call_args.args
    assert input_ids.shape == positions.shape == (num_tokens,)
    assert block == 5


@pytest.mark.parametrize("has_upstream_hook", [False, True])
def test_tpu_precompute_context_kv(device, has_upstream_hook):
    # Setup dimension sizes
    L = 2  # layers
    nkv = 2  # KV heads
    hd = 8  # head dim
    target_hidden_size = 32
    num_ctx = 3  # token count

    proposer = _make_proposer(draft_tp=1)

    # Create random hidden states and positions
    hidden_states = torch.randn(
        (num_ctx, target_hidden_size), dtype=torch.float32, device=device
    )
    positions = torch.tensor([10, 11, 12], dtype=torch.int32, device=device)

    # Create draft model mock
    self_model = mock.MagicMock()

    # Initialize fused weight matrix
    fused_weight = torch.randn(
        (L * 2 * nkv * hd, target_hidden_size), dtype=torch.float32, device=device
    )
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
        layer.self_attn.num_heads = nkv
        del layer.self_attn.k_norm

        # Extract individual weights for assertion comparison
        k_w = weight_structured[i, 0].reshape(nkv * hd, target_hidden_size)
        v_w = weight_structured[i, 1].reshape(nkv * hd, target_hidden_size)

        layer.self_attn.k_proj = torch.nn.Linear(
            target_hidden_size, nkv * hd, bias=False
        ).to(device)
        layer.self_attn.k_proj.weight.data.copy_(k_w)

        layer.self_attn.v_proj = torch.nn.Linear(
            target_hidden_size, nkv * hd, bias=False
        ).to(device)
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

    proposer.draft_model = SimpleNamespace(model=self_model)
    upstream_hook = mock.Mock(
        side_effect=AssertionError("The upstream context-KV hook must not run on TPU")
    )
    if has_upstream_hook:
        proposer.draft_model.precompute_and_store_context_kv = upstream_hook

    # Initialize mock forward implementations on the layers to record outputs
    for layer in layers:
        layer.self_attn.attn.impl = mock.MagicMock()
        layer.self_attn.attn.impl.forward = mock.MagicMock(
            side_effect=lambda **kwargs: kwargs["key"]
        )

    # 1. Run compiled/batched math eagerly inside test suite
    fn_eager = torch._dynamo.disable(DFlashProposer._tpu_precompute_and_update_kv_cache)
    dummy_md = mock.MagicMock()
    fn_eager(proposer, hidden_states, positions, tuple([dummy_md] * L))
    upstream_hook.assert_not_called()

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
        k_proj_out = layer.self_attn.k_proj(hidden_states).view(num_ctx, nkv, hd)
        v_proj_out = layer.self_attn.v_proj(hidden_states).view(num_ctx, nkv, hd)

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


def test_tpu_precompute_context_kv_uses_explicit_tpu_hook(device):
    proposer = _make_proposer(draft_tp=1)
    hidden_states = torch.randn(3, 32, device=device)
    positions = torch.arange(3, device=device)
    metadata = (SimpleNamespace(),)
    expected = torch.randn(3, 16, device=device)
    tpu_hook = mock.Mock(return_value=expected)
    upstream_hook = mock.Mock(
        side_effect=AssertionError("The upstream context-KV hook must not run on TPU")
    )
    proposer.draft_model = SimpleNamespace(
        model=SimpleNamespace(),
        tpu_precompute_and_store_context_kv=tpu_hook,
        precompute_and_store_context_kv=upstream_hook,
    )

    fn_eager = torch._dynamo.disable(DFlashProposer._tpu_precompute_and_update_kv_cache)
    result = fn_eager(proposer, hidden_states, positions, metadata)

    assert result is expected
    tpu_hook.assert_called_once_with(hidden_states, positions, metadata)
    upstream_hook.assert_not_called()


def _make_chunk(
    num_reqs,
    start_index,
    device,
    query_start_loc_np,
    position_ids,
    aux_hidden_states=None,
    seq_lens=None,
):
    query_start_loc = (
        torch.from_numpy(query_start_loc_np).to(device)
        if query_start_loc_np is not None
        else None
    )

    return DraftChunkInputs(
        input_ids=None,
        position_ids=position_ids,
        query_start_loc_np=query_start_loc_np,
        attn_ctx=SimpleNamespace(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            use_max_model_len=True,
            position_ids_override=None,
        ),
        start_index=start_index,
        num_reqs=num_reqs,
        aux_hidden_states=aux_hidden_states,
        attn_metadata={
            "dummy_layer": SimpleNamespace(
                block_tables=torch.ones(10, dtype=torch.int32, device=device),
                seq_lens=seq_lens,
            )
        },
    )


def test_prepare_dflash_inputs(device):
    proposer = _make_proposer(draft_tp=1)
    K = 3
    proposer.speculative_config.num_speculative_tokens = K
    proposer.runner = SimpleNamespace(
        device=device,
        num_tokens_paddings=[8, 16, 32],
        max_num_reqs=8,
        _dp_lockstep_enabled=lambda: False,
    )
    chunk = _make_chunk(
        num_reqs=2,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 3, 5], dtype=np.int32),
        position_ids=torch.tensor(
            [10, 11, 12, 40, 41, 0, 0, 0], dtype=torch.int32, device=device
        ),
    )

    next_tokens_device = torch.tensor(
        [[101, 102, 103, -1], [201, 202, -1, -1]], dtype=torch.int32, device=device
    )

    input_ids, position_ids, seq_lens = proposer._prepare_dflash_inputs(
        chunk, next_tokens_device=next_tokens_device
    )

    assert input_ids.shape == (8,)
    assert position_ids.shape == (8,)
    assert torch.equal(
        input_ids,
        torch.tensor([103, 0, 0, 0, 202, 0, 0, 0], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        position_ids,
        torch.tensor(
            [13, 14, 15, 16, 42, 43, 44, 45], dtype=torch.int32, device=device
        ),
    )


def test_update_draft_kv_cache_from_target_unit(device):
    proposer = _make_proposer(draft_tp=1)
    proposer._tpu_precompute_and_update_kv_cache = mock.MagicMock(
        return_value=(None, [])
    )

    slot_mapping = torch.zeros(8, dtype=torch.int32, device=device)
    proposer._compute_slot_mapping = mock.MagicMock(return_value=slot_mapping)
    proposer._draft_attn_layer_names = {"dummy_layer"}
    proposer.runner = SimpleNamespace(
        device=device,
        kv_caches=[torch.zeros((1, 8, 2, 2, 16), dtype=torch.float32, device=device)],
        block_size=8,
        num_tokens_paddings=[8, 16, 32],
        mesh=None,
        _dp_lockstep_enabled=lambda: False,
    )

    draft_model = mock.MagicMock()
    layer_mock1 = mock.MagicMock()
    layer_mock1.self_attn.attn.impl = mock.MagicMock()
    layer_mock2 = mock.MagicMock()
    layer_mock2.self_attn.attn.impl = mock.MagicMock()
    # Draft layers resolve their metadata by name; an unnamed layer is an
    # error, not a cue to borrow another group's block table.
    layer_mock1.self_attn.attn.layer_name = "dummy_layer"
    layer_mock2.self_attn.attn.layer_name = "dummy_layer"
    draft_model.model.layers = [layer_mock1, layer_mock2]
    proposer.draft_model = draft_model
    proposer.num_target_layers = 2

    chunk = _make_chunk(
        num_reqs=1,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 3]),
        position_ids=torch.zeros(3),
        aux_hidden_states=torch.zeros((3, 32), dtype=torch.float32, device=device),
    )

    proposer._update_draft_kv_cache_from_target(chunk)

    proposer._tpu_precompute_and_update_kv_cache.assert_called_once()


def test_propose_unit(device):
    proposer = _make_proposer(draft_tp=1)
    K = 3
    proposer.speculative_config.num_speculative_tokens = K
    proposer._draft_attn_layer_names = {"dummy_layer"}
    proposer.runner = SimpleNamespace(
        device=device,
        input_batch=SimpleNamespace(num_reqs=2),
        _attn_metadata_builder_ctx=None,
        max_num_blocks_per_req=2,
        _attn_layer_names={"dummy_layer"},
        build_attention_metadata_for_layers=mock.MagicMock(return_value={}),
        mesh=None,
        _is_async_drafter=True,
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda num_tokens: None,
        _dp_step_num_chunks=0,
    )

    chunk = _make_chunk(
        num_reqs=2,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 2, 4]),
        position_ids=torch.zeros(4),
    )
    proposer.draft_chunks = [chunk]

    proposer._prepare_dflash_inputs = mock.MagicMock(
        return_value=(
            torch.zeros(8, dtype=torch.int32, device=device),
            torch.zeros(8, dtype=torch.int32, device=device),
            torch.zeros(2, dtype=torch.int32, device=device),
        )
    )
    proposer._update_draft_kv_cache_from_target = mock.MagicMock()

    logits = torch.zeros((8, 1000), dtype=torch.float32, device=device)
    logits[1, 500] = 100.0
    logits[2, 501] = 100.0
    logits[3, 502] = 100.0
    logits[5, 600] = 100.0
    logits[6, 601] = 100.0
    logits[7, 602] = 100.0

    expected_tokens = torch.tensor([[500, 501, 502], [600, 601, 602]], device=device)
    proposer._dflash_forward_and_sample = mock.MagicMock(
        return_value=(expected_tokens, torch.zeros(8, 32, device=device))
    )

    draft_tokens = proposer.propose(
        sampled_token_ids=[[10], [20]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=np.array([0, 0]),
        scheduler_output=None,
        next_tokens_per_chunk=[torch.zeros((2, 4), dtype=torch.int32, device=device)],
    )

    assert draft_tokens == [[500, 501, 502], [600, 601, 602]]
    proposer._prepare_dflash_inputs.assert_called_once()
    proposer._update_draft_kv_cache_from_target.assert_called_once()
    proposer._dflash_forward_and_sample.assert_called_once()


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
    draft_model = SimpleNamespace(model=SimpleNamespace(layers=[named, unnamed]))
    proposer.draft_model = draft_model

    md = {"dummy_layer": SimpleNamespace(block_tables=None)}

    with pytest.raises(RuntimeError, match="not_in_the_metadata"):
        proposer._build_draft_layer_metadata(md)

    # The resolvable layer on its own comes back in layer order.
    draft_model.model.layers = [named]
    assert proposer._build_draft_layer_metadata(md) == (md["dummy_layer"],)


def test_dp_lockstep_run_dummy_draft():
    proposer = _make_proposer(draft_tp=8, target_tp=8)
    proposer.draft_model = mock.MagicMock()
    proposer.runner = SimpleNamespace(
        _dp_lockstep_enabled=lambda: True,
        _dp_target_bucket=32,
        _dp_step_max_reqs=4,
        num_reqs_max_model_len=8,
    )
    proposer._dp_dummy_kv_update = mock.MagicMock()
    proposer._dp_dummy_forward = mock.MagicMock()

    proposer.run_dp_dummy_draft(num_chunks=2)

    assert proposer._dp_dummy_kv_update.call_count == 2
    assert proposer._dp_dummy_forward.call_count == 2


def test_prepare_dflash_inputs_dp_lockstep(device):
    proposer = _make_proposer(draft_tp=8, target_tp=8)
    K = 3
    proposer.speculative_config.num_speculative_tokens = K
    proposer.runner = SimpleNamespace(
        device=device,
        num_tokens_paddings=[8, 16, 32],
        _dp_lockstep_enabled=lambda: True,
        _dp_step_max_reqs=4,
        max_num_reqs=8,
    )
    chunk = _make_chunk(
        num_reqs=1,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 3], dtype=np.int32),
        position_ids=torch.tensor([10, 11, 12, 0], dtype=torch.int32, device=device),
    )
    next_tokens_device = torch.tensor(
        [[101, 102, 103, -1]], dtype=torch.int32, device=device
    )

    input_ids, position_ids, _ = proposer._prepare_dflash_inputs(
        chunk, next_tokens_device=next_tokens_device
    )

    # Coordinated reqs = 4 -> 4 * 4 = 16 tokens padded.
    assert input_ids.shape == (16,)
    assert position_ids.shape == (16,)


def test_build_draft_attn_metadata_propagates_max_active_seq_len():
    device = torch.device("cpu")
    proposer = _make_proposer(draft_tp=1)
    proposer._draft_attn_layer_names = {"dummy_layer"}
    captured_ctxs = []

    def fake_build(layer_names, padded_num_reqs):
        captured_ctxs.append(proposer.runner._attn_metadata_builder_ctx)
        return {"dummy_layer": "ok"}

    proposer.runner = SimpleNamespace(
        device=device,
        _attn_metadata_builder_ctx=None,
        build_attention_metadata_for_layers=fake_build,
    )
    chunk = _make_chunk(
        num_reqs=2,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 2, 4], dtype=np.int32),
        position_ids=torch.zeros(4, dtype=torch.int32, device=device),
    )

    # 1. Tuple seq_lens_cpu without pre-populated max_active_seq_len
    chunk.attn_ctx.seq_lens_cpu = (100, 250, 0, 0)
    proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=torch.tensor([105, 255], device=device)
    )
    assert captured_ctxs[-1].seq_lens_cpu == (100, 250, 0, 0)
    assert captured_ctxs[-1].max_active_seq_len == 250 + proposer.block_size

    # 2. Tensor seq_lens_cpu without pre-populated max_active_seq_len
    chunk.attn_ctx.seq_lens_cpu = torch.tensor([300, 120, 0, 0], dtype=torch.int32)
    proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=torch.tensor([305, 125], device=device)
    )
    assert captured_ctxs[-1].max_active_seq_len == 300 + proposer.block_size

    # 3. Pre-populated max_active_seq_len clamped by _max_model_len
    chunk.attn_ctx.max_active_seq_len = 2046
    proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=torch.tensor([2048, 125], device=device)
    )
    assert captured_ctxs[-1].max_active_seq_len == 2048


def test_build_draft_attn_metadata_reuses_target_staged_block_tables():
    device = torch.device("cpu")
    proposer = _make_proposer(draft_tp=1)
    proposer._draft_attn_layer_names = {"draft_layer"}
    captured_ctxs = []

    def fake_build(layer_names, padded_num_reqs):
        captured_ctxs.append(proposer.runner._attn_metadata_builder_ctx)
        return {"draft_layer": "ok"}

    class _FakeBuilder:
        def __init__(self, gid: int, tnb: int):
            self.kv_cache_group_id = gid
            self.tnb = tnb

        def _block_table_geometry(self, _ctx):
            return None, self.tnb

        def _active_block_table_columns(self, ctx, tnb):
            if getattr(ctx, "max_active_seq_len", 0) > 512:
                return min(tnb, 64)
            return min(tnb, 16)

    target_group = SimpleNamespace(
        layer_names=["target_layer"],
        metadata_builders=[_FakeBuilder(0, 64)],
    )
    draft_group = SimpleNamespace(
        layer_names=["draft_layer"],
        metadata_builders=[_FakeBuilder(1, 64)],
    )
    proposer.runner = SimpleNamespace(
        device=device,
        _attn_metadata_builder_ctx=None,
        attn_groups=[[target_group, draft_group]],
        build_attention_metadata_for_layers=fake_build,
    )
    chunk = _make_chunk(
        num_reqs=2,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 2, 4], dtype=np.int32),
        position_ids=torch.zeros(4, dtype=torch.int32, device=device),
    )
    chunk.attn_ctx.max_active_seq_len = 100
    # num_tokens_padded=10, block_size=5 -> padded_num_reqs=2; tnb=64 -> numel=128
    staged_tables = {1: torch.ones(128, dtype=torch.int32)}
    chunk.attn_ctx.staged_block_tables = staged_tables

    # 1. Matching shape and active_cols -> reuses staged_tables directly
    proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=torch.tensor([105, 105], device=device)
    )
    assert captured_ctxs[-1].staged_block_tables is staged_tables

    # 2. Active column bucket grows across chunk -> draft boundary (510 -> 515 > 512)
    chunk.attn_ctx.max_active_seq_len = 510
    proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=torch.tensor([515, 105], device=device)
    )
    assert captured_ctxs[-1].staged_block_tables is None

    # 3. Padded request count mismatch (num_tokens_padded=20 -> padded_num_reqs=4)
    chunk.attn_ctx.max_active_seq_len = 100
    proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=20, seq_lens=torch.tensor([105, 105], device=device)
    )
    assert captured_ctxs[-1].staged_block_tables is None

    # 4. Missing draft group id in staged_block_tables
    chunk.attn_ctx.staged_block_tables = {0: torch.ones(128, dtype=torch.int32)}
    proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=torch.tensor([105, 105], device=device)
    )
    assert captured_ctxs[-1].staged_block_tables is None

    # 5. No matching draft group in attn_groups
    chunk.attn_ctx.staged_block_tables = staged_tables
    proposer.runner.attn_groups = [[target_group]]
    proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=torch.tensor([105, 105], device=device)
    )
    assert captured_ctxs[-1].staged_block_tables is None

    # 6. Two draft groups with identical (target_block_size, tnb, is_mamba_group)
    # geometry skip duplicate _active_block_table_columns checks (354->342 branch),
    # and multi-list attn_groups exercises the len(attn_groups) > 1 cache key path.
    b1 = _FakeBuilder(1, 64)
    b1.target_block_size = 16
    b1.is_mamba_group = False
    b2 = _FakeBuilder(2, 64)
    b2.target_block_size = 16
    b2.is_mamba_group = False
    proposer._draft_attn_layer_names = {"draft_layer_1", "draft_layer_2"}
    g1 = SimpleNamespace(layer_names=["draft_layer_1"], metadata_builders=[b1])
    g2 = SimpleNamespace(layer_names=["draft_layer_2"], metadata_builders=[b2])
    proposer.runner.attn_groups = [[g1], [g2]]
    staged_multi = {
        1: torch.ones(128, dtype=torch.int32),
        2: torch.ones(128, dtype=torch.int32),
    }
    chunk.attn_ctx.staged_block_tables = staged_multi
    proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=torch.tensor([105, 105], device=device)
    )
    assert captured_ctxs[-1].staged_block_tables is staged_multi


def test_init_and_load_model_all_branches(monkeypatch):
    # 1. draft_tp is None defaults to target_tp (lines 73-74)
    p_none = _make_proposer(draft_tp=None, target_tp=4)
    assert p_none.speculative_config.draft_tensor_parallel_size == 4
    assert not p_none._draft_replicated
    assert p_none._draft_tp_matches_target

    # 2. Unsupported draft_tp raises ValueError (line 78)
    with pytest.raises(ValueError, match="unsupported"):
        _make_proposer(draft_tp=2, target_tp=4)

    # 3. _load_draft_model with replicated=True and replicated=False (lines 232-246)
    fake_loader = SimpleNamespace(
        load_model=mock.MagicMock(return_value="loaded_draft")
    )
    monkeypatch.setattr(
        "vllm_torchtpu.spec_decode.dflash.get_model_loader", lambda _cfg: fake_loader
    )
    monkeypatch.setattr(
        "vllm_torchtpu.spec_decode.dflash.set_model_tag",
        lambda _tag: mock.MagicMock(
            __enter__=lambda s: None, __exit__=lambda s, *a: False
        ),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.spec_decode.dflash.set_vllm_model_wrapper_context",
        lambda **_kw: mock.MagicMock(
            __enter__=lambda s: None, __exit__=lambda s, *a: False
        ),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.spec_decode.dflash.set_current_vllm_config",
        lambda _cfg: mock.MagicMock(
            __enter__=lambda s: None, __exit__=lambda s, *a: False
        ),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.spec_decode.dflash._force_draft_tp1",
        lambda: mock.MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: False),
    )
    p_rep = _make_proposer(draft_tp=1, target_tp=4)
    p_rep.vllm_config.load_config = "load_cfg"
    p_rep._load_draft_model()
    assert p_rep.draft_model == "loaded_draft"

    p_sharded = _make_proposer(draft_tp=4, target_tp=4)
    p_sharded.vllm_config.load_config = "load_cfg"
    p_sharded._load_draft_model()
    assert p_sharded.draft_model == "loaded_draft"

    # 4. load_model target aux hidden states fallback and error (lines 163-168)
    p_err = _make_proposer(draft_tp=1, target_tp=1)
    with pytest.raises(
        RuntimeError, match="does not support _set_aux_hidden_state_layers"
    ):
        p_err.load_model(SimpleNamespace())

    # 5. load_model with target.model._set_aux_hidden_state_layers,
    # draft.model.get_draft_attn_causal, and PallasAttentionBackendImpl layers (210-229)
    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl

    pallas_impl = PallasAttentionBackendImpl.__new__(PallasAttentionBackendImpl)
    pallas_impl.initialize_kernel = mock.MagicMock()
    layer_no_attn = SimpleNamespace(self_attn=None)
    layer_other_impl = SimpleNamespace(
        self_attn=SimpleNamespace(attn=SimpleNamespace(impl=SimpleNamespace()))
    )
    layer_pallas = SimpleNamespace(
        self_attn=SimpleNamespace(attn=SimpleNamespace(impl=pallas_impl))
    )

    draft, target = _make_vocab_models()
    del target.set_aux_hidden_state_layers
    aux_called = []
    target.model._set_aux_hidden_state_layers = lambda layers: aux_called.append(layers)
    del draft.get_draft_attn_causal
    draft.model.get_draft_attn_causal = lambda: [False, True, False]
    draft.model.layers = [layer_no_attn, layer_other_impl, layer_pallas]

    p_load = _make_proposer(draft_tp=1, target_tp=1)
    p_load._load_draft_model = lambda: setattr(p_load, "draft_model", draft)
    with mock.patch(
        "vllm_torchtpu.spec_decode.dflash.get_layers_from_vllm_config", return_value={}
    ):
        p_load.load_model(target)
    assert aux_called == [(2, 3)]
    assert pallas_impl.layer_idx == 2
    assert pallas_impl.use_causal_mask is False
    pallas_impl.initialize_kernel.assert_called_once_with(layer_pallas.self_attn.attn)

    # 6. load_model without get_draft_attn_causal raises RuntimeError (line 213)
    del draft.model.get_draft_attn_causal
    with (
        mock.patch(
            "vllm_torchtpu.spec_decode.dflash.get_layers_from_vllm_config",
            return_value={},
        ),
        pytest.raises(RuntimeError, match="get_draft_attn_causal"),
    ):
        p_load.load_model(target)


def test_prepare_inputs_and_propose_remaining_branches(device, monkeypatch):
    proposer = _make_proposer(draft_tp=1)
    K = 3
    proposer.speculative_config.num_speculative_tokens = K
    proposer._draft_attn_layer_names = {"dummy_layer"}
    proposer.runner = SimpleNamespace(
        device=device,
        num_tokens_paddings=[8, 16, 32, 64, 128],
        max_num_reqs=64,
        input_batch=SimpleNamespace(num_reqs=2),
        _attn_metadata_builder_ctx=None,
        build_attention_metadata_for_layers=lambda names, n: {"dummy_layer": "md"},
        mesh=None,
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda n: None,
        _dp_step_num_chunks=0,
    )

    # 1. _zero_draft_lengths cache hit and num_reqs > MIN_NUM_SEQS (lines 542-555)
    z1 = proposer._zero_draft_lengths(20)
    z2 = proposer._zero_draft_lengths(20)
    assert z1 is z2 and z1.shape == (32,)

    # 2. _get_draft_kv_cache_layer_names with get_draft_kv_cache_layer_names method
    # (cache miss + cache hit, lines 559-566) and layer_name fallback cache hit (572)
    draft_with_hook = SimpleNamespace(
        get_draft_kv_cache_layer_names=lambda: ["dummy_layer"]
    )
    proposer.draft_model = draft_with_hook
    assert proposer._get_draft_kv_cache_layer_names() == ("dummy_layer",)
    assert proposer._get_draft_kv_cache_layer_names() == ("dummy_layer",)

    layer_named = SimpleNamespace(
        self_attn=SimpleNamespace(attn=SimpleNamespace(layer_name="dummy_layer"))
    )
    proposer.draft_model = SimpleNamespace(model=SimpleNamespace(layers=[layer_named]))
    assert proposer._get_draft_kv_cache_layer_names() == ("dummy_layer",)
    assert proposer._get_draft_kv_cache_layer_names() == ("dummy_layer",)

    # 3. _prepare_dflash_inputs with 2D position_ids, explicit draft_lengths,
    # device_seed path (_tpu_build_dflash_inputs_seed lines 521-530, 1366-1379),
    # and missing inputs RuntimeError (line 532)
    chunk = _make_chunk(
        num_reqs=2,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 2, 4], dtype=np.int32),
        position_ids=torch.tensor(
            [[10, 11, 20, 21, 0, 0]], dtype=torch.int32, device=device
        ),
    )
    chunk.draft_lengths = torch.tensor([1, 1], dtype=torch.int32, device=device)
    next_tokens = torch.tensor([[101, -1], [201, -1]], dtype=torch.int32, device=device)
    ids_nt, pos_nt, _ = proposer._prepare_dflash_inputs(
        chunk, next_tokens_device=next_tokens
    )
    assert ids_nt.shape == (8,) and pos_nt.shape == (8,)

    seed = torch.tensor([77, 88], dtype=torch.int32, device=device)
    ids_sd, pos_sd, sl_sd = proposer._prepare_dflash_inputs(chunk, device_seed=seed)
    assert ids_sd.tolist() == [77, 0, 0, 0, 88, 0, 0, 0]
    assert pos_sd.tolist() == [12, 13, 14, 15, 22, 23, 24, 25]
    assert sl_sd.tolist() == [16, 26]

    with pytest.raises(RuntimeError, match="without next_tokens_device or device_seed"):
        proposer._prepare_dflash_inputs(chunk)

    # 4. _update_draft_kv_cache_from_target with aux_hidden_states=None (line 614),
    # tuple aux_hidden_states, 2D position_ids (626), positions.shape[0] > num_tokens
    proposer._tpu_precompute_and_update_kv_cache = mock.MagicMock()
    chunk_none = _make_chunk(
        num_reqs=1,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
        position_ids=torch.zeros((1, 4), dtype=torch.int32, device=device),
        aux_hidden_states=None,
    )
    proposer._update_draft_kv_cache_from_target(chunk_none)
    proposer._tpu_precompute_and_update_kv_cache.assert_not_called()

    chunk_tuple = _make_chunk(
        num_reqs=1,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
        position_ids=torch.zeros((1, 4), dtype=torch.int32, device=device),
        aux_hidden_states=(torch.zeros((2, 8), device=device),),
    )
    proposer._update_draft_kv_cache_from_target(chunk_tuple)
    proposer._tpu_precompute_and_update_kv_cache.assert_called_once()
    assert proposer._tpu_precompute_and_update_kv_cache.call_args.args[1].shape == (2,)

    # 5. propose error when draft_chunks is empty (line 665)
    proposer.draft_chunks = []
    with pytest.raises(RuntimeError, match="no draft chunks captured"):
        proposer.propose([], [], None, None)

    # 6. propose with extra_chunks > 0, device_seed slicing, and return_device=True
    # across single-chunk (exact & padded) and multi-chunk (exact & padded)
    proposer._dp_lockstep_sharded = lambda: True
    proposer.runner._dp_step_num_chunks = 2
    proposer._dp_dummy_kv_update = mock.MagicMock()
    proposer._dp_dummy_forward = mock.MagicMock()
    proposer._prepare_dflash_inputs = mock.MagicMock(
        return_value=(
            torch.zeros(8, dtype=torch.int32, device=device),
            torch.zeros(8, dtype=torch.int32, device=device),
            torch.zeros(2, dtype=torch.int32, device=device),
        )
    )
    # Single chunk with exact shape [2, 3] and device_seed matching chunk
    proposer.draft_chunks = [chunk]
    proposer._dflash_forward_and_sample = mock.MagicMock(
        return_value=(
            torch.ones((2, 3), dtype=torch.int32, device=device),
            torch.zeros((8, 8), device=device),
        )
    )
    out_dev_exact = proposer.propose(
        [], [], None, None, return_device=True, device_seed=seed
    )
    assert out_dev_exact.shape == (2, 3)
    proposer._dp_dummy_kv_update.assert_called_once()
    proposer._dp_dummy_forward.assert_called_once()

    # Single chunk with padded shape [4, 3] -> slices [:2]
    proposer._dflash_forward_and_sample = mock.MagicMock(
        return_value=(
            torch.ones((4, 3), dtype=torch.int32, device=device),
            torch.zeros((8, 8), device=device),
        )
    )
    out_dev_padded = proposer.propose(
        [], [], None, None, return_device=True, device_seed=seed
    )
    assert out_dev_padded.shape == (2, 3)

    # Multi-chunk with one exact [2, 3] and one padded [4, 3] and seed slice
    chunk2 = _make_chunk(
        num_reqs=2,
        start_index=2,
        device=device,
        query_start_loc_np=np.array([0, 2, 4], dtype=np.int32),
        position_ids=torch.zeros(4, dtype=torch.int32, device=device),
    )
    proposer.draft_chunks = [chunk, chunk2]
    proposer._dflash_forward_and_sample = mock.MagicMock(
        side_effect=[
            (torch.ones((2, 3), dtype=torch.int32, device=device), None),
            (torch.ones((4, 3), dtype=torch.int32, device=device), None),
        ]
    )
    seed4 = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
    out_multi = proposer.propose(
        [], [], None, None, return_device=True, device_seed=seed4
    )
    assert out_multi.shape == (4, 3)


def test_precompile_dummies_and_dp_lockstep_helpers(device, monkeypatch):
    proposer = _make_proposer(draft_tp=1, target_tp=4)
    proposer.speculative_config.num_speculative_tokens = 3

    # 1. precompile with draft_model is None returns early (line 761)
    proposer.draft_model = None
    proposer.precompile()

    # 2. _dp_lockstep_sharded when draft_model is None (1008) vs MoE model (1006-1012)
    proposer.runner = SimpleNamespace(
        device=device,
        mesh=None,
        num_reqs_max_model_len=4,
        num_reqs_most_model_len=2,
        most_model_len=1024,
        max_num_blocks_per_req=4,
        num_blocks_per_most_len_req=2,
        num_tokens_paddings=[4, 8],
        _dp_lockstep_enabled=lambda: True,
        _dp_target_bucket=8,
        _dp_step_max_reqs=2,
        _dp_num_tokens_across_dp=lambda n: None,
        _precompile_timed=lambda _label: contextlib.nullcontext(),
    )
    import contextlib

    assert proposer._dp_lockstep_sharded() is False
    # run_dp_dummy_draft returns early when num_chunks <= 0 or draft_model is None
    proposer.run_dp_dummy_draft(0)
    proposer.run_dp_dummy_draft(1)

    layer = SimpleNamespace()
    proposer.draft_model = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=SimpleNamespace(
                weight=torch.zeros((2, 2), dtype=torch.float32)
            ),
            layers=[layer],
        )
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.interfaces.is_mixture_of_experts",
        lambda _m: False,
    )
    # First call populates _draft_has_moe_cache=False; second call hits cache
    assert proposer._dp_lockstep_sharded() is False
    assert proposer._dp_lockstep_sharded() is False
    # run_dp_dummy_draft returns early when not _dp_lockstep_sharded() (line 1021)
    proposer.run_dp_dummy_draft(1)

    # 3. _dummy_kv_update_inputs cache miss & hit + _dummy_precompute (819-888)
    proposer.vllm_config.model_config.get_hidden_size = lambda: 8
    proposer._tpu_precompute_and_update_kv_cache = mock.MagicMock(
        return_value=torch.zeros(1)
    )
    monkeypatch.setattr(
        "vllm_torchtpu.spec_decode.dflash.synchronize_tensors",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.spec_decode.dflash.set_vllm_model_wrapper_context",
        lambda **_kw: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.spec_decode.dflash.set_forward_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    proposer._build_attn_metadata_for_draft = mock.MagicMock(return_value={})
    proposer._dflash_forward_and_sample = mock.MagicMock(
        return_value=(torch.zeros(1), None)
    )

    # Precompile with most_model_len != None (lines 787-789, 806)
    proposer.precompile()
    # Second call hits _dummy_kv_input_cache (822) and _dummy_draft_forward_cache (897)
    proposer.precompile()

    # 4. _dummy_draft_forward raises ValueError when num_tokens < block_size (line 921)
    with pytest.raises(ValueError, match="smaller than its query block size"):
        proposer._dummy_draft_forward(num_tokens=2, use_max_model_len=True, sync=False)

    # 5. _dp_dummy_kv_update and _dp_dummy_forward direct execution (lines 979-994)
    proposer._dp_dummy_kv_update()
    proposer._dp_dummy_forward()


def test_tpu_precompute_and_forward_sample_remaining_branches(device):
    proposer = _make_proposer(draft_tp=1)
    fn_precompute = torch._dynamo.disable(
        DFlashProposer._tpu_precompute_and_update_kv_cache
    )
    fn_sample = torch._dynamo.disable(DFlashProposer._dflash_forward_and_sample)

    # 1. owns_context_kv class attr on draft_model + list hidden_states (1048, 1064)
    class _OwnsKvDraft:
        owns_context_kv = True

        def __init__(self):
            self.model = SimpleNamespace()
            self.called_with = None

        def precompute_and_store_context_kv(self, hidden, pos, md):
            self.called_with = (hidden, pos, md)
            return "owned_kv"

    proposer.draft_model = _OwnsKvDraft()
    h1 = torch.ones((2, 4), device=device)
    h2 = torch.ones((2, 4), device=device)
    pos = torch.tensor([0, 1], dtype=torch.int32, device=device)
    # Single-element list and two-element list in hidden_states
    assert fn_precompute(proposer, [h1], pos, ()) == "owned_kv"
    assert proposer.draft_model.called_with[0].shape == (2, 4)
    assert fn_precompute(proposer, [h1, h2], pos, ()) == "owned_kv"
    assert proposer.draft_model.called_with[0].shape == (2, 8)

    # 2. combine_hidden_states, fc (returning tuple and tensor), hidden_norm, k_norm
    # (lines 1073, 1075-1077, 1080, 1101-1105)
    L, nkv, hd = 1, 2, 4
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            head_dim=hd,
            num_kv_heads=nkv,
            num_heads=nkv,
            k_norm=SimpleNamespace(
                weight=torch.ones(hd, device=device),
                variance_epsilon=1e-6,
            ),
            rotary_emb=lambda p, q, k: (q, k),
            attn=SimpleNamespace(
                kv_cache=None,
                impl=SimpleNamespace(forward=lambda **_kw: None),
            ),
        )
    )
    self_model = SimpleNamespace(
        layers=[layer],
        _fused_kv_weight=torch.ones((L * 2 * nkv * hd, 4), device=device),
        _fused_kv_bias=None,
        fc=lambda x: (x, "aux"),
        hidden_norm=lambda x: x * 2.0,
    )
    # 2a. With combine_hidden_states on draft_model
    proposer.draft_model = SimpleNamespace(
        model=self_model,
        combine_hidden_states=lambda x: x,
    )
    fn_precompute(proposer, h1, pos, (SimpleNamespace(),))

    # 2b. Without combine_hidden_states -> uses self_model.fc (tuple then tensor)
    proposer.draft_model = SimpleNamespace(model=self_model)
    fn_precompute(proposer, h1, pos, (SimpleNamespace(),))
    self_model.fc = lambda x: x
    fn_precompute(proposer, h1, pos, (SimpleNamespace(),))

    # 3. _dflash_forward_and_sample (lines 1157-1179)
    proposer.draft_model = SimpleNamespace(
        model=lambda input_ids, positions: torch.ones(
            (input_ids.shape[0], 8), dtype=torch.float32, device=device
        ),
        compute_logits=lambda draft_hidden: torch.arange(
            draft_hidden.shape[0] * 10, dtype=torch.float32, device=device
        ).view(draft_hidden.shape[0], 10),
    )
    tokens, hidden_out = fn_sample(
        proposer,
        torch.zeros(8, dtype=torch.int32, device=device),
        torch.zeros(8, dtype=torch.int32, device=device),
        block_size=4,
    )
    assert tokens.shape == (2, 3)
    assert hidden_out.shape == (8, 8)


def test_prepare_dflash_inputs_no_num_reqs_specialization_and_slice_bounds(device):
    proposer = _make_proposer(draft_tp=8, target_tp=8)
    proposer.speculative_config.num_speculative_tokens = 5
    proposer._query_block_size = lambda K: K
    proposer.runner = SimpleNamespace(
        device=device,
        num_tokens_paddings=[384, 768, 1152, 4096],
        max_num_reqs=192,
        _dp_lockstep_enabled=lambda: False,
    )
    assert proposer._draft_tokens_paddings() == [320, 384, 640, 768, 960, 1152, 4096]

    # Exercise device_seed < padded_num_reqs (F.pad) and >= padded_num_reqs (slice)
    qsl_pad8 = np.array([0, 2, 4, 4, 4, 4, 4, 4, 4], dtype=np.int32)
    pos_buf = torch.arange(100, 100 + 384, dtype=torch.int32, device=device)
    chunk_seed_small = _make_chunk(
        num_reqs=2,
        start_index=0,
        device=device,
        query_start_loc_np=qsl_pad8,
        position_ids=pos_buf,
    )
    ids_seed_pad, pos_seed_pad, sl_seed_pad = proposer._prepare_dflash_inputs(
        chunk_seed_small,
        device_seed=torch.tensor([11, 22], dtype=torch.int32, device=device),
    )
    assert ids_seed_pad.shape == (320,)
    assert ids_seed_pad[0].item() == 11 and ids_seed_pad[5].item() == 22
    assert pos_seed_pad[:10].tolist() == [
        102,
        103,
        104,
        105,
        106,
        104,
        105,
        106,
        107,
        108,
    ]
    assert sl_seed_pad[:3].tolist() == [107, 109, 5]

    ids_seed_slice, _, _ = proposer._prepare_dflash_inputs(
        chunk_seed_small,
        device_seed=torch.arange(128, dtype=torch.int32, device=device),
    )
    assert ids_seed_slice.shape == (320,)

    # Target compile bucket 384 with block_size=5 gives padded_num_reqs=76 while
    # target padded_num_reqs=128; verify clean slicing without negative F.pad and
    # zero per-num_reqs Dynamo recompilations across num_reqs=65..76.
    torch._dynamo.reset()
    compile_calls = {"next_tokens": 0, "seed": 0}

    def counting_next_tokens_backend(gm, _example_inputs):
        compile_calls["next_tokens"] += 1
        return gm.forward

    def counting_seed_backend(gm, _example_inputs):
        compile_calls["seed"] += 1
        return gm.forward

    raw_nt = DFlashProposer._tpu_build_dflash_inputs_next_tokens.__wrapped__
    raw_sd = DFlashProposer._tpu_build_dflash_inputs_seed.__wrapped__
    proposer._tpu_build_dflash_inputs_next_tokens = torch.compile(
        raw_nt.__get__(proposer, DFlashProposer),
        backend=counting_next_tokens_backend,
        fullgraph=True,
        dynamic=False,
    )
    proposer._tpu_build_dflash_inputs_seed = torch.compile(
        raw_sd.__get__(proposer, DFlashProposer),
        backend=counting_seed_backend,
        fullgraph=True,
        dynamic=False,
    )

    for n_reqs in (65, 70, 76):
        qsl = np.zeros(129, dtype=np.int32)
        qsl[1 : n_reqs + 1] = np.arange(1, n_reqs + 1, dtype=np.int32) * 2
        qsl[n_reqs + 1 :] = qsl[n_reqs]
        chunk_n = _make_chunk(
            num_reqs=n_reqs,
            start_index=0,
            device=device,
            query_start_loc_np=qsl,
            position_ids=pos_buf,
        )
        next_tokens_128 = torch.full((128, 6), -1, dtype=torch.int32, device=device)
        next_tokens_128[:n_reqs, 0] = 500
        ids_n, pos_n, sl_n = proposer._prepare_dflash_inputs(
            chunk_n, next_tokens_device=next_tokens_128
        )
        assert ids_n.shape == (384,)
        assert pos_n.shape == (384,)
        assert sl_n.shape == (76,)
        # Live rows have seq_len > block_size; padded rows clamp to block_size=5
        assert int((sl_n > 5).sum().item()) == n_reqs

        seed_128 = torch.ones(128, dtype=torch.int32, device=device) * 7
        ids_s, _, sl_s = proposer._prepare_dflash_inputs(chunk_n, device_seed=seed_128)
        assert ids_s.shape == (384,)
        assert int((sl_s > 5).sum().item()) == n_reqs

    assert compile_calls["next_tokens"] == 1
    assert compile_calls["seed"] == 1


def test_td8236_host_prep_fast_paths(device):
    proposer = _make_proposer(draft_tp=1)
    proposer.runner = SimpleNamespace(
        device=device,
        num_tokens_paddings=[8, 16, 32],
        _attn_metadata_builder_ctx=None,
        build_attention_metadata_for_layers=lambda names, n: {k: "ok" for k in names},
    )

    # 1. _get_padded_len memoization cache hit and out-of-range fallback
    assert proposer._get_padded_len(10) == 16
    assert proposer._draft_paddings_cache[2][10] == 16
    assert proposer._get_padded_len(10) == 16
    assert proposer._get_padded_len(64) == 64
    assert proposer._draft_paddings_cache[2][64] == 64

    # 2. Multi-layer _build_draft_layer_metadata itemgetter fast path + cache hit
    l0 = SimpleNamespace(
        self_attn=SimpleNamespace(attn=SimpleNamespace(layer_name="d0"))
    )
    l1 = SimpleNamespace(
        self_attn=SimpleNamespace(attn=SimpleNamespace(layer_name="d1"))
    )
    proposer.draft_model = SimpleNamespace(model=SimpleNamespace(layers=[l0, l1]))
    md = {"t0": "target_md", "d0": "md0", "d1": "md1"}
    assert proposer._build_draft_layer_metadata(md) == ("md0", "md1")
    assert proposer._build_draft_layer_metadata(md) == ("md0", "md1")

    # 3. _build_draft_attn_metadata with exact-length seq_lens_cpu tuple (no slice copy)
    proposer._draft_attn_layer_names = {"d0", "d1"}
    captured_ctxs = []

    def capture_build(names, padded_num_reqs):
        captured_ctxs.append(proposer.runner._attn_metadata_builder_ctx)
        return {"d0": "md0", "d1": "md1"}

    proposer.runner.build_attention_metadata_for_layers = capture_build
    chunk = _make_chunk(
        num_reqs=2,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 2, 4], dtype=np.int32),
        position_ids=torch.zeros(4, dtype=torch.int32, device=device),
    )
    chunk.attn_ctx.seq_lens_cpu = (120, 240)
    proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=torch.tensor([125, 245], device=device)
    )
    assert captured_ctxs[-1].max_active_seq_len == 240 + proposer.block_size


def test_td8252_int32_window_bounds_and_pad_layout(device):
    proposer = _make_proposer(draft_tp=1)
    proposer._max_model_len = 4096
    qsl = torch.tensor([0, 3, 5, 5, 5], dtype=torch.int32, device=device)
    pos_i64 = torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64, device=device)

    safe_idx, valid_mask, start_pos, last_pos = proposer._request_window_bounds(
        qsl, pos_i64, padded_num_reqs=4
    )
    assert safe_idx.dtype == torch.int32
    assert valid_mask.dtype == torch.int32
    assert start_pos.dtype == torch.int32
    assert last_pos.dtype == torch.int32
    assert start_pos.squeeze(1).tolist() == [10, 20, 20, 20]
    assert last_pos.squeeze(1).tolist() == [12, 21, 21, 21]

    base_tokens = torch.tensor([101, 202, 99, 99], dtype=torch.int32, device=device)
    base_pos = torch.tensor([[13], [22], [0], [0]], dtype=torch.int32, device=device)
    for padded_len in (20, 24):
        ids, pos, sl = proposer._layout_draft_block(
            base_tokens, base_pos, block_size=5, padded_len=padded_len, mask_token_id=99
        )
        ref_ids = torch.full((padded_len,), 99, dtype=torch.int32, device=device)
        scatter_idx = torch.arange(0, 20, 5, dtype=torch.int32, device=device)
        ref_ids.scatter_(0, scatter_idx, base_tokens)
        assert ids.dtype == torch.int32
        assert pos.dtype == torch.int32
        assert sl.dtype == torch.int32
        assert torch.equal(ids, ref_ids)

    assert proposer._can_reuse_staged_block_tables(None, None, 4) is None


def test_td8270_1d_flat_base_tokens_gather_and_skip_zero_tail_pad(device, monkeypatch):
    """TD8.270: Verify _layout_draft_block skips both tail F.pad ops when
    tail_pad == 0 and _tpu_build_dflash_inputs_next_tokens extracts base_tokens
    via 1D flat int32 index_select."""
    import torch.nn.functional as F

    proposer = _make_proposer(draft_tp=1)
    proposer._max_model_len = 4096

    base_tokens = torch.tensor([101, 202, 303, 99], dtype=torch.int32, device=device)
    base_pos = torch.tensor([[10], [20], [30], [0]], dtype=torch.int32, device=device)
    pad_calls: list[tuple[int, ...]] = []
    orig_pad = F.pad

    def tracking_pad(tensor, pad, mode="constant", value=None):
        pad_calls.append(tuple(pad))
        return orig_pad(tensor, pad, mode=mode, value=value)

    monkeypatch.setattr(F, "pad", tracking_pad)

    # 1) tail_pad == 0 (padded_len == 20 == 4 * 5): only 1 F.pad call
    pad_calls.clear()
    ids_20, pos_20, sl_20 = proposer._layout_draft_block(
        base_tokens, base_pos, block_size=5, padded_len=20, mask_token_id=99
    )
    assert pad_calls == [(0, 4)]
    assert ids_20.shape == pos_20.shape == (20,)
    assert sl_20.shape == (4,)

    # 2) tail_pad > 0 (padded_len == 24 > 20): 3 F.pad calls (1 block + 2 tail)
    pad_calls.clear()
    ids_24, pos_24, sl_24 = proposer._layout_draft_block(
        base_tokens, base_pos, block_size=5, padded_len=24, mask_token_id=99
    )
    assert pad_calls == [(0, 4), (0, 4), (0, 4)]
    assert torch.equal(ids_24[:20], ids_20)
    assert torch.all(ids_24[20:] == 99)
    assert torch.equal(pos_24[:20], pos_20)
    assert torch.all(pos_24[20:] == 0)
    assert torch.equal(sl_24, sl_20)

    # 3) _tpu_build_dflash_inputs_next_tokens 1D flat_col index_select parity
    fn_nt = torch._dynamo.eval_frame.innermost_fn(
        DFlashProposer._tpu_build_dflash_inputs_next_tokens
    )
    next_tokens = torch.tensor(
        [
            [11, 12, 13, -1],
            [21, -1, -1, -1],
            [31, 32, 33, 34],
            [-1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    qsl = torch.tensor([0, 3, 4, 8, 8], dtype=torch.int32, device=device)
    pos = torch.arange(100, 108, dtype=torch.int32, device=device)
    dl = torch.tensor([3, 1, 3, 0], dtype=torch.int32, device=device)
    out_ids, out_pos, out_sl = fn_nt(
        proposer,
        next_tokens,
        qsl,
        pos,
        dl,
        block_size=4,
        padded_len=16,
        mask_token_id=99,
        invalid_token_id=-1,
    )
    assert out_ids[::4].tolist() == [13, 21, 34, 99]
    assert out_ids.dtype == out_pos.dtype == out_sl.dtype == torch.int32


def test_td8306_dflash_int32_cast_guards(device):
    proposer = _make_proposer(draft_tp=1)
    proposer._max_model_len = 2048
    fn_nt = torch._dynamo.eval_frame.innermost_fn(
        DFlashProposer._tpu_build_dflash_inputs_next_tokens
    )
    fn_seed = torch._dynamo.eval_frame.innermost_fn(
        DFlashProposer._tpu_build_dflash_inputs_seed
    )

    for rows in (2, 4):
        nt_vals = [[11, 12, -1], [21, -1, -1], [31, 32, 33], [-1, -1, -1]][:rows]
        seed_vals = [101, 202, 303, 404][:rows]
        res_nt = []
        res_seed = []
        for dt in (torch.int32, torch.int64):
            qsl = torch.tensor([0, 2, 3, 6, 6], dtype=dt, device=device)
            pos = torch.arange(50, 56, dtype=dt, device=device)
            dl = torch.tensor([2, 1, 2, 0], dtype=dt, device=device)
            nt = torch.tensor(nt_vals, dtype=dt, device=device)
            seed = torch.tensor(seed_vals, dtype=dt, device=device)
            ids_nt, pos_nt, sl_nt = fn_nt(
                proposer,
                nt,
                qsl,
                pos,
                dl,
                block_size=3,
                padded_len=12,
                mask_token_id=99,
                invalid_token_id=-1,
            )
            ids_s, pos_s, sl_s = fn_seed(
                proposer,
                seed,
                qsl,
                pos,
                block_size=3,
                padded_len=12,
                mask_token_id=99,
            )
            assert ids_nt.dtype == pos_nt.dtype == sl_nt.dtype == torch.int32
            assert ids_s.dtype == pos_s.dtype == sl_s.dtype == torch.int32
            res_nt.append((ids_nt, pos_nt, sl_nt))
            res_seed.append((ids_s, pos_s, sl_s))
        for a, b in zip(res_nt[0], res_nt[1]):
            assert torch.equal(a, b)
        for a, b in zip(res_seed[0], res_seed[1]):
            assert torch.equal(a, b)


def test_td8312_dflash_2d_argmax_and_int32_guard(device):
    """TD8.312: Verify _dflash_forward_and_sample runs argmax on 2D
    [padded_num_reqs * (block_size - 1), vocab] logits before viewing as
    [padded_num_reqs, block_size - 1] and guards redundant int32 casts."""
    padded_num_reqs, block_size, hidden_size, vocab = 3, 4, 8, 16
    valid_hidden = torch.arange(
        padded_num_reqs * block_size * hidden_size,
        dtype=torch.float32,
        device=device,
    ).reshape(padded_num_reqs * block_size, hidden_size)
    lm_w = torch.randn(vocab, hidden_size, device=device)
    argmax_input_shapes: list[tuple[int, ...]] = []

    class _Int32ArgmaxLogits:
        def __init__(self, tensor: torch.Tensor):
            self._t = tensor

        def argmax(self, dim: int = -1) -> torch.Tensor:
            argmax_input_shapes.append(tuple(self._t.shape))
            res = self._t.argmax(dim=dim).to(torch.int32)
            return res

    proposer = _make_proposer(draft_tp=1)
    fn = torch._dynamo.eval_frame.innermost_fn(
        DFlashProposer._dflash_forward_and_sample
    )
    ids = torch.zeros(padded_num_reqs * block_size, dtype=torch.int32, device=device)
    pos = torch.arange(padded_num_reqs * block_size, dtype=torch.int32, device=device)

    # 1) Standard int64 argmax path
    proposer.draft_model = SimpleNamespace(
        model=lambda **_kw: valid_hidden,
        compute_logits=lambda dh: dh @ lm_w.t(),
    )
    tokens_i64, hidden_i64 = fn(proposer, ids, pos, block_size)
    assert tokens_i64.dtype == torch.int32
    assert tokens_i64.shape == (padded_num_reqs, block_size - 1)
    assert torch.equal(hidden_i64, valid_hidden)

    # 2) Pre-int32 argmax path (exercises raw_ids.dtype == torch.int32 fast branch
    # and records the 2D shape of logits passed to .argmax(dim=-1))
    proposer.draft_model = SimpleNamespace(
        model=lambda **_kw: valid_hidden,
        compute_logits=lambda dh: _Int32ArgmaxLogits(dh @ lm_w.t()),
    )
    tokens_i32, _ = fn(proposer, ids, pos, block_size)
    assert argmax_input_shapes == [(padded_num_reqs * (block_size - 1), vocab)]
    assert tokens_i32.dtype == torch.int32
    assert torch.equal(tokens_i32, tokens_i64)


def test_td8335_dflash_vectorized_active_max_and_single_chunk_host_return(
    device, monkeypatch
):
    """TD8.335: Verify _build_draft_attn_metadata uses vectorized active.max()
    when seq_lens_cpu is a numpy ndarray (never Python builtin max(active)) and
    propose(return_device=False) returns logits0.cpu()[:reqs0].tolist() on single
    chunk while preserving multi-chunk extend parity."""
    import builtins

    proposer = _make_proposer(draft_tp=1)
    proposer.speculative_config.num_speculative_tokens = 3
    proposer._draft_attn_layer_names = {"dummy_layer"}
    captured_ctxs = []

    def fake_build(_names, _padded_num_reqs):
        captured_ctxs.append(proposer.runner._attn_metadata_builder_ctx)
        return {"dummy_layer": "md"}

    proposer.runner = SimpleNamespace(
        device=device,
        num_tokens_paddings=[8, 16, 32],
        max_num_reqs=16,
        input_batch=SimpleNamespace(num_reqs=2),
        _attn_metadata_builder_ctx=None,
        build_attention_metadata_for_layers=fake_build,
        mesh=None,
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda _n: None,
        _dp_step_num_chunks=0,
    )

    chunk = _make_chunk(
        num_reqs=2,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 2, 4], dtype=np.int32),
        position_ids=torch.zeros(4, dtype=torch.int32, device=device),
    )

    orig_max = builtins.max

    def _forbid_ndarray_builtin_max(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            raise AssertionError("builtins.max called on np.ndarray")
        return orig_max(*args, **kwargs)

    monkeypatch.setattr(builtins, "max", _forbid_ndarray_builtin_max)

    # 1a) Padded np.ndarray seq_lens_cpu slices [:num_reqs] and uses active.max()
    chunk.attn_ctx.max_active_seq_len = None
    chunk.attn_ctx.seq_lens_cpu = np.array([140, 280, 999, 999], dtype=np.int32)
    proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=torch.tensor([145, 285], device=device)
    )
    assert captured_ctxs[-1].max_active_seq_len == 280 + proposer.block_size

    # 1b) Exact-length np.ndarray seq_lens_cpu uses active.max() without slice
    chunk.attn_ctx.max_active_seq_len = None
    chunk.attn_ctx.seq_lens_cpu = np.array([310, 190], dtype=np.int32)
    proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=torch.tensor([315, 195], device=device)
    )
    assert captured_ctxs[-1].max_active_seq_len == 310 + proposer.block_size

    # 2) propose(return_device=False) single-chunk padded vs multi-chunk host return
    proposer._prepare_dflash_inputs = mock.MagicMock(
        return_value=(
            torch.zeros(8, dtype=torch.int32, device=device),
            torch.zeros(8, dtype=torch.int32, device=device),
            torch.zeros(2, dtype=torch.int32, device=device),
        )
    )
    proposer._update_draft_kv_cache_from_target = mock.MagicMock()

    proposer.draft_chunks = [chunk]
    proposer._dflash_forward_and_sample = mock.MagicMock(
        return_value=(
            torch.tensor(
                [[11, 12, 13], [21, 22, 23], [99, 99, 99], [99, 99, 99]],
                dtype=torch.int32,
                device=device,
            ),
            None,
        )
    )
    single_host = proposer.propose(
        [],
        [],
        None,
        None,
        return_device=False,
        next_tokens_per_chunk=[torch.zeros((2, 4), dtype=torch.int32, device=device)],
    )
    assert single_host == [[11, 12, 13], [21, 22, 23]]

    chunk2 = _make_chunk(
        num_reqs=1,
        start_index=2,
        device=device,
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
        position_ids=torch.zeros(2, dtype=torch.int32, device=device),
    )
    proposer.draft_chunks = [chunk, chunk2]
    proposer._dflash_forward_and_sample = mock.MagicMock(
        side_effect=[
            (
                torch.tensor(
                    [[11, 12, 13], [21, 22, 23], [99, 99, 99]],
                    dtype=torch.int32,
                    device=device,
                ),
                None,
            ),
            (
                torch.tensor(
                    [[31, 32, 33], [99, 99, 99]],
                    dtype=torch.int32,
                    device=device,
                ),
                None,
            ),
        ]
    )
    multi_host = proposer.propose(
        [],
        [],
        None,
        None,
        return_device=False,
        next_tokens_per_chunk=[
            torch.zeros((2, 4), dtype=torch.int32, device=device),
            torch.zeros((1, 4), dtype=torch.int32, device=device),
        ],
    )
    assert multi_host == [[11, 12, 13], [21, 22, 23], [31, 32, 33]]


def test_td8351_build_attn_metadata_for_draft_single_group_fast_path():
    device = torch.device("cpu")
    proposer = _make_proposer(draft_tp=1)
    proposer._draft_attn_layer_names = {"draft_layer_0", "draft_layer_1"}
    fallback_calls = []

    def fake_build(layer_names, padded_num_reqs):
        fallback_calls.append((layer_names, padded_num_reqs))
        return dict.fromkeys(layer_names, "fallback")

    class _Builder:
        def __init__(self, gid: int, tnb: int, bs: int = 16):
            self.kv_cache_group_id = gid
            self.tnb = tnb
            self.target_block_size = bs
            self.is_mamba_group = False

        def _block_table_geometry(self, _ctx):
            return None, self.tnb

        def _active_block_table_columns(self, ctx, tnb):
            if getattr(ctx, "num_reqs", 1) <= 0 or tnb <= 16:
                return tnb
            ms = getattr(ctx, "max_active_seq_len", 0) or 0
            if ms <= 0:
                return min(16, tnb)
            needed = (ms + self.target_block_size - 1) // self.target_block_size
            return min(tnb, 1 << max(4, (needed - 1).bit_length()))

    target_group = SimpleNamespace(
        layer_names=["target_layer"],
        metadata_builders=[_Builder(0, 64)],
    )
    draft_group = SimpleNamespace(
        layer_names=["draft_layer_0", "draft_layer_1"],
        metadata_builders=[_Builder(1, 64)],
    )
    pos_ids = torch.arange(10, dtype=torch.int32, device=device)
    proposer.runner = SimpleNamespace(
        device=device,
        position_ids=pos_ids,
        _attn_metadata_builder_ctx=None,
        attn_groups=[[target_group, draft_group]],
        build_attention_metadata_for_layers=fake_build,
    )
    chunk = _make_chunk(
        num_reqs=2,
        start_index=0,
        device=device,
        query_start_loc_np=np.array([0, 2, 4], dtype=np.int32),
        position_ids=torch.zeros(4, dtype=torch.int32, device=device),
    )
    chunk.attn_ctx.seq_lens_cpu = (100, 100)
    chunk.attn_ctx.max_active_seq_len = 100
    staged_draft = torch.ones(128, dtype=torch.int32)
    chunk.attn_ctx.staged_block_tables = {1: staged_draft}
    seq_lens = torch.tensor([105, 105], dtype=torch.int32, device=device)

    out = proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=seq_lens
    )
    assert fallback_calls == []
    assert set(out.keys()) == {"draft_layer_0", "draft_layer_1"}
    meta = out["draft_layer_0"]
    assert out["draft_layer_1"] is meta
    assert meta.input_positions is pos_ids
    assert meta.block_tables is staged_draft
    assert meta.seq_lens is seq_lens
    assert meta.seq_lens_cpu == (100, 100)

    # Also exercise num_reqs <= 0 and max_active_seq_len <= 0 fast-path branches.
    qsl, req_dist = proposer._get_static_attn_tensors(2)
    chunk.attn_ctx.num_reqs = 0
    out_zero_reqs = proposer._build_attn_metadata_for_draft(
        num_reqs=0,
        start_index=0,
        use_max_model_len=chunk.attn_ctx.use_max_model_len,
        seq_lens=seq_lens,
        query_start_loc=qsl,
        request_distribution=req_dist,
        position_ids_override=None,
        padded_num_reqs=2,
        seq_lens_cpu=(0, 0),
        max_active_seq_len=0,
        chunk_ctx=chunk.attn_ctx,
    )
    assert fallback_calls == []
    assert out_zero_reqs is out
    assert out_zero_reqs["draft_layer_0"] is meta
    assert out_zero_reqs["draft_layer_0"].block_tables is staged_draft
    assert meta.seq_lens_cpu == (0, 0)

    chunk.attn_ctx.num_reqs = 2
    seq_lens_step2 = torch.tensor([110, 110], dtype=torch.int32, device=device)
    out_zero_max = proposer._build_attn_metadata_for_draft(
        num_reqs=2,
        start_index=0,
        use_max_model_len=chunk.attn_ctx.use_max_model_len,
        seq_lens=seq_lens_step2,
        query_start_loc=qsl,
        request_distribution=req_dist,
        position_ids_override=None,
        padded_num_reqs=2,
        seq_lens_cpu=(110, 110),
        max_active_seq_len=0,
        chunk_ctx=chunk.attn_ctx,
    )
    assert fallback_calls == []
    assert out_zero_max is out
    assert out_zero_max["draft_layer_0"] is meta
    assert out_zero_max["draft_layer_0"].block_tables is staged_draft
    assert meta.seq_lens is seq_lens_step2
    assert meta.seq_lens_cpu == (110, 110)

    # Active column growth across chunk -> draft boundary falls back cleanly.
    chunk.attn_ctx.max_active_seq_len = 254
    out_fallback = proposer._build_draft_attn_metadata(
        chunk, num_tokens_padded=10, seq_lens=seq_lens
    )
    assert len(fallback_calls) == 1
    assert out_fallback["draft_layer_0"] == "fallback"


def test_td8356_draft_layer_getter_direct_call_and_single_group_meta_cache():
    """TD8.356: _build_draft_layer_metadata invokes cached _draft_kv_layer_getter
    directly without calling _get_draft_kv_cache_layer_names on steady-state hits,
    and refreshes when draft_model or layers change."""
    proposer = _make_proposer(draft_tp=1)
    orig_get_names = proposer._get_draft_kv_cache_layer_names
    call_count = 0

    def counting_get_names():
        nonlocal call_count
        call_count += 1
        return orig_get_names()

    proposer._get_draft_kv_cache_layer_names = counting_get_names
    l0 = SimpleNamespace(
        self_attn=SimpleNamespace(attn=SimpleNamespace(layer_name="d0"))
    )
    l1 = SimpleNamespace(
        self_attn=SimpleNamespace(attn=SimpleNamespace(layer_name="d1"))
    )
    layers_list = [l0, l1]
    proposer.draft_model = SimpleNamespace(model=SimpleNamespace(layers=layers_list))
    md = {"t0": "target_md", "d0": "md0", "d1": "md1"}

    assert proposer._build_draft_layer_metadata(md) == ("md0", "md1")
    assert call_count == 1
    # Steady-state hit: direct getter call without _get_draft_kv_cache_layer_names.
    assert proposer._build_draft_layer_metadata(md) == ("md0", "md1")
    assert call_count == 1

    # Mutating layers length invalidates fast check and refreshes cache.
    layers_list.pop()
    assert proposer._build_draft_layer_metadata(md) == ("md0",)
    assert call_count == 2
    assert proposer._build_draft_layer_metadata(md) == ("md0",)
    assert call_count == 2

    # Hook-based draft_model (cached[1] is None) also hits direct getter path.
    proposer.draft_model = SimpleNamespace(
        get_draft_kv_cache_layer_names=lambda: ["d0", "d1"]
    )
    assert proposer._build_draft_layer_metadata(md) == ("md0", "md1")
    assert call_count == 3
    assert proposer._build_draft_layer_metadata(md) == ("md0", "md1")
    assert call_count == 3
