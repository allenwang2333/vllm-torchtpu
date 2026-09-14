from types import SimpleNamespace

import pytest
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_torchtpu.layers.adapter import attention

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def eligible_config(monkeypatch):
    monkeypatch.setattr(attention.envs, "USE_BATCHED_RPA_LONGCTX", True)
    monkeypatch.setattr(attention.envs, "VLLM_TPU_BLOCK_MAJOR_KV", False)
    monkeypatch.setattr(attention, "get_kv_cache_layout", lambda: "HND")

    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="ngram",
            num_speculative_tokens=2,
            draft_tensor_parallel_size=2,
            draft_model_config=SimpleNamespace(is_mla=False,
                                               get_head_size=lambda: 256),
        ),
        model_config=SimpleNamespace(
            is_mla=False,
            get_head_size=lambda: 256,
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.CUSTOM, ),
    )


@pytest.mark.parametrize(
    ("num_speculative_tokens", "expected"),
    [(1, 2), (2, 4), (3, 4), (4, 8), (5, 8), (6, 8), (7, 8)],
)
def test_decode_query_size_rounds_k_plus_one_to_power_of_two(
        eligible_config, num_speculative_tokens, expected):
    """The static RPAd query bucket must cover the target plus K drafts."""
    eligible_config.speculative_config.num_speculative_tokens = num_speculative_tokens
    assert attention.PallasBatchedRPAAttentionBackend.get_decode_query_size(
        eligible_config) == expected


@pytest.mark.parametrize("disabled", [
    "no_context",
    "no_spec",
    "no_model",
    "longctx_off",
    "other_backend",
    "other_layout",
    "pcp",
    "dcp",
    "block_major",
    "mla",
    "hd64",
])
def test_unsupported_configs_keep_single_token_decode(monkeypatch,
                                                      eligible_config,
                                                      disabled):
    """Unsupported kernels must never receive a widened decode prefix."""
    config = eligible_config
    if disabled == "no_context":
        config = None
    elif disabled == "no_spec":
        config.speculative_config = None
    elif disabled == "no_model":
        config.model_config = None
    elif disabled == "longctx_off":
        monkeypatch.setattr(attention.envs, "USE_BATCHED_RPA_LONGCTX", False)
    elif disabled == "other_backend":
        config.attention_config.backend = AttentionBackendEnum.FLASH_ATTN
    elif disabled == "other_layout":
        monkeypatch.setattr(attention, "get_kv_cache_layout", lambda: "NHD")
    elif disabled == "pcp":
        config.parallel_config.prefill_context_parallel_size = 2
    elif disabled == "dcp":
        config.parallel_config.decode_context_parallel_size = 2
    elif disabled == "block_major":
        monkeypatch.setattr(attention.envs, "VLLM_TPU_BLOCK_MAJOR_KV", True)
    elif disabled == "mla":
        config.model_config.is_mla = True
    elif disabled == "hd64":
        config.model_config.get_head_size = lambda: 64
    assert attention.PallasBatchedRPAAttentionBackend.get_decode_query_size(
        config) == 1


@pytest.mark.parametrize("method", ["eagle", "eagle3", "mtp"])
@pytest.mark.parametrize("draft_kind",
                         ["sharded", "tp1", "missing", "mla", "hd64"])
def test_shared_draft_metadata_requires_a_compatible_kernel(
        eligible_config, method, draft_kind):
    """Target routing must remain consumable by the first draft forward."""
    spec = eligible_config.speculative_config
    spec.method = method
    if draft_kind == "tp1":
        spec.draft_tensor_parallel_size = 1
    elif draft_kind == "missing":
        spec.draft_model_config = None
    elif draft_kind == "mla":
        spec.draft_model_config.is_mla = True
    elif draft_kind == "hd64":
        spec.draft_model_config.get_head_size = lambda: 64
    expected = 4 if draft_kind == "sharded" else 1
    assert attention.PallasBatchedRPAAttentionBackend.get_decode_query_size(
        eligible_config) == expected
