from types import SimpleNamespace

import pytest
from vllm.v1.attention.backends.utils import set_kv_cache_layout

from vllm_torchtpu.layers.vllm import attention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context


@pytest.mark.parametrize(
    ("num_speculative_tokens", "expected"),
    [
        (1, 2),
        (2, 4),
        (3, 4),
        (4, 8),
        (5, 8),
        (None, None),
    ],
)
def test_verify_q_bucket_rounds_k_plus_one_to_power_of_two(
        num_speculative_tokens, expected):
    """The static RPAv BQ bucket must cover the target token plus K drafts."""
    speculative_config = (None if num_speculative_tokens is None else
                          SimpleNamespace(
                              num_speculative_tokens=num_speculative_tokens))
    config = SimpleNamespace(speculative_config=speculative_config)

    assert attention._get_verify_q_bucket(config) == expected


@pytest.mark.parametrize(
    ("use_longctx", "use_seq_on_lane", "has_spec_config", "use_pcp_streaming",
     "expected_bucket"),
    [
        (False, True, True, False, None),
        (True, False, True, False, None),
        (True, True, False, False, None),
        (True, True, True, False, 4),
        (True, True, True, True, None),
    ],
)
def test_verify_bucket_requires_seq_on_lane_longctx_non_pcp_spec_decode(
        monkeypatch, use_longctx, use_seq_on_lane, has_spec_config,
        use_pcp_streaming, expected_bucket):
    """RPAv specialization is isolated to its supported attention path."""
    monkeypatch.setattr(attention.envs, "USE_BATCHED_RPA_LONGCTX", use_longctx)
    set_kv_cache_layout("HND" if use_seq_on_lane else "NHD")
    captured = {}

    def fake_build(_self, _q_scale, _k_scale, _v_scale, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(attention.PallasAttentionBackendImpl,
                        "_build_rpa_kernel", fake_build)
    impl = attention.PallasBatchedRPAAttentionBackendImpl(
        num_heads=2,
        head_size=128,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bfloat16",
    )
    # No `method` field: the kernel specialization depends on query width,
    # not the speculative-decoding algorithm that produced those queries.
    speculative_config = (SimpleNamespace(
        num_speculative_tokens=3) if has_spec_config else None)
    vllm_config = SimpleNamespace(speculative_config=speculative_config)

    with set_vllm_model_wrapper_context(mesh=object(),
                                        vllm_config=vllm_config):
        impl._build_rpa_kernel(None,
                               None,
                               None,
                               use_pcp_streaming=use_pcp_streaming)

    assert captured["verify_q_bucket"] == expected_bucket


def test_mainline_batched_kernel_does_not_pass_verify_bucket(monkeypatch):
    """Mainline batched RPA must call its wrapper without RPAv arguments."""
    monkeypatch.setattr(attention.envs, "USE_BATCHED_RPA_LONGCTX", False)

    def mainline_rpa():
        return "mainline"

    def invoke_rpa(*_args, rpa_func, **_kwargs):
        return rpa_func()

    monkeypatch.setattr(attention, "ragged_paged_attention_batched",
                        mainline_rpa)
    monkeypatch.setattr(attention, "_pallas_rpa_kernel_impl", invoke_rpa)

    result = attention._pallas_rpa_kernel_batched(
        *([None] * 12),
        mesh=None,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=attention.batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
        verify_q_bucket=None,
    )

    assert result == "mainline"
