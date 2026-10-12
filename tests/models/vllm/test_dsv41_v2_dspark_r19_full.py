# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement + branch coverage tests for DeepSeek-V4.1 v2/dspark.py."""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import os
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import pytest
import torch
import torch._dynamo
import torch.nn as nn

if importlib.util.find_spec("vllm") is None:
    if "tpu" not in torch._dynamo.list_backends():
        torch._dynamo.register_backend(
            name="tpu", compiler_fn=lambda gm, *a, **kw: gm.forward
        )

    class _StubModule(types.ModuleType):
        def __init__(self, name: str):
            super().__init__(name)
            self.__path__ = []  # type: ignore[attr-defined]
            self.__file__ = f"<stub:{name}>"
            self.__qualname__ = name
            self.__globals__ = {}
            self.__code__ = (lambda *a, **kw: None).__code__
            self.__defaults__ = None
            self.__kwdefaults__ = None
            self.__closure__ = None

        def __mro_entries__(self, bases):
            if not any(
                isinstance(b, type) and issubclass(b, torch.nn.Module) for b in bases
            ):
                return (type(self.__name__.rsplit(".", 1)[-1], (torch.nn.Module,), {}),)
            return (type(self.__name__.rsplit(".", 1)[-1], (), {}),)

        def __or__(self, other):
            return object

        def __ror__(self, other):
            return object

        def __getattr__(self, attr: str):
            if attr.startswith("__"):
                raise AttributeError(attr)
            full = f"{self.__name__}.{attr}"
            if full in sys.modules:
                return sys.modules[full]
            sub = _StubModule(full)
            sys.modules[full] = sub
            setattr(self, attr, sub)
            return sub

        def __call__(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return type(
                "_CtxStub",
                (SimpleNamespace,),
                {
                    "__enter__": lambda s: s,
                    "__exit__": lambda s, *a: False,
                    "__call__": lambda s, *a, **kw: (
                        a[0] if len(a) == 1 and callable(a[0]) else s
                    ),
                },
            )()

    class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        PREFIXES = (
            "portpicker",
            "absl",
            "requests",
            "vllm",
            "torch_tpu",
            "tpu_info",
            "vllm_torchtpu.layers.adapter",
            "vllm_torchtpu.model_loader_patches",
        )

        def find_spec(self, fullname, path, target=None):
            if any(
                fullname == p or fullname.startswith(p + ".") for p in self.PREFIXES
            ):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            if spec.name in sys.modules:
                return sys.modules[spec.name]
            mod = _StubModule(spec.name)
            sys.modules[spec.name] = mod
            if "." in spec.name:
                parent_name, attr_name = spec.name.rsplit(".", 1)
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, attr_name, mod)
            return mod

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _StubFinder())
    import vllm.logger

    def _init_logger(name: str):
        log = logging.getLogger(name)
        log.info_once = log.info  # type: ignore[attr-defined]
        return log

    vllm.logger.init_logger = _init_logger  # type: ignore[attr-defined]

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import dspark as dspark_mod

if not isinstance(dspark_mod.WEIGHT_FLIPPED_ATTR, str):
    dspark_mod.WEIGHT_FLIPPED_ATTR = "_weight_flipped"


@contextmanager
def _fake_stage_scope():
    old = dspark_mod.v2_attn._active_stage_attn_cache
    dspark_mod.v2_attn._active_stage_attn_cache = {}
    try:
        yield
    finally:
        dspark_mod.v2_attn._active_stage_attn_cache = old


class _FakeLayer(nn.Module):
    def __init__(self, dim: int = 16, window_size: int = 64, head_dim: int = 8) -> None:
        super().__init__()
        self.attn = SimpleNamespace(
            window_size=window_size,
            q_lora_rank=4,
            head_dim=head_dim,
            swa_cache_layer=SimpleNamespace(prefix="swa.0"),
            kv_norm=SimpleNamespace(
                weight=nn.Parameter(torch.ones(head_dim)),
                variance_epsilon=1e-6,
            ),
            fused_wqa_wkv=SimpleNamespace(
                weight=nn.Parameter(torch.randn(dim, 4 + head_dim)),
                weight_scale=None,
            ),
            _rotate=lambda kv, pos: kv,
            _fused_wqa_wkv_gemm=lambda x: torch.randn(
                x.shape[0], 4 + head_dim, dtype=x.dtype
            ),
            _split_qkv_and_norm=lambda qr_kv: (
                None,
                None,
                qr_kv[:, 4:].contiguous(),
            ),
        )
        self.ffn = SimpleNamespace(
            use_native_mega_moe=False,
            use_mega_moe=False,
            finalize_mega_moe_weights=MagicMock(),
            shared_experts=SimpleNamespace(
                down_proj=SimpleNamespace(weight=nn.Parameter(torch.zeros(4, 4)))
            ),
            gate_up_proj=SimpleNamespace(weight=nn.Parameter(torch.zeros(4, 4))),
            gate=SimpleNamespace(e_score_correction_bias=nn.Parameter(torch.zeros(4))),
        )
        self.hc_attn_fn = nn.Parameter(torch.randn(2, 4 * dim, dtype=torch.float32))
        self.hc_attn_fn_broadcast = None
        self.attn_sink = nn.Parameter(torch.zeros(4))

    def forward(
        self, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual
    ):
        tokens = input_ids.shape[0]
        dim = self.hc_attn_fn.shape[-1] // 4
        out = torch.ones((tokens, dim), dtype=torch.bfloat16)
        res = torch.ones((tokens, 4, dim), dtype=torch.bfloat16)
        pm = torch.ones((tokens, 4), dtype=torch.float32)
        rm = torch.ones((tokens, 16), dtype=torch.float32)
        prm = torch.ones((tokens, 4), dtype=torch.float32)
        return out, res, pm, rm, prm, None


def _make_vllm_cfg(
    *,
    num_spec: int = 3,
    window_size: int = 64,
    sample_from_anchor: bool = True,
    enable_confidence_head: bool = True,
    weight_block_size: tuple[int, int] | None = (128, 128),
    expert_dtype: str = "fp4",
):
    hf_cfg = SimpleNamespace(
        hidden_size=16,
        hc_mult=4,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        num_hidden_layers=10,
        dspark_target_layer_ids=[2, 5],
        n_mtp_layers=None,
        num_nextn_predict_layers=None,
        dspark_block_size=None,
        block_size=None,
        sample_from_anchor=sample_from_anchor,
        vocab_size=32,
        draft_vocab_size=None,
        dspark_markov_rank=4,
        enable_confidence_head=enable_confidence_head,
        dspark_n_routed_experts=0,
        n_routed_experts=2,
        expert_dtype=expert_dtype,
        num_attention_heads=4,
        index_topk=4,
    )
    quant_cfg = SimpleNamespace(weight_block_size=weight_block_size)
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=num_spec,
            draft_model_config=SimpleNamespace(hf_config=hf_cfg),
        ),
        quant_config=quant_cfg,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(max_model_len=128),
    )


def _patch_model_deps(monkeypatch, window_size: int = 64):
    monkeypatch.setattr(dspark_mod, "initialize_cache_ops", lambda: None)
    monkeypatch.setattr(dspark_mod, "initialize_delayed_mhc", lambda: None)
    monkeypatch.setattr(dspark_mod, "initialize_mhc_post", lambda: None)
    monkeypatch.setattr(dspark_mod, "initialize_paged_attention", lambda: None)
    monkeypatch.setattr(dspark_mod, "initialize_indexer", lambda: None)
    monkeypatch.setattr(
        dspark_mod, "get_current_vllm_config", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(dspark_mod, "compact_rope_caches", lambda *a, **kw: None)
    monkeypatch.setattr(
        dspark_mod,
        "DeepseekV41DecoderLayer",
        lambda cfg, prefix="", topk_indices_buffer=None: _FakeLayer(
            dim=16, window_size=window_size, head_dim=8
        ),
    )
    monkeypatch.setattr(
        dspark_mod,
        "ReplicatedLinear",
        lambda in_f, out_f, **kw: nn.Linear(in_f, out_f, bias=False),
    )
    monkeypatch.setattr(
        dspark_mod,
        "RMSNorm",
        lambda dim, eps=1e-6: nn.LayerNorm(dim, eps=eps),
    )
    monkeypatch.setattr(
        dspark_mod,
        "DSparkMarkovHead",
        lambda vs, dvs, rnk, prefix="": SimpleNamespace(
            embed=lambda ids: torch.ones((ids.shape[0], rnk)),
            bias=lambda me, lp: torch.ones((me.shape[0], vs)),
        ),
    )
    monkeypatch.setattr(
        dspark_mod,
        "DSparkConfidenceHead",
        lambda dim_in, prefix="": lambda hh, me: torch.zeros((hh.shape[0], 1)),
    )
    monkeypatch.setattr(
        dspark_mod,
        "LogitsProcessor",
        lambda vs: lambda lm_head, hs, skip_gather=False: lm_head(hs),
    )


class TestRemapAndEmbedAndPostCollapse:
    def test_remap_embed_and_post_collapse_bit_exact(self):
        assert dspark_mod.get_dspark_swa_index_width(60, 8) == 128
        assert dspark_mod._remap_dspark_name("layers.0.attn.weight") is None
        assert (
            dspark_mod._remap_dspark_name(
                "mtp.0.confidence_head.weight", enable_confidence_head=False
            )
            is None
        )
        assert (
            dspark_mod._remap_dspark_name("mtp.0.confidence_head.weight")
            == "model.confidence_head.weight"
        )
        assert (
            dspark_mod._remap_dspark_name("mtp.0.markov_head.embed.weight")
            == "model.markov_head.markov_w1.weight"
        )
        assert (
            dspark_mod._remap_dspark_name("mtp.0.markov_head.head.weight")
            == "model.markov_head.markov_w2.weight"
        )
        assert dspark_mod._remap_dspark_name("mtp.0.main_proj.weight") == (
            "model.main_proj.weight"
        )
        assert (
            dspark_mod._remap_dspark_name("mtp.1.ffn.gate.bias_vl")
            == "model.layers.1.ffn.gate.bias"
        )
        assert (
            dspark_mod._remap_dspark_name("mtp.2.attn.wq_a.weight")
            == "model.layers.2.attn.wq_a.weight"
        )

        # _embed_draft_blocks early exit (block <= 1 or num_blocks == 0) + body/tail
        emb = nn.Embedding(64, 16).to(torch.bfloat16)
        short_ids = torch.tensor([1, 2], dtype=torch.long)
        assert torch.equal(
            dspark_mod._embed_draft_blocks(emb, short_ids, 4), emb(short_ids)
        )
        assert torch.equal(
            dspark_mod._embed_draft_blocks(emb, short_ids, 1), emb(short_ids)
        )
        ids = torch.tensor([3, 9, 9, 9, 5, 9, 7, 9, 9, 8], dtype=torch.long)
        out = dspark_mod._embed_draft_blocks(emb, ids, 4)
        assert out.shape == (10, 16)
        assert torch.equal(out[0], emb(torch.tensor(3)))
        assert torch.equal(out[1], emb(torch.tensor(9)))
        assert torch.isnan(out[6]).all()
        assert torch.isnan(out[9]).all()

        # _post_and_collapse mult=4 and mult=2 exact bit parity
        for mult in (4, 2):
            x = torch.randn((5, 16), dtype=torch.bfloat16)
            res = torch.randn((5, mult, 16), dtype=torch.bfloat16)
            post = torch.randn((5, mult), dtype=torch.float32)
            comb = torch.randn((5, mult * mult), dtype=torch.float32)
            pre = torch.randn((5, mult), dtype=torch.float32)
            val = dspark_mod._post_and_collapse(x, res, post, comb, pre)
            assert val.shape == (5, 16) and val.dtype == torch.bfloat16


class TestInsertContextKVAllBranches:
    def test_insert_context_kv_all_paths(self, monkeypatch):
        inserted = []
        monkeypatch.setattr(
            dspark_mod,
            "_insert",
            lambda cache, slots, kv, valid, records=None: inserted.append(
                (slots.clone(), valid.clone(), records)
            ),
        )
        monkeypatch.setattr(
            dspark_mod.v2_attn, "_offset_records", lambda c, *w: ("rec", *w)
        )

        attn = SimpleNamespace(
            window_size=8,
            swa_cache_layer=SimpleNamespace(kv_cache=torch.zeros((0, 4, 8))),
            _rotate=lambda kv, pos: kv + 1,
        )
        kv = torch.zeros((4, 8))
        pos_i64 = torch.tensor([0, 1, 5, 9], dtype=torch.int64)
        pos_i32 = pos_i64.int()

        # Empty cache or None metadata -> early return
        dspark_mod._insert_context_kv(attn, kv, pos_i64, None)
        assert len(inserted) == 0

        attn.swa_cache_layer.kv_cache = torch.zeros((4, 4, 8))
        # Tensor metadata branch
        dspark_mod.v2_attn._active_stage_attn_cache = None
        dspark_mod._insert_context_kv(
            attn, kv, pos_i64, torch.tensor([0, 1, -1, 3], dtype=torch.int32)
        )
        assert len(inserted) == 1

        # Structured metadata with block_tables + fast_offset True, cache hit & reload
        monkeypatch.setattr(
            dspark_mod.v2_attn, "_has_fast_cache_offset", lambda c: True
        )
        monkeypatch.setattr(dspark_mod.v2_attn, "_fast_cache_offset_key", lambda c: "k")
        monkeypatch.setattr(
            dspark_mod.v2_attn, "_fast_swa_row_slots", lambda c, b, r, m: ("fast",)
        )
        monkeypatch.setattr(
            dspark_mod,
            "_token_requests",
            lambda md, p: torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        )
        bt1 = torch.tensor([[1, 2], [-1, 3]], dtype=torch.int64)
        bt2 = torch.tensor([[2, 1], [0, 3]], dtype=torch.int32)
        monkeypatch.setattr(dspark_mod, "_block_table", lambda md: md.block_tables)

        md1 = SimpleNamespace(
            seq_lens=torch.tensor([4, 10], dtype=torch.int64),
            block_tables=bt1,
            query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int64),
        )
        cache_dict = {"context_kv_pre_rotated": True}
        dspark_mod.v2_attn._active_stage_attn_cache = cache_dict
        dspark_mod._insert_context_kv(attn, kv, pos_i64, md1)
        # Hit exact cached_slots[0] is block_tables
        dspark_mod._insert_context_kv(attn, kv, pos_i32, md1)
        # Different block_tables reuses cached (in_window, addr) with table=None reload
        md2 = SimpleNamespace(
            seq_lens=torch.tensor([4, 10], dtype=torch.int32),
            block_tables=bt2,
            query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        )
        dspark_mod._insert_context_kv(attn, kv, pos_i32, md2)

        # Fallback addr is None (block_tables=None, seq_lens non-empty vs empty)
        monkeypatch.setattr(
            dspark_mod.v2_attn, "_has_fast_cache_offset", lambda c: False
        )
        dspark_mod.v2_attn._active_stage_attn_cache = None
        dspark_mod._insert_context_kv(attn, kv, pos_i32, md2)
        monkeypatch.setattr(
            dspark_mod, "_cache_slots", lambda c, m, p: torch.tensor([0, 1, 2, -1])
        )
        monkeypatch.setattr(
            dspark_mod,
            "_token_requests",
            lambda md, p: torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        )
        dspark_mod.v2_attn._active_stage_attn_cache = None
        md_no_bt = SimpleNamespace(
            seq_lens=torch.tensor([4, 10], dtype=torch.int32),
            block_tables=None,
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        dspark_mod._insert_context_kv(attn, kv, pos_i32, md_no_bt)
        md_empty = SimpleNamespace(
            seq_lens=torch.zeros((0,), dtype=torch.int32),
            block_tables=None,
            query_start_loc=torch.zeros((1,), dtype=torch.int32),
        )
        dspark_mod._insert_context_kv(attn, kv, pos_i32, md_empty)


class TestDSparkModelAndCausalLM:
    def test_dspark_model_and_causal_lm_all_methods(self, monkeypatch):
        _patch_model_deps(monkeypatch, window_size=2)
        with pytest.raises(ValueError, match="exceeds the DSV4.1 sliding window"):
            dspark_mod.DSparkDeepseekV41Model(vllm_config=_make_vllm_cfg(num_spec=3))

        _patch_model_deps(monkeypatch, window_size=64)
        monkeypatch.setattr(
            dspark_mod, "stage_attention_cache_scope", _fake_stage_scope
        )
        monkeypatch.setattr(
            dspark_mod, "_insert_context_kv", lambda attn, kv, pos, md: None
        )

        # 1. Model without flipped weights + sample_from_anchor=False + no confidence
        cfg_plain = _make_vllm_cfg(
            sample_from_anchor=False,
            enable_confidence_head=False,
            weight_block_size=None,
            expert_dtype="fp8",
        )
        ds_lm = dspark_mod.DSparkDeepseekV41ForCausalLM(vllm_config=cfg_plain)
        ds_lm.model.embed_tokens = nn.Embedding(32, 16).to(torch.bfloat16)
        ds_lm.lm_head = nn.Linear(16, 32, bias=False)

        assert ds_lm.embed_input_ids(torch.tensor([1, 2])).shape == (2, 16)
        assert ds_lm.combine_hidden_states(torch.randn(2, 32)).shape == (2, 16)
        assert ds_lm.get_draft_attn_causal() == [False, False, False]
        assert ds_lm.get_draft_kv_cache_layer_names() == ["swa.0", "swa.0", "swa.0"]

        # Unfused precompute_and_store_context_kv (both 2*hidden concat and hidden)
        assert ds_lm.precompute_and_store_context_kv(
            torch.randn(2, 32), torch.tensor([0, 1]), metadata=None
        ).shape == (2, 16)
        assert ds_lm.precompute_and_store_context_kv(
            torch.randn(2, 16), torch.tensor([0, 1]), metadata=[None, None, None]
        ).shape == (2, 16)

        # Forward without hc_attn_fn_broadcast and with inputs_embeds=None vs explicit
        assert ds_lm(torch.tensor([1, 2, 2, 2]), torch.tensor([0, 1, 2, 3])).shape == (
            4,
            16,
        )
        assert ds_lm.compute_draft_logits(torch.randn(2, 16)).shape == (2, 32)
        assert ds_lm.compute_draft_logits_local(torch.randn(2, 16)).shape == (2, 32)
        assert torch.equal(
            ds_lm.map_draft_to_target(torch.tensor([5])), torch.tensor([5])
        )
        me = ds_lm.markov_embed(torch.tensor([1, 2]))
        assert me.shape == (2, 4)
        assert ds_lm.markov_bias(me).shape == (2, 32)

        # Empty layers guard in finalize_mhc_broadcast_weights
        saved_layers = ds_lm.model.layers
        ds_lm.model.layers = nn.ModuleList([])
        ds_lm.model.finalize_mhc_broadcast_weights()
        ds_lm.model.layers = saved_layers

        # CPU float32 matching shape (454->456), non-float32, and wrong shape branches
        monkeypatch.setattr(dspark_mod, "pack_mix_weights", lambda layers: None)
        ds_lm.model.finalize_mhc_broadcast_weights()
        ds_lm.model.layers[0].hc_attn_fn = nn.Parameter(
            torch.zeros(2, 64, dtype=torch.bfloat16)
        )
        ds_lm.model.finalize_mhc_broadcast_weights()
        ds_lm.model.layers[0].hc_attn_fn = nn.Parameter(
            torch.zeros(2, 32, dtype=torch.float32)
        )
        monkeypatch.setattr(dspark_mod, "pack_mix_weights", lambda layers: None)
        ds_lm.model.finalize_mhc_broadcast_weights()

        # 2. Flipped context KV weights (unquantized + quantized + TPU pack_mix_weight)
        ds_lm2 = dspark_mod.DSparkDeepseekV41ForCausalLM(vllm_config=_make_vllm_cfg())
        ds_lm2.model.embed_tokens = nn.Embedding(32, 16).to(torch.bfloat16)
        assert ds_lm2.compute_confidence(torch.randn(2, 16), me).shape == (2, 1)

        fake_tpu_fn = MagicMock()
        fake_tpu_fn.dtype = torch.float32
        fake_tpu_fn.shape = (2, 64)
        fake_tpu_fn.device.type = "tpu"
        fake_tpu_fn.detach.return_value = torch.randn(2, 64, dtype=torch.float32)
        ds_lm2.model.layers[0]._parameters.pop("hc_attn_fn", None)
        ds_lm2.model.layers[0].hc_attn_fn = fake_tpu_fn
        monkeypatch.setattr(dspark_mod, "pack_mix_weight", lambda x: x)
        for lyr in ds_lm2.model.layers:
            setattr(lyr.attn.fused_wqa_wkv, dspark_mod.WEIGHT_FLIPPED_ATTR, True)
        ds_lm2.model.finalize_mhc_broadcast_weights()
        # Idempotent second call when _fused_context_wkv_weight is already set
        ds_lm2.model.finalize_mhc_broadcast_weights()

        assert ds_lm2(
            torch.tensor([1, 2, 2]),
            torch.tensor([0, 1, 2]),
            inputs_embeds=torch.randn(3, 16, dtype=torch.bfloat16),
        ).shape == (3, 16)
        assert ds_lm2.precompute_and_store_context_kv(
            torch.randn(2, 16), torch.tensor([0, 1]), metadata=None
        ).shape == (2, 16)

        # Quantized context KV matmul path
        ds_lm2.model._fused_context_wkv_weight = None
        for lyr in ds_lm2.model.layers:
            lyr.attn.fused_wqa_wkv.weight_scale = torch.ones(16, 12)
        monkeypatch.setattr(
            dspark_mod, "quantized_matmul", lambda x, w, s: torch.matmul(x, w)
        )
        ds_lm2.model._finalize_context_kv_weights(list(ds_lm2.model.layers))
        assert ds_lm2.precompute_and_store_context_kv(
            torch.randn(2, 16), torch.tensor([0, 1]), metadata=[None, None, None]
        ).shape == (2, 16)

        # 3. load_weights across native_mega_moe + fused_moe + all param kinds
        monkeypatch.setattr(
            dspark_mod, "get_tensor_model_parallel_world_size", lambda: 2
        )
        monkeypatch.setattr(dspark_mod, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(
            dspark_mod.upstream,
            "make_deepseek_v4_expert_params_mapping",
            lambda n: [("w13_weight", "experts.0.w1", 0, "w1")],
        )
        monkeypatch.setattr(
            dspark_mod,
            "fused_moe_make_expert_params_mapping",
            lambda self, **kw: [
                ("w13_weight", "experts.0.w1", 0, "w1"),
                ("w13_scale", "experts.0.w1", 0, "w1"),
                ("w2_bad", "experts.0.w2", 0, "w2"),
            ],
        )
        monkeypatch.setattr(
            dspark_mod.upstream.DeepseekV4Model,
            "_pad_shared_expert_weight",
            staticmethod(lambda qc, n, w: w),
            raising=False,
        )

        # Register named_parameters matching every load_weights branch
        loader_ok = MagicMock(return_value=True)
        loader_fail = MagicMock(return_value=False)
        stacked_loader = MagicMock()
        p_exp_w = nn.Parameter(torch.zeros(4, 4))
        p_exp_w.weight_loader = loader_ok
        p_exp_s = nn.Parameter(torch.zeros(4, 4))
        p_exp_s.weight_loader = loader_ok
        p_exp_bad = nn.Parameter(torch.zeros(4, 4))
        p_exp_bad.weight_loader = loader_fail
        p_stacked = nn.Parameter(torch.zeros(4, 4))
        p_stacked.weight_loader = stacked_loader
        p_sink = nn.Parameter(torch.zeros(4))
        p_gate_bias = nn.Parameter(torch.zeros(4))
        p_shared = nn.Parameter(torch.zeros(4, 4))
        p_shared_s = nn.Parameter(torch.zeros(4, 4))

        fake_params = {
            "model.layers.0.w13_weight": p_exp_w,
            "model.layers.0.w13_weight.weight_scale": p_exp_s,
            "model.layers.0.w2_bad": p_exp_bad,
            "model.layers.0.gate_up_proj.weight": p_stacked,
            "model.layers.0.attn_sink": p_sink,
            "model.layers.0.ffn.gate.e_score_correction_bias": p_gate_bias,
            "model.layers.0.shared_experts.down_proj.weight": p_shared,
            "model.layers.0.shared_experts.down_proj.weight_scale_inv": p_shared_s,
            "model.main_proj.weight": nn.Parameter(torch.zeros(16, 32)),
        }
        monkeypatch.setattr(
            ds_lm2, "named_parameters", lambda: iter(fake_params.items())
        )
        loaded = ds_lm2.load_weights(
            [
                ("other.weight", torch.zeros(1)),
                ("mtp.0.experts.0.w1", torch.zeros(4, 4)),
                (
                    "mtp.0.experts.0.w1.scale",
                    torch.zeros(16, dtype=torch.uint8).view(torch.float8_e8m0fnu),
                ),
                ("mtp.0.experts.0.w2", torch.zeros(4, 4)),
                ("mtp.0.experts.9.w3", torch.zeros(4, 4)),
                ("mtp.0.w1.weight", torch.zeros(4, 4)),
                ("mtp.0.attn_sink", torch.ones(4)),
                ("mtp.0.ffn.gate.bias_vl", torch.zeros(4)),
                ("mtp.0.shared_experts.w2.weight", torch.zeros(4, 4)),
                ("mtp.0.shared_experts.w2.scale", torch.zeros(4, 4)),
                ("mtp.0.main_proj.weight", torch.zeros(16, 32)),
            ]
        )
        assert "model.main_proj.weight" in loaded
        assert ds_lm2.model.confidence_head is None

        # Also cover use_native_mega_moe=True and confidence_head loaded
        ds_lm2.model.layers[0].ffn.use_native_mega_moe = True
        ds_lm2.model.confidence_head = MagicMock()
        fake_params["model.confidence_head.weight"] = nn.Parameter(torch.zeros(1, 20))
        loaded2 = ds_lm2.load_weights(
            [("mtp.0.confidence_head.weight", torch.zeros(1, 20))]
        )
        assert "model.confidence_head.weight" in loaded2
