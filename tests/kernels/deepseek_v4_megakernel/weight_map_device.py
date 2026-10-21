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
"""Check weight_map against a real DeepSeek-V4 module tree on one TPU.

    python -m tests.kernels.deepseek_v4_megakernel.weight_map_device CONFIG_JSON [TP]

Builds a DSv4-shaped model (the given HF config, cut to the first two layers:
HCA + CSA; DSv4-Pro widths) through vLLM itself -- in-process engine, dummy
weights, so every load-time transform runs -- then in the worker:

1. every `served_layer_spec` / `served_global_spec` entry against the live
   tree (path, shape, dtype, contiguity), listing every mismatch;
2. that the served layouts mean what the map assumes: each stock fp8 linear
   forward vs `x @ W * scale` with W read as [k, n]; the unquantized gate
   and indexer weights_proj vs `x @ W` in the orientation the map detected;
3. the torch_tpu jax-op bridge: packed / stacked operands built on device
   bit-identical to the same JAX fns run on host copies.

With TP > 1 the engine runs expert-parallel with the W4A8 fused EP op armed
(as served) and only the routed-expert entries are held to the spec: the
dense projections are then TP-sharded, unlike the served DP16 layout.

Prints RESULT lines; exits non-zero on any failure.
"""

import json
import os
import sys
import tempfile

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
os.environ.setdefault("NEW_MODEL_DESIGN", "1")
os.environ.setdefault("MOE_REQUANTIZE_BLOCK_SIZE", "512")
os.environ.setdefault("MOE_REQUANTIZE_WEIGHT_DTYPE", "fp4")

import torch  # noqa: E402

RANK, RANKS = 3, 16
N_EXPERTS = 128  # fewest the dense gate stream tiles (128 lanes); small loads


def _model_dir(config_json: str) -> str:
    with open(config_json) as f:
        cfg = json.load(f)
    cfg.update(
        num_hidden_layers=2,
        compress_ratios=[128, 4, 0],
        num_hash_layers=1,
        n_routed_experts=N_EXPERTS,
        num_nextn_predict_layers=0,
    )
    d = tempfile.mkdtemp(prefix="dsv4_tiny_")
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return d


def _rel(a, b) -> float:
    a, b = a.float().cpu(), b.float().cpu()
    return float((a - b).norm() / (b.norm() + 1e-30))


def probe(worker):  # runs inside the vLLM worker
    sys.path.insert(0, os.getcwd())
    from tests.kernels.deepseek_v4_megakernel.test_weight_map import host_run_jax
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import vllm_adapter as VA
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import weight_map as WM

    lm = worker.model_runner.model
    g = WM.Geometry.from_hf(lm.config)
    bad = 0
    # read here: the probe is pickled by value into spawned workers
    experts_only = os.environ.get("DSV4_PROBE_EXPERTS_ONLY") == "1"

    def say(*a):
        print(*a, flush=True)

    say("RESULT geometry", g)
    for name, t in list(lm.named_parameters()) + list(lm.named_buffers()):
        say("DUMP", name, tuple(t.shape), t.dtype, t.is_contiguous(), t.device)
    layers = list(lm.model.layers)
    E = int(layers[0].ffn.experts.routed_experts.w13_weight.shape[0])
    flags_all, served_all = [], []
    for i, layer in enumerate(layers):
        spec = WM.served_layer_spec(g, i, E)
        good, flags = {}, {}
        for key, sv in spec.items():
            try:
                t, f = VA.read_served(layer, {key: sv}, f"layers.{i}")
                good.update(t)
                flags.update(f)
                say("RESULT spec", i, key, "ok", "nk" if f[key] else "kn")
            except VA.Unsupported as e:  # collected: the report lists them all
                say("RESULT spec", i, key, "MISMATCH", e)
                expert = key.startswith(("w13", "w2"))
                bad += expert if experts_only else not expert
        served_all.append(good)
        flags_all.append(flags)
    for key, sv in WM.served_global_spec(g).items():
        try:
            VA.read_served(lm, {key: sv}, "model")
            say("RESULT spec global", key, "ok")
        except VA.Unsupported as e:
            say("RESULT spec global", key, "MISMATCH", e)
            bad += not experts_only
    if experts_only:
        # the one load-time copy on the expert side: pad_scale_rows
        for key in ("w13_scale", "w2_scale"):
            t = served_all[0][key]
            pad = WM.moe_experts.pad_scale_rows
            got = VA.torch_tpu_run_jax(f"probe_pad_{key}", pad, t).cpu()
            want = host_run_jax(key, pad, t.cpu())
            same = torch.equal(got, want)
            say(
                "RESULT bridge pad",
                key,
                tuple(got.shape),
                "bitexact" if same else "DIFF",
            )
            bad += not same
        say(f"RESULT failures={bad}")
        return bad

    # 2. semantics of the served layouts, on sane values
    torch.manual_seed(0)
    dev = layers[0].attn.wq_b.weight.device
    with torch.no_grad():
        for i, layer in enumerate(layers):
            lins = {
                "wqa": layer.attn.fused_wqa_wkv,
                "wq_b": layer.attn.wq_b,
                "wo_b": layer.attn.wo_b,
                "sh_w13": layer.ffn.shared_experts.gate_up_proj,
                "sh_w2": layer.ffn.shared_experts.down_proj,
            }
            if g.ratios[i] == 4:
                lins["idx_wq_b"] = layer.attn.indexer.wq_b
            for key, lin in lins.items():
                k, n = lin.weight.shape
                lin.weight.copy_((torch.randn(k, n) * 32).to(lin.weight.dtype).to(dev))
                lin.weight_scale.copy_(torch.full((n,), 1 / 512.0).to(dev))
                x = torch.randn(8, k, dtype=torch.bfloat16).to(dev)
                y = lin(x)
                y = y[0] if isinstance(y, tuple) else y
                ref = (x.float() @ lin.weight.float()) * lin.weight_scale.float()
                rel = _rel(y, ref)
                say(f"RESULT sem layer={i} {key} rel={rel:.4g} tol=0.05")
                bad += not rel < 0.05
            unq = {"gate": layer.ffn.gate}
            if g.ratios[i] == 4:
                unq["wp"] = layer.attn.indexer.weights_proj
            for key, lin in unq.items():
                w = lin.weight
                w.copy_(torch.randn(tuple(w.shape)).to(w.dtype).to(dev))
                kn = w.t() if flags_all[i].get(key) else w
                x = torch.randn(8, kn.shape[0], dtype=torch.bfloat16).to(dev)
                y = lin(x)
                y = y[0] if isinstance(y, tuple) else y
                rel = _rel(y, x.float() @ kn.float())
                say(f"RESULT sem layer={i} {key} rel={rel:.4g} tol=0.01")
                bad += not rel < 0.01

    # 3. the device bridge vs the host, for packed and stacked operands
    cfg = WM.dc.DenseChainConfig(ratios=g.ratios, tp=RANKS, n_experts=g.n_experts)
    s = WM.Shard(RANK, RANKS)
    dense_flags = [
        {k: v for k, v in f.items() if k in d} for f, d in zip(flags_all, served_all)
    ]
    for op in (
        "wq_b",
        "wo_a",
        "wo_b",
        "w13",
        "w2",
        "xproj_csa",
        "xproj_hca",
        "hc_attn",
        "smalls",
    ):
        keys = WM.pack_inputs(op)
        src = WM.plan_inputs(keys, dense_flags, g)
        tens = [served_all[x.layer][x.key] for x in src]
        fn = lambda *a, op=op, src=src: WM.packed_operand(op, src, a, g, s, cfg)  # noqa: E731
        got = VA.torch_tpu_run_jax(f"probe_{op}", fn, *tens).cpu()
        want = host_run_jax(op, fn, *[t.cpu() for t in tens])
        same = got.shape == want.shape and torch.equal(
            got.view(torch.uint8), want.view(torch.uint8)
        )
        say(
            "RESULT bridge",
            op,
            tuple(got.shape),
            got.dtype,
            "bitexact" if same else "DIFF",
        )
        bad += not same
    for name in ("attn_sink", "comp_ape", "moe_bias"):
        src = WM.plan_inputs((name,), dense_flags, g, stack=True)
        tens = [served_all[x.layer][x.key] for x in src]
        fn = lambda *a, name=name, src=src: WM.stacked_operand(name, src, a, g)  # noqa: E731
        got = VA.torch_tpu_run_jax(f"probe_{name}", fn, *tens).cpu()
        want = host_run_jax(name, fn, *[t.cpu() for t in tens])
        same = torch.equal(got, want)
        say(
            "RESULT bridge",
            name,
            tuple(got.shape),
            got.dtype,
            "bitexact" if same else "DIFF",
        )
        bad += not same
    say(f"RESULT failures={bad}")
    return bad


def main(config_json: str, tp: int) -> int:
    from vllm import LLM

    if tp > 1:
        os.environ.update(
            DSV4_PROBE_EXPERTS_ONLY="1",
            USE_MOE_FUSED_EP_KERNEL="1",
            MOE_FUSED_EP_ENABLE_W4A8="1",
            # served: 128 tokens x 16 DP ranks clears the default 1024
            MOE_FUSED_EP_KERNEL_MIN_TOKENS="0",
        )
    llm = LLM(
        tensor_parallel_size=tp,
        enable_expert_parallel=tp > 1,
        model=_model_dir(config_json),
        quantization="deepseek_v4_fp8",
        load_format="dummy",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=128,
        block_size=1024,
        kv_cache_dtype="fp8_e4m3",
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    return int(sum(llm.collective_rpc(probe)) > 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1))
