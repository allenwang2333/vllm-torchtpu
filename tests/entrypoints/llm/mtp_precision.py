# SPDX-License-Identifier: Apache-2.0
"""TL;DR：逐次比较 MTP logits 与独立 dense KV/attention 重放，生产代码零改动。

通过 worker_extension_cls 注入，启动完成后 enable_mtp_precision，生成结束后
finish_mtp_precision。仅支持 Qwen3.5 dense、单 DP/PP/PCP/DCP；复用加载的
MTP 权重和 upstream projection/MLP，以 CPU SDPA 替换 paged attention。
参考 KV 按 request/逻辑位置保存，后续 draft 使用参考自己的 hidden carry。

捕获只保留真实请求的 logits，逐调用写临时文件，重放后删除；磁盘占用随
调用数增长，RAM 不随生成轮数增长。CPU 捕获会同步，因此不能证明异步
竞态不存在。它检查实际计算的所有词表 logits，但不独立验证权重/MLP。
默认误差阈值是待 TPU 实测校准的候选值，失败不能自动放宽或跳过。
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F


def kv_roundtrip(tensor, dtype, scale):
    """按实际 KV 存储的 dtype/静态 scale 量化，再独立解量化到 FP32。"""
    info = torch.finfo(dtype)
    return (tensor.float() / scale).clamp(info.min, info.max).to(dtype).float() * scale


def check_logits(actual, reference, *, atol, rtol, context):
    """全词表逐元素门禁；返回摘要，不存储词表大小的诊断。"""
    actual, reference = actual.float(), reference.float()
    if actual.shape != reference.shape:
        return f"{context}: logits shape {actual.shape} != {reference.shape}"
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        return f"{context}: non-finite real-request logits"
    error = (actual - reference).abs()
    bad = error > atol + rtol * reference.abs()
    if bad.any():
        return (
            f"{context}: {bad.sum().item()}/{bad.numel()} logits exceed "
            f"atol={atol}, rtol={rtol}; max_abs={error.max().item():.6g}"
        )
    return None


class DenseHistory:
    """独立 CPU KV；覆盖位置时丢弃之后的 speculative 分支。"""

    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, positions, keys, values):
        positions = positions.to(device="cpu", dtype=torch.long)
        start = int(positions[0])
        expected = torch.arange(start, start + len(positions))
        assert torch.equal(positions, expected), (
            "reference positions must be contiguous"
        )
        previous_len = 0 if self.keys is None else len(self.keys)
        assert 0 <= start <= previous_len, (
            f"reference KV history has a gap: start={start}, cached={previous_len}"
        )
        if self.keys is None:
            self.keys, self.values = keys.clone(), values.clone()
        else:
            self.keys = torch.cat((self.keys[:start], keys), dim=0)
            self.values = torch.cat((self.values[:start], values), dim=0)
        return self.keys, self.values

    def attend(self, query, key, value, positions, *, scale):
        key, value = self.update(positions, key, value)
        # CPU SDPA 的 query-specific mask 同时覆盖 chunked prefill 和 decode。
        ratio = query.shape[1] // key.shape[1]
        assert query.shape[1] % key.shape[1] == 0, "GQA heads must divide query heads"
        key = key.repeat_interleave(ratio, dim=1).transpose(0, 1).unsqueeze(0)
        value = value.repeat_interleave(ratio, dim=1).transpose(0, 1).unsqueeze(0)
        query = query.transpose(0, 1).unsqueeze(0)
        outputs = []
        # 限制显式 causal mask 的大小；不构造完整 [context, context] 矩阵。
        for start in range(0, query.shape[2], 128):
            pos = positions[start : start + 128].cpu()
            mask = torch.arange(key.shape[2])[None, :] <= pos[:, None]
            outputs.append(
                F.scaled_dot_product_attention(
                    query[:, :, start : start + 128],
                    key,
                    value,
                    attn_mask=mask,
                    scale=scale,
                )
            )
        return torch.cat(outputs, dim=2).squeeze(0).transpose(0, 1)


class StepCoverage:
    def __init__(self, k):
        self.k = k
        self.steps = {}

    def add(self, round_id, request, step):
        key = (round_id, request)
        steps = self.steps.setdefault(key, [])
        assert step == len(steps), (
            f"{key}: expected draft step {len(steps)}, got {step}"
        )
        steps.append(step)

    def finish(self):
        assert self.steps, "no real MTP calls captured"
        for key, steps in self.steps.items():
            assert steps == list(range(self.k)), f"{key}: missing draft steps: {steps}"
        return len(self.steps)


class MTPPrecisionWorkerExtension:
    def enable_mtp_precision(self, *, atol=0.5, rtol=0.03):
        if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
            raise ValueError("MTP precision tolerances must be finite and nonnegative")
        runner = self.model_runner
        proposer = runner.drafter
        config = proposer.draft_vllm_config
        parallel = runner.vllm_config.parallel_config
        if (
            type(proposer.draft_model).__name__ != "Qwen3_5MTP"
            or proposer.speculative_config.method != "mtp"
            or len(proposer.draft_model.model.layers) != 1
            or max(
                parallel.data_parallel_size,
                parallel.pipeline_parallel_size,
                parallel.prefill_context_parallel_size,
                parallel.decode_context_parallel_size,
            )
            != 1
            or config.quant_config is not None
            or config.cache_config.enable_prefix_caching
            or config.cache_config.cache_dtype
            not in ("auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2")
        ):
            raise ValueError(
                "MTP precision oracle requires unquantized dense Qwen3.5, "
                "one MTP layer, disabled prefix caching and DP=PP=PCP=DCP=1"
            )
        attention = proposer.draft_model.model.layers[0].self_attn.attn
        if attention.query_quant is not None or not all(
            math.isfinite(x) and x > 0
            for x in (attention._k_scale_float, attention._v_scale_float)
        ):
            raise ValueError(
                "MTP oracle requires unquantized Q and positive static KV scales"
            )
        recorder = _Recorder(proposer, atol, rtol)
        recorder.install()
        self._mtp_precision_recorder = recorder
        return {"enabled": True, "k": recorder.k}

    def finish_mtp_precision(self):
        recorder = self._mtp_precision_recorder
        try:
            return recorder.finish()
        finally:
            recorder.restore()
            recorder.directory.cleanup()
            del self._mtp_precision_recorder


class _Recorder:
    def __init__(self, proposer, atol, rtol):
        self.proposer = proposer
        self.atol, self.rtol = atol, rtol
        self.k = proposer.speculative_config.num_speculative_tokens
        self.directory = tempfile.TemporaryDirectory(prefix="mtp-precision-")
        self.paths = []
        self.round_id = -1
        self.active = False
        self.pending = None
        self.gathers = {}
        self.coverage = StepCoverage(self.k)
        self.originals = {}

    @staticmethod
    def cpu(tensor):
        return tensor.detach().to("cpu").clone()

    def install(self):
        from vllm_torchtpu.spec_decode.eagle3 import _static_shape_args

        proposer = self.proposer
        for name in (
            "propose",
            "_prepare_draft_inputs",
            "_forward_draft",
            "_draft_propose_token",
        ):
            self.originals[name] = getattr(proposer, name)

        def logits_and_token(hidden):
            logits = proposer.draft_model.compute_logits(hidden)
            return logits, logits.argmax(dim=-1).to(torch.int32)

        self.head = _static_shape_args(
            torch.compile(
                logits_and_token, backend="tpu", fullgraph=True, dynamic=False
            )
        )

        def propose(instance, *args, **kwargs):
            self.round_id += 1
            self.active = True
            self.gathers.clear()
            try:
                return self.originals["propose"](*args, **kwargs)
            finally:
                self.active = False

        def prepare(instance, chunk, *args, **kwargs):
            result = self.originals["_prepare_draft_inputs"](chunk, *args, **kwargs)
            if self.active:
                rejected = torch.as_tensor(result[3]).to("cpu", dtype=torch.long)
                expected = (
                    torch.as_tensor(
                        chunk.query_start_loc_np[1 : chunk.num_reqs + 1],
                        dtype=torch.long,
                    )
                    - 1
                    - rejected[: chunk.num_reqs]
                )
                actual = self.cpu(result[2])[: chunk.num_reqs]
                assert torch.equal(actual, expected), (
                    "MTP gather must select the accepted prefix end for each request"
                )
                self.gathers[id(chunk)] = expected
            return result

        def forward(instance, **kwargs):
            if self.active:
                assert self.pending is None, "draft forward without preceding logits"
                chunk = kwargs["chunk"]
                step = kwargs["step_idx"]
                qsl = chunk.query_start_loc_np[: chunk.num_reqs + 1].tolist()
                if step:
                    qsl = list(range(chunk.num_reqs + 1))
                count = qsl[-1]
                req_ids = instance.runner.input_batch.req_ids[
                    chunk.start_index : chunk.start_index + chunk.num_reqs
                ]
                self.pending = {
                    "round": self.round_id,
                    "step": step,
                    "req_ids": list(req_ids),
                    "qsl": qsl,
                    "positions": self.cpu(kwargs["positions"][..., :count]),
                    "input_ids": self.cpu(kwargs["input_ids"][:count]),
                    "hidden": (
                        self.cpu(kwargs["target_hidden_states"][:count])
                        if step == 0
                        else None
                    ),
                    "gathers": self.gathers[id(chunk)].clone() if step == 0 else None,
                }
                for req in req_ids:
                    self.coverage.add(self.round_id, req, step)
            return self.originals["_forward_draft"](**kwargs)

        def sample(instance, hidden):
            if not self.active:
                return self.originals["_draft_propose_token"](hidden)
            assert self.pending is not None, "draft logits without recorded forward"
            logits, token = self.head(hidden)
            record = self.pending
            n = len(record["req_ids"])
            record["logits"] = self.cpu(logits[:n])
            record["tokens"] = self.cpu(token[:n])
            path = Path(self.directory.name) / f"{len(self.paths):08d}.pt"
            torch.save(record, path)
            self.paths.append(path)
            self.pending = None
            return token

        for name, function in (
            ("propose", propose),
            ("_prepare_draft_inputs", prepare),
            ("_forward_draft", forward),
            ("_draft_propose_token", sample),
        ):
            setattr(proposer, name, MethodType(function, proposer))

    def restore(self):
        for name, original in self.originals.items():
            setattr(self.proposer, name, original)

    def finish(self):
        from vllm.compilation.backends import set_model_tag
        from vllm.forward_context import set_forward_context

        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        assert self.pending is None, "last draft forward has no logits"
        rounds = self.coverage.finish()
        proposer = self.proposer
        model = proposer.draft_model
        attention = model.model.layers[0].self_attn.attn
        original_attention = attention.forward
        histories, carries = {}, {}
        current = {}
        failures = []
        checked = 0
        max_abs_error = 0.0
        max_tolerance_ratio = 0.0

        def dense_attention(
            layer, query, key, value, output_shape=None, output_dtype=None
        ):
            query_cpu = self.cpu(query).view(-1, layer.num_heads, layer.head_size)
            key_cpu = self.cpu(key).view(-1, layer.num_kv_heads, layer.head_size)
            value_cpu = self.cpu(value).view(-1, layer.num_kv_heads, layer.head_size)
            quant_dtype = layer.impl.kv_cache_quantized_dtype
            if quant_dtype is not None:
                query_cpu = query_cpu.float()
                key_cpu = kv_roundtrip(key_cpu, quant_dtype, layer._k_scale_float)
                value_cpu = kv_roundtrip(value_cpu, quant_dtype, layer._v_scale_float)
            history = histories.setdefault(current["request"], DenseHistory())
            result = history.attend(
                query_cpu,
                key_cpu,
                value_cpu,
                current["positions"],
                scale=layer.impl.scale,
            )
            return result.reshape(query.shape[0], -1).to(
                device=query.device,
                dtype=query.dtype if output_dtype is None else output_dtype,
            )

        attention.forward = MethodType(dense_attention, attention)
        try:
            for path in self.paths:
                record = torch.load(path, weights_only=True)
                for i, req in enumerate(record["req_ids"]):
                    start, end = record["qsl"][i : i + 2]
                    positions = record["positions"][..., start:end]
                    logical = positions[0] if positions.ndim == 2 else positions
                    current.update(request=req, positions=logical)
                    ids = record["input_ids"][start:end]
                    if record["step"] == 0:
                        hidden = record["hidden"][start:end]
                        gather = int(record["gathers"][i]) - start
                        assert 0 <= gather < end - start, (
                            f"{req}: invalid accepted-prefix gather"
                        )
                    else:
                        hidden, previous_pos, previous_token = carries[req]
                        assert int(logical[0]) == previous_pos + 1, (
                            f"{req}: draft position did not advance"
                        )
                        assert int(ids[0]) == previous_token, (
                            f"{req}: draft input token is not previous proposal"
                        )
                        gather = 0
                    with (
                        torch.inference_mode(),
                        set_model_tag("eagle_head"),
                        set_forward_context(
                            {},
                            proposer.draft_vllm_config,
                            num_tokens=len(ids),
                            skip_compiled=True,
                        ),
                        set_vllm_model_wrapper_context(
                            mesh=proposer.runner.mesh,
                            vllm_config=proposer.draft_vllm_config,
                        ),
                    ):
                        out = model(
                            input_ids=ids.to(proposer.runner.device),
                            positions=positions.to(proposer.runner.device),
                            hidden_states=hidden.to(proposer.runner.device),
                            spec_step_idx=record["step"],
                        )
                        carry = out[gather : gather + 1]
                        reference = self.cpu(model.compute_logits(carry))[0]
                        carries[req] = (
                            self.cpu(carry),
                            int(logical[gather]),
                            int(record["tokens"][i]),
                        )
                    context = (
                        f"round={record['round']} request={req} step={record['step']}"
                    )
                    error = check_logits(
                        record["logits"][i],
                        reference,
                        atol=self.atol,
                        rtol=self.rtol,
                        context=context,
                    )
                    difference = (record["logits"][i].float() - reference.float()).abs()
                    tolerance = self.atol + self.rtol * reference.float().abs()
                    if torch.isfinite(difference).all():
                        max_abs_error = max(max_abs_error, float(difference.max()))
                        ratio = difference / tolerance.clamp_min(
                            torch.finfo(torch.float32).tiny
                        )
                        max_tolerance_ratio = max(
                            max_tolerance_ratio, float(ratio.max())
                        )
                    else:
                        max_abs_error = max_tolerance_ratio = float("inf")
                    if error and len(failures) < 10:
                        failures.append(error)
                    checked += 1
                path.unlink()
        finally:
            attention.forward = original_attention
        return {
            "checked_request_steps": checked,
            "request_rounds": rounds,
            "k": self.k,
            "atol": self.atol,
            "rtol": self.rtol,
            "max_abs_error": max_abs_error,
            "max_tolerance_ratio": max_tolerance_ratio,
            "failures": failures,
        }
