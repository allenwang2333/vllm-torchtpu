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
"""vLLM runtime driver (`MegakernelDriver`) for `megakernel`.

Integrates the modular `megakernel` library with `TPUModelRunner`:
- Extracts/packs model weights via `weight_collector.pack_weights` (preserving
  3D expert `Parameter` base buffers when `stack_e=False`).
- Binds unified KV+GDN `float8_e4m3fn` raw pools (`bind_caches`) and builds
  per-bucket ops via `exported_bridge.build_pooled_step_op` +
  `execute_pooled_step_inplace`.
- Verifies pool donation in lowered HLO (`verify_pooled_step_hlo`) and runs a
  warmup sentinel check confirming both KV and GDN pages are updated in-place.
- Dispatches eligible `T=1` decode and `T=4` MTP verify steps from
  `TPUModelRunner.forward_model`, with optional A/B calibration (`MKABCalibrator`).
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.ops import rup
from vllm_torchtpu.megakernel.spec import ExecSpec
from vllm_torchtpu.megakernel_runtime.exported_bridge import (
    build_pooled_step_op,
    execute_pooled_step_inplace,
    verify_pooled_step_hlo,
)
from vllm_torchtpu.megakernel_runtime.weight_collector import (
    find_backbone,
    get_routed_experts,
    pack_weights,
    reshard_layer_experts_ep_to_tp,
)

logger = init_logger(__name__)

_DEFAULT_BUCKETS = (8, 16, 32)
_DEFAULT_VERIFY_BUCKETS = (4, 8, 12, 16, 20, 24, 32)


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"MegakernelDriver: {msg}")


def _unwrap(x: Any) -> Any:
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


def parse_mk_buckets(env_val: str | None = None) -> tuple[int, ...]:
    """Parse comma-separated `T=1` decode batch buckets (or `'0'` to disable)."""
    raw = env_val if env_val is not None else envs.VLLM_TPU_QWEN35_MK_BUCKETS
    if raw.strip().lower() in ("0", "none", "off", "false", ""):
        return ()
    buckets = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    _need(
        len(buckets) > 0 and all(1 <= b <= 64 for b in buckets),
        f"invalid VLLM_TPU_QWEN35_MK_BUCKETS={raw!r}",
    )
    return tuple(buckets)


def parse_mk_verify_buckets(env_val: str | None = None) -> tuple[int, ...]:
    """Parse comma-separated `T=4` MTP verify row buckets (or `'0'` to disable)."""
    raw = env_val if env_val is not None else envs.VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS
    if raw.strip().lower() in ("0", "none", "off", "false", ""):
        return ()
    buckets = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    _need(
        len(buckets) > 0 and all(4 <= b <= 64 and b % 4 == 0 for b in buckets),
        f"invalid VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS={raw!r}",
    )
    return tuple(buckets)


def select_bucket(num_tokens: int, buckets: Sequence[int] = _DEFAULT_BUCKETS) -> int:
    """Return the smallest compiled bucket `>= num_tokens`."""
    for b in buckets:
        if b >= num_tokens:
            return b
    raise ValueError(f"num_tokens={num_tokens} exceeds max bucket in {buckets}")


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class _LatencySeries:
    """Bounded-memory sample accumulator keeping count, mean, and quantiles."""

    __slots__ = ("cap", "count", "samples", "sum_ms")

    def __init__(self, cap: int = 4000) -> None:
        self.cap = int(cap)
        self.count = 0
        self.sum_ms = 0.0
        self.samples: list[float] = []

    def add(self, dt_ms: float) -> None:
        v = float(dt_ms)
        idx = self.count
        self.count = idx + 1
        self.sum_ms += v
        if len(self.samples) < self.cap:
            self.samples.append(v)
        else:
            self.samples[idx % self.cap] = v

    def summary(self) -> list[int | float]:
        if self.count == 0:
            return [0, 0.0, 0.0, 0.0, 0.0]
        s = sorted(self.samples)
        mean_ms = self.sum_ms / self.count
        return [
            self.count,
            round(mean_ms, 3),
            round(_percentile(s, 0.10), 3),
            round(_percentile(s, 0.50), 3),
            round(_percentile(s, 0.90), 3),
        ]


class MKABCalibrator:
    """Deterministic per-key MK vs stock alternation and latency accumulator."""

    def __init__(
        self,
        enabled: bool = False,
        period_s: float = 60.0,
        cap: int = 4000,
    ) -> None:
        self.enabled = bool(enabled)
        self.period_s = float(period_s)
        self.cap = int(cap)
        self.counts: dict[str, int] = {}
        self.last_decision: tuple[str, str] = ("stock_early", "stock")
        self._window: dict[str, dict[str, _LatencySeries]] = {}
        self._cumulative: dict[str, dict[str, _LatencySeries]] = {}
        self._t0 = time.monotonic()

    def reset(self) -> None:
        self.counts.clear()
        self.last_decision = ("stock_early", "stock")
        self._window.clear()
        self._cumulative.clear()
        self._t0 = time.monotonic()

    def choose(self, key: str, mk_eligible: bool) -> tuple[bool, str, str]:
        """Return `(actual_run_mk, mix_key, arm)` for the current step."""
        if not mk_eligible:
            self.last_decision = (key, "stock")
            return False, key, "stock"
        if not self.enabled:
            self.last_decision = (key, "mk")
            return True, key, "mk"
        c = self.counts.get(key, 0)
        self.counts[key] = c + 1
        if (c & 1) == 0:
            self.last_decision = (key, "mk")
            return True, f"{key}_ab_mk", "mk"
        self.last_decision = (key, "stock")
        return False, f"{key}_ab_stock", "stock"

    def record(
        self, dt_ms: float, padded_tokens: int | None = None
    ) -> tuple[float, str, str] | None:
        """Record a timed step and return `(elapsed_s, window_json, cum_json)`."""
        if not self.enabled:
            return None
        key, arm = self.last_decision
        rec_key = f"{key}_p{int(padded_tokens)}" if padded_tokens is not None else key
        for store in (self._window, self._cumulative):
            bucket_map = store.get(rec_key)
            if bucket_map is None:
                bucket_map = {}
                store[rec_key] = bucket_map
            series = bucket_map.get(arm)
            if series is None:
                series = _LatencySeries(self.cap)
                bucket_map[arm] = series
            series.add(dt_ms)

        now = time.monotonic()
        elapsed = now - self._t0
        if elapsed < self.period_s:
            return None
        win_json = self.format_summary(self._window)
        cum_json = self.format_summary(self._cumulative)
        self._window.clear()
        self._t0 = now
        return elapsed, win_json, cum_json

    @staticmethod
    def format_summary(store: dict[str, dict[str, _LatencySeries]]) -> str:
        out: dict[str, dict[str, Any]] = {}
        for k in sorted(store):
            arms = store[k]
            entry: dict[str, Any] = {}
            for arm in ("mk", "stock"):
                if arm in arms:
                    entry[arm] = arms[arm].summary()
            if "mk" in arms and "stock" in arms:
                mk_s = arms["mk"].summary()
                st_s = arms["stock"].summary()
                entry["d_mean_ms"] = round(float(mk_s[1]) - float(st_s[1]), 3)
                entry["d_p50_ms"] = round(float(mk_s[3]) - float(st_s[3]), 3)
            out[k] = entry
        return json.dumps(out, separators=(",", ":"))


def _slice_or_pad_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    x1 = x.reshape(-1)
    n = int(x1.shape[0])
    if n >= target_len:
        return x1[:target_len]
    return F.pad(x1, (0, target_len - n), value=0)


def _slice_or_pad_2d(
    x: torch.Tensor, target_rows: int, target_cols: int
) -> torch.Tensor:
    r, c = int(x.shape[0]), int(x.shape[1])
    out = x[: min(r, target_rows), : min(c, target_cols)]
    pad_r = max(0, target_rows - r)
    pad_c = max(0, target_cols - c)
    if pad_r > 0 or pad_c > 0:
        out = F.pad(out, (0, pad_c, 0, pad_r), value=0)
    return out


def prepare_step_inputs_eager(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
    s0: torch.Tensor,
    c0: torch.Tensor,
    r0: torch.Tensor,
    s1: torch.Tensor,
    c1: torch.Tensor,
    r1: torch.Tensor,
    s2: torch.Tensor,
    c2: torch.Tensor,
    r2: torch.Tensor,
    bucket_t: int,
    tokens_per_seq: int,
    max_pages_per_seq: int,
    has_roff: bool,
    use_ckpt_2d: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Pack `(ids_b, pos_b, pt_b, ckpt_b, roff_b)` for a bucket of `bucket_t` tokens."""
    T = tokens_per_seq
    B = bucket_t
    nseq_bkt = B // T

    bt = block_tables
    if bt.ndim == 1:
        if bt.numel() % max_pages_per_seq == 0:
            bt = bt.reshape(-1, max_pages_per_seq)
        else:
            bt = bt.reshape(int(s0.shape[0]), -1)
    pt_b = _slice_or_pad_2d(bt, nseq_bkt, max_pages_per_seq).to(torch.int32)

    ckpt_rows: list[torch.Tensor] = []
    roff_rows: list[torch.Tensor] = []
    for s_raw, c_raw, r_raw in ((s0, c0, r0), (s1, c1, r1), (s2, c2, r2)):
        s_b = (
            _slice_or_pad_1d(s_raw[:, 0], nseq_bkt).to(torch.int32)
            if s_raw.ndim == 2
            else _slice_or_pad_1d(s_raw, nseq_bkt).to(torch.int32)
        )
        if has_roff:
            if r_raw.numel() > s_raw.shape[0]:
                s_idx = s_b.to(torch.long).clamp(min=0, max=r_raw.numel() - 1)
                roff_g = r_raw.reshape(-1)[s_idx].to(torch.int32)
            else:
                roff_g = _slice_or_pad_1d(r_raw, nseq_bkt).to(torch.int32)
        else:
            roff_g = torch.zeros_like(s_b)

        if T > 1:
            if use_ckpt_2d and c_raw.ndim == 2 and c_raw.shape[1] >= T:
                ckpt_g = _slice_or_pad_2d(c_raw, nseq_bkt, T).to(torch.int32)
            else:
                src_1d = c_raw[:, 0] if c_raw.ndim == 2 else c_raw.reshape(-1)
                ckpt_1d = _slice_or_pad_1d(src_1d, nseq_bkt).to(torch.int32)
                ckpt_g = ckpt_1d.unsqueeze(1).expand(nseq_bkt, T).contiguous()
                roff_g = torch.zeros_like(roff_g)
        else:
            if use_ckpt_2d and has_roff and c_raw.ndim == 2 and c_raw.shape[1] > 1:
                c_b = _slice_or_pad_2d(c_raw, nseq_bkt, int(c_raw.shape[1])).to(
                    torch.int32
                )
                r_idx = roff_g.to(torch.long).clamp(min=0, max=c_raw.shape[1] - 1)
                ckpt_g = c_b.gather(1, r_idx.unsqueeze(1))
            else:
                ckpt_g = s_b.unsqueeze(1)
            roff_g = torch.zeros_like(roff_g)

        active_g = ckpt_g[:, 0] > 0
        roff_g = torch.where(active_g, roff_g, torch.zeros_like(roff_g))
        ckpt_rows.append(ckpt_g)
        roff_rows.append(roff_g)

    ckpt_b = torch.stack(ckpt_rows, dim=0)
    roff_b = torch.stack(roff_rows, dim=0)
    active_seq = ckpt_b[0, :, 0] > 0
    active_tok = active_seq.unsqueeze(1).expand(nseq_bkt, T).reshape(B)
    zero_b = torch.zeros((B,), dtype=torch.int32, device=input_ids.device)
    ids_b = torch.where(
        active_tok, _slice_or_pad_1d(input_ids, B).to(torch.int32), zero_b
    )
    pos_b = torch.where(
        active_tok, _slice_or_pad_1d(positions, B).to(torch.int32), zero_b
    )
    return ids_b, pos_b, pt_b, ckpt_b, roff_b


_compiled_prepare_step_inputs = torch.compile(
    prepare_step_inputs_eager, backend="tpu", fullgraph=True, dynamic=False
)


def _extract_group_mamba_tensors(
    m_md: Any,
    fallback_1d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]:
    state_ids_raw = getattr(m_md, "mamba_state_indices", None)
    ckpt_raw = getattr(m_md, "mamba_ckpt_indices", None)
    roff = getattr(m_md, "mamba_slot_read_offsets", None)

    if state_ids_raw is not None:
        s_1d = (
            state_ids_raw[:, 0]
            if state_ids_raw.ndim == 2
            else state_ids_raw.reshape(-1)
        )
    elif ckpt_raw is not None:
        s_1d = ckpt_raw[:, 0] if ckpt_raw.ndim == 2 else ckpt_raw.reshape(-1)
    else:
        s_1d = fallback_1d

    if ckpt_raw is not None and ckpt_raw.ndim == 2:
        c_2d = ckpt_raw
        use_ckpt_2d = True
    else:
        src_1d = ckpt_raw.reshape(-1) if ckpt_raw is not None else s_1d
        c_2d = src_1d.unsqueeze(1)
        use_ckpt_2d = False

    has_roff = roff is not None
    roff_t = roff.reshape(-1) if roff is not None else fallback_1d
    return s_1d, c_2d, roff_t, has_roff, use_ckpt_2d


def prepare_step_metadata(
    attn_md: Any,
    mamba_mds: Sequence[Any],
    positions: torch.Tensor | None,
    num_reqs: int,
    bucket_t: int,
    max_pages_per_seq: int,
    tokens_per_seq: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded `(pos_b, pt_b, ckpt_b, roff_b)` tensors for `bucket_t`."""
    _need(
        len(mamba_mds) == 3,
        f"expected 3 GDN group metadata objects, got {len(mamba_mds)}",
    )
    T = int(tokens_per_seq)
    B = int(bucket_t)
    _need(B % T == 0, f"bucket_t={B} must be divisible by tokens_per_seq={T}")
    bt = attn_md.block_tables
    dev = bt.device
    active_rows = num_reqs * T

    fb_1d = getattr(attn_md, "seq_lens", None)
    if fb_1d is None:
        fb_1d = torch.zeros((max(1, num_reqs),), dtype=torch.int32, device=dev)

    if positions is not None and positions.numel() >= active_rows:
        pos_in = positions.reshape(-1)[:active_rows]
    elif T > 1:
        seq_l = _slice_or_pad_1d(fb_1d, num_reqs).to(torch.int32)
        offsets = torch.arange(1 - T, 1, dtype=torch.int32, device=dev).unsqueeze(0)
        pos_in = (seq_l.unsqueeze(1) + offsets).clamp(min=0).reshape(-1)
    else:
        seq_l = _slice_or_pad_1d(fb_1d, num_reqs).to(torch.int32)
        pos_in = (seq_l - 1).clamp(min=0)

    s0, c0, r0, has_r0, use_c0 = _extract_group_mamba_tensors(mamba_mds[0], fb_1d)
    s1, c1, r1, has_r1, use_c1 = _extract_group_mamba_tensors(mamba_mds[1], fb_1d)
    s2, c2, r2, has_r2, use_c2 = _extract_group_mamba_tensors(mamba_mds[2], fb_1d)
    has_roff = bool(has_r0 or has_r1 or has_r2)
    use_ckpt_2d = bool(use_c0 or use_c1 or use_c2)

    dummy_ids = torch.ones((active_rows,), dtype=torch.int32, device=dev)
    prep_fn = (
        _compiled_prepare_step_inputs
        if dev.type == "tpu"
        else prepare_step_inputs_eager
    )
    _, pos_b, pt_b, ckpt_b, roff_b = prep_fn(
        dummy_ids,
        pos_in,
        bt,
        s0[:num_reqs],
        c0[:num_reqs],
        r0,
        s1[:num_reqs],
        c1[:num_reqs],
        r1,
        s2[:num_reqs],
        c2[:num_reqs],
        r2,
        B,
        T,
        int(max_pages_per_seq),
        has_roff,
        use_ckpt_2d,
    )
    return pos_b, pt_b, ckpt_b, roff_b


class MegakernelDriver:
    """End-to-end Qwen3.5 megakernel driver for `TPUModelRunner`."""

    def __init__(
        self,
        model: Any,
        mesh: Any,
        *,
        tp_rank: int = 0,
        mode: str = "ep",
        geom: qwen35.Qwen35Geometry | None = None,
        base_exec: ExecSpec | None = None,
        buckets: Sequence[int] = _DEFAULT_BUCKETS,
        verify_buckets: Sequence[int] = _DEFAULT_VERIFY_BUCKETS,
        reshard_ep_to_tp: bool = False,
    ) -> None:
        _need(mode in ("ep", "tp"), f"invalid mode={mode!r}; expected 'ep' or 'tp'")
        self.model = model
        self.backbone = find_backbone(model)
        self.embed_tokens = self.backbone.embed_tokens
        self.mesh = mesh
        self.tp_rank = int(tp_rank)
        self.mode = mode
        self.buckets = tuple(sorted(buckets))
        self.max_bucket = max(self.buckets) if self.buckets else 0
        self.verify_buckets = tuple(sorted(verify_buckets))
        self.max_verify_bucket = max(self.verify_buckets) if self.verify_buckets else 0
        _need(
            bool(self.buckets or self.verify_buckets),
            "at least one of decode buckets or verify_buckets must be non-empty",
        )

        r_cnt = int(mesh.devices.size)
        if geom is None:
            e_slots = 0
            if mode == "ep":
                w13_0 = get_routed_experts(self.backbone.layers[0].mlp).w13_weight
                e_slots = int(w13_0.shape[0])
            geom = qwen35.Qwen35Geometry(
                moe_tp=(mode == "tp"),
                e_slots=0 if mode == "tp" else e_slots,
            )
        self.geom = geom
        self.spec = qwen35.qwen35_spec(geom)

        if base_exec is None:
            base_exec = ExecSpec(
                R=r_cnt,
                B=max(self.max_bucket, self.max_verify_bucket, 8),
                T=1,
                paged_kv=True,
                stack_e=False,
            )
        else:
            base_exec = dataclasses.replace(
                base_exec, R=r_cnt, paged_kv=True, stack_e=False
            )
        self.base_exec = base_exec

        if reshard_ep_to_tp and mode == "tp":
            logger.info(
                "Resharding Qwen3.5 MoE experts from EP to TP across %d layers "
                "(rank=%d)...",
                self.geom.L,
                self.tp_rank,
            )
            for l_idx in range(self.geom.L):
                reshard_layer_experts_ep_to_tp(
                    self.backbone.layers[l_idx].mlp,
                    tp_rank=self.tp_rank,
                    g=self.geom,
                    ex=self.base_exec,
                )

        t0 = time.perf_counter()
        self.weights = pack_weights(
            self.backbone, g=self.geom, ex=self.base_exec, tp_rank=self.tp_rank
        )
        self.w_flat: list[torch.Tensor] = self.weights["w_flat"]
        logger.info(
            "Packed Qwen3.5 megakernel weights (%d flat tensors, mode=%s) in %.2fs",
            len(self.w_flat),
            self.mode,
            time.perf_counter() - t0,
        )

        self.raw_pools: list[torch.Tensor] = []
        self.num_pages: int = 0
        self.page_size: int = 128
        self.max_pages_per_seq: int = 0
        self.mamba_layer_names: list[str] = []
        self.attn_layer_name: str = ""
        self._ckpt_window: int = 1
        self._zero_roff_1d: torch.Tensor | None = None
        self._runner_token_paddings: tuple[int, ...] = ()
        self._bucket_steps: dict[Any, Any] = {}
        self._bucket_ops: dict[tuple[int, int], Any] = {}
        self._bucket_out_rows: dict[tuple[int, int], int] = {}
        self._bound = False
        self._hlo_checked = False
        self._prepared_step: tuple[Any, ...] | None = None
        self._step_mix: dict[str, int] = {}
        self._step_mix_t0: float = time.monotonic()
        self._step_mix_period_s: float = float(envs.VLLM_TPU_QWEN35_MK_STATS_PERIOD_S)
        self.ab_calibrator = MKABCalibrator(
            enabled=bool(envs.VLLM_TPU_QWEN35_MK_AB),
            period_s=self._step_mix_period_s,
        )

    @classmethod
    def maybe_create(cls, runner: Any) -> MegakernelDriver | None:
        """Create a `MegakernelDriver` for `runner` if enabled and compatible."""
        if not envs.VLLM_TPU_QWEN35_MEGAKERNEL:
            return None
        try:
            backbone = find_backbone(runner.model)
        except ValueError:
            return None
        if len(getattr(backbone, "layers", ())) != 60:
            return None

        r_cnt = int(runner.mesh.devices.size)
        if r_cnt != 8:
            logger.warning(
                "Qwen3.5 megakernel requires 8 TPU devices in mesh, got %d; skipping.",
                r_cnt,
            )
            return None

        from vllm.distributed import get_tensor_model_parallel_rank

        tp_rank = int(get_tensor_model_parallel_rank())
        routed0 = get_routed_experts(backbone.layers[0].mlp)
        w13_0 = routed0.w13_weight
        loaded_ep = int(w13_0.shape[0]) < 512

        req_mode = envs.VLLM_TPU_QWEN35_MK_MODE.lower()
        if req_mode not in ("", "ep", "tp"):
            raise ValueError(
                f"Invalid VLLM_TPU_QWEN35_MK_MODE={req_mode!r}; expected 'ep' or 'tp'"
            )
        mode = req_mode if req_mode else ("ep" if loaded_ep else "tp")
        reshard = mode == "tp" and loaded_ep
        mesh = build_ep_mesh() if mode == "ep" else runner.mesh
        buckets = parse_mk_buckets()
        verify_buckets = parse_mk_verify_buckets()
        if not buckets and not verify_buckets:
            logger.info("Qwen3.5 megakernel disabled (both bucket lists empty).")
            return None

        return cls(
            model=runner.model,
            mesh=mesh,
            tp_rank=tp_rank,
            mode=mode,
            buckets=buckets,
            verify_buckets=verify_buckets,
            reshard_ep_to_tp=reshard,
        )

    def bind_caches(self, runner: Any) -> None:
        """Bind the `NATT` shared `float8_e4m3fn` raw pools from `backbone.layers`."""
        natt = self.geom.NATT
        g_per_att = self.geom.att_every - 1
        pools: list[torch.Tensor] = []
        for r in range(natt):
            att_layer = self.backbone.layers[self.geom.att_every * r + g_per_att]
            raw = _unwrap(att_layer.self_attn.attn.kv_cache)
            if isinstance(raw, torch.Tensor) and raw.ndim == 6 and raw.shape[0] == 1:
                raw = raw[0]
            pools.append(raw)

        _need(len(pools) == natt, f"expected {natt} pools, got {len(pools)}")
        num_pages = int(pools[0].shape[0])
        for r, pool in enumerate(pools):
            _need(
                isinstance(pool, torch.Tensor) and pool.dtype == torch.float8_e4m3fn,
                f"pool[{r}] must be float8_e4m3fn tensor, got {type(pool)}",
            )
            _need(
                pool.storage_offset() == 0,
                f"pool[{r}] has non-zero storage_offset={pool.storage_offset()}",
            )
            _need(
                tuple(pool.shape) == (num_pages, 2, 64, 4, 128),
                f"pool[{r}] shape {tuple(pool.shape)} != ({num_pages}, 2, 64, 4, 128)",
            )
            for g_idx in range(g_per_att):
                la_layer = self.backbone.layers[self.geom.att_every * r + g_idx]
                la_cache = _unwrap(la_layer.linear_attn.kv_cache)
                if isinstance(la_cache, torch.Tensor) and la_cache.numel() > 0:
                    _need(
                        la_cache.data_ptr() == pool.data_ptr(),
                        f"region {r} group {g_idx} linear_attn.kv_cache != pool[{r}]",
                    )

        self.raw_pools = pools
        self.num_pages = num_pages
        self.page_size = int(pools[0].shape[-1])
        self._zero_roff_1d = torch.zeros(
            (1,), dtype=torch.int32, device=pools[0].device
        )
        k_spec = (
            int(runner.speculative_config.num_speculative_tokens)
            if getattr(runner, "speculative_config", None) is not None
            else 0
        )
        self._ckpt_window = max(
            1, int(getattr(runner, "_mamba_ckpt_window", k_spec + 1))
        )
        attn_shapes_fn = getattr(runner, "_attention_kernel_shapes", None)
        if callable(attn_shapes_fn):
            _, max_pages_per_seq, _ = attn_shapes_fn(True)
            self.max_pages_per_seq = int(max_pages_per_seq)
        else:
            self.max_pages_per_seq = int(getattr(runner, "max_num_blocks_per_req", 2))
        self.mamba_layer_names = [
            self.backbone.layers[g_idx].linear_attn.prefix for g_idx in range(g_per_att)
        ]
        self.attn_layer_name = self.backbone.layers[g_per_att].self_attn.attn.layer_name

        self.base_exec = dataclasses.replace(
            self.base_exec,
            ctx=max(self.base_exec.ctx, self.max_pages_per_seq * self.page_size),
            num_pages=self.num_pages,
            page_size=self.page_size,
            max_pages_per_seq=self.max_pages_per_seq,
            paged_kv=True,
        )
        self._runner_token_paddings = tuple(
            int(x) for x in getattr(runner, "num_tokens_paddings", ())
        )
        self._build_all_buckets()
        self._bound = True
        logger.info(
            "Qwen3.5 megakernel bound to %d unified pools (num_pages=%d, "
            "page_size=%d, max_pages_per_seq=%d, decode_buckets=%s, verify_buckets=%s)",
            len(self.raw_pools),
            self.num_pages,
            self.page_size,
            self.max_pages_per_seq,
            self.buckets,
            self.verify_buckets,
        )

    def _runner_out_rows(self, b: int, m_rows: int) -> int:
        for p in self._runner_token_paddings:
            if p >= b:
                return p
        return m_rows

    def _build_all_buckets(self) -> None:
        self._bucket_steps.clear()
        self._bucket_ops.clear()
        self._bucket_out_rows.clear()
        for b in self.buckets:
            m_rows = max(16, rup(b, 16))
            out_rows = self._runner_out_rows(b, m_rows)
            step_1 = self._build_bucket_op(b, tokens_per_seq=1, out_pad_rows=out_rows)
            self._bucket_steps[(1, b)] = step_1
            self._bucket_steps[b] = step_1
            self._bucket_out_rows[(1, b)] = out_rows
        for b in self.verify_buckets:
            m_rows = max(16, rup(b, 16))
            out_rows = self._runner_out_rows(b, m_rows)
            self._bucket_steps[(4, b)] = self._build_bucket_op(
                b, tokens_per_seq=4, out_pad_rows=out_rows
            )
            self._bucket_out_rows[(4, b)] = out_rows

    def _build_bucket_op(
        self,
        bucket_t: int,
        tokens_per_seq: int = 1,
        out_pad_rows: int | None = None,
    ) -> Any:
        """Build the custom op and step wrapper for `(tokens_per_seq, bucket_t)`."""
        ex = dataclasses.replace(
            self.base_exec,
            B=bucket_t,
            T=tokens_per_seq,
        )
        if not self._hlo_checked:
            self._hlo_checked = True
            ok_hlo, reason_hlo = verify_pooled_step_hlo(
                self.mesh,
                self.spec,
                ex,
                self.w_flat,
                self.raw_pools,
                max_pages_per_seq=self.max_pages_per_seq,
                compile_hlo=bool(envs.VLLM_TPU_QWEN35_MK_COMPILE_HLO_CHECK),
            )
            if not ok_hlo:
                logger.warning(
                    "Qwen3.5 megakernel HLO pool-donation check reported: %s",
                    reason_hlo,
                )
            else:
                logger.info(
                    "Qwen3.5 megakernel HLO pool-donation check passed (%s)",
                    reason_hlo,
                )

        op_name = (
            f"megakernel_runtime::qwen35_t{tokens_per_seq}_b{bucket_t}_{self.mode}"
        )
        op, _, _, _ = build_pooled_step_op(
            self.mesh,
            self.spec,
            ex,
            op_name=op_name,
            donate_pools=True,
        )
        self._bucket_ops[(tokens_per_seq, bucket_t)] = op

        m_rows = ex.M
        out_rows = int(out_pad_rows) if out_pad_rows is not None else m_rows
        max_pages = self.max_pages_per_seq
        embed = self.embed_tokens
        is_tpu = self.raw_pools[0].device.type == "tpu"

        def _step_impl(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            block_tables: torch.Tensor,
            s0: torch.Tensor,
            c0: torch.Tensor,
            s1: torch.Tensor,
            c1: torch.Tensor,
            s2: torch.Tensor,
            c2: torch.Tensor,
            roff: torch.Tensor,
            weights: list[torch.Tensor],
            pools: list[torch.Tensor],
        ) -> torch.Tensor:
            c0_2d = c0 if c0.ndim == 2 else c0.reshape(-1, 1)
            c1_2d = c1 if c1.ndim == 2 else c1.reshape(-1, 1)
            c2_2d = c2 if c2.ndim == 2 else c2.reshape(-1, 1)
            r_1d = roff.reshape(-1)
            has_roff = r_1d.numel() > 1
            use_ckpt_2d = c0_2d.shape[1] > 1
            ids_b, pos_b, pt_b, ckpt_b, roff_b = prepare_step_inputs_eager(
                input_ids,
                positions,
                block_tables,
                s0,
                c0_2d,
                r_1d,
                s1,
                c1_2d,
                r_1d,
                s2,
                c2_2d,
                r_1d,
                bucket_t,
                tokens_per_seq,
                max_pages,
                has_roff,
                use_ckpt_2d,
            )
            hid = embed(ids_b).to(torch.bfloat16)
            if bucket_t < m_rows:
                hid = F.pad(hid, (0, 0, 0, m_rows - bucket_t))
            hid_out = execute_pooled_step_inplace(
                op,
                pos_b,
                hid,
                pt_b,
                ckpt_b,
                roff_b,
                weights,
                pools,
            )
            return hid_out[:out_rows]

        if is_tpu:
            return torch.compile(
                _step_impl, backend="tpu", fullgraph=True, dynamic=False
            )
        return _step_impl

    @torch.no_grad()
    def warmup(self, runner: Any = None) -> None:
        """Warm compiled bucket steps and run the sentinel pool-write check."""
        if not self._bound:
            raise RuntimeError("bind_caches() must be called before warmup()")
        dev = self.raw_pools[0].device
        ckpt_cols = max(1, int(self._ckpt_window))
        n_save = min(25, self.num_pages)
        saved_page0 = [p[:n_save].clone() for p in self.raw_pools]
        sentinel_checked = False
        try:
            for T, b_list in ((1, self.buckets), (4, self.verify_buckets)):
                for b in b_list:
                    step_fn = self._bucket_steps[(T, b)]
                    nseq = b // T
                    ids_b = torch.zeros((b,), dtype=torch.int32, device=dev)
                    pos_b = torch.zeros((b,), dtype=torch.int32, device=dev)
                    bt_b = torch.zeros(
                        (nseq, self.max_pages_per_seq),
                        dtype=torch.int32,
                        device=dev,
                    )
                    s0_b = torch.zeros((nseq,), dtype=torch.int32, device=dev)
                    s1_b = torch.zeros((nseq,), dtype=torch.int32, device=dev)
                    s2_b = torch.zeros((nseq,), dtype=torch.int32, device=dev)
                    c0_b = torch.zeros(
                        (nseq, max(T, ckpt_cols)), dtype=torch.int32, device=dev
                    )
                    c1_b = torch.zeros(
                        (nseq, max(T, ckpt_cols)), dtype=torch.int32, device=dev
                    )
                    c2_b = torch.zeros(
                        (nseq, max(T, ckpt_cols)), dtype=torch.int32, device=dev
                    )
                    r_b = torch.zeros((1,), dtype=torch.int32, device=dev)

                    do_sentinel = not sentinel_checked
                    sent_kv = None
                    sent_gdn = None
                    if do_sentinel:
                        if dev.type != "tpu":
                            self.raw_pools[0][0].view(torch.uint8).fill_(0x5A)
                            self.raw_pools[0][5:10].view(torch.uint8).fill_(0x5A)
                        sent_kv = self.raw_pools[0][0].view(torch.uint8).cpu().clone()
                        sent_gdn = (
                            self.raw_pools[0][5:10].view(torch.uint8).cpu().clone()
                        )
                        ids_b[0] = 1
                        s0_b[0] = 1
                        s1_b[0] = 1
                        s2_b[0] = 1
                        c0_b[0, :T] = torch.arange(
                            1, 1 + T, dtype=torch.int32, device=dev
                        )
                        c1_b[0, :T] = torch.arange(
                            1, 1 + T, dtype=torch.int32, device=dev
                        )
                        c2_b[0, :T] = torch.arange(
                            1, 1 + T, dtype=torch.int32, device=dev
                        )

                    _ = step_fn(
                        ids_b,
                        pos_b,
                        bt_b,
                        s0_b,
                        c0_b,
                        s1_b,
                        c1_b,
                        s2_b,
                        c2_b,
                        r_b,
                        self.w_flat,
                        self.raw_pools,
                    )
                    if do_sentinel:
                        sentinel_checked = True
                        assert sent_kv is not None and sent_gdn is not None
                        kv_ok = not torch.equal(
                            self.raw_pools[0][0].view(torch.uint8).cpu(), sent_kv
                        )
                        gdn_ok = not torch.equal(
                            self.raw_pools[0][5:10].view(torch.uint8).cpu(),
                            sent_gdn,
                        )
                        _need(
                            kv_ok and gdn_ok,
                            "warmup sentinel check failed "
                            f"(kv_ok={kv_ok}, gdn_ok={gdn_ok})",
                        )
        finally:
            for p, saved in zip(self.raw_pools, saved_page0, strict=True):
                p[:n_save].copy_(saved)

    @staticmethod
    def _inspect_step_detail(
        attn_md: Any, m0: Any, runner: Any = None
    ) -> tuple[bool, int, int, str]:
        if runner is not None and getattr(runner, "input_batch", None) is not None:
            ib = runner.input_batch
            num_reqs = int(getattr(ib, "num_reqs", 0))
            if num_reqs <= 0:
                return False, 0, 0, "stock_n0"
            num_sched = getattr(runner, "_Last_num_scheduled_tokens_np", None)
            if num_sched is None:
                num_sched = getattr(runner, "_last_num_scheduled_tokens", None)
            if num_sched is not None and len(num_sched) >= num_reqs:
                q0 = int(num_sched[0])
                if all(int(num_sched[i]) == q0 for i in range(num_reqs)):
                    return True, num_reqs, q0, "ok"
                return False, num_reqs, 0, "stock_mixed"

        qsl_cpu = getattr(attn_md, "query_start_loc_cpu", None)
        num_seqs_val = getattr(attn_md, "num_seqs", None)
        if qsl_cpu is not None and num_seqs_val is not None:
            n_val = (
                int(num_seqs_val.item())
                if isinstance(num_seqs_val, torch.Tensor)
                else int(num_seqs_val)
            )
            if n_val <= 0 or int(qsl_cpu.shape[0]) <= n_val:
                return False, 0, 0, "stock_n0"
            qsl_list = qsl_cpu[: n_val + 1].tolist()
            total_toks = int(qsl_list[n_val])
            if total_toks == n_val:
                if all(qsl_list[i] == i for i in range(n_val + 1)):
                    return True, n_val, 1, "ok"
                return False, n_val, 0, "stock_mixed"
            if total_toks == 4 * n_val:
                if all(qsl_list[i] == 4 * i for i in range(n_val + 1)):
                    return True, n_val, 4, "ok"
                return False, n_val, 0, "stock_mixed"
            return (
                False,
                n_val,
                0,
                f"stock_q{total_toks // max(1, n_val)}",
            )

        s_idx = getattr(m0, "mamba_state_indices", None)
        if s_idx is None:
            s_idx = getattr(m0, "mamba_ckpt_indices", None)
        if s_idx is None:
            return False, 0, 0, "stock_nomamba"
        n_rows = int(s_idx.shape[0])
        c_idx = getattr(m0, "mamba_ckpt_indices", None)
        t_per_seq = (
            int(c_idx.shape[1])
            if (c_idx is not None and c_idx.ndim == 2 and c_idx.shape[1] == 4)
            else 1
        )
        return True, n_rows, t_per_seq, "ok"

    def can_run(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        runner: Any = None,
    ) -> bool:
        """Return `True` if the current forward step should run on the megakernel."""
        self._prepared_step = None
        if not self._bound:
            return False
        if (
            input_ids is None
            or inputs_embeds is not None
            or intermediate_tensors is not None
        ):
            return False

        ctx = get_forward_context()
        md = getattr(ctx, "attn_metadata", None)
        if not isinstance(md, dict) or not md:
            return False

        attn_md = md.get(self.attn_layer_name)
        if attn_md is None:
            return False
        m_names = self.mamba_layer_names
        m0 = md.get(m_names[0])
        m1 = md.get(m_names[1])
        m2 = md.get(m_names[2])
        if m0 is None or m1 is None or m2 is None:
            return False

        if (
            getattr(m0, "mamba_state_indices", None) is None
            and getattr(m0, "mamba_ckpt_indices", None) is None
        ):
            return False

        ok, num_reqs, tokens_per_seq, reason = self._inspect_step_detail(
            attn_md, m0, runner=runner
        )
        if not ok or num_reqs <= 0:
            return self._note_step(reason, False)

        active_rows = num_reqs * tokens_per_seq
        if tokens_per_seq == 4:
            if not self.verify_buckets or active_rows > self.max_verify_bucket:
                return self._note_step(f"stock_t4_n{num_reqs}", False)
            bkt = select_bucket(active_rows, self.verify_buckets)
            step_fn = self._bucket_steps.get((4, bkt))
            if step_fn is None:
                return self._note_step("stock_t4_nobkt", False)
            if self._bucket_out_rows.get((4, bkt), input_ids.shape[0]) != int(
                input_ids.shape[0]
            ):
                return self._note_step(f"stock_t4_rows{bkt}", False)
            self._prepared_step = (
                step_fn,
                attn_md,
                m0,
                m1,
                m2,
                num_reqs,
                4,
                bkt,
            )
            return self._note_step(f"mk_t4_b{bkt}", True)
        if tokens_per_seq == 1:
            if not self.buckets or active_rows > self.max_bucket:
                return self._note_step(f"stock_t1_n{num_reqs}", False)
            bkt = select_bucket(active_rows, self.buckets)
            step_fn = self._bucket_steps.get((1, bkt)) or self._bucket_steps.get(bkt)
            if step_fn is None:
                return self._note_step("stock_t1_nobkt", False)
            self._prepared_step = (
                step_fn,
                attn_md,
                m0,
                m1,
                m2,
                num_reqs,
                1,
                bkt,
            )
            return self._note_step(f"mk_t1_b{bkt}", True)
        return self._note_step(f"stock_q{tokens_per_seq}", False)

    def _note_step(self, key: str, run: bool) -> bool:
        actual_run, mix_key, _ = self.ab_calibrator.choose(key, run)
        self._step_mix[mix_key] = self._step_mix.get(mix_key, 0) + 1
        now = time.monotonic()
        if now - self._step_mix_t0 >= self._step_mix_period_s:
            if self.tp_rank == 0:
                total = sum(self._step_mix.values())
                mk_cnt = sum(
                    v for k, v in self._step_mix.items() if k.startswith("mk_")
                )
                logger.info(
                    "Qwen3.5 megakernel step mix over %.0fs: %d/%d MK (%.1f%%) %s",
                    now - self._step_mix_t0,
                    mk_cnt,
                    total,
                    100.0 * mk_cnt / max(total, 1),
                    dict(sorted(self._step_mix.items())),
                )
            self._step_mix = {}
            self._step_mix_t0 = now
        return actual_run

    def record_ab_step(self, dt_ms: float, padded_tokens: int | None = None) -> None:
        res = self.ab_calibrator.record(dt_ms, padded_tokens=padded_tokens)
        if res is not None and self.tp_rank == 0:
            elapsed, win_json, cum_json = res
            logger.info(
                "Qwen3.5 megakernel AB latency (window %.0fs): %s",
                elapsed,
                win_json,
            )
            logger.info(
                "Qwen3.5 megakernel AB latency (cumulative): %s",
                cum_json,
            )

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int | None = None,
        runner: Any = None,
    ) -> torch.Tensor:
        """Execute one full-backbone decode (`T=1`) or MTP verify (`T=4`) step."""
        prepared = self._prepared_step
        self._prepared_step = None
        if prepared is not None:
            step_fn, attn_md, m0, m1, m2, num_reqs, tokens_per_seq, _ = prepared
        else:
            md = get_forward_context().attn_metadata
            attn_md = md[self.attn_layer_name]
            m0 = md[self.mamba_layer_names[0]]
            m1 = md[self.mamba_layer_names[1]]
            m2 = md[self.mamba_layer_names[2]]
            _, num_reqs, tokens_per_seq, _ = self._inspect_step_detail(
                attn_md, m0, runner=runner
            )
            active_rows = num_reqs * tokens_per_seq
            if tokens_per_seq == 4:
                bkt = select_bucket(
                    max(4, min(active_rows, self.max_verify_bucket)),
                    self.verify_buckets,
                )
                step_fn = self._bucket_steps[(4, bkt)]
            else:
                bkt = select_bucket(
                    max(1, min(active_rows, self.max_bucket)), self.buckets
                )
                step_fn = self._bucket_steps.get((1, bkt)) or self._bucket_steps[bkt]

        out_rows = int(num_tokens if num_tokens is not None else input_ids.shape[0])
        fb_1d = getattr(attn_md, "seq_lens", None)
        if fb_1d is None:
            fb_1d = torch.zeros(
                (max(1, num_reqs),), dtype=torch.int32, device=input_ids.device
            )

        s0 = getattr(m0, "mamba_state_indices", None)
        c0 = getattr(m0, "mamba_ckpt_indices", None)
        if s0 is None:
            s0 = c0 if c0 is not None else fb_1d
        if c0 is None:
            c0 = s0

        s1 = getattr(m1, "mamba_state_indices", None)
        c1 = getattr(m1, "mamba_ckpt_indices", None)
        if s1 is None:
            s1 = c1 if c1 is not None else fb_1d
        if c1 is None:
            c1 = s1

        s2 = getattr(m2, "mamba_state_indices", None)
        c2 = getattr(m2, "mamba_ckpt_indices", None)
        if s2 is None:
            s2 = c2 if c2 is not None else fb_1d
        if c2 is None:
            c2 = s2

        roff = getattr(m0, "mamba_slot_read_offsets", None)
        if roff is None and runner is not None:
            roff = getattr(runner, "mamba_slot_read_offsets", None)
        if roff is None:
            assert self._zero_roff_1d is not None
            roff = self._zero_roff_1d

        active_rows = num_reqs * tokens_per_seq
        if positions is not None and positions.numel() >= active_rows:
            pos_in = positions
        elif tokens_per_seq > 1:
            offsets = torch.arange(
                1 - tokens_per_seq, 1, dtype=torch.int32, device=fb_1d.device
            ).unsqueeze(0)
            pos_in = (fb_1d.unsqueeze(1) + offsets).clamp(min=0).reshape(-1)
        else:
            pos_in = (fb_1d - 1).clamp(min=0)

        hidden_out = step_fn(
            input_ids,
            pos_in,
            attn_md.block_tables,
            s0,
            c0,
            s1,
            c1,
            s2,
            c2,
            roff,
            self.w_flat,
            self.raw_pools,
        )

        if out_rows == hidden_out.shape[0]:
            return hidden_out
        if out_rows <= hidden_out.shape[0]:
            return hidden_out[:out_rows]
        return F.pad(hidden_out, (0, 0, 0, out_rows - hidden_out.shape[0]))


Qwen35Megakernel = MegakernelDriver
