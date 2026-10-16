# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic cache pool pipeline (`CachePipe`, `DirectHBM`, `PallasWriter`).

Architecture (Communicating vs. Non-Communicating Split):
---------------------------------------------------------
Mosaic TPU enforces a hardware/compiler invariant (`custom_call_importer.cc`):
a `pallas_call` that carries a `collective_id` or issues cross-device DMAs
(`all-reduce-start`, `collective-permute-start`, `semaphore_signal` with
`device_id`) **cannot** also alias an `HBM` state buffer in-place via
`input_output_aliases`.

Therefore, the pooled cache pipeline splits execution into three phases via the
model-injected `ModelSpec.pool_backend`:
1. **Non-communicating gather (`pool_backend.gather_states`)**: Reads per-sequence
   recurrent/conv checkpoints from unified block pools into dense per-step tiles.
2. **Communicating megakernel (`kernel.build(..., paged_kv=True)`)**: Takes
   unified block pools as **read-only** `HBM` inputs (`access="r"`) for paged
   KV reads, performs all layer compute and cross-chip communication, and emits
   small per-step outputs.
3. **Non-communicating scatter (`pool_backend.scatter_states`)**: Runs with
   `collective_id=None` and `input_output_aliases` to scatter new KV and
   recurrent/conv states in-place into the unified block pools.
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from . import kernel
from .spec import ExecSpec, ModelSpec, PoolBackend


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ModeTag:
    """Zero-leaf pytree wrapper for string modes like `'from_ckpt'` in `jax.jit`."""

    mode: str

    def tree_flatten(self) -> tuple[tuple[()], str]:
        return (), self.mode

    @classmethod
    def tree_unflatten(cls, aux_data: str, children: tuple[()]) -> _ModeTag:
        del children
        return cls(aux_data)


def _wrap_active_seqs(active_seqs: Any) -> Any:
    if isinstance(active_seqs, str):
        if active_seqs not in ("from_ckpt", "all"):
            raise ValueError(
                f"Unsupported active_seqs mode {active_seqs!r}; "
                "expected 'from_ckpt', 'all', None, or an int32/bool mask array"
            )
        return _ModeTag(active_seqs)
    return active_seqs


def _unwrap_active_seqs(active_seqs: Any) -> Any:
    if isinstance(active_seqs, _ModeTag):
        return active_seqs.mode
    return active_seqs


def _resolve_model_spec(g_or_spec: Any) -> ModelSpec:
    if isinstance(g_or_spec, ModelSpec):
        return g_or_spec
    if hasattr(g_or_spec, "to_model_spec"):
        return g_or_spec.to_model_spec()
    raise TypeError(
        "Expected a ModelSpec or a geometry config implementing .to_model_spec(), "
        f"got {type(g_or_spec).__name__}"
    )


def _require_pool_backend(model_spec: ModelSpec) -> PoolBackend:
    if model_spec.pool_backend is None:
        raise ValueError(
            f"ModelSpec {model_spec.name!r} does not configure a pool_backend"
        )
    return model_spec.pool_backend


def _validate_page_table_shape(page_table: Any, ex: ExecSpec) -> None:
    ns, r = ex.NS, ex.R
    if page_table.ndim == 2:
        if page_table.shape[0] not in (ns, r * ns) or page_table.shape[1] < 1:
            raise ValueError(
                f"page_table has shape {page_table.shape}, expected "
                f"({ns}, >=1) or ({r * ns}, >=1)"
            )
    elif page_table.ndim == 3:
        if page_table.shape[:2] not in ((1, ns), (r, ns)) or page_table.shape[2] < 1:
            raise ValueError(
                f"page_table has shape {page_table.shape}, expected ({r}, {ns}, >=1)"
            )
    else:
        raise ValueError(f"page_table must be 2D or 3D, got shape {page_table.shape}")


class _PooledStepFn:
    """Callable step function that unpacks as `(step_fn, info)` and has `.lower()`."""

    def __init__(self, call_fn: Any, lower_fn: Any, info: dict[str, Any]):
        self._call_fn = call_fn
        self._lower_fn = lower_fn
        self.info = info

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_fn(*args, **kwargs)

    def lower(self, *args: Any, **kwargs: Any) -> Any:
        return self._lower_fn(*args, **kwargs)

    def __iter__(self) -> Any:
        yield self
        yield self.info


def check_pooled_step_hlo(
    hlo_text: str,
    page_size: int = 128,
    *,
    kvh: int = 1,
    hd: int = 256,
    min_pool_pages: int = 16,
    require_scatter: bool = True,
) -> tuple[bool, str]:
    """Verify that a compiled pooled step HLO has no full-pool HBM or VMEM copies.

    `min_pool_pages` (default 16) filters out small per-token KV (1-page) or GDN
    checkpoint (5-page) staging slices materialized in CPU interpret mode while
    flagging any full-pool HBM copy (`num_pages >= min_pool_pages`).
    """
    n_tpu_cc = hlo_text.count("tpu_custom_call")
    n_cpu_cc = hlo_text.count("xla_ffi_python_cpu_callback")
    is_cpu_interp = n_cpu_cc > 0
    min_cc = 2 if require_scatter else 1
    if require_scatter:
        has_scatter = (
            "scatter_pool_states" in hlo_text
            or n_tpu_cc >= 2
            or (is_cpu_interp and n_cpu_cc >= 2)
        )
        if not has_scatter:
            return False, "missing scatter_pool_states in HLO"
    has_mk = (
        "decode_megakernel" in hlo_text
        or n_tpu_cc >= min_cc
        or (is_cpu_interp and n_cpu_cc >= min_cc)
        or ("pallas_call" in hlo_text and "_body" in hlo_text)
    )
    if not has_mk:
        msg = (
            f"missing megakernel call in HLO "
            f"(found {n_tpu_cc} tpu_custom_call, need >={min_cc})"
        )
        return False, msg
    pool_suffix = f",{2 * kvh},{hd // 4},4,{page_size}]"
    for line in hlo_text.splitlines():
        if pool_suffix in line and (
            "copy-start(" in line or ("= f8e4m3fn[" in line and "copy(" in line)
        ):
            prefix = line.split(pool_suffix, 1)[0]
            dim_str = prefix.rsplit("[", 1)[-1].rsplit(",", 1)[-1].strip()
            if dim_str.isdigit() and int(dim_str) < min_pool_pages:
                continue
            if "S(1)" in line:
                return False, f"pool operand copied to VMEM S(1): {line.strip()}"
            return False, f"unexpected full-pool HBM copy in HLO: {line.strip()}"
    return True, "ok"


def make_flat_pooled_step(
    mesh: Any,
    model_spec: Any,
    ex: ExecSpec,
    *,
    donate_pools: bool = True,
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...], tuple[int, ...]]:
    """Build a flat-signature `shard_map` step for `torch_xla` bridge integration.

    Flat input signature:
      `(pos, hidden, *w_flat, *raw_pools, page_table, ckpt_indices, read_offsets)`
    Flat output signature:
      `(out_hidden, *raw_pools_out)`
    """
    ms = _resolve_model_spec(model_spec)
    pb = _require_pool_backend(ms)
    ex_paged = dataclasses.replace(ex, paged_kv=True)
    call, info = kernel.build(ms, ex_paged)
    p = info["plan"]
    w_names = p.w_names
    s_names = p.s_names
    o_names = p.o_names
    num_pools = pb.num_pools
    has_rec = pb.has_recurrent_pool

    n_w_flat = len(w_names)
    pool_start = 2 + n_w_flat
    pool_end = pool_start + num_pools
    donate_argnums = tuple(range(pool_start, pool_end)) if donate_pools else ()

    in_specs = (
        (P(), P("tp"))
        + (P("tp"),) * n_w_flat
        + (P("tp"),) * num_pools
        + (P(), P(), P())
    )
    out_specs = (P("tp"),) + (P("tp"),) * num_pools
    max_pt = ex_paged.resolved_max_pages_per_seq

    def _body(pos: Any, hidden: Any, *rest: Any) -> tuple[Any, ...]:
        w_flat = rest[:n_w_flat]
        raw_pools = tuple(
            r.squeeze(0) if r.ndim == 6 else r
            for r in rest[n_w_flat : n_w_flat + num_pools]
        )
        page_table, ckpt_indices, read_offsets = rest[
            n_w_flat + num_pools : n_w_flat + num_pools + 3
        ]

        w_args = [w_flat[idx][0] for idx in range(n_w_flat)]

        _validate_page_table_shape(page_table, ex_paged)
        if page_table.ndim == 3:
            page_table = page_table.squeeze(0)
        if page_table.shape[-1] > max_pt:
            page_table = page_table[..., :max_pt]

        gathered = (
            pb.gather_states(
                ex_paged,
                raw_pools,
                ckpt_indices,
                read_offsets,
                "from_ckpt",
            )
            if (has_rec and pb.gather_states is not None)
            else {}
        )

        st_map: dict[str, Any] = {
            **gathered,
            pb.pool_state_name: raw_pools,
            pb.page_table_state_name: page_table,
        }
        s_args = [st_map[k] for k in s_names]
        out_vals = call(pos, hidden, *w_args, *s_args)
        if not isinstance(out_vals, (tuple, list)):
            out_vals = (out_vals,)
        out_map = dict(zip(o_names, out_vals))
        hid = out_map["hidden"]

        raw_out = pb.scatter_states(
            ex_paged,
            raw_pools,
            pos,
            page_table,
            out_map,
            ckpt_indices if has_rec else None,
            "from_ckpt" if has_rec else None,
        )
        return (hid, *raw_out)

    smap = jax.jit(
        jax.shard_map(
            _body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
        ),
        donate_argnums=donate_argnums,
    )
    return smap, in_specs, out_specs, donate_argnums


def make_pooled_step(
    mesh: Any,
    model_spec: Any,
    ex: ExecSpec,
    *,
    donate_pools: bool = False,
) -> Any:
    """Build the `jax.jit` `shard_map` step over unified block pools.

    All calling conventions route through a single `jax.jit` function with
    `donate_argnums=(3,) if donate_pools else ()`, where argument 3 is always
    `raw_pools: tuple[jax.Array, ...]`:
      1. States dict (`PallasWriter.make_step` / `make_step` compatible):
         `hid, states = step(pos, hidden, weights, states, [rbias])`
      2. Recurrent + attention positional:
         `hid, raw_pools = step(pos, hidden, weights, raw_pools, page_table,
                                ckpt_indices, read_offsets, [rbias], [active_seqs])`
      3. Attention-only (`not pool_backend.has_recurrent_pool`) positional:
         `hid, raw_pools = step(pos, hidden, weights, raw_pools, page_table,
                                [rbias], [active_seqs])`
    """
    ms = _resolve_model_spec(model_spec)
    pb = _require_pool_backend(ms)
    ex_paged = dataclasses.replace(ex, paged_kv=True)
    call, info = kernel.build(ms, ex_paged)
    p = info["plan"]
    w_names = p.w_names
    s_names = p.s_names
    o_names = p.o_names
    has_rec = pb.has_recurrent_pool
    rec_slots = pb.recurrent_slots_per_pool
    pool_key = pb.pool_state_name
    pt_key = pb.page_table_state_name
    max_pt = ex_paged.resolved_max_pages_per_seq

    def _run_local(
        pos: Any,
        hidden: Any,
        weights: dict[str, Any],
        raw_pools: tuple[jax.Array, ...],
        page_table: jax.Array,
        ckpt_indices: jax.Array | None,
        read_offsets: jax.Array | None,
        rb_tuple: tuple[Any, ...],
        active_seqs: Any,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        active_seqs = _unwrap_active_seqs(active_seqs)
        raw_6d = raw_pools[0].ndim == 6
        raw_in = tuple(r.squeeze(0) for r in raw_pools) if raw_6d else tuple(raw_pools)
        if page_table.ndim == 3:
            page_table = page_table.squeeze(0)
        if page_table.shape[-1] > max_pt:
            page_table = page_table[..., :max_pt]
        if ckpt_indices is not None and ckpt_indices.ndim == 4:
            ckpt_indices = ckpt_indices.squeeze(0)
        if read_offsets is not None and read_offsets.ndim == 3:
            read_offsets = read_offsets.squeeze(0)
        if (
            active_seqs is not None
            and not isinstance(active_seqs, str)
            and active_seqs.ndim == 2
        ):
            active_seqs = active_seqs.squeeze(0)

        w = [
            tuple(x[0] for x in weights[k])
            if isinstance(weights[k], tuple)
            else weights[k][0]
            for k in w_names
        ]

        gathered = (
            pb.gather_states(
                ex_paged,
                raw_in,
                ckpt_indices,
                read_offsets,
                active_seqs,
            )
            if (has_rec and pb.gather_states is not None)
            else {}
        )

        st_map: dict[str, Any] = {
            **gathered,
            pool_key: raw_in,
            pt_key: page_table,
        }
        s_args = [st_map[k] for k in s_names]
        out_vals = call(pos, hidden, *w, *s_args, *rb_tuple)
        if not isinstance(out_vals, (tuple, list)):
            out_vals = (out_vals,)
        out_map = dict(zip(o_names, out_vals))
        hid = out_map["hidden"]

        raw_out = pb.scatter_states(
            ex_paged,
            raw_in,
            pos,
            page_table,
            out_map,
            ckpt_indices if has_rec else None,
            active_seqs,
        )

        if raw_6d:
            raw_out = tuple(jnp.expand_dims(r, 0) for r in raw_out)
        return hid, raw_out

    def _meta_spec(arr: Any, ns_dim: int, max_unsharded_ndim: int = 2) -> P:
        if isinstance(arr, _ModeTag) or arr is None or ex_paged.R <= 1:
            return P()
        if arr.ndim > max_unsharded_ndim:
            return P("tp") if arr.shape[0] == ex_paged.R else P()
        return P("tp") if arr.shape[0] == ex_paged.R * ns_dim else P()

    def _raw_step(
        pos: Any,
        hidden: Any,
        weights: dict[str, Any],
        raw_pools: tuple[jax.Array, ...],
        page_table: jax.Array,
        ckpt_indices: jax.Array | None,
        read_offsets: jax.Array | None,
        active_wrapped: Any,
        *rb_tuple: Any,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        _validate_page_table_shape(page_table, ex_paged)
        ws = {
            k: tuple(P("tp") for _ in weights[k])
            if isinstance(weights[k], tuple)
            else P("tp")
            for k in weights
        }
        raw_spec = tuple(P("tp") for _ in raw_pools)
        rb_specs = (P("tp"),) if ex_paged.forced_routing else ()

        meta: dict[str, Any] = {pt_key: page_table}
        meta_specs: dict[str, Any] = {
            pt_key: _meta_spec(page_table, ex_paged.NS, max_unsharded_ndim=2)
        }
        if has_rec:
            meta["ckpt_indices"] = ckpt_indices
            meta["read_offsets"] = read_offsets
            meta_specs["ckpt_indices"] = _meta_spec(
                ckpt_indices,
                ex_paged.NS
                if (ckpt_indices is not None and ckpt_indices.ndim == 2)
                else rec_slots,
                max_unsharded_ndim=3,
            )
            meta_specs["read_offsets"] = _meta_spec(
                read_offsets,
                ex_paged.NS
                if (read_offsets is not None and read_offsets.ndim == 1)
                else rec_slots,
                max_unsharded_ndim=2,
            )
        if active_wrapped is not None:
            meta["active_seqs"] = active_wrapped
            meta_specs["active_seqs"] = _meta_spec(
                active_wrapped, ex_paged.NS, max_unsharded_ndim=1
            )

        def local_pos(
            pos_l: Any,
            hid_l: Any,
            w_l: Any,
            raw_l: Any,
            meta_l: dict[str, Any],
            *rb_l: Any,
        ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
            return _run_local(
                pos_l,
                hid_l,
                w_l,
                raw_l,
                meta_l[pt_key],
                meta_l.get("ckpt_indices"),
                meta_l.get("read_offsets"),
                rb_l,
                meta_l.get("active_seqs"),
            )

        return jax.shard_map(
            local_pos,
            mesh=mesh,
            in_specs=(P(), P("tp"), ws, raw_spec, meta_specs) + rb_specs,
            out_specs=(P("tp"), raw_spec),
            check_vma=False,
        )(pos, hidden, weights, raw_pools, meta, *rb_tuple)

    _jitted_step = jax.jit(
        _raw_step,
        donate_argnums=(3,) if donate_pools else (),
    )

    def _parse_call_args(
        states_or_raw: Any, rest: tuple[Any, ...]
    ) -> tuple[
        bool,
        dict[str, Any] | None,
        tuple[jax.Array, ...],
        jax.Array,
        jax.Array | None,
        jax.Array | None,
        Any,
        tuple[Any, ...],
    ]:
        if isinstance(states_or_raw, dict):
            states = states_or_raw
            raw_pools = tuple(states[pool_key])
            page_table = states[pt_key]
            ckpt_indices = states.get("ckpt_indices")
            read_offsets = states.get("read_offsets")
            active_seqs = _wrap_active_seqs(states.get("active_seqs"))
            rb_tuple = rest[:1] if ex_paged.forced_routing else ()
            return (
                True,
                states,
                raw_pools,
                page_table,
                ckpt_indices,
                read_offsets,
                active_seqs,
                rb_tuple,
            )

        raw_pools = tuple(states_or_raw)
        page_table = rest[0]
        if has_rec:
            ckpt_indices = rest[1]
            read_offsets = rest[2]
            tail = rest[3:]
        else:
            ckpt_indices = None
            read_offsets = None
            tail = rest[1:]

        if ex_paged.forced_routing:
            rb_tuple = (tail[0],)
            active_seqs = _wrap_active_seqs(tail[1] if len(tail) > 1 else None)
        else:
            rb_tuple = ()
            active_seqs = _wrap_active_seqs(tail[0] if len(tail) > 0 else None)

        return (
            False,
            None,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            active_seqs,
            rb_tuple,
        )

    def _call(
        pos: Any, hidden: Any, weights: dict[str, Any], states_or_raw: Any, *rest: Any
    ) -> Any:
        (
            is_dict,
            states_dict,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            active_seqs,
            rb_tuple,
        ) = _parse_call_args(states_or_raw, rest)
        hid, new_raw = _jitted_step(
            pos,
            hidden,
            weights,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            active_seqs,
            *rb_tuple,
        )
        if is_dict:
            assert states_dict is not None
            return hid, {**states_dict, pool_key: new_raw}
        return hid, new_raw

    def _lower(
        pos: Any, hidden: Any, weights: dict[str, Any], states_or_raw: Any, *rest: Any
    ) -> Any:
        (
            _,
            _,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            active_seqs,
            rb_tuple,
        ) = _parse_call_args(states_or_raw, rest)
        return _jitted_step.lower(
            pos,
            hidden,
            weights,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            active_seqs,
            *rb_tuple,
        )

    return _PooledStepFn(_call, _lower, info)


# ---------------------------------------------------------------------------
# CachePipe Hierarchy (`DirectHBM` & `PallasWriter`)
# ---------------------------------------------------------------------------
class CachePipe(abc.ABC):
    """Abstract base class for megakernel cache I/O pipelines."""

    @abc.abstractmethod
    def make_step(
        self,
        mesh: Any,
        g_or_spec: Any,
        ex: ExecSpec,
        *,
        donate: bool = False,
    ) -> Any:
        """Build a step callable `(step_fn, info)` for `(mesh, g_or_spec, ex)`."""


@dataclass(frozen=True)
class DirectHBM(CachePipe):
    """M1/M2 direct-HBM state cache pipe (`kc`/`vc`, `conv`/`rec` in HBM buffers)."""

    def make_step(
        self,
        mesh: Any,
        g_or_spec: Any,
        ex: ExecSpec,
        *,
        donate: bool = False,
        donate_states: bool | None = None,
    ) -> Any:
        use_donate = donate if donate_states is None else bool(donate_states)
        ms = _resolve_model_spec(g_or_spec)
        ex_direct = dataclasses.replace(ex, paged_kv=False)
        return kernel.make_step(mesh, ms, ex_direct, donate_states=use_donate)


@dataclass(frozen=True)
class PallasWriter(CachePipe):
    """M3 pooled cache pipe (`gather_states` -> `megakernel` -> `scatter_states`)."""

    def make_step(
        self,
        mesh: Any,
        g_or_spec: Any,
        ex: ExecSpec,
        *,
        donate: bool = False,
        donate_pools: bool | None = None,
    ) -> Any:
        use_donate = donate if donate_pools is None else bool(donate_pools)
        ms = _resolve_model_spec(g_or_spec)
        return make_pooled_step(mesh, ms, ex, donate_pools=use_donate)


__all__ = [
    "CachePipe",
    "DirectHBM",
    "PallasWriter",
    "check_pooled_step_hlo",
    "make_flat_pooled_step",
    "make_pooled_step",
]
