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
"""Can per-layer tensors be views of one stacked buffer on torch_tpu?

    python -m tests.kernels.deepseek_v4_megakernel.view_operand_probe

One TPU device. A base int32 [L=4, P, 1024, 128] (1 GiB per layer slice);
layer views `base[l]` (storage offset l GiB). Each probe runs in its own
process with its own XLA dump and prints one RESULT line:

  p1_read        eager Pallas kernel reads a page of base[2] (offset view)
  p1_inplace     eager Pallas kernel DMA-writes a page of base[2] in place
  p1_time        per-call time of p1_read / p1_inplace: view vs own buffer
  p1_base        the in-place kernel on the whole base (writes layer 2)
  p1_alias_eager stock pattern, eager: donated + aliased kernel, view.copy_
  p2_plain       torch.compile graph writing pages of 3 views of one base
  p2_alias       torch.compile graph: stock jax_op (donated, aliased) on 2
                 views of one base, each copied back (the DSv4 KV pattern)
  p3_fp4         u8 raw stack -> base[l].view(float4_e2m1fn_x2) Parameter
                 read by a JAX op
  p4_readonly    torch.compile graph passing 2 views (read only, like stock
                 per-layer weights) to a jax_op: time vs own buffers
  p5_fp4         routed-expert-shaped fp4 operand [24, 7168, 3072] x2: a whole
                 u8 tensor viewed as float4_e2m1fn_x2 (offset 0), vs the u8
                 tensor itself, vs an fp4 tensor a JAX op produced: torch_tpu
                 layout report, in-place DMA write lands, per-call time

Writes are checked page by page against the pre-call pages (other layers
must be untouched). The parent lists every optimized-HLO instruction whose
result has the layer-slice or the base shape (a materialized view or a
cache-sized copy shows up there).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

L, P, R, C = 4, 2048, 1024, 128  # int32: one layer slice = 1 GiB
LAYER = 2
SRC, DST = 5, 9
PROBES = (
    "p1_read",
    "p1_inplace",
    "p1_time",
    "p1_base",
    "p1_alias_eager",
    "p2_plain",
    "p2_alias",
    "p3_fp4",
    "p4_readonly",
    "p5_fp4",
)


def _kernels():
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    anyspace = pl.BlockSpec(memory_space=pl.ANY)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    dma = [pltpu.SemaphoreType.DMA(())]
    tile = jax.ShapeDtypeStruct((8, 128), jnp.int32)

    def _dma(src, dst, sem):
        cp = pltpu.make_async_copy(src, dst, sem)
        cp.start()
        cp.wait()

    def read(x: jax.Array) -> jax.Array:
        def k(x_hbm, o_ref, sem):
            _dma(x_hbm.at[SRC, pl.ds(0, 8)], o_ref, sem)

        return pl.pallas_call(
            k, in_specs=[anyspace], out_specs=vmem, out_shape=tile,
            scratch_shapes=dma, name="probe_read",
        )(x)  # fmt: skip

    def _page_copy(at):
        def k(x_hbm, o_ref, sem):
            _dma(at(x_hbm, SRC), at(x_hbm, DST), sem)
            o_ref[...] = jnp.ones((8, 128), jnp.int32)

        return k

    def inplace(x: jax.Array) -> jax.Array:
        return pl.pallas_call(
            _page_copy(lambda r, p: r.at[p]), in_specs=[anyspace], out_specs=vmem,
            out_shape=tile, scratch_shapes=dma, name="probe_inplace",
        )(x)  # fmt: skip

    def inplace_base(x: jax.Array) -> jax.Array:
        return pl.pallas_call(
            _page_copy(lambda r, p: r.at[LAYER, p]), in_specs=[anyspace],
            out_specs=vmem, out_shape=tile, scratch_shapes=dma,
            name="probe_inplace_base",
        )(x)  # fmt: skip

    def aliased(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        def k(x_hbm, o_ref, y_hbm, sem):
            _dma(y_hbm.at[SRC], y_hbm.at[DST], sem)
            o_ref[...] = jnp.ones((8, 128), jnp.int32)

        return pl.pallas_call(
            k, in_specs=[anyspace], out_specs=[vmem, anyspace],
            out_shape=[tile, jax.ShapeDtypeStruct(x.shape, x.dtype)],
            input_output_aliases={0: 1}, scratch_shapes=dma, name="probe_aliased",
        )(x)  # fmt: skip

    def read_base(x: jax.Array) -> jax.Array:
        def k(x_hbm, o_ref, sem):
            _dma(x_hbm.at[LAYER, SRC, pl.ds(0, 8)], o_ref, sem)

        return pl.pallas_call(
            k, in_specs=[anyspace], out_specs=vmem, out_shape=tile,
            scratch_shapes=dma, name="probe_read_base",
        )(x)  # fmt: skip

    def identity(x: jax.Array) -> jax.Array:
        return x + jnp.zeros((), x.dtype)

    return dict(read=read, read_base=read_base, inplace=inplace,
                inplace_base=inplace_base, aliased=aliased,
                identity=identity)  # fmt: skip


def _sync(t):
    return t.reshape(-1)[:1].cpu()


def _base(torch):
    n = L * P * R * C
    x = torch.arange(n, dtype=torch.int32, device="tpu")
    return (x * 1103515245 + 12345).reshape(L, P, R, C)


def _pages(base):
    return {(li, p): base[li, p].cpu() for li in range(L) for p in (SRC, DST)}


def _landed(torch, base, before, layers):
    """`layers` had page DST <- page SRC; every other page is unchanged."""
    ok = True
    for li in range(L):
        want_dst = before[(li, SRC)] if li in layers else before[(li, DST)]
        ok &= torch.equal(base[li, DST].cpu(), want_dst)
        ok &= torch.equal(base[li, SRC].cpu(), before[(li, SRC)])
    return bool(ok)


def _timed(fn, n=30):
    _sync(fn())
    t0 = time.perf_counter()
    for _ in range(n):
        _sync(fn())
    return (time.perf_counter() - t0) / n * 1e6


def run_probe(name: str) -> dict:
    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import pallas as tp
    from torch_tpu._internal.pallas import pallas as tpp

    K = _kernels()

    def eager(fn, donate=()):
        return tpp.custom_jax_kernel(
            fn, f"probe_{name}::{fn.__name__}", donate_argnums=list(donate),
            static_argnums=(), warn_deprecated=False,
        )  # fmt: skip

    if name == "p5_fp4":
        return _fp4_probe(torch, eager)
    base = _base(torch)
    view = base[LAYER]
    out: dict = dict(view_offset=view.storage_offset(), contiguous=view.is_contiguous())
    before = _pages(base)
    if name == "p1_read":
        got = eager(K["read"])(view).cpu()
        out["read_correct"] = bool(torch.equal(got, before[(LAYER, SRC)][:8]))
        out["read_is_layer0"] = bool(torch.equal(got, before[(0, SRC)][:8]))
    elif name == "p1_inplace":
        _sync(eager(K["inplace"])(view))
        out["writes_land"] = _landed(torch, base, before, {LAYER})
        out["view_sees_write"] = bool(
            torch.equal(view[DST].cpu(), before[(LAYER, SRC)])
        )
    elif name == "p1_time":
        own = torch.empty((P, R, C), dtype=torch.int32, device="tpu")
        rd, ip = eager(K["read"]), eager(K["inplace"])
        out["read_us_view"] = _timed(lambda: rd(view))
        out["read_us_own"] = _timed(lambda: rd(own))
        out["inplace_us_view"] = _timed(lambda: ip(view))
        out["inplace_us_own"] = _timed(lambda: ip(own))
        out["read_us_base0_view"] = _timed(lambda: rd(base[0]))
        rb = eager(K["read_base"])
        out["read_us_whole_base"] = _timed(lambda: rb(base))
    elif name == "p1_base":
        _sync(eager(K["inplace_base"])(base))
        out["writes_land"] = _landed(torch, base, before, {LAYER})
    elif name == "p1_alias_eager":
        new = eager(K["aliased"], donate=(0,))(view)[1]
        view.copy_(new)
        _sync(view)
        out["writes_land"] = _landed(torch, base, before, {LAYER})
    elif name == "p2_plain":

        def f(v0, v1, v2):
            v0[DST] = v0[SRC]
            v1[DST] = v1[SRC]
            v2[DST] = v2[SRC]

        g = torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
        g(base[0], base[1], base[3])
        _sync(base)
        out["writes_land"] = _landed(torch, base, before, {0, 1, 3})
        own = [
            torch.empty((P, R, C), dtype=torch.int32, device="tpu") for _ in range(3)
        ]
        out["us_views"] = _timed(lambda: g(base[0], base[1], base[3]) or base)
        out["us_own"] = _timed(lambda: g(*own) or own[0])
    elif name == "p2_alias":
        op = tp.jax_op(f"probe_{name}::aliased", K["aliased"], donate_argnums=(0,))

        def f(v0, v1):
            n0 = op(v0)[1]
            v0.copy_(n0)
            n1 = op(v1)[1]
            v1.copy_(n1)

        g = torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
        g(base[1], base[2])
        _sync(base)
        out["writes_land"] = _landed(torch, base, before, {1, 2})
        own = [
            torch.empty((P, R, C), dtype=torch.int32, device="tpu") for _ in range(2)
        ]
        out["us_views"] = _timed(lambda: g(base[1], base[2]) or base)
        out["us_own"] = _timed(lambda: g(*own) or own[0])
    elif name == "p4_readonly":
        op = tp.jax_op(f"probe_{name}::read", K["read"])

        def f(v0, v1):
            return op(v0) + op(v1)

        g = torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
        got = g(base[1], base[2]).cpu()
        out["read_correct"] = bool(
            torch.equal(got, before[(1, SRC)][:8] + before[(2, SRC)][:8])
        )
        own = [
            torch.empty((P, R, C), dtype=torch.int32, device="tpu") for _ in range(2)
        ]
        out["us_views"] = _timed(lambda: g(base[1], base[2]))
        out["us_own"] = _timed(lambda: g(*own))
    elif name == "p3_fp4":
        nbytes = 1 << 20
        raw = (torch.arange(L * nbytes, device="tpu", dtype=torch.int32) * 7 % 251).to(
            torch.uint8
        )
        raw = raw.reshape(L, nbytes)
        v = raw[LAYER].view(torch.float4_e2m1fn_x2).reshape(1024, 1024)
        param = torch.nn.Parameter(v, requires_grad=False)
        out["param_is_view"] = param.data_ptr() == v.data_ptr()
        got = eager(K["identity"])(param)
        out["dtype_roundtrip"] = str(got.dtype)
        out["fp4_bytes_correct"] = bool(
            torch.equal(got.view(torch.uint8).reshape(-1).cpu(), raw[LAYER].cpu())
        )
    else:
        raise ValueError(name)
    return out


def _fp4_probe(torch, eager) -> dict:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from torch_tpu._internal.compile import tpu_torch_compile as ttc

    E, H, I2 = 24, 7168, 3072  # served w13: [E, H, I] fp4 pairs  # noqa: N806
    n = E * H * I2
    u8 = (torch.arange(n, dtype=torch.int32, device="tpu") * 7 % 251).to(torch.uint8)
    u8 = u8.reshape(E, H, I2)

    def to_fp4(x: jax.Array) -> jax.Array:
        return x.view(jnp.float4_e2m1fn)

    made = eager(to_fp4)(u8.clone())  # an fp4 tensor as a JAX op returns one
    view = u8.view(torch.float4_e2m1fn_x2)

    def copy_rows(x: jax.Array) -> jax.Array:  # in place: expert 1 rows <- expert 0
        def k(x_hbm, o_ref, sem):
            cp = pltpu.make_async_copy(x_hbm.at[0, pl.ds(0, 64)],
                                       x_hbm.at[1, pl.ds(0, 64)], sem)  # fmt: skip
            cp.start()
            cp.wait()
            o_ref[...] = jnp.ones((8, 128), jnp.int32)

        return pl.pallas_call(
            k, in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
            scratch_shapes=[pltpu.SemaphoreType.DMA(())], name="probe_fp4_rows",
        )(x)  # fmt: skip

    out = {}
    for label, t in (("u8", u8), ("fp4_view", view), ("fp4_made", made)):
        lay = ttc.get_device_layout_if_materialized(t)
        out[f"{label}_layout"] = None if lay is None else repr(lay)
        out[f"{label}_default"] = repr(ttc.get_default_layout(t.dtype, t.shape))
        as_u8 = t if t.dtype == torch.uint8 else t.view(torch.uint8)
        before0 = as_u8[0, :64].cpu()
        call = eager(copy_rows)
        _sync(call(t))
        out[f"{label}_write_lands"] = bool(torch.equal(as_u8[1, :64].cpu(), before0))
        out[f"{label}_us"] = _timed(lambda c=call, x=t: c(x))
    return out


def _hlo_shapes(dump: Path) -> list[str]:
    """Optimized-HLO instructions producing a layer-slice / base shaped s32."""
    want = (f"s32[{P},{R},{C}]", f"s32[{L},{P},{R},{C}]", f"s32[{L * P},{R},{C}]")
    hits = []
    for f in sorted(dump.glob("*after_optimizations.txt")) if dump.exists() else []:
        mod = f.name.split(".")[1] if "." in f.name else f.name
        for ln in f.read_text(errors="replace").splitlines():
            m = re.match(r"\s*(?:ROOT )?%\S+ = (\S+?)(\{[^}]*\})? ([\w-]+)\(", ln)
            if m and m.group(1) in want and m.group(3) not in ("parameter",):
                hits.append(f"{mod}:{m.group(3)}:{m.group(1)}")
    return hits


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--probe")
    p.add_argument("--out", type=Path, default=Path("/tmp/view_probe"))
    p.add_argument("--only", default=",".join(PROBES))
    a = p.parse_args()
    if a.probe:
        try:
            res = run_probe(a.probe)
        except BaseException as exc:  # noqa: BLE001 - reported to the parent
            traceback.print_exc()
            res = {"error": repr(exc)[:400]}
        (a.out / f"{a.probe}.json").write_text(json.dumps(res))
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.only.split(","):
        dump = a.out / f"xla_{name}"
        env = dict(os.environ)
        env["XLA_FLAGS"] = (
            env.get("XLA_FLAGS", "") + f" --xla_dump_to={dump} --xla_dump_hlo_as_text"
        )
        me = "tests.kernels.deepseek_v4_megakernel.view_operand_probe"
        cmd = [sys.executable, "-m", me, "--probe", name, "--out", str(a.out)]
        code = subprocess.run(cmd, env=env, check=False, timeout=1200).returncode
        f = a.out / f"{name}.json"
        res = json.loads(f.read_text()) if f.exists() else {"error": f"exit {code}"}
        hits = _hlo_shapes(dump)
        res["hlo_slice_or_base_ops"] = {h: hits.count(h) for h in sorted(set(hits))}
        print("RESULT", name, json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
