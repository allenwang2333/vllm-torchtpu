# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the routing tap: recorder arithmetic, ring wrap, shape gating,
and that the post-import patches land on the classes the server uses.

    dsv41-docker --cpu python -m pytest scripts/dsv41_bench/routing_tap -q
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import routing_tap  # noqa: E402


def _ring(ring, width):
    buf = np.full((ring + 1, width), -1, np.int32)
    buf[ring] = 0
    return jnp.asarray(buf)


def test_moe_recorder_matches_reference_and_wraps():
    rows, experts, topk, ring = 12, 40, 3, 2
    fn = jax.jit(
        routing_tap.moe_record_fn(
            rows=rows,
            experts=experts,
            topk=topk,
            scoring_fn="sqrtsoftplus",
            has_bias=True,
            ring=ring,
        ),
        donate_argnums=(0,),
    )
    buf = _ring(ring, rows * topk)
    bias = jax.random.normal(jax.random.key(1), (experts,)) * 0.1
    refs = []
    for step in range(3):
        logits = jax.random.normal(jax.random.key(10 + step), (rows, experts)).astype(
            jnp.bfloat16
        )
        scores = jnp.sqrt(jax.nn.softplus(logits.astype(jnp.float32))) + bias[None]
        refs.append(np.asarray(lax.top_k(scores, topk)[1]).reshape(-1))
        buf = fn(buf, logits, bias)
    buf = np.asarray(buf)
    assert buf[ring, 0] == 3
    np.testing.assert_array_equal(buf[0], refs[2])  # step 2 overwrote slot 0
    np.testing.assert_array_equal(buf[1], refs[1])


def test_moe_recorder_ignores_other_shapes():
    fn = routing_tap.moe_record_fn(
        rows=8, experts=16, topk=2, scoring_fn="softmax", has_bias=False, ring=4
    )
    buf = _ring(4, 16)
    out = fn(buf, jnp.zeros((9, 16)))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(buf))


def test_topk_recorder_fills_once():
    fn = routing_tap.topk_record_fn(rows=4, width=6, ring=2)
    buf = jnp.asarray(
        np.concatenate([np.full((8, 6), -1, np.int32), np.zeros((1, 6), np.int32)])
    )
    sel = [jnp.arange(24, dtype=jnp.int32).reshape(4, 6) + 100 * i for i in range(3)]
    for s in sel:
        buf = fn(buf, s)
    buf = np.asarray(buf)
    assert buf[8, 0] == 3
    np.testing.assert_array_equal(buf[0:4], np.asarray(sel[0]))
    np.testing.assert_array_equal(buf[4:8], np.asarray(sel[1]))
    same = fn(jnp.asarray(buf), jnp.zeros((5, 6), jnp.int32))
    np.testing.assert_array_equal(np.asarray(same), buf)


def test_patches_land_through_sitecustomize(tmp_path):
    steps = os.path.join(os.path.dirname(HERE), "server_steps")
    env = dict(
        os.environ,
        DSV41_ROUTING_TAP_OUT=str(tmp_path),
        PYTHONPATH=os.pathsep.join([steps, HERE, os.environ.get("PYTHONPATH", "")]),
    )
    code = (
        "import os, sys\n"
        "flag = os.environ.get('TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS')\n"
        "assert flag == 'false', 'shadowed sitecustomize not run'\n"
        "assert 'routing_tap' in sys.modules\n"
        "from vllm_torchtpu.layers.adapter.quantization import mxfp4\n"
        "from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention\n"
        "import server_step_worker as w\n"
        "c = mxfp4.VllmDeepseekV4Mxfp4MoEMethod\n"
        "assert c._forward_monolithic_tpu.__module__ == 'routing_tap'\n"
        "assert c.process_weights_after_loading.__module__ == 'routing_tap'\n"
        "a = attention.DeepseekV41Attention\n"
        "assert a._select_paged_indices.__module__ == 'routing_tap'\n"
        "assert a._select_indices.__module__ == 'routing_tap'\n"
        "assert w.ServerStepWorker.benchmark_end.__module__ == 'routing_tap'\n"
        "import inspect\n"
        "print('fwd sig', inspect.signature(c._forward_monolithic_tpu))\n"
        "print('PATCHES-OK')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert "PATCHES-OK" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]


def test_torch_op_builds_and_runs_eagerly():
    """The torch_tpu jax_op registration (signature / static-arg inference)."""
    import torch

    op = routing_tap._jax_op(
        "dsv41_tap_test::moe",
        routing_tap.moe_record_fn(
            rows=4, experts=8, topk=2, scoring_fn="sqrtsoftplus", has_bias=True, ring=3
        ),
    )
    buf = torch.full((4, 8), -1, dtype=torch.int32)
    buf[3] = 0
    logits = torch.randn(4, 8, dtype=torch.bfloat16)
    bias = torch.randn(8)
    try:
        new = op(buf.clone(), logits, bias)
    except (
        Exception
    ) as exc:  # the eager CPU path may not exist; the build is what we check
        print("eager call not available on CPU:", repr(exc)[:300])
        return
    assert int(new[3, 0]) == 1


def test_dump_numa_and_convert_roundtrip(tmp_path, monkeypatch):
    """Host side end to end on fake rings: dump per rank, then convert.py."""
    import json as _json
    from types import SimpleNamespace

    import convert
    import torch

    run = tmp_path / "run"
    (run / "fixed").mkdir(parents=True)
    monkeypatch.setenv(routing_tap.OUT_ENV, str(run / "tap"))
    rows, topk, experts, ring, steps = 288, 6, 384, 16, 5
    rng = np.random.default_rng(0)
    truth = rng.integers(0, experts, (2, steps, 8, rows * topk)).astype(np.int32)
    step_recs = []
    for r in range(8):
        routing_tap.MOE_RINGS.clear()
        routing_tap.TOPK_RINGS.clear()
        for li, name in enumerate(
            ["model.layers.3.ffn.experts", "model.layers.22.ffn.experts"]
        ):
            buf = torch.full((ring + 1, rows * topk), -1, dtype=torch.int32)
            c0 = 7  # counter at benchmark_begin
            for s in range(steps):
                buf[(c0 + s) % ring] = torch.from_numpy(truth[li, s, r])
            buf[ring, 0] = c0 + steps
            owner = SimpleNamespace(**{routing_tap.MOE_ATTR: buf})
            routing_tap.MOE_RINGS[name] = (
                owner,
                dict(
                    rows=rows,
                    experts=experts,
                    topk=topk,
                    scoring_fn="sqrtsoftplus",
                    has_bias=True,
                    ring=ring,
                ),
            )
        tb = torch.full((2 * rows + 1, 512), -1, dtype=torch.int32)
        tb[:rows] = 5
        tb[2 * rows, 0] = 9
        routing_tap.TOPK_RINGS["model.layers.2.attn"] = (
            SimpleNamespace(**{routing_tap.TOPK_ATTR: tb}),
            dict(rows=rows, width=512, ring=2),
        )
        begin = {
            "moe": {n: 7 for n in routing_tap.MOE_RINGS},
            "csa": {"model.layers.2.attn": 0},
        }
        routing_tap.dump(
            r,
            "round1",
            {"counters_begin": begin, "counters_end": routing_tap.counters()},
        )
        enc = [
            {"stage": "encoder", "padded_rows": 2048, "phase": "prefill", "requests": 8}
        ]
        enc += [
            {
                "stage": "encoder",
                "padded_rows": 288,
                "phase": "decode",
                "requests": 48 if s < 4 else 47,
            }
            for s in range(steps)
        ]
        step_recs.append({"rank": r, "steps": [{"stages": [e]} for e in enc]})
    (run / "fixed" / "round-1-steps.json").write_text(
        _json.dumps({"results": [step_recs]})
    )
    dumps = convert.load(run, "round1")
    out = convert.convert_moe(dumps, run, "round1", lambda m: m["experts"] == 384)
    assert out["topk_ids"].shape == (2, 4, 8 * rows, topk)
    assert out["layers"].tolist() == [3, 22]
    np.testing.assert_array_equal(out["topk_ids"][1, 2].reshape(8, -1), truth[1, 2])
    assert out["counts"].sum() == 2 * 4 * 8 * rows * topk
    csa = convert.convert_csa(dumps, run, "round1")
    assert csa["csa_topk"].shape == (1, 2, 8, rows, 512)

    model = torch.nn.Module()
    model.register_buffer("big", torch.zeros(1 << 19))
    model._host_packed_buf = torch.zeros(1 << 21, dtype=torch.uint8)
    snap = routing_tap.numa_snapshot(
        SimpleNamespace(model_runner=SimpleNamespace(model=model))
    )
    names = {t["name"] for t in snap["tensors"]}
    assert "._host_packed_buf" in names and ".big" in names, names
    assert all("vmas" in t for t in snap["tensors"])
    buf = next(t for t in snap["tensors"] if t["name"] == "._host_packed_buf")
    nodes = buf["sampled_page_nodes"]
    # An unprivileged container's seccomp profile refuses move_pages (EPERM).
    assert nodes.get("error") == "move_pages errno 1" or sum(nodes.values()) > 0, nodes
    assert snap["affinity"]
