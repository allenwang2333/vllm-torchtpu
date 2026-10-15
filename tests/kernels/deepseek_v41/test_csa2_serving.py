# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for the actual serving cache-to-CSA2 bridge."""

import importlib.util
import itertools
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "csa2_oracle", Path(__file__).with_name("test_csa2.py")
)
assert _spec is not None and _spec.loader is not None
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


def serving_inputs(inputs):
    """Encode native BF16 SWA records, including invalid cache overlay rows."""
    positions = np.asarray(inputs["positions"])
    lower, upper = np.asarray(inputs["swa_lower"]), np.asarray(inputs["swa_upper"])
    starts = np.asarray(inputs["starts"])
    swa = np.asarray(inputs["swa_kv"])
    width = 192 if swa.shape[1] == 256 else 128
    records = np.full((len(positions), width), -1, np.int32)
    rows = []
    for begin, end in zip(starts[:-1], starts[1:]):
        addresses = {}
        for t in range(begin, end):
            for column in range(width):
                logical = positions[t] - 127 + column if width == 128 else column
                if lower[t] <= logical < upper[t]:
                    if logical not in addresses:
                        addresses[logical] = len(rows)
                        base = max(0, int(positions[t]) - 127) if width == 128 else 0
                        rows.append(swa[t, logical - base])
                    records[t, column] = addresses[logical]
    rows.extend(np.full((32, 512), np.nan, dtype=jnp.bfloat16))
    bits = np.asarray(rows, dtype=jnp.bfloat16).view(np.uint16).astype(np.uint32)
    pool = (bits[:, ::2] | (bits[:, 1::2] << 16)).view(np.int32).reshape(-1, 128)
    counts = np.asarray(inputs["main_lengths"])
    main_records = np.where(
        np.arange(512)[None, :] < counts[:, None], np.arange(512)[None, :], -1
    )
    main_records[int(starts[-1]) :] = -1
    return dict(
        q=inputs["q"],
        swa_pool=jnp.asarray(pool),
        main_pool=inputs["main_kv"].reshape(-1, 128),
        rope_pool=inputs["main_rope"].reshape(-1, 128),
        swa_records=jnp.asarray(records),
        main_records=jnp.asarray(main_records, jnp.int32),
        pages=jnp.zeros((len(starts) - 1, 1), jnp.int32),
        query_start_loc=inputs["starts"],
        sink=inputs["sink"],
        scale=512**-0.5,
        page_size=4,
        max_requests=max(1, int(np.sum(np.diff(starts) > 0))),
        precompacted=True,
        gathered_nope=inputs["main_kv"],
        gathered_rope=inputs["main_rope"],
        positions=inputs["positions"],
    )


class Csa2ServingTest(unittest.TestCase):
    @unittest.skipUnless(
        jax.default_backend() == "tpu" and len(jax.local_devices()) == 1,
        "TorchTPU serving uses one local TPU per worker",
    )
    def test_torch_tp1_wrapper(self):
        """Run the real TorchTPU custom op, including its optional position input."""
        import torch
        import torch_tpu  # noqa: F401
        from jax.sharding import Mesh

        from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels as ak
        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        def to_torch(value):
            array = np.asarray(value)
            if value.dtype == jnp.bfloat16:
                return torch.from_numpy(array.astype(np.float32)).to(
                    device="tpu", dtype=torch.bfloat16
                )
            return torch.from_numpy(array.copy()).to("tpu")

        inputs, dense = oracle._packed_inputs(noncausal=False, heads=64)
        args = serving_inputs(inputs)
        data = {
            key: to_torch(value)
            for key, value in args.items()
            if isinstance(value, jax.Array)
        }
        swa = SimpleNamespace(raw=data["swa_pool"], raw_hi=None)
        main = SimpleNamespace(
            raw=data["main_pool"],
            raw_hi=None,
            rope=data["rope_pool"],
            rope_hi=None,
            shape=(1, args["page_size"]),
        )
        mesh = Mesh(np.asarray(jax.local_devices()), ("tp",))
        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch.object(
                ak,
                "_gather_fp8_for_stage",
                return_value=(data["gathered_nope"], data["gathered_rope"]),
            ),
            mock.patch.object(
                ak, "_csa2_attention_jax", wraps=ak._csa2_attention_jax
            ) as native,
        ):
            ak.initialize_paged_attention()
            actual = (
                ak.sparse_fp8_attention(
                    data["q"],
                    swa,
                    main,
                    data["swa_records"],
                    data["main_records"],
                    data["sink"],
                    args["scale"],
                    data["pages"],
                    data["query_start_loc"],
                    args["max_requests"],
                    precompacted=True,
                    positions=data["positions"],
                )
                .cpu()
                .float()
                .numpy()
            )
            self.assertGreater(native.call_count, 0)
        np.testing.assert_allclose(
            actual, oracle._packed_reference(inputs, dense), atol=2e-3, rtol=2e-2
        )

    def test_tp1_registered_attention_bridge(self):
        """TP1 bypasses local-Q fusion; exercise its registered JAX body too."""
        from jax.sharding import Mesh

        from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels as ak
        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        mesh = Mesh(np.asarray(jax.devices()[:1]), ("tp",))
        registered = {}

        def capture(name, function, **kwargs):
            registered[name] = function
            return mock.Mock()

        group = mock.Mock()
        group.size.return_value = 1
        with (
            mock.patch.multiple(
                ak,
                _attention_op=None,
                _attention_2bank_op=None,
                _fp8_attention_op=None,
                _fp8_attention_2bank_op=None,
                _fp8_attention_qo_op=None,
                _fp8_attention_qo_2bank_op=None,
                _gather_bf16_op=None,
                _gather_fp8_op=None,
                _attention_tp_size=1,
            ),
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch(
                "vllm.distributed.get_tp_group",
                return_value=SimpleNamespace(device_group=group),
            ),
            mock.patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
            mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=capture),
        ):
            ak.initialize_paged_attention()

        for noncausal, two_banks in itertools.product((False, True), repeat=2):
            inputs, dense = oracle._packed_inputs(noncausal=noncausal, heads=64)
            args = serving_inputs(inputs)
            name = "pallas::dsv41v2_sparse_fp8_attention"
            pools = [args[key] for key in ("swa_pool", "main_pool", "rope_pool")]
            if two_banks:
                name += "_2bank"
                pools = [part for pool in pools for part in jnp.split(pool, 2)]
            function = registered[name]
            with mock.patch.object(
                ak, "_csa2_attention_jax", wraps=ak._csa2_attention_jax
            ) as native:
                actual = function(
                    args["q"],
                    *pools,
                    args["swa_records"],
                    args["main_records"],
                    args["pages"],
                    args["query_start_loc"],
                    args["sink"],
                    args["sink"],
                    jnp.zeros((1,), jnp.int32),
                    args["scale"],
                    args["page_size"],
                    args["max_requests"],
                    True,
                    False,
                    False,
                    4096,
                    args["gathered_nope"],
                    args["gathered_rope"],
                    args["positions"],
                )
                self.assertGreater(native.call_count, 0)
            np.testing.assert_allclose(
                np.asarray(actual).astype(np.float32),
                oracle._packed_reference(inputs, dense),
                atol=2e-3,
                rtol=2e-2,
            )

    def test_native_cache_bridge(self):
        from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
            sparse_fp8_attention_jax,
        )

        for noncausal in (False, True):
            with self.subTest(noncausal=noncausal):
                inputs, dense = oracle._packed_inputs(noncausal=noncausal, heads=64)
                args = serving_inputs(inputs)
                arrays = {k: v for k, v in args.items() if isinstance(v, jax.Array)}
                static = {k: v for k, v in args.items() if k not in arrays}
                run = jax.jit(
                    lambda data, static=static: sparse_fp8_attention_jax(
                        **data, **static
                    )
                )
                actual = np.asarray(run(arrays)).astype(np.float32)
                expected = oracle._packed_reference(inputs, dense)
                np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-2)
                self.assertLess(
                    np.linalg.norm(actual - expected) / np.linalg.norm(expected), 0.006
                )


if __name__ == "__main__":
    unittest.main()
