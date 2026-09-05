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
"""GLM-5.3-Flash KDA parity against the HuggingFace reference.

GLM-5.3-Flash's linear-attention layers are Kimi Delta Attention, the same
recurrence this repo already implements for Kimi-K3. This test pins that claim
to numbers: it replays activations captured from a real
``transformers.Glm5NextForConditionalGeneration`` forward through
``vllm_torchtpu``'s KDA kernels and diffs the result.

The oracle is produced by ``scripts/glm5_next/dump_reference.py`` against the
tiny fixture from ``scripts/glm5_next/build_tiny_fixture.py``. Both run on CPU;
so does this test, which is the point -- it catches KDA porting bugs without a
TPU queue wait.

Boundaries that are easy to get wrong and are each asserted here:

* ``mixed_qkv`` enters the op **before** the short convolution, laid out
  ``(3, H, D)`` row-major over q, k, v.
* ``raw_gate`` is the bare ``f_b_proj`` output; the decay activation
  ``lower_bound * sigmoid(exp(A_log) * (raw + dt_bias))`` happens inside.
* ``beta`` is the bare ``b_proj`` output; the sigmoid happens inside. Applying
  it in the model too is a silent double-activation bug.
* The op's output is already gated-normalised, i.e. it corresponds to HF's
  ``o_norm(core_attn_out, gate)``, not to the raw recurrence output.

Run:
    PYTHONPATH=src:$VLLM:$GLM5_NEXT_TRANSFORMERS_SRC \\
    GLM5_NEXT_ORACLE=/path/to/oracle.npz pytest tests/kernels/test_glm5_next_kda_parity.py
"""

import os
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from vllm_torchtpu.kernels.kimi_k3.kda import ragged_kda  # noqa: E402

_DEFAULT_ORACLE = pathlib.Path(
    os.environ.get(
        "GLM5_NEXT_ORACLE",
        pathlib.Path.home() / "data/glm5_next_tiny/oracle.npz",
    ))


@pytest.fixture(scope="module")
def oracle():
    if not _DEFAULT_ORACLE.exists():
        pytest.skip(f"no glm5_next oracle at {_DEFAULT_ORACLE}; build it with "
                    "scripts/glm5_next/build_tiny_fixture.py + "
                    "scripts/glm5_next/dump_reference.py")
    return np.load(_DEFAULT_ORACLE)


def _causal_depthwise_conv(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Match HF's ``causal_conv1d_fn``: left-pad by K-1, keep the first T rows.

    ``x`` is ``[T, C]`` and ``weight`` is ``[C, K]``. HF spells this as
    ``F.conv1d(..., padding=K-1, groups=C)[:, :, :seq_len]``, which is the
    standard causal form -- tap ``j`` reads ``x[t - (K-1) + j]``.
    """
    num_tokens, channels = x.shape
    kernel = weight.shape[-1]
    padded = np.concatenate([np.zeros((kernel - 1, channels), x.dtype), x],
                            axis=0)
    out = np.zeros_like(x)
    for j in range(kernel):
        out += padded[j:j + num_tokens] * weight[:, j]
    return out


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _kda_layer_ids(oracle) -> list[int]:
    return [int(i) for i in oracle["kda_layers"]]


def _run_ragged_kda(oracle, layer: int):
    """Drive `ragged_kda` with one KDA layer's captured activations."""
    num_heads = int(oracle["cfg.linear_num_heads"])
    head_dim = int(oracle["cfg.linear_head_dim"])
    lower_bound = float(oracle["cfg.lower_bound"])
    eps = float(oracle["cfg.rms_norm_eps"])

    tag = f"kda{layer:02d}"
    # HF captures are [1, T, ...]; the op is flat over tokens.
    q = oracle[f"{tag}.q"][0]
    k = oracle[f"{tag}.k"][0]
    v = oracle[f"{tag}.v"][0]
    num_tokens = q.shape[0]

    # Concatenate in q,k,v order, convolve, SiLU -- exactly what the module
    # does before handing the result to the kernel.
    mixed = np.concatenate([q, k, v], axis=-1).astype(np.float32)
    conv_w = oracle[f"{tag}.w.conv1d"]  # [3*H*D, 1, K]
    mixed = _silu(_causal_depthwise_conv(mixed, conv_w[:, 0, :]))

    raw_gate = oracle[f"{tag}.f_b_out"][0].astype(np.float32)
    beta_raw = oracle[f"{tag}.beta_raw"][0].astype(np.float32)
    out_gate = oracle[f"{tag}.out_gate"][0].astype(np.float32)

    # One request, no carried-in state: seq_lens == query_lens.
    state = jnp.zeros((2, num_heads, head_dim, head_dim), jnp.float32)
    output, new_state = ragged_kda(
        jnp.asarray(mixed),
        jnp.asarray(raw_gate),
        jnp.asarray(beta_raw),
        jnp.asarray(out_gate),
        state,
        jnp.asarray(oracle[f"{tag}.w.A_log"], jnp.float32),
        jnp.asarray(oracle[f"{tag}.w.dt_bias"], jnp.float32),
        jnp.asarray(oracle[f"{tag}.w.o_norm"], jnp.float32),
        jnp.asarray([0, num_tokens], jnp.int32),
        jnp.asarray([1], jnp.int32),
        jnp.asarray([num_tokens], jnp.int32),
        lower_bound=lower_bound,
        eps=eps,
    )
    return np.asarray(output), np.asarray(new_state)


def _report(name: str, got: np.ndarray, want: np.ndarray) -> float:
    denom = max(float(np.abs(want).max()), 1e-9)
    rel = float(np.abs(got - want).max()) / denom
    print(f"  {name:28s} max|diff|={np.abs(got-want).max():.3e} "
          f"rel={rel:.3e} ref|max|={np.abs(want).max():.3e}")
    return rel


def test_conv_and_gate_preprocessing_match(oracle):
    """The forget gate the op computes internally must equal HF's."""
    num_heads = int(oracle["cfg.linear_num_heads"])
    head_dim = int(oracle["cfg.linear_head_dim"])
    lower_bound = float(oracle["cfg.lower_bound"])

    worst = 0.0
    for layer in _kda_layer_ids(oracle):
        tag = f"kda{layer:02d}"
        raw = oracle[f"{tag}.f_b_out"][0].astype(np.float32)
        dt_bias = oracle[f"{tag}.w.dt_bias"].astype(np.float32)
        a_log = oracle[f"{tag}.w.A_log"].astype(np.float32)

        gate_input = (raw + dt_bias).reshape(-1, num_heads, head_dim)
        decay = np.exp(a_log)[None, :, None]
        got = lower_bound / (1.0 + np.exp(-decay * gate_input))

        want = oracle[f"{tag}.g"][0].astype(np.float32)
        worst = max(worst, _report(f"{tag}.forget_gate", got, want))

    assert worst < 1e-5, f"forget gate diverges (rel {worst:.3e})"


def test_ragged_kda_matches_hf_reference(oracle):
    """`ragged_kda` in fp32 should reproduce HF's gated-normalised output."""
    num_heads = int(oracle["cfg.linear_num_heads"])
    head_dim = int(oracle["cfg.linear_head_dim"])

    worst = 0.0
    for layer in _kda_layer_ids(oracle):
        got, _ = _run_ragged_kda(oracle, layer)
        want = oracle[f"kda{layer:02d}.o_norm_out"][0].astype(np.float32)
        assert got.shape == (want.shape[0], num_heads, head_dim)
        worst = max(worst, _report(f"kda{layer:02d}.o_norm_out", got, want))

    # fp32 both sides; the only slack is reduction order.
    assert worst < 2e-4, f"ragged_kda diverges from HF (rel {worst:.3e})"


_NEEDS_TPU = (
    "chunk_kda needs a real TPU. PALLAS_INTERPRET=1 is not enough: the tile "
    "planner calls get_tpu_info(), which rejects a CPU device kind, and "
    "registering a fake v7x entry only gets as far as the interpreter's "
    "discharge rules, which cannot handle this kernel's dynamically-sized "
    "slices. The repo's own tests/kernels/test_kimi_chunk_kda.py fails on CPU "
    "the same way. Recurrence parity is covered on CPU by "
    "test_ragged_kda_matches_hf_reference instead.")


def test_chunk_kda_matches_hf_reference(oracle):
    """The chunked prefill Pallas kernel, on device.

    `chunk_kda` returns the bare recurrence output, before the gated output
    norm, so this compares against `core_attn_out` rather than `o_norm_out`.
    It is bf16-only, hence the far looser tolerance than the fp32 reference.
    """
    # Probed here, not in a `skipif`: touching jax.devices() at collection
    # time claims the accelerator for the process and breaks spawned workers.
    # conftest.py enforces that.
    if not any(d.platform == "tpu" for d in jax.devices()):
        pytest.skip(_NEEDS_TPU)

    from vllm_torchtpu.kernels.kimi_k3.chunk_kda import chunk_kda

    num_heads = int(oracle["cfg.linear_num_heads"])
    head_dim = int(oracle["cfg.linear_head_dim"])
    lower_bound = float(oracle["cfg.lower_bound"])

    worst = 0.0
    for layer in _kda_layer_ids(oracle):
        tag = f"kda{layer:02d}"
        q = oracle[f"{tag}.q"][0]
        k = oracle[f"{tag}.k"][0]
        v = oracle[f"{tag}.v"][0]
        num_tokens = q.shape[0]

        mixed = np.concatenate([q, k, v], axis=-1).astype(np.float32)
        conv_w = oracle[f"{tag}.w.conv1d"]
        mixed = _silu(_causal_depthwise_conv(mixed, conv_w[:, 0, :]))
        qc, kc, vc = np.split(mixed, 3, axis=-1)

        def to_hbtd(x):
            # [T, H*D] -> [H, 1, T, D]
            return jnp.asarray(
                x.reshape(num_tokens, num_heads,
                          head_dim).transpose(1, 0, 2)[:, None], jnp.bfloat16)

        gate = oracle[f"{tag}.g"][0].astype(np.float32)  # already activated
        gate_hbtd = jnp.asarray(
            gate.reshape(num_tokens, num_heads,
                         head_dim).transpose(1, 0, 2)[:, None], jnp.bfloat16)
        beta = oracle[f"{tag}.beta"][0].astype(np.float32)  # post-sigmoid
        beta_hbt = jnp.asarray(beta.transpose(1, 0)[:, None], jnp.bfloat16)

        segment_ids = jnp.ones((1, num_tokens), jnp.int32)
        initial_state = jnp.zeros((1, 1, num_heads, head_dim, head_dim),
                                  jnp.float32)

        out, _ = chunk_kda(
            to_hbtd(qc),
            to_hbtd(kc),
            to_hbtd(vc),
            gate_hbtd,
            beta_hbt,
            initial_state=initial_state,
            output_final_state=True,
            segment_ids=segment_ids,
            lower_bound=lower_bound,
            use_qk_l2norm_in_kernel=True,
        )
        # [H, 1, T, D] -> [T, H, D]
        got = np.asarray(out, np.float32)[:, 0].transpose(1, 0, 2)
        want = oracle[f"{tag}.core_attn_out"][0].astype(np.float32)
        worst = max(worst, _report(f"{tag}.chunk_kda", got, want))

    assert worst < 5e-2, f"chunk_kda diverges from HF (rel {worst:.3e})"


def test_beta_is_passed_raw(oracle):
    """Guard the double-sigmoid trap: the op applies sigmoid to beta itself."""
    for layer in _kda_layer_ids(oracle):
        tag = f"kda{layer:02d}"
        raw = oracle[f"{tag}.beta_raw"]
        activated = oracle[f"{tag}.beta"]
        np.testing.assert_allclose(1.0 / (1.0 + np.exp(-raw)),
                                   activated,
                                   rtol=1e-6,
                                   atol=1e-6)
        # If they were interchangeable the guard would be vacuous.
        assert np.abs(raw - activated).max() > 1e-3
