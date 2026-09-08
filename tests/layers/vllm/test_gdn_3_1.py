import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp

from vllm_torchtpu.kernels.gdn.v3 import wrapper as v3_wrapper
from vllm_torchtpu.kernels.gdn.v3_1 import wrapper

try:
  from google3.perftools.accelerators.xprof.api.python import xprof_session
except ImportError:
  class DummyXprofSession:
    def start_session(self, *args, **kwargs): pass
    def end_session_and_get_session_id(self): return "dummy_session"
  class DummyXprofModule:
    XprofSession = DummyXprofSession
  xprof_session = DummyXprofModule()


@dataclasses.dataclass(frozen=True)
class RuntimeConfigs:
  max_reqs: int
  lengths: tuple[int, ...]
  q_loc: tuple[int, ...]
  distribution: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ModelConfigs:
  kq_head_dim: int
  v_head_dim: int
  n_kq: int
  n_v: int
  kernel_size: int


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class GdnAttentionTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="dp8_prefill_1024",
          runtime_configs=RuntimeConfigs(
              max_reqs=1,
              lengths=[1024],
              q_loc=[0, 1024],
              distribution=[0, 0, 1],
          ),
          model_configs=ModelConfigs(
              kq_head_dim=128,
              v_head_dim=128,
              n_kq=2 * 8,
              n_v=8 * 8,
              kernel_size=4,
          ),
      ),
      dict(
          testcase_name="dp8_decode_64",
          runtime_configs=RuntimeConfigs(
              max_reqs=64,
              lengths=[1] * 64,
              q_loc=list(range(65)),
              distribution=[64, 64, 64],
          ),
          model_configs=ModelConfigs(
              kq_head_dim=128,
              v_head_dim=128,
              n_kq=2 * 8,
              n_v=8 * 8,
              kernel_size=4,
          ),
      ),
      dict(
          testcase_name="tp8_prefill_8192",
          runtime_configs=RuntimeConfigs(
              max_reqs=1,
              lengths=[8192],
              q_loc=[0, 8192],
              distribution=[0, 0, 1],
          ),
          model_configs=ModelConfigs(
              kq_head_dim=128,
              v_head_dim=128,
              n_kq=2 * 1,
              n_v=8 * 1,
              kernel_size=4,
          ),
      ),
      dict(
          testcase_name="tp8_decode_64",
          runtime_configs=RuntimeConfigs(
              max_reqs=64,
              lengths=[1] * 64,
              q_loc=list(range(65)),
              distribution=[64, 64, 64],
          ),
          model_configs=ModelConfigs(
              kq_head_dim=128,
              v_head_dim=128,
              n_kq=2 * 1,
              n_v=8 * 1,
              kernel_size=4,
          ),
      ),
  )
  def test_gdn_attention(
      self, runtime_configs: RuntimeConfigs, model_configs: ModelConfigs
  ):
    num_tokens = sum(runtime_configs.lengths)
    state_indices = jnp.arange(1, runtime_configs.max_reqs + 1)
    q_loc = jnp.array(runtime_configs.q_loc)
    distribution = jnp.array(runtime_configs.distribution, dtype=jnp.int32)

    key = jax.random.key(0)

    kq_size = model_configs.n_kq * model_configs.kq_head_dim
    v_size = model_configs.n_v * model_configs.v_head_dim

    # Step 1: Create tensors.
    # Declare activation qkv and ba.
    q = jax.random.normal(key, (num_tokens, kq_size), dtype=jnp.bfloat16)
    k = jax.random.normal(key, (num_tokens, kq_size), dtype=jnp.bfloat16)
    v = jax.random.normal(key, (num_tokens, v_size), dtype=jnp.bfloat16)
    b = jax.random.normal(
        key, (num_tokens, model_configs.n_v), dtype=jnp.bfloat16
    )
    a = jax.random.normal(
        key, (num_tokens, model_configs.n_v), dtype=jnp.bfloat16
    )

    # Declare states conv nand recurrent.
    conv_state_q = jax.random.normal(
        key,
        (runtime_configs.max_reqs + 1, model_configs.kernel_size - 1, kq_size),
        dtype=jnp.float32,
    )
    conv_state_k = jax.random.normal(
        key,
        (runtime_configs.max_reqs + 1, model_configs.kernel_size - 1, kq_size),
        dtype=jnp.float32,
    )
    conv_state_v = jax.random.normal(
        key,
        (runtime_configs.max_reqs + 1, model_configs.kernel_size - 1, v_size),
        dtype=jnp.float32,
    )
    recurrent_state = jax.random.normal(
        key,
        (
            runtime_configs.max_reqs + 1,
            model_configs.n_v,
            model_configs.kq_head_dim,
            model_configs.v_head_dim,
        ),
        dtype=jnp.float32,
    )
    # Declare weights and bias for conv1d.
    conv_weight_q = jax.random.normal(
        key,
        (kq_size, 1, model_configs.kernel_size),
        dtype=jnp.bfloat16,
    )
    conv_weight_k = jax.random.normal(
        key,
        (kq_size, 1, model_configs.kernel_size),
        dtype=jnp.bfloat16,
    )
    conv_weight_v = jax.random.normal(
        key,
        (v_size, 1, model_configs.kernel_size),
        dtype=jnp.bfloat16,
    )

    conv_bias_q = jax.random.normal(key, (kq_size,), dtype=jnp.bfloat16)
    conv_bias_k = jax.random.normal(key, (kq_size,), dtype=jnp.bfloat16)
    conv_bias_v = jax.random.normal(key, (v_size,), dtype=jnp.bfloat16)

    a_log = jax.random.normal(key, (model_configs.n_v,), dtype=jnp.bfloat16)
    dt_bias = jax.random.normal(key, (model_configs.n_v,), dtype=jnp.bfloat16)

    seq_lens = (
        jnp.asarray(
            q_loc[1 : runtime_configs.max_reqs + 1]
            - q_loc[: runtime_configs.max_reqs],
            dtype=jnp.int32,
        )
        + 10
    )

    # Step 2: Pre-process tensors.
    mixed_qkv = jnp.concatenate([q, k, v], axis=-1)
    conv_state = jnp.concatenate(
        [conv_state_q, conv_state_k, conv_state_v], axis=-1
    )
    conv_weight = jnp.concatenate(
        [conv_weight_q, conv_weight_k, conv_weight_v], axis=0
    )
    conv_bias = jnp.concatenate(
        [conv_bias_q, conv_bias_k, conv_bias_v], axis=-1
    )

    kv_head_dim = 256
    padding = 3

    @jax.jit
    def fn_gdn_attention(
        mixed_qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
    ):
      num_blocks = runtime_configs.max_reqs + 1
      orig_conv_shape = conv_state.shape
      orig_conv_dtype = conv_state.dtype
      orig_recurrent_shape = recurrent_state.shape
      orig_recurrent_dtype = recurrent_state.dtype
      conv_state = jnp.expand_dims(conv_state, axis=-2)

      conv_state = jax.lax.bitcast_convert_type(conv_state, jnp.uint32)
      recurrent_state = jax.lax.bitcast_convert_type(
          recurrent_state, jnp.uint32
      )
      reshaped_conv_state = conv_state.reshape(num_blocks, -1, 1, 128)
      reshaped_recurrent_state = recurrent_state.reshape(num_blocks, -1, 1, 128)

      offset = reshaped_conv_state.shape[1]
      shared_state = jnp.concat(
          [reshaped_conv_state, reshaped_recurrent_state], axis=1
      )

      # Make things complicated.
      # NOTE: last second dim must have size of packing.
      shared_state = shared_state.reshape(num_blocks, -1, 1, kv_head_dim)
      shared_state = jnp.pad(
          shared_state, ((0, 0), (0, padding), (0, 0), (0, 0))
      )

      out_states, out = wrapper.fused_conv1d_gdn(
          mixed_qkv,
          b,
          a,
          shared_state,
          conv_weight,
          conv_bias,
          A_log,
          dt_bias,
          query_start_loc,
          state_indices,
          distribution,
          seq_lens,
          n_kq=model_configs.n_kq,
          n_v=model_configs.n_v,
          d_k=model_configs.kq_head_dim,
          d_v=model_configs.v_head_dim,
          kernel_size=model_configs.kernel_size,
          conv_dtype=jnp.float32.dtype,
          recurrent_dtype=jnp.float32.dtype,
      )

      # Undo changes.
      out_states = out_states[:, :-padding]
      out_states = out_states.reshape(num_blocks, -1, 1, 128)

      # Get real values.
      conv_state, recurrent_state = jnp.split(out_states, [offset], axis=1)
      conv_state = jax.lax.bitcast_convert_type(conv_state, orig_conv_dtype)
      recurrent_state = jax.lax.bitcast_convert_type(
          recurrent_state, orig_recurrent_dtype
      )

      conv_state = conv_state.reshape(orig_conv_shape)
      recurrent_state = recurrent_state.reshape(orig_recurrent_shape)

      return (conv_state, recurrent_state), out

    @jax.jit
    def fn_gdn_attention_ref(
        mixed_qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
    ):
      orig_conv_dtype = conv_state.dtype
      orig_recurrent_dtype = recurrent_state.dtype
      conv_state = conv_state.astype(jnp.bfloat16)
      recurrent_state = recurrent_state.astype(jnp.bfloat16)
      (conv_state, recurren_state), out = v3_wrapper.fused_conv1d_gdn(
          mixed_qkv,
          b,
          a,
          conv_state,
          recurrent_state,
          conv_weight,
          conv_bias,
          A_log,
          dt_bias,
          query_start_loc,
          state_indices,
          distribution,
          seq_lens,
          n_kq=model_configs.n_kq,
          n_v=model_configs.n_v,
          d_k=model_configs.kq_head_dim,
          d_v=model_configs.v_head_dim,
          kernel_size=model_configs.kernel_size,
      )
      conv_state = conv_state.astype(orig_conv_dtype)
      recurrent_state = recurren_state.astype(orig_recurrent_dtype)

      return (conv_state, recurrent_state), out

    ref_args = dict(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        conv_state=conv_state.astype(jnp.bfloat16),
        recurrent_state=recurrent_state.astype(jnp.bfloat16),
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        A_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=q_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=seq_lens,
    )
    args = dict(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        A_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=q_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=seq_lens,
    )

    # Warmup
    _ = fn_gdn_attention(**jax.tree.map(lambda x: x.copy(), args))

    # Warmup ref
    _ = fn_gdn_attention_ref(**jax.tree.map(lambda x: x.copy(), args))

    session = xprof_session.XprofSession()
    session.start_session(
        device_name="",
        enable_python_tracer=True,
        host_trace_level=2,
        trace_mode="TRACE_COMPUTE_AND_DMA",
    )

    new_states, output = fn_gdn_attention(
        **jax.tree.map(lambda x: x.copy(), args)
    )
    output.block_until_ready()

    self.assertEqual(output.shape, (num_tokens, v_size))

    # Run reference implementation for parity check
    new_states_ref, output_ref = fn_gdn_attention_ref(
        **jax.tree.map(lambda x: x.copy(), args)
    )
    output_ref.block_until_ready()

    session_id = session.end_session_and_get_session_id()
    print(f"[XProf Session ID]: http://xprof/?session_id={session_id}")
    print(f"[Percale] http://percale/llo/debug?id={session_id}")

    # Also check states
    new_cs, new_recurrent_state = new_states
    new_cs_ref, new_recurrent_state_ref = new_states_ref

    new_cs_q, new_cs_k, new_cs_v = jnp.split(
        new_cs, [kq_size, 2 * kq_size], axis=-1
    )

    new_cs_q_ref, new_cs_k_ref, new_cs_v_ref = jnp.split(
        new_cs_ref, [kq_size, 2 * kq_size], axis=-1
    )

    rtol = 3e-2
    atol = 3e-2

    self.assertArraysAllClose(output, output_ref, rtol=rtol, atol=atol)
    self.assertArraysAllClose(new_cs_q, new_cs_q_ref, rtol=rtol, atol=atol)
    self.assertArraysAllClose(new_cs_k, new_cs_k_ref, rtol=rtol, atol=atol)
    self.assertArraysAllClose(new_cs_v, new_cs_v_ref, rtol=rtol, atol=atol)
    self.assertArraysAllClose(
        new_recurrent_state,
        new_recurrent_state_ref,
        rtol=rtol,
        atol=atol,
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
