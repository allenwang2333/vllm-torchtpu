import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, List, Optional

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from vllm_torchtpu import utils

if TYPE_CHECKING:
    from vllm.v1.configs.vllm_config import VllmConfig

MESH_AXIS_NAMES = ("attn_dp", "expert", "model")
MESH_AXIS_NAMES_2D = ('model', )


class ShardingAxisNameBase:
    """Base class for sharding axis names."""
    SEQUENCE = 'attn_dp'
    ATTN_DATA = 'attn_dp'
    MLP_DATA = None
    ATTN_HEAD = 'model'
    ATTN_TENSOR = None
    MLP_TENSOR = ('attn_dp', 'model', 'expert')
    MOE_TENSOR = ('attn_dp', 'model')
    EXPERT = ('attn_dp', 'expert', 'model')
    VOCAB = ('expert', 'model')


class ShardingAxisName2D:
    """Sharding axis names for 2D data parallelism scenarios.
    """
    SEQUENCE = None
    ATTN_DATA = None
    MLP_DATA = None
    ATTN_HEAD = 'model'
    ATTN_TENSOR = None
    MLP_TENSOR = 'model'
    MOE_TENSOR = 'model'
    EXPERT = 'model'
    VOCAB = 'model'


ShardingAxisName = ShardingAxisName2D


@dataclass
class ShardingStrategy:
    """Defines parallelism degrees across tensor, expert, sequence, and data dimensions."""
    tensor_parallelism: int = 1
    expert_parallelism: int = 1
    sequence_parallelism: int = 1
    data_parallelism: int = 1
    attention_data_parallelism: int = 1


class ShardingConfigManager:
    """Manages sharding configuration parsing and access from VllmConfig.

    Stores parsed sharding parameters as a JSON-native dict under
    ``vllm_config.additional_config["_sharding_config"]`` to ensure compatibility
    with ``vllm.config.utils.replace`` and JSON-based config hashing.
    """

    # Key under vllm_config.additional_config holding the JSON-native sharding
    # data. Underscore-prefixed to avoid colliding with the user-supplied
    # "sharding" input key that from_vllm_config reads.
    _ADDITIONAL_CONFIG_KEY = "_sharding_config"

    @classmethod
    def initialize(cls, vllm_config: 'VllmConfig') -> 'ShardingConfigManager':
        """Parse the sharding config and stash a JSON-native copy on the config.

        Idempotent and re-entrant. ``check_and_update_config`` (which calls this)
        runs inside ``VllmConfig.__post_init__``, so it re-fires every time the
        config is reconstructed -- e.g. vLLM's ``replace()`` on the Qwen3-VL
        ``with_hf_config`` path. If the stored key is already present, reuse it
        via ``get()`` instead of re-parsing, so the parse happens exactly once
        per config and downstream reads stay stable across reconstructions.
        Returns the manager for convenience.
        """
        if cls._ADDITIONAL_CONFIG_KEY in vllm_config.additional_config:
            return cls.get(vllm_config)
        manager = cls.from_vllm_config(vllm_config)
        device_indexes = manager.device_indexes
        vllm_config.additional_config[cls._ADDITIONAL_CONFIG_KEY] = {
            "sharding_strategy":
            asdict(manager.sharding_strategy),
            "device_indexes":
            (list(device_indexes) if device_indexes is not None else None),
        }
        return manager

    @classmethod
    def get(cls, vllm_config: 'VllmConfig') -> 'ShardingConfigManager':
        """Reconstruct the manager from the JSON-native dict on the config."""
        data = vllm_config.additional_config.get(cls._ADDITIONAL_CONFIG_KEY)
        if data is None:
            return cls.initialize(vllm_config)
        return cls(ShardingStrategy(**data["sharding_strategy"]),
                   data["device_indexes"])

    def __init__(self,
                 sharding_strategy: ShardingStrategy,
                 device_indexes: Optional[List] = None):

        self.sharding_strategy: ShardingStrategy = sharding_strategy
        self.device_indexes: Optional[List[int]] = device_indexes
        self._total_devices: int = int(
            math.prod(asdict(sharding_strategy).values()))
        if device_indexes:
            assert self._total_devices == len(device_indexes)

    @classmethod
    def from_vllm_config(cls,
                         vllm_config: 'VllmConfig') -> 'ShardingConfigManager':

        sharding_strategy = vllm_config.additional_config.get(
            "sharding", {}).get("sharding_strategy", {})
        parallel_config = vllm_config.parallel_config
        tensor_parallelism = parallel_config.tensor_parallel_size
        # Native vLLM multi-engine DP is not represented in the JAX mesh.
        data_parallelism = 1
        expert_parallelism = sharding_strategy.get("expert_parallelism", 1)
        sequence_parallelism = sharding_strategy.get("sequence_parallelism", 1)
        device_indexes = sharding_strategy.get("device_indexes", None)

        enable_dp_attention = sharding_strategy.get("enable_dp_attention",
                                                    False)
        if enable_dp_attention:
            # Replicate attention layer when num_kv_heads < TP
            num_kv_heads = 1 if vllm_config.model_config.use_mla else vllm_config.model_config.get_total_num_kv_heads(
            )
            cache_dtype = vllm_config.cache_config.cache_dtype
            if cache_dtype == 'auto':
                cache_dtype = vllm_config.model_config.dtype
            kv_dtype = utils.get_jax_dtype_from_str_dtype(
                cache_dtype) or jnp.bfloat16
            packing = 4 // jnp.dtype(kv_dtype).itemsize
            # Convert excess TP to Attention DP when packed K+V head count is smaller than TP to prevent KV cache duplication.
            num_kv_heads_per_device_in_kv_cache = (num_kv_heads * 2) / packing
            attn_dp = max(
                int(tensor_parallelism // num_kv_heads_per_device_in_kv_cache),
                1)
            tensor_parallelism = tensor_parallelism // attn_dp
        else:
            attn_dp = 1

        sharding_strategy = ShardingStrategy(
            tensor_parallelism=tensor_parallelism,
            data_parallelism=data_parallelism,
            expert_parallelism=expert_parallelism,
            sequence_parallelism=sequence_parallelism,
            attention_data_parallelism=attn_dp)

        cls.validate(vllm_config, sharding_strategy)
        return cls(sharding_strategy, device_indexes)

    @classmethod
    def validate(cls, vllm_config, sharding_strategy):
        total_dp_size = sharding_strategy.data_parallelism * sharding_strategy.attention_data_parallelism
        if total_dp_size > 1:
            if vllm_config.speculative_config is not None:
                raise ValueError(
                    f"Speculative decoding is not supported with data parallelism "
                    f"(DP size: {total_dp_size}). Please disable speculative decoding or "
                    f"set data parallelism to 1.")
            if vllm_config.lora_config is not None:
                raise ValueError(
                    f"LoRA is not supported with data parallelism "
                    f"(DP size: {total_dp_size}). Please disable LoRA or "
                    f"set data parallelism to 1.")

    @property
    def total_dp_size(self) -> int:
        return self.sharding_strategy.data_parallelism * self.sharding_strategy.attention_data_parallelism

    @property
    def model_dp_size(self) -> int:
        return self.sharding_strategy.data_parallelism

    @property
    def attn_dp_size(self) -> int:
        return self.sharding_strategy.attention_data_parallelism

    @property
    def tp_size(self) -> int:
        return self.sharding_strategy.tensor_parallelism

    @property
    def expert_size(self) -> int:
        return self.sharding_strategy.expert_parallelism

    @property
    def sequence_size(self) -> int:
        return self.sharding_strategy.sequence_parallelism

    @property
    def total_devices(self) -> int:
        return self._total_devices

    def __str__(self):
        return (f"ShardingConfigManager(total_devices={self.total_devices}, "
                f"sharding_strategy={self.sharding_strategy}, "
                f"device_indexes={self.device_indexes})")


#TODO split this into block unique sharding config, i.e. attentionShardingConfig, MoEShardingConfig
@dataclass
class ShardingRulesConfig:
    """Logical mesh partitioning rules for model tensors.

    Attributes map tensor dimensions (suffix letters: t=tokens, d=dim, h=head_dim,
    n/k=heads, e=experts, f=ffw_dim, v=vocab) to logical mesh axes ('dp', 'tp', 'ep').
    """

    # Attention activations & weights
    activation_attention_td: tuple = (None, None)
    activation_attention_out_td: tuple = (None, None)
    activation_q_td: tuple = (None, None)
    attn_o_tnh: tuple = (None, None, None)
    query_tnh: tuple = (None, None, None)
    keyvalue_skh: tuple = (None, None, None)

    attn_q_weight_dnh: tuple = (None, None, None)
    attn_k_weight_dkh: tuple = (None, None, None)
    attn_v_weight_dkh: tuple = (None, None, None)
    attn_o_weight_nhd: tuple = (None, None, None)

    # Feed-forward & MoE activations & weights
    activation_ffw_td: tuple = (None, None)
    activation_ffw_ted: tuple = (None, None, None)
    ffw_hidden_tf: tuple = (None, None)

    ffw_weight_df: tuple = (None, None)
    ffw_weight_fd: tuple = (None, None)
    moe_weights_edf: tuple = (None, None, None)
    moe_weights_efd: tuple = (None, None, None)
    moe_router_de: tuple = (None, None)
    moe_router_bias_e: tuple = (None, )

    # Embedding, norms, and logits
    emb_weight_vd: tuple = (None, None)
    activation_td: tuple = (None, None)
    prelogit_td: tuple = (None, None)
    logits_tv: tuple = (None, None)
    norm_scale: tuple = (None)
    vocab_vd: tuple = (None, None)
    vocab_dv: tuple = (None, None)


class ShardingConfig:
    """Container holding prefill and generation ShardingRulesConfig instances."""

    def __init__(self,
                 prefill_rules=None,
                 generate_rules=None,
                 default_rules_cls=ShardingRulesConfig):
        # Use a factory pattern to avoid mutable default arguments
        self.default_rules_cls = default_rules_cls
        self.prefill_rules = prefill_rules if prefill_rules is not None else default_rules_cls(
        )
        self.generate_rules = generate_rules if generate_rules is not None else default_rules_cls(
        )

    def __repr__(self):
        return (f"ShardingConfig(prefill_rules={self.prefill_rules!r}, "
                f"generate_rules={self.generate_rules!r})")


def build_mesh(devices, strategy: dict[str, int]) -> Mesh:
    """Constructs a JAX device mesh from a sharding strategy.

    This method creates a logical grid of devices based on the parallelism
    degrees defined in the strategy. The logical axis names ('dp', 'ep',
    'sp', 'tp') are used to map tensor dimensions to the physical device grid.

    Args:
        strategy: A dictionary from upper level config.

    Returns:
        A JAX `Mesh` object.
    """

    axis_order = {
        "expert": strategy.get("expert_parallelism", 1),
        "seq": strategy.get("sequence_parallelism", 1),
        "model": strategy.get("tensor_parallelism", 1),
    }

    num_devices = len(devices)
    known_product = 1
    inferred_key = None
    for k, v in axis_order.items():
        if v == -1:
            if inferred_key is not None:
                raise ValueError("At most one parallelism degree can be -1 for automatic inference.")
            inferred_key = k
        else:
            known_product *= max(1, v)

    if inferred_key is not None:
        if num_devices % known_product != 0:
            raise ValueError(
                f"Total devices ({num_devices}) is not divisible by product of "
                f"specified parallelism degrees ({known_product})."
            )
        axis_order[inferred_key] = max(1, num_devices // known_product)

    mesh_axis_names = []
    mesh_shape = []
    for axis, dim in axis_order.items():
        mesh_axis_names.append(axis)
        mesh_shape.append(dim)

    if not mesh_shape:
        mesh_shape = [1]
        mesh_axis_names = ['model']

    devices = np.asarray(devices).reshape(mesh_shape)
    return Mesh(devices, axis_names=tuple(mesh_axis_names))


class Sharding:
    """Generates prefill and generation sharding rules from a ShardingStrategy."""

    def __init__(self,
                 prefill_rules: dict | None = None,
                 generate_rules: dict | None = None,
                 default_rules_cls=ShardingRulesConfig,
                 vllm_config: 'VllmConfig' = None):
        self.vllm_config = vllm_config
        self.default_rules_cls = default_rules_cls
        self.sharding_cfg = self.make_sharding_config(
            default_rules_cls=default_rules_cls,
            prefill_overrides=prefill_rules,
            generate_overrides=generate_rules)

    def _get_overrides(self, sharding_phase: str):
        """Return the overrides from the vLLM config for the given sharding phase."""
        if self.vllm_config is None:
            return {}
        overrides = {}
        try:
            overrides = self.vllm_config.additional_config["sharding"][
                "logical_rules"]["all"]
        except KeyError:
            pass

        try:
            additional_overrides = self.vllm_config.additional_config[
                "sharding"]["logical_rules"][f"{sharding_phase}"]
            overrides.update(additional_overrides)
        except KeyError:
            pass
        return overrides

    def __str__(self):
        """Succinct representation of relevant Sharding settings and overrides."""
        output_str = f"  Using {self.default_rules_cls.__name__} logical rules.\n"
        output_str += f"  {self.__class__.__name__:} overrides:\n"
        output_str += f"    prefill logical_rule overrides:\n    {json.dumps(self._get_overrides('prefill'), indent=4, default=str)}\n\n"
        output_str += f"    generate logical_rule overrides:\n    {json.dumps(self._get_overrides('generate'), indent=4, default=str)}\n\n"
        return output_str

    def validate_sharding_strategy(self, ):
        """Validates if the sharding strategy is compatible with the environment.

        This method is a placeholder now, and will check if the product of parallelism degrees
        matches the number of available devices.
        """
        #TODO: check num_devices % parallelism == 0
        #TODO: check num_devices == multiply(parallelism(with inferred))
        return

    def get_sharding_cfg(self) -> ShardingConfig:
        """Returns the generated sharding configuration."""
        return self.sharding_cfg

    def _apply_overrides(self, config_obj: ShardingRulesConfig,
                         overrides: dict | None):
        """Applies runtime overrides to a sharding configuration object.

        Args:
            config_obj: The sharding configuration object (e.g., prefill_rules)
                to be updated.
            overrides: A dictionary where keys are attribute names of the config
                object and values are the new sharding tuples.

        Raises:
            AttributeError: If a key in the overrides dictionary is not a valid
                attribute of the configuration object.
        """
        for key, value in overrides.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                # Raise an error for invalid keys to prevent silent failures
                raise AttributeError(
                    f"'{key}' is not a valid attribute of {type(config_obj).__name__}"
                )

    def _make_default_sharding_config(self, prefill_rules, generate_rules):

        # Populate Prefill Config
        # During prefill, sequence length is long, so we shard along the sequence axis.
        prefill_rules.activation_attention_td = (ShardingAxisName.ATTN_DATA,
                                                 ShardingAxisName.ATTN_TENSOR)
        prefill_rules.activation_attention_out_td = (
            ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_TENSOR)
        prefill_rules.activation_q_td = (ShardingAxisName.ATTN_DATA,
                                         ShardingAxisName.ATTN_TENSOR)
        # Note: default QKV/KV-cache is sharded along ATTN_HEAD (dim 1); non-standard attention architectures like MLA require custom sharding rules.
        prefill_rules.attn_o_tnh = (ShardingAxisName.ATTN_DATA,
                                    ShardingAxisName.ATTN_HEAD, None)
        prefill_rules.query_tnh = (ShardingAxisName.ATTN_DATA,
                                   ShardingAxisName.ATTN_HEAD, None)
        prefill_rules.keyvalue_skh = (ShardingAxisName.ATTN_DATA,
                                      ShardingAxisName.ATTN_HEAD, None)

        # Populate Generate (Decode) Config
        # During decode, batch size is the large dimension, so we shard along the batch axis.
        generate_rules.activation_attention_td = (ShardingAxisName.ATTN_DATA,
                                                  ShardingAxisName.ATTN_TENSOR)
        generate_rules.activation_attention_out_td = (
            ShardingAxisName.MLP_DATA, ShardingAxisName.ATTN_TENSOR)
        generate_rules.activation_q_td = (ShardingAxisName.ATTN_DATA,
                                          ShardingAxisName.ATTN_TENSOR)
        # Note: default QKV/KV-cache is sharded along ATTN_HEAD (dim 1); non-standard attention architectures like MLA require custom sharding rules.
        generate_rules.attn_o_tnh = (ShardingAxisName.ATTN_DATA,
                                     ShardingAxisName.ATTN_HEAD, None)
        generate_rules.query_tnh = (ShardingAxisName.ATTN_DATA,
                                    ShardingAxisName.ATTN_HEAD, None)
        generate_rules.keyvalue_skh = (ShardingAxisName.ATTN_DATA,
                                       ShardingAxisName.ATTN_HEAD, None)
        generate_rules.attn_q_weight_dnh = (None, ShardingAxisName.ATTN_HEAD,
                                            ShardingAxisName.ATTN_TENSOR)
        generate_rules.attn_k_weight_dkh = (None, ShardingAxisName.ATTN_HEAD,
                                            ShardingAxisName.ATTN_TENSOR)
        generate_rules.attn_v_weight_dkh = (None, ShardingAxisName.ATTN_HEAD,
                                            ShardingAxisName.ATTN_TENSOR)
        generate_rules.attn_o_weight_nhd = (ShardingAxisName.ATTN_HEAD, None,
                                            ShardingAxisName.ATTN_TENSOR)
        generate_rules.activation_ffw_td = (ShardingAxisName.MLP_DATA, None)
        generate_rules.activation_ffw_ted = (ShardingAxisName.MLP_DATA,
                                             ShardingAxisName.EXPERT, None)
        generate_rules.ffw_hidden_tf = (ShardingAxisName.MLP_DATA,
                                        ShardingAxisName.MLP_TENSOR)
        # FFW weights are typically sharded along the hidden dimension (F).
        generate_rules.ffw_weight_df = (None, ShardingAxisName.MLP_TENSOR)
        generate_rules.ffw_weight_fd = (ShardingAxisName.MLP_TENSOR, None)
        # MoE weights are sharded along the expert axis and the hidden dimension.
        generate_rules.moe_weights_edf = (ShardingAxisName.EXPERT, None,
                                          ShardingAxisName.MOE_TENSOR)
        generate_rules.moe_weights_efd = (ShardingAxisName.EXPERT,
                                          ShardingAxisName.MOE_TENSOR, None)
        generate_rules.moe_router_de = (None, ShardingAxisName.EXPERT)

        # Embedding weight: (VocabSize, Dim)
        generate_rules.emb_weight_vd = (ShardingAxisName.MLP_TENSOR, None)
        generate_rules.activation_td = (ShardingAxisName.MLP_DATA,
                                        ShardingAxisName.ATTN_TENSOR)
        generate_rules.prelogit_td = (ShardingAxisName.MLP_DATA,
                                      ShardingAxisName.MLP_TENSOR)
        generate_rules.logits_tv = (ShardingAxisName.MLP_DATA,
                                    ShardingAxisName.MLP_TENSOR)
        generate_rules.vocab_vd = (ShardingAxisName.VOCAB, None)
        generate_rules.vocab_dv = (None, ShardingAxisName.VOCAB)

    def make_sharding_config(
            self,
            default_rules_cls: ShardingRulesConfig,
            prefill_overrides: dict | None = None,
            generate_overrides: dict | None = None) -> ShardingConfig:
        """Creates the detailed `ShardingConfig` with specific partitioning rules
        and applies any runtime overrides.

        This method populates the `prefill_rules` and
        `generate_rules` with hardcoded sharding rules that are generally
        effective for transformer models, and then updates them with any provided
        overrides.

        Args:
            prefill_overrides: A dictionary with attribute names and their new values
                for the prefill sharding configuration.
            generate_overrides: A dictionary with attribute names and their new values
                for the generate sharding configuration.

        Returns:
            The populated and overridden `ShardingConfig` object.
        """
        #TODO: organize into update_prefill() and update_decode for each axis
        #TODO: verify the sharding axes
        sharding_cfg = ShardingConfig(default_rules_cls=default_rules_cls)
        prefill_rules = sharding_cfg.prefill_rules
        generate_rules = sharding_cfg.generate_rules

        # Extract the overrides from the vllm_config if they are not provided programatically.
        if prefill_overrides is None:
            prefill_overrides = self._get_overrides("prefill")
        if generate_overrides is None:
            generate_overrides = self._get_overrides("generate")

        # Apply default sharding configs
        self._make_default_sharding_config(prefill_rules, generate_rules)

        # Apply overriding the runtime sharding rules
        self._apply_overrides(prefill_rules, prefill_overrides)
        self._apply_overrides(generate_rules, generate_overrides)

        return sharding_cfg

    def __repr__(self):
        return (f"Sharding(default_rules_cls={self.default_rules_cls.__name__}, "
                f"sharding_cfg={self.sharding_cfg!r})")


class ShardingInfo:
    #TODO a sharding info class for visualizing & debugging the sharding performance
    # Will implement it for the next version
    pass
