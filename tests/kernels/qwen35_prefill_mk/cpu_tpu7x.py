"""Import first: makes Pallas TPU interpret mode on CPU size everything for a TPU7x core."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
from jax._src import tpu_info as _ti
_ti.registry["cpu"] = lambda: _ti._get_tpu_info_impl(_ti.ChipVersion.TPU_7X, 1)
