import functools
import glob
import os

import requests
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

GCE_TPU_ACCELERATOR_ENDPOINT = (
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/"
)
GCE_TPU_HEADERS = {"Metadata-Flavor": "Google"}
_CHIP_VERSION_ALIASES = {"v5litepod": "v5e", "v7x": "7x"}
_CHIP_VERSION_BY_NAME: dict[str, pltpu.ChipVersion] = {
    member.value: member for member in pltpu.ChipVersion
}
_CHIP_VERSION_BY_NAME.update(
    (alias, _CHIP_VERSION_BY_NAME[target])
    for alias, target in _CHIP_VERSION_ALIASES.items()
)
_DEFAULT_CHIP_VERSION = pltpu.ChipVersion.TPU_7X
_SINGLE_CORE_TPU_PREFIXES = ("v5litepod", "v6e")

_get_tpu_accel_type_env = envs.environment_variables["TPU_ACCELERATOR_TYPE"]
_get_tpu_name_env = envs.environment_variables["TPU_NAME"]
_get_tpu_worker_id_env = envs.environment_variables["TPU_WORKER_ID"]


def get_tpu_metadata(key: str = "") -> str:
    try:
        accelerator_type_request = requests.get(
            f"{GCE_TPU_ACCELERATOR_ENDPOINT}{key}",
            headers=GCE_TPU_HEADERS,
        )
        status_code = accelerator_type_request.status_code
        text = accelerator_type_request.text
        if status_code == 200 and text:
            return text
        else:
            logger.error(
                "Unable to poll TPU GCE Metadata. Got status code: %s and content: %s",
                status_code,
                text,
            )
    except requests.RequestException as e:
        logger.error("Unable to poll the TPU GCE Metadata: %s", e)
    return None


def get_tpu_type() -> str:
    tpu_type = _get_tpu_accel_type_env()
    if tpu_type is None:
        tpu_type = get_tpu_metadata("accelerator-type")
    return tpu_type


@functools.cache
def get_chip_version(accelerator_type: str | None = None) -> pltpu.ChipVersion:
    """Resolves the TPU chip version without initializing a TPU client.

    `pltpu.get_tpu_info()` would initialize PJRT and claim the chips, so
    host-side callers (the scheduler, KV offloading connectors) resolve the
    chip from `TPU_ACCELERATOR_TYPE` instead. Falls back to TPU7x with a
    warning when the accelerator type is unrecognized.
    """
    accelerator_type = accelerator_type or get_tpu_type() or ""
    # ChipVersion's values are bare chip names ("7x", "v6e"); accelerator types
    # add a chip-count suffix and sometimes a "tpu" prefix ("tpu7x", "v6e-8").
    name = accelerator_type.strip().lower().partition("-")[0].removeprefix("tpu")
    # v5litepod and v7x are the only two that still differ after stripping.
    chip = _CHIP_VERSION_BY_NAME.get(name)
    if chip is not None:
        return chip
    logger.warning(
        "Unrecognized TPU accelerator type %r; defaulting to TPU7x static geometry.",
        accelerator_type,
    )
    return _DEFAULT_CHIP_VERSION


def get_node_name() -> str:
    tpu_name = _get_tpu_name_env()
    if not tpu_name:
        tpu_name = get_tpu_metadata("instance-id")
    return tpu_name


def get_node_worker_id() -> int:
    """For multi-host TPU VM, this returns the worker id for the current node."""
    worker_id = _get_tpu_worker_id_env()
    if worker_id is None:
        worker_id = get_tpu_metadata("agent-worker-number")
        if worker_id is None:
            return 0
    return int(worker_id)


def get_num_cores_per_chip() -> int:
    tpu_type = get_tpu_type()
    if tpu_type and tpu_type.startswith(_SINGLE_CORE_TPU_PREFIXES):
        return 1
    return 2


def get_num_chips() -> int:
    accel_files = glob.glob("/dev/accel*")
    if accel_files:
        return len(accel_files)
    try:
        vfio_entries = os.listdir("/dev/vfio")
    except FileNotFoundError as e:
        logger.warning("Failed to detect number of TPUs: %s", e)
        return 0
    count = 0
    for entry in vfio_entries:
        if entry.isdigit():
            count += 1
    return count
