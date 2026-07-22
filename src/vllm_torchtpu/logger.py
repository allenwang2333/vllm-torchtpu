# SPDX-License-Identifier: Apache-2.0


def init_logger(name: str):
    from vllm.logger import init_logger as init_vllm_logger
    patched_name = "vllm." + name
    return init_vllm_logger(patched_name)
