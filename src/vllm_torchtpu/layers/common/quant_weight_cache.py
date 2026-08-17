"""On-disk cache for weights transformed at load time.

Several quantization methods rebuild the same tensors on every start --
dequantize, pad, requantize to the layout a kernel wants. For a large MoE that
is minutes of every run (~9 min for DeepSeek-V4-Flash across 8 ranks), which
dominates iteration time during model bring-up.

Point ``VLLM_QUANT_WEIGHT_CACHE_DIR`` at a directory and the result is stored
there, keyed on everything that determines it. Unset (the default) this module
is inert: no reads, no writes, no behaviour change. It is a development
convenience, never a source of truth -- every failure path falls back to
recomputing.

Usage from a quantization method::

    key = {"layer": layer.layer_name, "rank": rank,
           "w13_shape": tuple(w13.shape), "block_size": block_size, ...}
    path = cache_path("dsv4_mxfp4", key)
    tensors = load(path, key)           # None on miss/mismatch/corruption
    if tensors is None:
        tensors = expensive_requantization(...)
        save(path, key, tensors)        # best effort, never raises

**The key must name every input that changes the output.** An earlier,
DSv4-specific version of this cache keyed only on rank and tensor shapes, so a
cache built under one requantization config was silently reused by a run
expecting another -- a documented cause of garbage output
(``docs/dsv4_handoff_20260806.md`` section 7c). Entries also record their key
and are rejected when it disagrees, so a hash collision or a stale format is a
miss rather than a wrong hit.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

CACHE_DIR_ENV = "VLLM_QUANT_WEIGHT_CACHE_DIR"
# The DSv4 bring-up shipped with this name; honoured so existing dev setups and
# scripts keep working.
LEGACY_CACHE_DIR_ENV = "VLLM_FP4_CACHE_DIR"

# Bump when the on-disk layout changes in a way that makes old entries
# unreadable or wrong. Entries carrying a different version are ignored.
FORMAT_VERSION = 1


def cache_dir() -> Path | None:
    """The configured cache directory, or None when caching is off."""
    for env in (CACHE_DIR_ENV, LEGACY_CACHE_DIR_ENV):
        value = os.environ.get(env)
        if value:
            return Path(value)
    return None


def _canonical(value: Any) -> Any:
    """Render a key part so equal inputs always produce equal text."""
    if isinstance(value, torch.dtype):
        return str(value)
    # A list, not a tuple: the sequence branch below renders plain tuples as
    # lists too, and `torch.Size([2, 3])` must key identically to `(2, 3)`.
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def key_digest(namespace: str, key: Mapping[str, Any]) -> str:
    canonical = f"{namespace}|{FORMAT_VERSION}|{_canonical(dict(key))!r}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def cache_path(namespace: str, key: Mapping[str, Any]) -> Path | None:
    """Path for this entry, or None when caching is off or unusable."""
    directory = cache_dir()
    if directory is None:
        return None
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning_once(
            "Quantized-weight cache directory %s is unusable (%s); "
            "continuing without it.", directory, exc)
        return None
    return directory / f"{namespace}_{key_digest(namespace, key)}.pt"


def load(
    path: Path | None,
    key: Mapping[str, Any],
    *,
    expect: Sequence[tuple[tuple[int, ...], torch.dtype]] | None = None,
    device: torch.device | str | None = None,
) -> list[torch.Tensor] | None:
    """Load an entry, or None on miss, mismatch, or unreadable file.

    `key` is re-checked against what the entry recorded, so an entry only hits
    when it was produced by the same inputs. `expect`, when given, additionally
    requires each tensor's (shape, dtype) to match.
    """
    if path is None or not path.exists():
        if path is not None:
            logger.info("[QUANT_CACHE] miss (absent) %s", path.name)
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # dev cache: never fail the run over it
        logger.warning("Ignoring unreadable quantized-weight cache %s: %s",
                       path, exc)
        return None

    if (not isinstance(payload, Mapping)
            or payload.get("format_version") != FORMAT_VERSION
            or payload.get("key") != _canonical(dict(key))):
        logger.warning_once(
            "Ignoring stale quantized-weight cache entry %s (key or format "
            "mismatch).", path)
        return None

    tensors = payload.get("tensors")
    if not isinstance(tensors, (list, tuple)) or not all(
            isinstance(t, torch.Tensor) for t in tensors):
        logger.warning("Ignoring malformed quantized-weight cache %s", path)
        return None

    if expect is not None:
        if len(expect) != len(tensors):
            logger.warning(
                "Ignoring quantized-weight cache %s: expected %d "
                "tensors, found %d", path, len(expect), len(tensors))
            return None
        for tensor, (shape, dtype) in zip(tensors, expect):
            if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
                logger.warning(
                    "Ignoring quantized-weight cache %s: tensor %s/%s does not "
                    "match expected %s/%s", path, tuple(tensor.shape),
                    tensor.dtype, tuple(shape), dtype)
                return None

    logger.info("[QUANT_CACHE] hit %s (%d tensors)", path.name, len(tensors))
    if device is not None:
        return [t.to(device) for t in tensors]
    return list(tensors)


def save(path: Path | None, key: Mapping[str, Any],
         tensors: Sequence[torch.Tensor]) -> bool:
    """Store an entry. Returns whether it was written; never raises.

    Writes through a temp file and renames, so a killed run cannot leave a
    truncated entry for a later run to load. The temp name carries the pid and
    rank so concurrent writers of the same entry cannot clobber each other.
    """
    if path is None:
        return False
    rank = "0"
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            rank = str(dist.get_rank())
    except Exception:
        pass
    tmp = path.with_name(f"{path.stem}_{os.getpid()}_{rank}.pt.tmp")
    try:
        torch.save(
            {
                "format_version": FORMAT_VERSION,
                "key": _canonical(dict(key)),
                "tensors": [t.detach().cpu() for t in tensors],
            }, tmp)
        tmp.rename(path)
        return True
    except Exception as exc:
        # A full disk is ordinary here: entries run to hundreds of MB per layer
        # per rank, and pointing this at a tmpfs will exhaust it.
        logger.warning_once(
            "Could not write quantized-weight cache to %s (%s); continuing "
            "without it. Point %s at a disk with room for the model's "
            "transformed weights.", path, exc, CACHE_DIR_ENV)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
