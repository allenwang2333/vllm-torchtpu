"""Tests for the generic on-disk quantized-weight cache."""
import torch

from vllm_torchtpu.layers.common import quant_weight_cache as qwc


def _key(**overrides):
    key = {
        "layer": "model.layers.0.mlp.experts",
        "rank": 0,
        "w13_shape": (8, 4),
        "block_size": 512,
        "weight_dtype": "fp4",
    }
    key.update(overrides)
    return key


def test_disabled_by_default(monkeypatch):
    """Unset env => inert: no path, no read, no write."""
    monkeypatch.delenv(qwc.CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv(qwc.LEGACY_CACHE_DIR_ENV, raising=False)
    assert qwc.cache_dir() is None
    assert qwc.cache_path("ns", _key()) is None
    assert qwc.load(None, _key()) is None
    assert qwc.save(None, _key(), [torch.zeros(2)]) is False


def test_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv(qwc.CACHE_DIR_ENV, str(tmp_path))
    key = _key()
    path = qwc.cache_path("ns", key)
    assert qwc.load(path, key) is None  # miss before write

    tensors = [torch.arange(8).float().reshape(2, 4), torch.ones(3)]
    assert qwc.save(path, key, tensors) is True

    loaded = qwc.load(path, key)
    assert loaded is not None
    for got, want in zip(loaded, tensors):
        assert torch.equal(got, want)


def test_legacy_env_still_honoured(tmp_path, monkeypatch):
    monkeypatch.delenv(qwc.CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv(qwc.LEGACY_CACHE_DIR_ENV, str(tmp_path))
    assert qwc.cache_dir() == tmp_path


def test_different_key_is_a_miss_not_a_bad_hit(tmp_path, monkeypatch):
    """The whole point: a changed input must not reuse an old entry.

    Guards the documented failure where a cache built under one requantization
    config was silently reused by a run expecting another.
    """
    monkeypatch.setenv(qwc.CACHE_DIR_ENV, str(tmp_path))
    key = _key(block_size=512)
    qwc.save(qwc.cache_path("ns", key), key, [torch.ones(4)])

    other = _key(block_size=128)
    assert qwc.cache_path("ns", other) != qwc.cache_path("ns", key)
    assert qwc.load(qwc.cache_path("ns", other), other) is None


def test_key_mismatch_inside_entry_is_rejected(tmp_path, monkeypatch):
    """Even at the same path, a disagreeing key must not hit."""
    monkeypatch.setenv(qwc.CACHE_DIR_ENV, str(tmp_path))
    key = _key()
    path = qwc.cache_path("ns", key)
    qwc.save(path, key, [torch.ones(4)])
    assert qwc.load(path, _key(rank=7)) is None


def test_shape_dtype_expectation_rejects_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv(qwc.CACHE_DIR_ENV, str(tmp_path))
    key = _key()
    path = qwc.cache_path("ns", key)
    qwc.save(path, key, [torch.ones(4, dtype=torch.float32)])

    assert qwc.load(path, key, expect=[((4, ), torch.float32)]) is not None
    assert qwc.load(path, key, expect=[((8, ), torch.float32)]) is None
    assert qwc.load(path, key, expect=[((4, ), torch.int8)]) is None
    assert qwc.load(path, key, expect=[((4, ), torch.float32)] * 2) is None


def test_corrupt_entry_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setenv(qwc.CACHE_DIR_ENV, str(tmp_path))
    key = _key()
    path = qwc.cache_path("ns", key)
    path.write_bytes(b"not a torch file")
    assert qwc.load(path, key) is None


def test_stale_format_version_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setenv(qwc.CACHE_DIR_ENV, str(tmp_path))
    key = _key()
    path = qwc.cache_path("ns", key)
    torch.save(
        {
            "format_version": qwc.FORMAT_VERSION + 1,
            "key": key,
            "tensors": [torch.ones(2)]
        }, path)
    assert qwc.load(path, key) is None


def test_save_failure_does_not_raise(tmp_path, monkeypatch):
    """A full disk is ordinary; it must never take the run down."""
    monkeypatch.setenv(qwc.CACHE_DIR_ENV, str(tmp_path))
    key = _key()
    path = qwc.cache_path("ns", key)

    def boom(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(torch, "save", boom)
    assert qwc.save(path, key, [torch.ones(2)]) is False
    assert not path.exists()


def test_no_temp_file_left_behind_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv(qwc.CACHE_DIR_ENV, str(tmp_path))
    key = _key()
    path = qwc.cache_path("ns", key)

    def write_then_fail(obj, target):
        open(target, "wb").write(b"partial")
        raise OSError("No space left on device")

    monkeypatch.setattr(torch, "save", write_then_fail)
    assert qwc.save(path, key, [torch.ones(2)]) is False
    assert list(tmp_path.glob("*.tmp")) == []


def test_torch_size_and_dtype_keys_are_canonical(tmp_path, monkeypatch):
    """torch.Size and torch.dtype in a key must compare by value."""
    monkeypatch.setenv(qwc.CACHE_DIR_ENV, str(tmp_path))
    a = {"shape": torch.Size([2, 3]), "dtype": torch.float32}
    b = {"shape": (2, 3), "dtype": torch.float32}
    assert qwc.cache_path("ns", a) == qwc.cache_path("ns", b)
    qwc.save(qwc.cache_path("ns", a), a, [torch.ones(2)])
    assert qwc.load(qwc.cache_path("ns", b), b) is not None
