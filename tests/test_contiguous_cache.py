import pytest


from pathlib import Path
import sys



sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from engine.contiguous_cache import ContiguousKVCache


def test_allocate_reserves_max_seq_len():
    cache = ContiguousKVCache(max_seq_len=512)

    allocation = cache.allocate(
        request_id="r1",
        used_tokens=128,
    )

    assert allocation.request_id == "r1"
    assert allocation.reserved_tokens == 512
    assert allocation.used_tokens == 128
    assert allocation.wasted_tokens == 384


def test_multiple_requests():
    cache = ContiguousKVCache(max_seq_len=512)

    cache.allocate("r1", used_tokens=128)
    cache.allocate("r2", used_tokens=256)

    assert cache.reserved_tokens == 1024
    assert cache.used_tokens == 384
    assert cache.wasted_tokens == 640


def test_waste_ratio():
    cache = ContiguousKVCache(max_seq_len=512)

    cache.allocate("r1", used_tokens=128)
    cache.allocate("r2", used_tokens=256)

    assert cache.waste_ratio == pytest.approx(640 / 1024)


def test_no_waste_when_fully_used():
    cache = ContiguousKVCache(max_seq_len=512)

    cache.allocate(
        request_id="r1",
        used_tokens=512,
    )

    assert cache.reserved_tokens == 512
    assert cache.used_tokens == 512
    assert cache.wasted_tokens == 0
    assert cache.waste_ratio == 0.0


def test_free_request():
    cache = ContiguousKVCache(max_seq_len=512)

    cache.allocate("r1", used_tokens=128)
    cache.allocate("r2", used_tokens=256)

    cache.free("r1")

    assert "r1" not in cache.allocations
    assert cache.reserved_tokens == 512
    assert cache.used_tokens == 256
    assert cache.wasted_tokens == 256


def test_duplicate_request_rejected():
    cache = ContiguousKVCache(max_seq_len=512)

    cache.allocate("r1", used_tokens=128)

    with pytest.raises(ValueError):
        cache.allocate("r1", used_tokens=256)


def test_negative_used_tokens_rejected():
    cache = ContiguousKVCache(max_seq_len=512)

    with pytest.raises(ValueError):
        cache.allocate(
            request_id="r1",
            used_tokens=-1,
        )


def test_used_tokens_above_max_seq_len_rejected():
    cache = ContiguousKVCache(max_seq_len=512)

    with pytest.raises(ValueError):
        cache.allocate(
            request_id="r1",
            used_tokens=513,
        )


def test_invalid_max_seq_len_rejected():
    with pytest.raises(ValueError):
        ContiguousKVCache(max_seq_len=0)

    with pytest.raises(ValueError):
        ContiguousKVCache(max_seq_len=-1)


def test_free_unknown_request_rejected():
    cache = ContiguousKVCache(max_seq_len=512)

    with pytest.raises(KeyError):
        cache.free("unknown")