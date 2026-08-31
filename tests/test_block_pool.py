from __future__ import annotations

import pytest
import torch



from pathlib import Path
import sys



sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)


from engine.block_pool import BlockPool


@pytest.fixture
def pool() -> BlockPool:
    return BlockPool(
        num_blocks=4,
        block_size=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        dtype=torch.float32,
        device="cpu",
    )

def test_pool_initializes_with_all_blocks_free(
    pool: BlockPool,
):
    assert pool.free_count == 4
    assert pool.allocated_count == 0
def test_allocate_block_reduces_free_count(
    pool: BlockPool,
):
    block_id = pool.allocate_block()

    assert 0 <= block_id < 4
    assert pool.free_count == 3
    assert pool.allocated_count == 1


def test_multiple_allocations_return_unique_blocks(
    pool: BlockPool,
):
    block_ids = [
        pool.allocate_block()
        for _ in range(4)
    ]

    assert len(set(block_ids)) == 4

    assert pool.free_count == 0



def test_allocation_exhaustion_raises(
    pool: BlockPool,
):
    for _ in range(4):
        pool.allocate_block()

    with pytest.raises(RuntimeError, match="exhausted"):
        pool.allocate_block()


def test_free_block_makes_block_available_again(
    pool: BlockPool,
):
    block_id = pool.allocate_block()

    assert pool.free_count == 3

    pool.free_block(block_id)

    assert pool.free_count == 4
    assert pool.allocated_count == 0



def test_freed_block_is_reused(
    pool: BlockPool,
):
    block_id = pool.allocate_block()

    pool.free_block(block_id)

    reused_id = pool.allocate_block()

    assert reused_id == block_id


def test_double_free_is_rejected(
    pool: BlockPool,
):
    block_id = pool.allocate_block()

    pool.free_block(block_id)

    with pytest.raises(
        ValueError,
        match="already free",
    ):
        pool.free_block(block_id)


def test_invalid_block_id_is_rejected(
    pool: BlockPool,
):
    with pytest.raises(ValueError):
        pool.free_block(999)


def test_sequence_allocation_tracks_owned_blocks(
    pool: BlockPool,
):
    block_ids = pool.allocate_sequence(
        seq_id="r1",
        num_blocks=2,
    )

    assert len(block_ids) == 2
    assert len(set(block_ids)) == 2

    assert pool.free_count == 2
    assert pool.allocated_count == 2

    assert "r1" in pool.allocated_sequences

    assert (
        pool.allocated_sequences["r1"].block_ids
        == block_ids
    )



def test_sequence_free_returns_all_blocks(
    pool: BlockPool,
):
    block_ids = pool.allocate_sequence(
        seq_id="r1",
        num_blocks=2,
    )

    assert pool.free_count == 2

    pool.free_sequence("r1")

    assert pool.free_count == 4
    assert pool.allocated_count == 0

    assert "r1" not in pool.allocated_sequences


def test_sequence_exhaustion_is_rejected(
    pool: BlockPool,
):
    pool.allocate_sequence(
        seq_id="r1",
        num_blocks=3,
    )

    with pytest.raises(RuntimeError, match="exhausted"):
        pool.allocate_sequence(
            seq_id="r2",
            num_blocks=2,
        )


def test_sequence_reuse_after_free(
    pool: BlockPool,
):
    first = pool.allocate_sequence(
        seq_id="r1",
        num_blocks=2,
    )

    pool.free_sequence("r1")

    second = pool.allocate_sequence(
        seq_id="r2",
        num_blocks=2,
    )

    assert set(first) == set(second)
    assert pool.free_count == 2


def test_sequence_double_allocation_is_rejected(
    pool: BlockPool,
):
    pool.allocate_sequence(
        seq_id="r1",
        num_blocks=1,
    )

    with pytest.raises(
        ValueError,
        match="already allocated",
    ):
        pool.allocate_sequence(
            seq_id="r1",
            num_blocks=1,
        )

def test_sequence_free_unknown_id_is_rejected(
    pool: BlockPool,
):
    with pytest.raises(KeyError):
        pool.free_sequence("unknown")


def test_kv_block_tensor_shapes(
    pool: BlockPool,
):
    expected_shape = (
        2,
        4,
        2,
        16,
        64,
    )

    assert pool.key_blocks.shape == expected_shape
    assert pool.value_blocks.shape == expected_shape


def test_repeated_sequence_lifecycles_do_not_leak_blocks(
    pool: BlockPool,
):
    initial_free_count = pool.free_count

    for iteration in range(100):

        seq_id = f"request-{iteration}"

        block_ids = pool.allocate_sequence(
            seq_id=seq_id,
            num_blocks=2,
        )

        assert len(block_ids) == 2

        pool.free_sequence(seq_id)

        assert pool.free_count == initial_free_count
        assert pool.allocated_count == 0