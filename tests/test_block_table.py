import pytest
import torch


from pathlib import Path
import sys



sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)




from engine.block_pool import BlockPool
from engine.block_table import BlockTable



def make_block_table():
    pool = BlockPool(
        num_blocks=8,
        block_size=16,
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    table = BlockTable(
        block_pool=pool,
        block_size=16,
    )

    return table


def test_position_zero_translation():
    table = make_block_table()

    table.ensure_capacity(
        seq_id="r1",
        next_pos=0,
    )

    expected_block = (
        table.tables["r1"].physical_blocks[0]
    )

    block_id, offset = table.translate(
        seq_id="r1",
        logical_pos=0,
    )

    assert block_id == expected_block
    assert offset == 0


def test_position_zero_translation():
    table = make_block_table()

    table.ensure_capacity(
        seq_id="r1",
        next_pos=0,
    )

    expected_block = (
        table.tables["r1"].physical_blocks[0]
    )

    block_id, offset = table.translate(
        seq_id="r1",
        logical_pos=0,
    )

    assert block_id == expected_block
    assert offset == 0


def test_first_position_of_second_block():
    table = make_block_table()

    table.ensure_capacity(
        seq_id="r1",
        next_pos=16,
    )

    expected_block = (
        table.tables["r1"].physical_blocks[1]
    )

    block_id, offset = table.translate(
        seq_id="r1",
        logical_pos=16,
    )

    assert block_id == expected_block
    assert offset == 0


def test_multi_block_translation():
    table = make_block_table()

    table.ensure_capacity(
        seq_id="r1",
        next_pos=32,
    )

    blocks = table.tables["r1"].physical_blocks

    assert len(blocks) == 3

    assert table.translate("r1", 0) == (
        blocks[0],
        0,
    )

    assert table.translate("r1", 15) == (
        blocks[0],
        15,
    )

    assert table.translate("r1", 16) == (
        blocks[1],
        0,
    )

    assert table.translate("r1", 31) == (
        blocks[1],
        15,
    )

    assert table.translate("r1", 32) == (
        blocks[2],
        0,
    )


def test_new_block_allocated_at_boundary():
    table = make_block_table()

    table.ensure_capacity(
        seq_id="r1",
        next_pos=15,
    )

    assert len(
        table.tables["r1"].physical_blocks
    ) == 1

    table.ensure_capacity(
        seq_id="r1",
        next_pos=16,
    )

    assert len(
        table.tables["r1"].physical_blocks
    ) == 2


def test_sequences_do_not_share_blocks():
    table = make_block_table()

    table.ensure_capacity(
        seq_id="r1",
        next_pos=0,
    )

    table.ensure_capacity(
        seq_id="r2",
        next_pos=0,
    )

    r1_block = table.tables[
        "r1"
    ].physical_blocks[0]

    r2_block = table.tables[
        "r2"
    ].physical_blocks[0]

    assert r1_block != r2_block



def test_free_sequence_clears_table():
    table = make_block_table()

    table.ensure_capacity(
        seq_id="r1",
        next_pos=32,
    )

    assert "r1" in table.tables

    table.free_sequence("r1")

    assert "r1" not in table.tables


def test_multi_block_translation():
    table = make_block_table()

    table.ensure_capacity(
        seq_id="r1",
        next_pos=32,
    )

    blocks = table.tables["r1"].physical_blocks

    assert len(blocks) == 3

    assert table.translate(
        "r1",
        0,
    ) == (blocks[0], 0)

    assert table.translate(
        "r1",
        15,
    ) == (blocks[0], 15)

    assert table.translate(
        "r1",
        16,
    ) == (blocks[1], 0)

    assert table.translate(
        "r1",
        31,
    ) == (blocks[1], 15)

    assert table.translate(
        "r1",
        32,
    ) == (blocks[2], 0)


def test_free_sequence_returns_blocks_to_pool():
    table = make_block_table()

    initial_free = table.block_pool.free_count

    table.ensure_capacity(
        seq_id="r1",
        next_pos=32,
    )

    assert table.block_pool.free_count == (
        initial_free - 3
    )

    table.free_sequence("r1")

    assert table.block_pool.free_count == initial_free