import torch

from pathlib import Path
import sys



sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)


from engine.block_pool import BlockPool
from engine.block_table import BlockTable
from engine.paged_kv_cache import PagedKVCache


def make_paged_cache():
    pool = BlockPool(
        num_blocks=8,
        block_size=16,
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    table = BlockTable(
        block_pool=pool,
        block_size=16,
    )

    cache = PagedKVCache(
        block_pool=pool,
        block_table=table,
    )

    return cache


def test_write_and_gather_single_position():
    cache = make_paged_cache()

    key = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    value = torch.tensor(
        [
            [10.0, 11.0, 12.0, 13.0],
            [14.0, 15.0, 16.0, 17.0],
        ]
    )

    cache.write_kv(
        seq_id="r1",
        pos=0,
        layer=0,
        key=key,
        value=value,
    )

    gathered_k, gathered_v = (
        cache.gather_kv_for_sequence(
            seq_id="r1",
            layer=0,
            length=1,
        )
    )

    assert gathered_k.shape == (1, 2, 4)
    assert gathered_v.shape == (1, 2, 4)

    torch.testing.assert_close(
        gathered_k[0],
        key,
    )

    torch.testing.assert_close(
        gathered_v[0],
        value,
    )


def test_multi_block_gather_preserves_logical_order():
    cache = make_paged_cache()

    for pos in range(32):

        key = torch.full(
            (2, 4),
            float(pos),
        )

        value = torch.full(
            (2, 4),
            float(pos + 1000),
        )

        cache.write_kv(
            seq_id="r1",
            pos=pos,
            layer=0,
            key=key,
            value=value,
        )

    gathered_k, gathered_v = (
        cache.gather_kv_for_sequence(
            seq_id="r1",
            layer=0,
            length=32,
        )
    )

    assert gathered_k.shape == (32, 2, 4)
    assert gathered_v.shape == (32, 2, 4)

    for pos in range(32):

        torch.testing.assert_close(
            gathered_k[pos],
            torch.full(
                (2, 4),
                float(pos),
            ),
        )

        torch.testing.assert_close(
            gathered_v[pos],
            torch.full(
                (2, 4),
                float(pos + 1000),
            ),
        )


def test_non_contiguous_physical_blocks_gather_correctly():
    cache = make_paged_cache()

    # Occupy some blocks first so r1 cannot receive
    # consecutive physical block IDs.
    cache.write_kv(
        seq_id="other",
        pos=0,
        layer=0,
        key=torch.full((2, 4), -1.0),
        value=torch.full((2, 4), -1.0),
    )

    for pos in range(32):
        key = torch.full(
            (2, 4),
            float(pos),
        )

        value = torch.full(
            (2, 4),
            float(pos + 1000),
        )

        cache.write_kv(
            seq_id="r1",
            pos=pos,
            layer=0,
            key=key,
            value=value,
        )

    blocks = cache.block_table.tables[
        "r1"
    ].physical_blocks

    assert len(blocks) == 2

    # The physical blocks must not be assumed to be
    # [0, 1]. The exact IDs depend on the free-list.
    assert blocks[0] != blocks[1]

    gathered_k, gathered_v = (
        cache.gather_kv_for_sequence(
            seq_id="r1",
            layer=0,
            length=32,
        )
    )

    for pos in range(32):
        torch.testing.assert_close(
            gathered_k[pos],
            torch.full(
                (2, 4),
                float(pos),
            ),
        )

        torch.testing.assert_close(
            gathered_v[pos],
            torch.full(
                (2, 4),
                float(pos + 1000),
            ),
        )


def test_gather_across_block_boundaries():
    cache = make_paged_cache()

    positions = [15, 16, 31, 32]

    for pos in positions:
        cache.write_kv(
            seq_id="r1",
            pos=pos,
            layer=0,
            key=torch.full(
                (2, 4),
                float(pos),
            ),
            value=torch.full(
                (2, 4),
                float(pos + 1000),
            ),
        )

    gathered_k, gathered_v = (
        cache.gather_kv_for_sequence(
            seq_id="r1",
            layer=0,
            length=33,
        )
    )

    for pos in positions:
        torch.testing.assert_close(
            gathered_k[pos],
            torch.full(
                (2, 4),
                float(pos),
            ),
        )

        torch.testing.assert_close(
            gathered_v[pos],
            torch.full(
                (2, 4),
                float(pos + 1000),
            ),
        )


import math


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Reference scaled dot-product attention.

    Shapes:
        query: [num_heads, head_dim]
        key:   [seq_len, num_heads, head_dim]
        value: [seq_len, num_heads, head_dim]

    Returns:
        [num_heads, head_dim]
    """

    num_heads, head_dim = query.shape
    seq_len = key.shape[0]

    assert key.shape == (
        seq_len,
        num_heads,
        head_dim,
    )

    assert value.shape == (
        seq_len,
        num_heads,
        head_dim,
    )

    # [num_heads, seq_len]
    scores = torch.einsum(
        "hd,thd->ht",
        query,
        key,
    )

    scores = scores / math.sqrt(head_dim)

    # [num_heads, seq_len]
    weights = torch.softmax(
        scores,
        dim=-1,
    )

    # [num_heads, head_dim]
    output = torch.einsum(
        "ht,thd->hd",
        weights,
        value,
    )

    return output


def test_paged_attention_matches_contiguous_attention():
    cache = make_paged_cache()

    seq_len = 32

    keys = []
    values = []

    for pos in range(seq_len):
        key = torch.tensor(
            [
                [float(pos), 1.0, 2.0, 3.0],
                [4.0, float(pos), 5.0, 6.0],
            ]
        )

        value = torch.tensor(
            [
                [10.0 + pos, 11.0, 12.0, 13.0],
                [14.0, 15.0 + pos, 16.0, 17.0],
            ]
        )

        keys.append(key)
        values.append(value)

        cache.write_kv(
            seq_id="r1",
            pos=pos,
            layer=0,
            key=key,
            value=value,
        )

    contiguous_k = torch.stack(
        keys,
        dim=0,
    )

    contiguous_v = torch.stack(
        values,
        dim=0,
    )

    paged_k, paged_v = (
        cache.gather_kv_for_sequence(
            seq_id="r1",
            layer=0,
            length=seq_len,
        )
    )

    query = torch.tensor(
        [
            [0.5, 1.0, 1.5, 2.0],
            [2.0, 1.5, 1.0, 0.5],
        ]
    )

    contiguous_output = reference_attention(
        query=query,
        key=contiguous_k,
        value=contiguous_v,
    )

    paged_output = reference_attention(
        query=query,
        key=paged_k,
        value=paged_v,
    )

    torch.testing.assert_close(
        paged_output,
        contiguous_output,
    )