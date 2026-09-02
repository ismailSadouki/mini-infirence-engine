import torch


from pathlib import Path
import sys



sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)



from engine.block_pool import BlockPool
from engine.block_table import BlockTable
from engine.paged_kv_cache import PagedKVCache


def make_paged_cache(
    *,
    num_layers=1,
    num_blocks=8,
    block_size=16,
    num_kv_heads=2,
    head_dim=64,
    dtype=torch.bfloat16,
    device="cuda",
):
    pool = BlockPool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    table = BlockTable(
        block_pool=pool,
        block_size=block_size,
    )

    cache = PagedKVCache(
        block_pool=pool,
        block_table=table,
    )

    return cache


def test_paged_cache_can_store_multi_block_sequence():
    """
    A sequence of length 20 with block_size=16 must use
    at least two physical blocks.
    """

    cache = make_paged_cache()

    seq_id = "seq0"

    for pos in range(20):
        key = torch.full(
            (2, 64),
            float(pos),
            device="cuda",
            dtype=torch.bfloat16,
        )

        value = torch.full(
            (2, 64),
            float(pos + 100),
            device="cuda",
            dtype=torch.bfloat16,
        )

        cache.write_kv(
            seq_id=seq_id,
            pos=pos,
            layer=0,
            key=key,
            value=value,
        )

    keys, values = cache.gather_kv_for_sequence(
        seq_id=seq_id,
        layer=0,
        length=20,
    )

    assert keys.shape == (20, 2, 64)
    assert values.shape == (20, 2, 64)

    # Logical position must be preserved.
    for pos in range(20):
        assert torch.all(keys[pos] == pos)
        assert torch.all(values[pos] == pos + 100)


def test_paged_cache_crosses_block_boundary():
    """
    Explicitly test positions 15 -> 16.

    Position 15 belongs to logical block 0.
    Position 16 belongs to logical block 1.
    """

    cache = make_paged_cache()

    seq_id = "seq_boundary"

    for pos in [15, 16]:
        key = torch.full(
            (2, 64),
            float(pos),
            device="cuda",
            dtype=torch.bfloat16,
        )

        value = torch.full(
            (2, 64),
            float(pos),
            device="cuda",
            dtype=torch.bfloat16,
        )

        cache.write_kv(
            seq_id=seq_id,
            pos=pos,
            layer=0,
            key=key,
            value=value,
        )

    keys, values = cache.gather_kv_for_sequence(
        seq_id=seq_id,
        layer=0,
        length=17,
    )

    assert torch.all(keys[15] == 15)
    assert torch.all(keys[16] == 16)

    assert torch.all(values[15] == 15)
    assert torch.all(values[16] == 16)


def test_paged_attention_matches_contiguous_attention():
    """
    Core M3.4 correctness test.

    The same K/V are stored in:

        1. normal contiguous tensors
        2. paged KV cache

    Gathered paged K/V must produce the same attention output.
    """

    torch.manual_seed(0)

    device = "cuda"
    dtype = torch.bfloat16

    T = 20
    H = 2
    D = 64

    # ---------------------------------------------------------
    # Create reference contiguous K/V
    # ---------------------------------------------------------

    key = torch.randn(
        T,
        H,
        D,
        device=device,
        dtype=dtype,
    )

    value = torch.randn(
        T,
        H,
        D,
        device=device,
        dtype=dtype,
    )

    query = torch.randn(
        H,
        D,
        device=device,
        dtype=dtype,
    )

    # ---------------------------------------------------------
    # Reference attention
    # ---------------------------------------------------------

    scores = torch.einsum(
        "hd,thd->ht",
        query,
        key,
    ) / (D ** 0.5)

    weights = torch.softmax(
        scores,
        dim=-1,
        dtype=torch.float32,
    ).to(dtype)

    reference_output = torch.einsum(
        "ht,thd->hd",
        weights,
        value,
    )

    # ---------------------------------------------------------
    # Store exactly the same K/V in paged cache
    # ---------------------------------------------------------

    cache = make_paged_cache(
        num_layers=1,
        num_blocks=8,
        block_size=16,
        num_kv_heads=H,
        head_dim=D,
        dtype=dtype,
        device=device,
    )

    seq_id = "attention_test"

    for pos in range(T):
        cache.write_kv(
            seq_id=seq_id,
            pos=pos,
            layer=0,
            key=key[pos],
            value=value[pos],
        )

    # ---------------------------------------------------------
    # Gather
    # ---------------------------------------------------------

    gathered_key, gathered_value = (
        cache.gather_kv_for_sequence(
            seq_id=seq_id,
            layer=0,
            length=T,
        )
    )

    # ---------------------------------------------------------
    # Attention using gathered paged K/V
    # ---------------------------------------------------------

    paged_scores = torch.einsum(
        "hd,thd->ht",
        query,
        gathered_key,
    ) / (D ** 0.5)

    paged_weights = torch.softmax(
        paged_scores,
        dim=-1,
        dtype=torch.float32,
    ).to(dtype)

    paged_output = torch.einsum(
        "ht,thd->hd",
        paged_weights,
        gathered_value,
    )

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    torch.testing.assert_close(
        paged_output,
        reference_output,
        rtol=2e-2,
        atol=2e-2,
    )


def test_paged_cache_uses_multiple_physical_blocks():
    """
    Verify that a sequence longer than block_size actually
    receives multiple physical blocks.
    """

    cache = make_paged_cache(
        num_blocks=8,
        block_size=16,
    )

    seq_id = "multi_block"

    for pos in range(33):
        key = torch.zeros(
            2,
            64,
            device="cuda",
            dtype=torch.bfloat16,
        )

        value = torch.zeros(
            2,
            64,
            device="cuda",
            dtype=torch.bfloat16,
        )

        cache.write_kv(
            seq_id=seq_id,
            pos=pos,
            layer=0,
            key=key,
            value=value,
        )

    blocks = cache.block_table.tables[seq_id].physical_blocks

    # 33 positions with B=16 => 3 logical blocks.
    assert len(blocks) == 3

    # Physical blocks must be distinct.
    assert len(set(blocks)) == 3