import torch



from pathlib import Path
import sys



sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)



from engine.model_adapter import ModelAdapter
from engine.kv_cache import KVCache
from engine.block_pool import BlockPool
from engine.block_table import BlockTable
from engine.paged_kv_cache import PagedKVCache


def make_paged_cache(adapter, block_size=16, num_blocks=8):
    config = adapter.cached_model.model.config

    pool = BlockPool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=adapter.dtype,
        device=adapter.device,
    )

    table = BlockTable(
        block_pool=pool,
        block_size=block_size,
    )

    return PagedKVCache(
        block_pool=pool,
        block_table=table,
    )


def make_contiguous_cache(adapter, max_seq_len):
    config = adapter.cached_model.model.config

    return KVCache(
        num_layers=config.num_hidden_layers,
        max_seq_len=max_seq_len,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=adapter.dtype,
        device=adapter.device,
    )


@torch.inference_mode()
def test_paged_prefill_crosses_block_boundary():

    adapter = ModelAdapter(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda",
        dtype=torch.bfloat16,
    )

    # ---------------------------------------------------------
    # Create > 1 block worth of tokens.
    #
    # block_size = 16
    # sequence length = 20
    #
    # Therefore:
    #
    # logical positions 0..15  -> block 1
    # logical positions 16..19 -> block 2
    # ---------------------------------------------------------

    token_ids = adapter.tokenize(
        "The capital of France is Paris. "
        "The capital of Germany is Berlin. "
        "The capital of Italy is Rome."
    )

    assert len(token_ids) >= 20

    input_ids = adapter.to_tensor(
        token_ids[:20]
    )

    assert input_ids.shape == (1, 20)

    # ---------------------------------------------------------
    # Contiguous cache
    # ---------------------------------------------------------

    contiguous_cache = make_contiguous_cache(
        adapter,
        max_seq_len=20,
    )

    contiguous_logits = (
        adapter.forward_prefill_cached(
            input_ids=input_ids,
            cache=contiguous_cache,
        )
    )

    # ---------------------------------------------------------
    # Paged cache
    # ---------------------------------------------------------

    paged_cache = make_paged_cache(
        adapter,
        block_size=16,
        num_blocks=8,
    )

    seq_id = 0

    paged_logits = (
        adapter.forward_prefill_paged(
            input_ids=input_ids,
            cache=paged_cache,
            seq_id=seq_id,
        )
    )

    # ---------------------------------------------------------
    # Check that multiple physical blocks were actually used.
    # ---------------------------------------------------------

    physical_blocks = (
        paged_cache.block_table.tables[
            seq_id
        ].physical_blocks
    )

    assert len(physical_blocks) == 2

    # ---------------------------------------------------------
    # Compare outputs.
    # ---------------------------------------------------------

    assert contiguous_logits.shape == paged_logits.shape

    max_diff = (
        contiguous_logits.float()
        - paged_logits.float()
    ).abs().max().item()

    mean_diff = (
        contiguous_logits.float()
        - paged_logits.float()
    ).abs().mean().item()

    print(f"\nMAX DIFF  = {max_diff}")
    print(f"MEAN DIFF = {mean_diff}")

    torch.testing.assert_close(
        paged_logits.float(),
        contiguous_logits.float(),
        rtol=2e-2,
        atol=2e-2,
    )


@torch.inference_mode()
def test_paged_decode_matches_contiguous_decode_after_multiblock_prefill():

    adapter = ModelAdapter(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda",
        dtype=torch.bfloat16,
    )

    # ---------------------------------------------------------
    # 20-token prompt
    #
    # block_size = 16
    #
    # positions:
    #   0..15  -> block 0
    #   16..19 -> block 1
    # ---------------------------------------------------------

    token_ids = adapter.tokenize(
        "The capital of France is Paris. "
        "The capital of Germany is Berlin. "
        "The capital of Italy is Rome."
    )

    assert len(token_ids) >= 20

    input_ids = adapter.to_tensor(
        token_ids[:20]
    )

    assert input_ids.shape == (1, 20)

    # ---------------------------------------------------------
    # Create both caches
    # ---------------------------------------------------------

    contiguous_cache = make_contiguous_cache(
        adapter,
        max_seq_len=32,
    )

    paged_cache = make_paged_cache(
        adapter,
        block_size=16,
        num_blocks=8,
    )

    seq_id = 0

    # ---------------------------------------------------------
    # PREFILL
    # ---------------------------------------------------------

    contiguous_logits = adapter.forward_prefill_cached(
        input_ids=input_ids,
        cache=contiguous_cache,
    )

    paged_logits = adapter.forward_prefill_paged(
        input_ids=input_ids,
        cache=paged_cache,
        seq_id=seq_id,
    )

    # ---------------------------------------------------------
    # Make sure paged prefill actually used 2 blocks.
    # ---------------------------------------------------------

    physical_blocks = (
        paged_cache.block_table
        .tables[seq_id]
        .physical_blocks
    )

    assert len(physical_blocks) == 2

    # ---------------------------------------------------------
    # First generated token.
    #
    # Use the SAME token for both decode paths.
    # ---------------------------------------------------------

    next_token = torch.argmax(
        contiguous_logits[0, -1]
    ).item()

    last_token = torch.tensor(
        [[next_token]],
        dtype=torch.long,
        device=adapter.device,
    )

    # ---------------------------------------------------------
    # DECODE POSITION 20
    #
    # Existing prompt occupies:
    #
    #   positions 0..19
    #
    # Therefore the new token is at position 20.
    # ---------------------------------------------------------

    contiguous_decode_logits = (
        adapter.forward_decode_cached(
            last_token=last_token,
            cache=contiguous_cache,
            position=20,
        )
    )

    paged_decode_logits = (
        adapter.forward_decode_paged(
            last_token=last_token,
            cache=paged_cache,
            seq_id=seq_id,
            position=20,
        )
    )

    # ---------------------------------------------------------
    # Compare decode logits
    # ---------------------------------------------------------

    assert (
        contiguous_decode_logits.shape
        == paged_decode_logits.shape
    )

    diff = (
        contiguous_decode_logits.float()
        - paged_decode_logits.float()
    ).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nDECODE MAX DIFF  = {max_diff}")
    print(f"DECODE MEAN DIFF = {mean_diff}")

    torch.testing.assert_close(
        paged_decode_logits.float(),
        contiguous_decode_logits.float(),
        rtol=2e-2,
        atol=2e-2,
    )


@torch.inference_mode()
def test_paged_decode_crosses_block_boundary():

    adapter = ModelAdapter(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda",
        dtype=torch.bfloat16,
    )

    # ---------------------------------------------------------
    # 20-token prompt
    #
    # block_size = 16
    #
    # block 0 -> positions 0..15
    # block 1 -> positions 16..31
    # ---------------------------------------------------------

    token_ids = adapter.tokenize(
        "The capital of France is Paris. "
        "The capital of Germany is Berlin. "
        "The capital of Italy is Rome."
    )

    assert len(token_ids) >= 20

    input_ids = adapter.to_tensor(
        token_ids[:20]
    )

    assert input_ids.shape == (1, 20)

    # ---------------------------------------------------------
    # Create caches
    # ---------------------------------------------------------

    contiguous_cache = make_contiguous_cache(
        adapter,
        max_seq_len=40,
    )

    paged_cache = make_paged_cache(
        adapter,
        block_size=16,
        num_blocks=8,
    )

    seq_id = 0

    # ---------------------------------------------------------
    # PREFILL positions 0..19
    # ---------------------------------------------------------

    contiguous_logits = adapter.forward_prefill_cached(
        input_ids=input_ids,
        cache=contiguous_cache,
    )

    paged_logits = adapter.forward_prefill_paged(
        input_ids=input_ids,
        cache=paged_cache,
        seq_id=seq_id,
    )

    # We should have two blocks.
    blocks = (
        paged_cache.block_table
        .tables[seq_id]
        .physical_blocks
    )

    assert len(blocks) == 2

    # ---------------------------------------------------------
    # Start decoding from the same token.
    # ---------------------------------------------------------

    next_token = torch.argmax(
        contiguous_logits[0, -1]
    ).item()

    # ---------------------------------------------------------
    # Decode positions 20..32
    # ---------------------------------------------------------

    for position in range(20, 33):

        last_token = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=adapter.device,
        )

        # ---------------------------------------------
        # Contiguous
        # ---------------------------------------------

        contiguous_logits = (
            adapter.forward_decode_cached(
                last_token=last_token,
                cache=contiguous_cache,
                position=position,
            )
        )

        # ---------------------------------------------
        # Paged
        # ---------------------------------------------

        paged_logits = (
            adapter.forward_decode_paged(
                last_token=last_token,
                cache=paged_cache,
                seq_id=seq_id,
                position=position,
            )
        )

        # ---------------------------------------------
        # Compare
        # ---------------------------------------------

        diff = (
            contiguous_logits.float()
            - paged_logits.float()
        ).abs()

        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(
            f"\nposition={position}"
            f"  max_diff={max_diff}"
            f"  mean_diff={mean_diff}"
        )

        torch.testing.assert_close(
            paged_logits.float(),
            contiguous_logits.float(),
            rtol=2e-2,
            atol=2e-2,
        )

        # ---------------------------------------------
        # Greedy next token
        #
        # Both implementations receive the SAME token
        # at every position.
        # ---------------------------------------------

        next_token = torch.argmax(
            contiguous_logits[0, -1]
        ).item()

    # ---------------------------------------------------------
    # Position 32 requires a THIRD physical block.
    #
    # 0..15  -> block 0
    # 16..31 -> block 1
    # 32     -> block 2
    # ---------------------------------------------------------

    blocks = (
        paged_cache.block_table
        .tables[seq_id]
        .physical_blocks
    )

    assert len(blocks) == 3












@torch.inference_mode()
def test_paged_generation_matches_contiguous_generation():

    adapter = ModelAdapter(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda",
        dtype=torch.bfloat16,
    )

    # ---------------------------------------------------------
    # Prompt > 16 tokens so we exercise multiple paged blocks.
    # ---------------------------------------------------------

    token_ids = adapter.tokenize(
        "The capital of France is Paris. "
        "The capital of Germany is Berlin. "
        "The capital of Italy is Rome."
    )

    assert len(token_ids) >= 20

    input_ids = adapter.to_tensor(
        token_ids[:20]
    )

    assert input_ids.shape == (1, 20)

    num_new_tokens = 5

    # ---------------------------------------------------------
    # CONTIGUOUS CACHE
    # ---------------------------------------------------------

    contiguous_cache = make_contiguous_cache(
        adapter,
        max_seq_len=20 + num_new_tokens,
    )

    contiguous_logits = (
        adapter.forward_prefill_cached(
            input_ids=input_ids,
            cache=contiguous_cache,
        )
    )

    contiguous_generated = []

    next_token = torch.argmax(
        contiguous_logits[0, -1]
    ).item()

    contiguous_generated.append(next_token)

    position = 20

    for _ in range(num_new_tokens - 1):

        last_token = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=adapter.device,
        )

        logits = adapter.forward_decode_cached(
            last_token=last_token,
            cache=contiguous_cache,
            position=position,
        )

        next_token = torch.argmax(
            logits[0, -1]
        ).item()

        contiguous_generated.append(
            next_token
        )

        position += 1

    # ---------------------------------------------------------
    # PAGED CACHE
    # ---------------------------------------------------------

    paged_cache = make_paged_cache(
        adapter,
        block_size=16,
        num_blocks=8,
    )

    seq_id = 0

    paged_logits = (
        adapter.forward_prefill_paged(
            input_ids=input_ids,
            cache=paged_cache,
            seq_id=seq_id,
        )
    )

    paged_generated = []

    next_token = torch.argmax(
        paged_logits[0, -1]
    ).item()

    paged_generated.append(next_token)

    position = 20

    for _ in range(num_new_tokens - 1):

        last_token = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=adapter.device,
        )

        logits = adapter.forward_decode_paged(
            last_token=last_token,
            cache=paged_cache,
            seq_id=seq_id,
            position=position,
        )

        next_token = torch.argmax(
            logits[0, -1]
        ).item()

        paged_generated.append(
            next_token
        )

        position += 1

    # ---------------------------------------------------------
    # Compare generated token IDs
    # ---------------------------------------------------------

    print(
        f"\nContiguous tokens: {contiguous_generated}"
    )

    print(
        f"Paged tokens:      {paged_generated}"
    )

    assert paged_generated == contiguous_generated

    # ---------------------------------------------------------
    # Compare decoded text
    # ---------------------------------------------------------

    contiguous_text = adapter.decode(
        contiguous_generated
    )

    paged_text = adapter.decode(
        paged_generated
    )

    print(
        f"\nContiguous text: {contiguous_text!r}"
    )

    print(
        f"Paged text:      {paged_text!r}"
    )

    assert paged_text == contiguous_text

    # ---------------------------------------------------------
    # Verify we crossed into the third block.
    #
    # Prompt: positions 0..19
    # Decode: positions 20..23
    #
    # This test doesn't reach position 32, so we expect
    # exactly two physical blocks.
    # ---------------------------------------------------------

    physical_blocks = (
        paged_cache.block_table
        .tables[seq_id]
        .physical_blocks
    )

    assert len(physical_blocks) == 2