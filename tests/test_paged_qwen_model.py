import torch


from pathlib import Path
import sys



sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)



from engine.model_adapter import ModelAdapter
from engine.block_pool import BlockPool
from engine.block_table import BlockTable
from engine.paged_kv_cache import PagedKVCache
from engine.kv_cache import KVCache


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def make_paged_cache(adapter, num_blocks=64, block_size=16):

    config = adapter.model.config

    pool = BlockPool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=adapter.model.config.hidden_size
        // config.num_attention_heads,
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

    config = adapter.model.config

    return KVCache(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size
        // config.num_attention_heads,
        max_seq_len=max_seq_len,
        dtype=adapter.dtype,
        device=adapter.device,
    )


def test_paged_prefill_matches_contiguous_prefill():

    adapter = ModelAdapter(
        MODEL_NAME,
        device="cuda",
        dtype=torch.bfloat16,
    )

    prompt = "The capital of France is"

    token_ids = adapter.tokenize(prompt)
    input_ids = adapter.to_tensor(token_ids)

    # ---------------------------------------------------------
    # Contiguous cache
    # ---------------------------------------------------------

    contiguous_cache = make_contiguous_cache(
        adapter,
        max_seq_len=input_ids.shape[1],
    )

    contiguous_logits = adapter.forward_prefill_cached(
        input_ids=input_ids,
        cache=contiguous_cache,
    )

    # ---------------------------------------------------------
    # Paged cache
    # ---------------------------------------------------------

    paged_cache = make_paged_cache(adapter)

    paged_logits = adapter.forward_prefill_paged(
        input_ids=input_ids,
        cache=paged_cache,
        seq_id="test_sequence",
    )

    # ---------------------------------------------------------
    # Compare final token logits
    # ---------------------------------------------------------

    contiguous_last = contiguous_logits[:, -1, :]
    paged_last = paged_logits[:, -1, :]

    max_diff = (
        contiguous_last - paged_last
    ).abs().max().item()

    print(f"\nMAX LOGIT DIFF: {max_diff}")

    torch.testing.assert_close(
        paged_last,
        contiguous_last,
        rtol=2e-2,
        atol=2e-2,
    )