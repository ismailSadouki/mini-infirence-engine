from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parent.parent))


from engine.kv_cache import KVCache


def create_request_cache(
    request,
    adapter,
    max_new_tokens,
):
    model = adapter.model

    return KVCache(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        max_seq_len=request.prompt_len + max_new_tokens,
        head_dim=(
            model.config.hidden_size
            // model.config.num_attention_heads
        ),
        dtype=adapter.dtype,
        device=adapter.device,
    )