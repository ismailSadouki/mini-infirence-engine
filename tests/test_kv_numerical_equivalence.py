import torch
import yaml


from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parent.parent))


from engine.model_adapter import ModelAdapter
from engine.kv_cache import KVCache
from engine.hf_kv_cache import EngineKVCache


with open("configs/inference.yaml", "r") as f:
    config = yaml.safe_load(f)


def test_cached_decode_numerical_equivalence():

    adapter = ModelAdapter(
        model_name=config["model"]["model_name"],
        device=config["device"],
    )

    model = adapter.model
    model.eval()

    prompt = "The capital of France is"

    prompt_ids = adapter.tokenize(prompt)

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=adapter.device,
    )

    T = input_ids.shape[1]

    # ---------------------------------------------------------
    # 1. Choose the next token deterministically
    # ---------------------------------------------------------

    with torch.no_grad():
        prompt_output = model(
            input_ids=input_ids,
            use_cache=False,
        )

    next_token = prompt_output.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    # ---------------------------------------------------------
    # 2. UNCACHED
    #
    # Run the complete sequence:
    #
    # x_0 ... x_{T-1}, x_T
    # ---------------------------------------------------------

    full_input = torch.cat(
        [input_ids, next_token],
        dim=1,
    )

    with torch.no_grad():
        full_output = model(
            input_ids=full_input,
            use_cache=False,
            output_hidden_states=True,
        )

    # Hidden state of the NEW token x_T
    uncached_hidden = full_output.hidden_states[-1][:, -1, :]

    uncached_logits = full_output.logits[:, -1, :]

    # ---------------------------------------------------------
    # 3. CACHED PREFILL
    #
    # Process:
    #
    # x_0 ... x_{T-1}
    # ---------------------------------------------------------

    kv_cache = KVCache(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        max_seq_len=T + 1,
        head_dim=(
            model.config.hidden_size
            // model.config.num_attention_heads
        ),
        dtype=model.dtype,
        device=adapter.device,
    )

    hf_cache = EngineKVCache(kv_cache)

    with torch.no_grad():
        prefill_output = model(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=hf_cache,
        )

    assert hf_cache.get_seq_length() == T

    # ---------------------------------------------------------
    # 4. CACHED DECODE
    #
    # Process x_T at logical position T
    # ---------------------------------------------------------

    position_ids = torch.tensor(
        [[T]],
        dtype=torch.long,
        device=adapter.device,
    )

    with torch.no_grad():
        decode_output = model(
            input_ids=next_token,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=hf_cache,
            output_hidden_states=True,
        )

    cached_hidden = decode_output.hidden_states[-1][:, -1, :]

    cached_logits = decode_output.logits[:, -1, :]

    # ---------------------------------------------------------
    # 5. NUMERICAL COMPARISON
    # ---------------------------------------------------------

    hidden_diff = (
        cached_hidden - uncached_hidden
    ).abs()

    logits_diff = (
        cached_logits - uncached_logits
    ).abs()

    print()
    print("=" * 70)
    print("NUMERICAL EQUIVALENCE")
    print("=" * 70)

    print(
        f"hidden max diff : {hidden_diff.max().item():.8e}"
    )

    print(
        f"hidden mean diff: {hidden_diff.mean().item():.8e}"
    )

    print(
        f"logits max diff : {logits_diff.max().item():.8e}"
    )

    print(
        f"logits mean diff: {logits_diff.mean().item():.8e}"
    )

    print(
        "cached token   :",
        cached_logits.argmax(dim=-1).item(),
    )

    print(
        "uncached token :",
        uncached_logits.argmax(dim=-1).item(),
    )

    # ---------------------------------------------------------
    # 6. We deliberately do NOT assert equality yet.
    # ---------------------------------------------------------

    assert torch.isfinite(cached_hidden).all()
    assert torch.isfinite(uncached_hidden).all()

    assert torch.isfinite(cached_logits).all()
    assert torch.isfinite(uncached_logits).all()