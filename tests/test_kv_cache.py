

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parent.parent))


import torch
import pytest

from engine.kv_cache import KVCache
from engine.hf_kv_cache import EngineKVCache
from engine.model_adapter import ModelAdapter

import yaml

with open("configs/inference.yaml", "r") as f:
    config = yaml.safe_load(f)






@pytest.fixture
def cache():
    return KVCache(
        num_layers=24,
        num_kv_heads=2,
        max_seq_len=16,
        head_dim=64,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

def test_cache_allocation(cache):
    assert len(cache.key_cache) == 24
    assert len(cache.value_cache) == 24

    assert cache.key_cache[0].shape == (
        1,
        2,
        16,
        64,
    )

    assert cache.value_cache[0].shape == (
        1,
        2,
        16,
        64,
    )





def test_prefill_update(cache):
    seq_len = 5

    key = torch.ones(
        1, 2, seq_len, 64,
        dtype=torch.bfloat16,
    )
    value = torch.full(
        (1, 2, seq_len, 64),
        2,
        dtype=torch.bfloat16,
    )

    cache.update(
        layer=0,
        key=key,
        value=value,
        start_position=0,
    )

    cached_k, cached_v = cache.read_prefix(
        layer=0,
        length=5,
    )

    assert torch.equal(cached_k, key)
    assert torch.equal(cached_v, value)






def test_decode_update(cache):
    # Prefill positions 0..4
    key = torch.ones(
        1, 2, 5, 64,
        dtype=torch.bfloat16,
    )

    value = torch.ones(
        1, 2, 5, 64,
        dtype=torch.bfloat16,
    )

    cache.update(
        layer=0,
        key=key,
        value=value,
        start_position=0,
    )
    # Decode token at position 5
    new_key = torch.full(
        (1, 2, 1, 64),
        3,
        dtype=torch.bfloat16,
    )

    new_value = torch.full(
        (1, 2, 1, 64),
        4,
        dtype=torch.bfloat16,
    )

    cache.update(
        layer=0,
        key=new_key,
        value=new_value,
        start_position=5,
    )

    cached_k, cached_v = cache.read_prefix(
        layer=0,
        length=6,
    )
    assert torch.equal(
        cached_k[:, :, :5, :],
        key,
    )

    assert torch.equal(
        cached_v[:, :, :5, :],
        value,
    )

    assert torch.equal(
        cached_k[:, :, 5:6, :],
        new_key,
    )

    assert torch.equal(
        cached_v[:, :, 5:6, :],
        new_value,
    )


def test_wrong_shape_rejected(cache):
    key = torch.randn(
        1, 14, 1, 64,
        dtype=torch.bfloat16,
    )

    value = torch.randn(
        1, 14, 1, 64,
        dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError):
        cache.update(
            layer=0,
            key=key,
            value=value,
            start_position=0,
        )
def test_cache_overflow_rejected(cache):
    key = torch.randn(
        1, 2, 2, 64,
        dtype=torch.bfloat16,
    )

    value = torch.randn(
        1, 2, 2, 64,
        dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError):
        cache.update(
            layer=0,
            key=key,
            value=value,
            start_position=15,
        )


def test_invalid_read_rejected(cache):
    with pytest.raises(ValueError):
        cache.read_prefix(
            layer=0,
            length=17,
        )














import torch

from engine.kv_cache import KVCache


def test_update_and_read_prefix():
    cache = KVCache(
        num_layers=2,
        num_kv_heads=2,
        max_seq_len=8,
        head_dim=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    key = torch.randn(
        1, 2, 1, 4,
        dtype=torch.bfloat16,
    )

    value = torch.randn(
        1, 2, 1, 4,
        dtype=torch.bfloat16,
    )

    cache.update(
        layer=0,
        key=key,
        value=value,
        start_position=3,
    )

    cached_k, cached_v = cache.read_prefix(
        layer=0,
        length=4,
    )

    assert cached_k.shape == (1, 2, 4, 4)
    assert cached_v.shape == (1, 2, 4, 4)

    torch.testing.assert_close(
        cached_k[:, :, 3:4, :],
        key,
    )

    torch.testing.assert_close(
        cached_v[:, :, 3:4, :],
        value,
    )


def test_update_rejects_invalid_position():
    cache = KVCache(
        num_layers=2,
        num_kv_heads=2,
        max_seq_len=8,
        head_dim=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    key = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16)
    value = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16)

    try:
        cache.update(
            layer=0,
            key=key,
            value=value,
            start_position=8,
        )
        assert False
    except ValueError:
        pass




def test_engine_kv_cache_prefill_and_decode():

    cache = KVCache(
        num_layers=24,
        num_kv_heads=2,
        max_seq_len=16,
        head_dim=64,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    hf_cache = EngineKVCache(cache)

    # Prefill: 3 tokens
    key = torch.ones(
        1, 2, 3, 64,
        dtype=torch.bfloat16,
    )

    value = torch.ones(
        1, 2, 3, 64,
        dtype=torch.bfloat16,
    )

    hf_cache.update(
        key_states=key,
        value_states=value,
        layer_idx=0,
    )

    assert hf_cache.get_seq_length() == 3

    # Decode: 1 token
    key_decode = torch.full(
        (1, 2, 1, 64),
        2.0,
        dtype=torch.bfloat16,
    )

    value_decode = torch.full(
        (1, 2, 1, 64),
        2.0,
        dtype=torch.bfloat16,
    )

    hf_cache.update(
        key_states=key_decode,
        value_states=value_decode,
        layer_idx=0,
    )

    assert hf_cache.get_seq_length() == 4

    # First 3 positions came from prefill.
    assert torch.allclose(
        cache.key_cache[0][:, :, :3, :],
        key,
    )

    # Position 3 came from decode.
    assert torch.allclose(
        cache.key_cache[0][:, :, 3:4, :],
        key_decode,
    )


def test_model_prefill_populates_cache():

    adapter = ModelAdapter(
        model_name=config['model']['model_name'],
        device=config['device'],
    )

    prompt = "The capital of France is"

    prompt_ids = adapter.tokenize(prompt)

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=adapter.device,
    )

    kv_cache = KVCache(
        num_layers=adapter.model.config.num_hidden_layers,
        num_kv_heads=adapter.model.config.num_key_value_heads,
        max_seq_len=len(prompt_ids) + 16,
        head_dim=(
            adapter.model.config.hidden_size
            // adapter.model.config.num_attention_heads
        ),
        dtype=adapter.model.dtype,
        device=adapter.device,
    )

    hf_cache = EngineKVCache(kv_cache)

    outputs = adapter.model(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=hf_cache,
    )

    assert outputs.logits.shape[1] == len(prompt_ids)

    assert (
        hf_cache.get_seq_length()
        == len(prompt_ids)
    )


def test_model_decode_appends_to_cache():

    adapter = ModelAdapter(
        model_name=config["model"]["model_name"],
        device=config["device"],
    )

    prompt = "The capital of France is"

    prompt_ids = adapter.tokenize(prompt)

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=adapter.device,
    )

    kv_cache = KVCache(
        num_layers=adapter.model.config.num_hidden_layers,
        num_kv_heads=adapter.model.config.num_key_value_heads,
        max_seq_len=len(prompt_ids) + 16,
        head_dim=(
            adapter.model.config.hidden_size
            // adapter.model.config.num_attention_heads
        ),
        dtype=adapter.model.dtype,
        device=adapter.device,
    )

    hf_cache = EngineKVCache(kv_cache)

    # ----------------
    # PREFILL
    # ----------------

    outputs = adapter.model(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=hf_cache,
    )

    prompt_length = len(prompt_ids)

    assert hf_cache.get_seq_length() == prompt_length

    # ----------------
    # DECODE
    # ----------------

    last_token = torch.tensor(
        [[1]],
        dtype=torch.long,
        device=adapter.device,
    )

    outputs = adapter.model(
        input_ids=last_token,
        use_cache=True,
        past_key_values=hf_cache,
    )

    # One new token should now exist.
    assert hf_cache.get_seq_length() == prompt_length + 1



def test_decode_uses_correct_position_ids():

    adapter = ModelAdapter(
        model_name=config["model"]["model_name"],
        device=config["device"],
    )

    prompt = "The capital of France is"

    prompt_ids = adapter.tokenize(prompt)

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=adapter.device,
    )

    kv_cache = KVCache(
        num_layers=adapter.model.config.num_hidden_layers,
        num_kv_heads=adapter.model.config.num_key_value_heads,
        max_seq_len=len(prompt_ids) + 16,
        head_dim=(
            adapter.model.config.hidden_size
            // adapter.model.config.num_attention_heads
        ),
        dtype=adapter.model.dtype,
        device=adapter.device,
    )

    hf_cache = EngineKVCache(kv_cache)

    # ----------------
    # PREFILL
    # ----------------

    adapter.model(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=hf_cache,
    )

    prompt_length = len(prompt_ids)

    assert hf_cache.get_seq_length() == prompt_length

    # ----------------
    # DECODE
    # ----------------

    decode_position_ids = torch.tensor(
        [[prompt_length]],
        dtype=torch.long,
        device=adapter.device,
    )

    last_token = torch.tensor(
        [[1]],
        dtype=torch.long,
        device=adapter.device,
    )

    adapter.model(
        input_ids=last_token,
        # position_ids=decode_position_ids,
        use_cache=True,
        past_key_values=hf_cache,
    )

    assert hf_cache.get_seq_length() == prompt_length + 1



def test_cached_decode_matches_uncached():

    adapter = ModelAdapter(
        model_name=config["model"]["model_name"],
        device=config["device"],
    )

    prompt = "The capital of France is"

    prompt_ids = adapter.tokenize(prompt)

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=adapter.device,
    )

    # ------------------------------------------------
    # 1. UNCACHED
    # ------------------------------------------------

    uncached_outputs = adapter.model(
        input_ids=input_ids,
        use_cache=False,
    )

    first_token = torch.argmax(
        uncached_outputs.logits[:, -1, :],
        dim=-1,
    )

    # ------------------------------------------------
    # 2. CACHED PREFILL
    # ------------------------------------------------

    kv_cache = KVCache(
        num_layers=adapter.model.config.num_hidden_layers,
        num_kv_heads=adapter.model.config.num_key_value_heads,
        max_seq_len=len(prompt_ids) + 16,
        head_dim=(
            adapter.model.config.hidden_size
            // adapter.model.config.num_attention_heads
        ),
        dtype=adapter.model.dtype,
        device=adapter.device,
    )

    hf_cache = EngineKVCache(kv_cache)

    prefill_outputs = adapter.model(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=hf_cache,
    )

    cached_first_token = torch.argmax(
        prefill_outputs.logits[:, -1, :],
        dim=-1,
    )


    

    assert torch.equal(
        first_token,
        cached_first_token,
    )


    print(
        "BEFORE DECODE",
        hf_cache.get_seq_length(),
    )

    for layer in range(
        adapter.model.config.num_hidden_layers
    ):
        k = hf_cache.kv_cache.key_cache[layer][
            :, :, :5, :
        ]

        v = hf_cache.kv_cache.value_cache[layer][
            :, :, :5, :
        ]

        print(
            layer,
            "K:",
            k.shape,
            "V:",
            v.shape,
        )
    # ------------------------------------------------
    # 3. CACHED DECODE
    # ------------------------------------------------

    print(
        "CACHE LENGTH BEFORE DECODE:",
        hf_cache.get_seq_length(),
    )

    print(
        "LAYER LENGTHS:",
        hf_cache._layer_seq_lengths,
    )
    cached_states = {}
    hooks = []

    for i, layer in enumerate(adapter.model.model.layers):

        def make_hook(i):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                cached_states[i] = hidden.detach().clone()
            return hook

        hooks.append(
            layer.register_forward_hook(make_hook(i))
        )
    decode_outputs = adapter.model(
        input_ids=first_token.unsqueeze(0),
        position_ids=torch.tensor(
            [[len(prompt_ids)]],
            dtype=torch.long,
            device=adapter.device,
        ),
        use_cache=True,
        past_key_values=hf_cache,
        output_hidden_states=True,
    )
    for hook in hooks:
        hook.remove()
    cached_hidden = decode_outputs.hidden_states[-1][:, -1, :]

    cached_decode_logits = decode_outputs.logits[:, -1, :]

    # ------------------------------------------------
    # 4. UNCACHED FULL SEQUENCE
    # ------------------------------------------------

    full_input_ids = torch.cat(
        [
            input_ids,
            first_token.unsqueeze(0),
        ],
        dim=1,
    )


    full_states = {}
    hooks = []

    for i, layer in enumerate(adapter.model.model.layers):

        def make_hook(i):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                full_states[i] = hidden.detach().clone()
            return hook

        hooks.append(
            layer.register_forward_hook(make_hook(i))
        )

    full_outputs = adapter.model(
        input_ids=full_input_ids,
        use_cache=False,
        output_hidden_states=True,
    )
    for hook in hooks:
        hook.remove()


    for i in range(
        adapter.model.config.num_hidden_layers
    ):
        cached = cached_states[i][:, -1, :]
        full = full_states[i][:, -1, :]

        diff = (cached - full).abs()

        print(
            f"LAYER {i:02d} | "
            f"max={diff.max().item():.6f} | "
            f"mean={diff.mean().item():.6f}"
        )
    uncached_hidden = full_outputs.hidden_states[-1][:, -1, :]


    diff = (cached_hidden - uncached_hidden).abs()

    print("HIDDEN MAX DIFF :", diff.max().item())
    print("HIDDEN MEAN DIFF:", diff.mean().item())

    uncached_decode_logits = (
        full_outputs.logits[:, -1, :]
    )

    # ------------------------------------------------
    # 5. COMPARE
    # ------------------------------------------------

    max_diff = (
        cached_decode_logits - uncached_decode_logits
    ).abs().max().item()

    mean_diff = (
        cached_decode_logits - uncached_decode_logits
    ).abs().mean().item()

    cached_token = cached_decode_logits.argmax(dim=-1)
    uncached_token = uncached_decode_logits.argmax(dim=-1)

    print(f"LOGITS MAX DIFF:  {max_diff}")
    print(f"LOGITS MEAN DIFF: {mean_diff}")
    print(f"CACHED TOKEN:    {cached_token.item()}")
    print(f"UNCACHED TOKEN:  {uncached_token.item()}")

    assert torch.equal(
        cached_token,
        uncached_token,
    )


def test_cached_generation_matches_full_generation():

    adapter = ModelAdapter(
        model_name=config["model"]["model_name"],
        device=config["device"],
    )

    model = adapter.model
    model.eval()

    prompt = "The capital of France is"

    input_ids = torch.tensor(
        [adapter.tokenize(prompt)],
        dtype=torch.long,
        device=adapter.device,
    )

    prompt_len = input_ids.shape[1]

    # ============================================================
    # ENGINE CACHE
    # ============================================================

    kv_cache = KVCache(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        max_seq_len=64,
        head_dim=(
            model.config.hidden_size
            // model.config.num_attention_heads
        ),
        dtype=model.dtype,
        device=adapter.device,
    )

    hf_cache = EngineKVCache(kv_cache)

    # ============================================================
    # PREFILL
    # ============================================================

    with torch.no_grad():
        cached_output = model(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=hf_cache,
        )

    cached_tokens = input_ids.clone()

    # ============================================================
    # CACHED GENERATION
    # ============================================================

    NUM_NEW_TOKENS = 10

    for step in range(NUM_NEW_TOKENS):

        next_token = torch.argmax(
            cached_output.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        cached_tokens = torch.cat(
            [cached_tokens, next_token],
            dim=1,
        )

        position_ids = torch.tensor(
            [[cached_tokens.shape[1] - 1]],
            dtype=torch.long,
            device=adapter.device,
        )

        with torch.no_grad():
            cached_output = model(
                input_ids=next_token,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=hf_cache,
            )

    # ============================================================
    # FULL / NO-CACHE GENERATION
    # ============================================================

    full_tokens = input_ids.clone()

    for step in range(NUM_NEW_TOKENS):

        with torch.no_grad():
            full_output = model(
                input_ids=full_tokens,
                use_cache=False,
            )

        next_token = torch.argmax(
            full_output.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        full_tokens = torch.cat(
            [full_tokens, next_token],
            dim=1,
        )

    # ============================================================
    # COMPARE
    # ============================================================

    print()
    print("=" * 80)
    print("CACHED VS FULL GENERATION")
    print("=" * 80)

    print("cached:", cached_tokens.tolist())
    print("full:  ", full_tokens.tolist())

    generated_cached = cached_tokens[:, prompt_len:]
    generated_full = full_tokens[:, prompt_len:]

    print()
    print("cached generated:", generated_cached.tolist())
    print("full generated:  ", generated_full.tolist())

    assert torch.equal(
        generated_cached,
        generated_full,
    )