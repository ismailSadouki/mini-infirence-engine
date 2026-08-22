
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





# ============================================================
# CONFIG
# ============================================================

ATOL = 1e-2
RTOL = 1e-2

PROMPT = "The capital of France is"


# ============================================================
# HELPERS
# ============================================================

def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def mean_diff(a, b):
    return (a.float() - b.float()).abs().mean().item()


def compare(name, cached, uncached):
    diff = (cached.float() - uncached.float()).abs()

    print(
        f"{name:30s} "
        f"shape_cached={tuple(cached.shape)} "
        f"shape_full={tuple(uncached.shape)} "
        f"max={diff.max().item():.8f} "
        f"mean={diff.mean().item():.8f}"
    )


def get_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    return output


# ============================================================
# TEST
# ============================================================

def test_kv_cache_attention_internals():

    adapter = ModelAdapter(
        model_name=config["model"]["model_name"],
        device=config["device"],
    )

    model = adapter.model
    model.eval()

    device = adapter.device

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    prompt_ids = adapter.tokenize(PROMPT)

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    prompt_len = input_ids.shape[1]

    print("\n" + "=" * 100)
    print("INPUT")
    print("=" * 100)

    print("prompt:", PROMPT)
    print("input_ids:", input_ids)
    print("prompt_len:", prompt_len)

    # --------------------------------------------------------
    # PREFILL WITHOUT CACHE
    # --------------------------------------------------------

    with torch.no_grad():

        uncached_prefill = model(
            input_ids=input_ids,
            use_cache=False,
        )

    first_token = torch.argmax(
        uncached_prefill.logits[:, -1, :],
        dim=-1,
    )

    print("\nfirst generated token:", first_token.tolist())

    # --------------------------------------------------------
    # CREATE CACHE
    # --------------------------------------------------------

    kv_cache = KVCache(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        max_seq_len=prompt_len + 16,
        head_dim=(
            model.config.hidden_size
            // model.config.num_attention_heads
        ),
        dtype=model.dtype,
        device=device,
    )

    hf_cache = EngineKVCache(kv_cache)

    # --------------------------------------------------------
    # CACHED PREFILL
    # --------------------------------------------------------

    with torch.no_grad():

        cached_prefill = model(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=hf_cache,
        )

    assert torch.equal(
        torch.argmax(
            cached_prefill.logits[:, -1, :],
            dim=-1,
        ),
        first_token,
    )

    # --------------------------------------------------------
    # WE WILL INSPECT LAYER 4
    # --------------------------------------------------------

    layer_idx = 4

    layer = model.model.layers[layer_idx]

    print("\n" + "=" * 100)
    print(f"INSPECTING LAYER {layer_idx}")
    print("=" * 100)

    print(layer.self_attn)

    # --------------------------------------------------------
    # CAPTURE PROJECTION OUTPUTS
    # --------------------------------------------------------

    cached = {}
    full = {}

    cached_hooks = []
    full_hooks = []

    # ========================================================
    # HOOK FACTORY
    # ========================================================

    def make_projection_hook(storage, name):

        def hook(module, inputs, output):

            if isinstance(output, tuple):
                output = output[0]

            storage[name] = output.detach().clone()

        return hook

    # --------------------------------------------------------
    # REGISTER HOOKS ON Q/K/V/O
    # --------------------------------------------------------

    for name in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]:

        module = getattr(layer.self_attn, name)

        cached_hooks.append(
            module.register_forward_hook(
                make_projection_hook(cached, name)
            )
        )

        full_hooks.append(
            module.register_forward_hook(
                make_projection_hook(full, name)
            )
        )

    # ========================================================
    # CACHED DECODE
    # ========================================================

    decode_input = first_token.unsqueeze(0)

    position_ids = torch.tensor(
        [[prompt_len]],
        dtype=torch.long,
        device=device,
    )

    print("\n" + "=" * 100)
    print("CACHED DECODE")
    print("=" * 100)

    with torch.no_grad():

        cached_decode = model(
            input_ids=decode_input,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=hf_cache,
        )

    # ========================================================
    # UNCACHED FULL FORWARD
    # ========================================================

    full_input_ids = torch.cat(
        [
            input_ids,
            first_token.unsqueeze(0),
        ],
        dim=1,
    )

    print("\n" + "=" * 100)
    print("UNCACHED FULL FORWARD")
    print("=" * 100)

    with torch.no_grad():

        uncached_full = model(
            input_ids=full_input_ids,
            use_cache=False,
        )

    # --------------------------------------------------------
    # REMOVE HOOKS
    # --------------------------------------------------------

    for hook in cached_hooks:
        hook.remove()

    for hook in full_hooks:
        hook.remove()

    # ========================================================
    # PROJECTION COMPARISON
    # ========================================================

    print("\n" + "=" * 100)
    print("PROJECTION COMPARISON")
    print("=" * 100)

    for name in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]:

        c = cached[name]
        f = full[name]

        print(f"\n{name}")

        print("cached shape :", tuple(c.shape))
        print("full shape   :", tuple(f.shape))

        # Last token only
        c_last = c[:, -1, :]
        f_last = f[:, -1, :]

        compare(
            f"{name} LAST TOKEN",
            c_last,
            f_last,
        )

    # ========================================================
    # Q/K/V LAST TOKEN EXACT TEST
    # ========================================================

    print("\n" + "=" * 100)
    print("LAST-TOKEN Q/K/V")
    print("=" * 100)

    for name in [
        "q_proj",
        "k_proj",
        "v_proj",
    ]:

        c = cached[name][:, -1, :]
        f = full[name][:, -1, :]

        diff = (c.float() - f.float()).abs()

        print(
            f"{name}: "
            f"max={diff.max().item():.10f} "
            f"mean={diff.mean().item():.10f}"
        )

    # ========================================================
    # CACHE ENTRY VS FULL K/V
    # ========================================================

    print("\n" + "=" * 100)
    print("CACHE K/V VS FULL K/V")
    print("=" * 100)

    cache_k = hf_cache.kv_cache.key_cache[layer_idx]
    cache_v = hf_cache.kv_cache.value_cache[layer_idx]

    print("cache K:", tuple(cache_k.shape))
    print("cache V:", tuple(cache_v.shape))

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The cache stores:
    #
    # K: [B, H_kv, S, D]
    # V: [B, H_kv, S, D]
    #
    # Projection output is normally:
    #
    # [B, S, H_kv * D]
    #
    # We need to reshape the full K/V.
    # --------------------------------------------------------

    num_kv_heads = model.config.num_key_value_heads

    head_dim = (
        model.config.hidden_size
        // model.config.num_attention_heads
    )

    full_k = full["k_proj"].view(
        1,
        -1,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    full_v = full["v_proj"].view(
        1,
        -1,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    print(
        "full K reshaped:",
        tuple(full_k.shape),
    )

    print(
        "full V reshaped:",
        tuple(full_v.shape),
    )

    # --------------------------------------------------------
    # COMPARE ENTIRE CACHE
    # --------------------------------------------------------

    cached_k = cache_k[:, :, :prompt_len + 1, :]
    cached_v = cache_v[:, :, :prompt_len + 1, :]

    compare(
        "CACHE K vs FULL K",
        cached_k,
        full_k,
    )

    compare(
        "CACHE V vs FULL V",
        cached_v,
        full_v,
    )

    # ========================================================
    # COMPARE OLD TOKENS
    # ========================================================

    print("\n" + "=" * 100)
    print("OLD K/V ENTRIES")
    print("=" * 100)

    compare(
        "OLD K",
        cached_k[:, :, :prompt_len, :],
        full_k[:, :, :prompt_len, :],
    )

    compare(
        "OLD V",
        cached_v[:, :, :prompt_len, :],
        full_v[:, :, :prompt_len, :],
    )

    # ========================================================
    # COMPARE NEW TOKEN K/V
    # ========================================================

    print("\n" + "=" * 100)
    print("NEW K/V ENTRY")
    print("=" * 100)

    compare(
        "NEW K",
        cached_k[:, :, prompt_len:prompt_len + 1, :],
        full_k[:, :, prompt_len:prompt_len + 1, :],
    )

    compare(
        "NEW V",
        cached_v[:, :, prompt_len:prompt_len + 1, :],
        full_v[:, :, prompt_len:prompt_len + 1, :],
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 100)
    print("FINAL OUTPUT")
    print("=" * 100)

    cached_logits = cached_decode.logits[:, -1, :]
    full_logits = uncached_full.logits[:, -1, :]

    compare(
        "FINAL LOGITS",
        cached_logits,
        full_logits,
    )

    cached_token = cached_logits.argmax(dim=-1)
    full_token = full_logits.argmax(dim=-1)

    print(
        "cached token:",
        cached_token.tolist(),
    )

    print(
        "full token:",
        full_token.tolist(),
    )

    print(
        "same token:",
        torch.equal(cached_token, full_token),
    )


def test_kv_cache_against_post_rope():
    import torch

    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb,
    )

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
    # 1. FIRST TOKEN
    # ============================================================

    with torch.no_grad():
        prefill = model(
            input_ids=input_ids,
            use_cache=False,
        )

    first_token = torch.argmax(
        prefill.logits[:, -1, :],
        dim=-1,
    )

    full_input_ids = torch.cat(
        [
            input_ids,
            first_token.unsqueeze(0),
        ],
        dim=1,
    )

    total_len = full_input_ids.shape[1]

    print()
    print("=" * 100)
    print("BASIC INFO")
    print("=" * 100)

    print("prompt_len:", prompt_len)
    print("total_len:", total_len)
    print("first_token:", first_token.tolist())

    # ============================================================
    # 2. ENGINE CACHE
    # ============================================================

    kv_cache = KVCache(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        max_seq_len=prompt_len + 16,
        head_dim=(
            model.config.hidden_size
            // model.config.num_attention_heads
        ),
        dtype=model.dtype,
        device=adapter.device,
    )

    hf_cache = EngineKVCache(kv_cache)

    # ============================================================
    # 3. CAPTURE PREFILL HIDDEN STATES
    # ============================================================

    cached_prefill_hidden = {}

    prefill_handles = []

    for layer_idx, layer in enumerate(model.model.layers):

        def make_prefill_hook(idx):
            def hook(module, inputs, kwargs):
                cached_prefill_hidden[idx] = (
                    kwargs["hidden_states"]
                    .detach()
                    .clone()
                )

            return hook

        handle = layer.self_attn.register_forward_pre_hook(
            make_prefill_hook(layer_idx),
            with_kwargs=True,
        )

        prefill_handles.append(handle)

    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=hf_cache,
            )
    finally:
        for handle in prefill_handles:
            handle.remove()

    # ============================================================
    # 4. CACHED DECODE
    # ============================================================

    decode_input = first_token.unsqueeze(0)

    position_ids = torch.tensor(
        [[prompt_len]],
        dtype=torch.long,
        device=adapter.device,
    )

    cached_decode_hidden = {}
    cached_decode_pos = {}

    decode_handles = []

    for layer_idx, layer in enumerate(model.model.layers):

        def make_decode_hook(idx):
            def hook(module, inputs, kwargs):

                cached_decode_hidden[idx] = (
                    kwargs["hidden_states"]
                    .detach()
                    .clone()
                )

                if "position_ids" in kwargs:
                    cached_decode_pos[idx] = (
                        kwargs["position_ids"]
                        .detach()
                        .clone()
                    )

            return hook

        handle = layer.self_attn.register_forward_pre_hook(
            make_decode_hook(layer_idx),
            with_kwargs=True,
        )

        decode_handles.append(handle)

    # ------------------------------------------------------------
    # Layer 0 detailed outputs
    # ------------------------------------------------------------

    cached_layer0 = {}

    layer0 = model.model.layers[0]

    def cached_layer0_hook(module, inputs, output):

        if isinstance(output, tuple):
            output = output[0]

        cached_layer0["output"] = (
            output.detach()
            .clone()
        )

    layer0_output_handle = layer0.register_forward_hook(
        cached_layer0_hook
    )

    # ------------------------------------------------------------
    # Layer 0 submodule outputs
    # ------------------------------------------------------------

    cached_layer0_submodules = {}

    def make_submodule_hook(name):
        def hook(module, inputs, output):

            if isinstance(output, tuple):
                output = output[0]

            cached_layer0_submodules[name] = (
                output.detach()
                .clone()
            )

        return hook

    cached_submodule_handles = []

    for name in [
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "mlp",
    ]:

        if hasattr(layer0, name):

            module = getattr(layer0, name)

            handle = module.register_forward_hook(
                make_submodule_hook(name)
            )

            cached_submodule_handles.append(handle)

    try:

        with torch.no_grad():

            cached_output = model(
                input_ids=decode_input,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=hf_cache,
            )

    finally:

        for handle in decode_handles:
            handle.remove()

        layer0_output_handle.remove()

        for handle in cached_submodule_handles:
            handle.remove()

    # ============================================================
    # 5. UNCACHED FULL FORWARD
    # ============================================================

    full_hidden = {}
    full_pos = {}

    full_handles = []

    for layer_idx, layer in enumerate(model.model.layers):

        def make_full_hook(idx):
            def hook(module, inputs, kwargs):

                full_hidden[idx] = (
                    kwargs["hidden_states"]
                    .detach()
                    .clone()
                )

                if "position_ids" in kwargs:
                    full_pos[idx] = (
                        kwargs["position_ids"]
                        .detach()
                        .clone()
                    )

            return hook

        handle = layer.self_attn.register_forward_pre_hook(
            make_full_hook(layer_idx),
            with_kwargs=True,
        )

        full_handles.append(handle)

    # ------------------------------------------------------------
    # Full Layer 0 output
    # ------------------------------------------------------------

    full_layer0 = {}

    def full_layer0_hook(module, inputs, output):

        if isinstance(output, tuple):
            output = output[0]

        full_layer0["output"] = (
            output.detach()
            .clone()
        )

    full_layer0_output_handle = layer0.register_forward_hook(
        full_layer0_hook
    )

    # ------------------------------------------------------------
    # Full Layer 0 submodules
    # ------------------------------------------------------------

    full_layer0_submodules = {}

    def make_full_submodule_hook(name):
        def hook(module, inputs, output):

            if isinstance(output, tuple):
                output = output[0]

            full_layer0_submodules[name] = (
                output.detach()
                .clone()
            )

        return hook

    full_submodule_handles = []

    for name in [
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "mlp",
    ]:

        if hasattr(layer0, name):

            module = getattr(layer0, name)

            handle = module.register_forward_hook(
                make_full_submodule_hook(name)
            )

            full_submodule_handles.append(handle)

    try:

        with torch.no_grad():

            full_output = model(
                input_ids=full_input_ids,
                use_cache=False,
            )

    finally:

        for handle in full_handles:
            handle.remove()

        full_layer0_output_handle.remove()

        for handle in full_submodule_handles:
            handle.remove()

    # ============================================================
    # 6. FIND FIRST DIVERGING LAYER
    # ============================================================

    print()
    print("=" * 100)
    print("LAYER-BY-LAYER DECODE HIDDEN STATE")
    print("=" * 100)

    first_bad_layer = None

    for layer_idx in range(
        model.config.num_hidden_layers
    ):

        cached_h = cached_decode_hidden[layer_idx]

        full_h = full_hidden[layer_idx][:, -1:, :]

        diff = (
            cached_h.float()
            - full_h.float()
        ).abs()

        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        exact = torch.equal(
            cached_h,
            full_h,
        )

        print(
            f"layer={layer_idx:2d} "
            f"max={max_diff:.10f} "
            f"mean={mean_diff:.10f} "
            f"exact={exact}"
        )

        if (
            first_bad_layer is None
            and max_diff != 0.0
        ):
            first_bad_layer = layer_idx

    print()
    print("=" * 100)
    print("FIRST DIVERGING LAYER")
    print("=" * 100)

    print(
        "first_bad_layer:",
        first_bad_layer,
    )

    # ============================================================
    # 7. LAYER 0 COMPLETE OUTPUT
    # ============================================================

    print()
    print("=" * 100)
    print("LAYER 0 COMPLETE OUTPUT")
    print("=" * 100)

    cached_l0 = cached_layer0["output"]

    full_l0 = full_layer0["output"][:, -1:, :]

    layer0_output_diff = (
        cached_l0.float()
        - full_l0.float()
    ).abs()

    print(
        "cached shape:",
        tuple(cached_l0.shape),
    )

    print(
        "full shape:",
        tuple(full_l0.shape),
    )

    print(
        "max diff:",
        layer0_output_diff.max().item(),
    )

    print(
        "mean diff:",
        layer0_output_diff.mean().item(),
    )

    print(
        "exact:",
        torch.equal(
            cached_l0,
            full_l0,
        ),
    )

    # ============================================================
    # 8. LAYER 0 SUBMODULE COMPARISON
    # ============================================================

    print()
    print("=" * 100)
    print("LAYER 0 SUBMODULE COMPARISON")
    print("=" * 100)

    for name in [
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "mlp",
    ]:

        if (
            name not in cached_layer0_submodules
            or name not in full_layer0_submodules
        ):
            print(
                f"{name}: NOT CAPTURED"
            )
            continue

        cached_x = cached_layer0_submodules[name]

        full_x = full_layer0_submodules[name]

        # Full forward has sequence dimension.
        if (
            cached_x.ndim == 3
            and full_x.ndim == 3
            and full_x.shape[1] != cached_x.shape[1]
        ):
            full_x = full_x[:, -1:, :]

        diff = (
            cached_x.float()
            - full_x.float()
        ).abs()

        print(
            f"{name:28s} "
            f"max={diff.max().item():.10f} "
            f"mean={diff.mean().item():.10f} "
            f"exact={torch.equal(cached_x, full_x)}"
        )

    # ============================================================
    # 9. POSITION IDS
    # ============================================================

    print()
    print("=" * 100)
    print("POSITION IDS")
    print("=" * 100)

    if first_bad_layer is not None:

        print(
            "cached:",
            cached_decode_pos[first_bad_layer],
        )

        print(
            "full:",
            full_pos[first_bad_layer],
        )

    # ============================================================
    # 10. LAYER 0 ATTENTION DIAGNOSTIC
    # ============================================================

    layer_idx = 0

    cached_h = cached_decode_hidden[layer_idx]

    full_h = full_hidden[layer_idx][:, -1:, :]

    layer0_input_diff = (
        cached_h.float()
        - full_h.float()
    ).abs()

    print()
    print("=" * 100)
    print("LAYER 0 INPUT")
    print("=" * 100)

    print(
        "cached shape:",
        tuple(cached_h.shape),
    )

    print(
        "full shape:",
        tuple(full_h.shape),
    )

    print(
        "hidden max:",
        layer0_input_diff.max().item(),
    )

    print(
        "hidden mean:",
        layer0_input_diff.mean().item(),
    )

    print(
        "hidden exact:",
        torch.equal(
            cached_h,
            full_h,
        ),
    )

    # ============================================================
    # 11. RAW Q/K/V
    # ============================================================

    attn = layer0.self_attn

    cached_q = attn.q_proj(cached_h)

    cached_k = attn.k_proj(cached_h)

    cached_v = attn.v_proj(cached_h)

    full_q = attn.q_proj(full_h)

    full_k = attn.k_proj(full_h)

    full_v = attn.v_proj(full_h)

    print()
    print("=" * 100)
    print("LAYER 0 RAW Q/K/V")
    print("=" * 100)

    for name, a, b in [
        ("Q", cached_q, full_q),
        ("K", cached_k, full_k),
        ("V", cached_v, full_v),
    ]:

        diff = (
            a.float()
            - b.float()
        ).abs()

        print(
            f"{name}: "
            f"shape={tuple(a.shape)} "
            f"max={diff.max().item():.10f} "
            f"mean={diff.mean().item():.10f}"
        )

    # ============================================================
    # 12. RESHAPE K/V
    # ============================================================

    bsz = 1
    seq_len = 1

    num_kv_heads = (
        attn.config.num_key_value_heads
    )

    head_dim = attn.head_dim

    cached_k = cached_k.view(
        bsz,
        seq_len,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    cached_v = cached_v.view(
        bsz,
        seq_len,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    full_k = full_k.view(
        bsz,
        seq_len,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    full_v = full_v.view(
        bsz,
        seq_len,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    # ============================================================
    # 13. RoPE
    # ============================================================

    cached_position_ids = torch.tensor(
        [[prompt_len]],
        dtype=torch.long,
        device=adapter.device,
    )

    rotary_emb = model.model.rotary_emb

    cos, sin = rotary_emb(
        cached_k,
        cached_position_ids,
    )

    _, cached_k_rope = apply_rotary_pos_emb(
        cached_k,
        cached_k,
        cos,
        sin,
    )

    _, full_k_rope = apply_rotary_pos_emb(
        full_k,
        full_k,
        cos,
        sin,
    )

    print()
    print("=" * 100)
    print("LAYER 0 POST-RoPE")
    print("=" * 100)

    rope_diff = (
        cached_k_rope.float()
        - full_k_rope.float()
    ).abs()

    print(
        "K max:",
        rope_diff.max().item(),
    )

    print(
        "K mean:",
        rope_diff.mean().item(),
    )

    # ============================================================
    # 14. ENGINE CACHE
    # ============================================================

    engine_k = kv_cache.key_cache[0][
        :,
        :,
        prompt_len:total_len,
        :,
    ]

    engine_v = kv_cache.value_cache[0][
        :,
        :,
        prompt_len:total_len,
        :,
    ]

    print()
    print("=" * 100)
    print("LAYER 0 ENGINE CACHE")
    print("=" * 100)

    cache_k_diff = (
        engine_k.float()
        - cached_k_rope.float()
    ).abs()

    cache_v_diff = (
        engine_v.float()
        - cached_v.float()
    ).abs()

    print(
        "K max:",
        cache_k_diff.max().item(),
    )

    print(
        "K mean:",
        cache_k_diff.mean().item(),
    )

    print(
        "V max:",
        cache_v_diff.max().item(),
    )

    print(
        "V mean:",
        cache_v_diff.mean().item(),
    )

    # ============================================================
    # 15. FINAL OUTPUT
    # ============================================================

    cached_logits = (
        cached_output.logits[:, -1, :]
    )

    full_logits = (
        full_output.logits[:, -1, :]
    )

    logits_diff = (
        cached_logits.float()
        - full_logits.float()
    ).abs()

    cached_token = torch.argmax(
        cached_logits,
        dim=-1,
    )

    full_token = torch.argmax(
        full_logits,
        dim=-1,
    )

    print()
    print("=" * 100)
    print("FINAL OUTPUT")
    print("=" * 100)

    print(
        "logits max diff:",
        logits_diff.max().item(),
    )

    print(
        "logits mean diff:",
        logits_diff.mean().item(),
    )

    print(
        "cached token:",
        cached_token.tolist(),
    )

    print(
        "full token:",
        full_token.tolist(),
    )

    print(
        "same token:",
        torch.equal(
            cached_token,
            full_token,
        ),
    )

    # ============================================================
    # 16. DO NOT FAIL ON NUMERICAL DIVERGENCE YET
    # ============================================================

    # This test is currently diagnostic.
    #
    # The purpose is to locate the first operation that diverges.
    #
    # Once we know whether the divergence occurs in:
    #
    #   attention
    #   residual
    #   post-attention norm
    #   MLP
    #   layer output
    #
    # we can replace this with the appropriate numerical assertion.

    assert torch.equal(
        cached_token,
        full_token,
    )
def test_kv_cache_attention_diagnostic():
    import torch

    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb,
    )

    adapter = ModelAdapter(
        model_name=config["model"]["model_name"],
        device=config["device"],
    )

    model = adapter.model
    model.eval()

    # ============================================================
    # CONFIG
    # ============================================================

    prompt = "The capital of France is"
    diagnostic_layer = 0

    # ============================================================
    # INPUT
    # ============================================================

    input_ids = torch.tensor(
        [adapter.tokenize(prompt)],
        dtype=torch.long,
        device=adapter.device,
    )

    prompt_len = input_ids.shape[1]

    # ============================================================
    # FIRST TOKEN
    # ============================================================

    with torch.no_grad():
        prefill = model(
            input_ids=input_ids,
            use_cache=False,
        )

    first_token = torch.argmax(
        prefill.logits[:, -1, :],
        dim=-1,
    )

    full_input_ids = torch.cat(
        [
            input_ids,
            first_token.unsqueeze(0),
        ],
        dim=1,
    )

    total_len = full_input_ids.shape[1]

    print()
    print("=" * 100)
    print("BASIC INFO")
    print("=" * 100)

    print("prompt_len:", prompt_len)
    print("total_len:", total_len)
    print("first_token:", first_token.tolist())
    print("total_len:", total_len)

    # ============================================================
    # ENGINE CACHE
    # ============================================================

    kv_cache = KVCache(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        max_seq_len=prompt_len + 16,
        head_dim=(
            model.config.hidden_size
            // model.config.num_attention_heads
        ),
        dtype=model.dtype,
        device=adapter.device,
    )

    hf_cache = EngineKVCache(kv_cache)

    # ============================================================
    # HOOK FACTORY
    # ============================================================

    def install_attention_hooks(storage):

        handles = []

        for layer_idx, layer in enumerate(model.model.layers):

            def make_hook(idx):

                def hook(module, inputs, kwargs):

                    storage[idx] = {
                        "hidden_states": (
                            kwargs["hidden_states"]
                            .detach()
                            .clone()
                        ),
                        "position_ids": (
                            kwargs["position_ids"]
                            .detach()
                            .clone()
                            if "position_ids" in kwargs
                            else None
                        ),
                    }

                return hook

            handle = (
                layer.self_attn.register_forward_pre_hook(
                    make_hook(layer_idx),
                    with_kwargs=True,
                )
            )

            handles.append(handle)

        return handles

    # ============================================================
    # 1. CACHED PREFILL
    # ============================================================

    cached_prefill = {}

    handles = install_attention_hooks(cached_prefill)

    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=hf_cache,
            )
    finally:
        for handle in handles:
            handle.remove()

    # ============================================================
    # 2. CACHED DECODE
    # ============================================================

    decode_input = first_token.unsqueeze(0)

    decode_position_ids = torch.tensor(
        [[prompt_len]],
        dtype=torch.long,
        device=adapter.device,
    )

    cached_decode = {}

    handles = install_attention_hooks(cached_decode)

    try:
        with torch.no_grad():
            cached_output = model(
                input_ids=decode_input,
                position_ids=decode_position_ids,
                use_cache=True,
                past_key_values=hf_cache,
            )
    finally:
        for handle in handles:
            handle.remove()

    # ============================================================
    # 3. FULL FORWARD
    # ============================================================

    full_forward = {}

    handles = install_attention_hooks(full_forward)

    try:
        with torch.no_grad():
            full_output = model(
                input_ids=full_input_ids,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    # ============================================================
    # 4. LAYER-BY-LAYER INPUT COMPARISON
    # ============================================================

    print()
    print("=" * 100)
    print("LAYER-BY-LAYER DECODE HIDDEN STATE")
    print("=" * 100)

    first_bad_layer = None

    for layer_idx in range(model.config.num_hidden_layers):

        cached_h = cached_decode[layer_idx]["hidden_states"]

        full_h = full_forward[layer_idx][
            "hidden_states"
        ][:, -1:, :]

        diff = (
            cached_h.float()
            - full_h.float()
        ).abs()

        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        exact = torch.equal(
            cached_h,
            full_h,
        )

        print(
            f"layer={layer_idx:2d} "
            f"max={max_diff:.10f} "
            f"mean={mean_diff:.10f} "
            f"exact={exact}"
        )

        if (
            first_bad_layer is None
            and max_diff != 0.0
        ):
            first_bad_layer = layer_idx

    print()
    print("=" * 100)
    print("FIRST DIVERGING LAYER")
    print("=" * 100)

    print(
        "first_bad_layer:",
        first_bad_layer,
    )

    # ============================================================
    # 5. WE EXPECT LAYER 0 TO BE THE FIRST COMPUTATIONAL TEST
    # ============================================================

    layer_idx = diagnostic_layer

    layer = model.model.layers[layer_idx]
    attn = layer.self_attn

    cached_h = cached_decode[layer_idx]["hidden_states"]
    full_h = full_forward[layer_idx][
        "hidden_states"
    ][:, -1:, :]

    print()
    print("=" * 100)
    print(f"LAYER {layer_idx} INPUT")
    print("=" * 100)

    print("cached shape:", tuple(cached_h.shape))
    print("full shape:", tuple(full_h.shape))

    hidden_diff = (
        cached_h.float()
        - full_h.float()
    ).abs()

    print(
        "hidden max:",
        hidden_diff.max().item(),
    )

    print(
        "hidden mean:",
        hidden_diff.mean().item(),
    )

    print(
        "hidden exact:",
        torch.equal(cached_h, full_h),
    )

    # ============================================================
    # 6. RAW Q/K/V
    # ============================================================

    cached_q = attn.q_proj(cached_h)
    cached_k = attn.k_proj(cached_h)
    cached_v = attn.v_proj(cached_h)

    full_q = attn.q_proj(full_h)
    full_k = attn.k_proj(full_h)
    full_v = attn.v_proj(full_h)

    print()
    print("=" * 100)
    print(f"LAYER {layer_idx} RAW Q/K/V")
    print("=" * 100)

    for name, cached, full in [
        ("Q", cached_q, full_q),
        ("K", cached_k, full_k),
        ("V", cached_v, full_v),
    ]:

        diff = (
            cached.float()
            - full.float()
        ).abs()

        print(
            f"{name}: "
            f"shape={tuple(cached.shape)} "
            f"max={diff.max().item():.10f} "
            f"mean={diff.mean().item():.10f}"
        )

    # ============================================================
    # 7. RESHAPE
    # ============================================================

    bsz = 1
    seq_len = 1

    num_heads = attn.config.num_attention_heads
    num_kv_heads = attn.config.num_key_value_heads
    head_dim = attn.head_dim

    cached_q = cached_q.view(
        bsz,
        seq_len,
        num_heads,
        head_dim,
    ).transpose(1, 2)

    cached_k = cached_k.view(
        bsz,
        seq_len,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    cached_v = cached_v.view(
        bsz,
        seq_len,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    full_q = full_q.view(
        bsz,
        seq_len,
        num_heads,
        head_dim,
    ).transpose(1, 2)

    full_k = full_k.view(
        bsz,
        seq_len,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    full_v = full_v.view(
        bsz,
        seq_len,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    # ============================================================
    # 8. RoPE
    # ============================================================

    rotary_emb = model.model.rotary_emb

    position_ids = torch.tensor(
        [[prompt_len]],
        dtype=torch.long,
        device=adapter.device,
    )

    cos, sin = rotary_emb(
        cached_k,
        position_ids,
    )

    cached_q_rope, cached_k_rope = (
        apply_rotary_pos_emb(
            cached_q,
            cached_k,
            cos,
            sin,
        )
    )

    full_q_rope, full_k_rope = (
        apply_rotary_pos_emb(
            full_q,
            full_k,
            cos,
            sin,
        )
    )

    print()
    print("=" * 100)
    print(f"LAYER {layer_idx} POST-RoPE")
    print("=" * 100)

    for name, cached, full in [
        ("Q", cached_q_rope, full_q_rope),
        ("K", cached_k_rope, full_k_rope),
    ]:

        diff = (
            cached.float()
            - full.float()
        ).abs()

        print(
            f"{name}: "
            f"max={diff.max().item():.10f} "
            f"mean={diff.mean().item():.10f}"
        )

    # ============================================================
    # 9. ENGINE CACHE
    # ============================================================

    engine_k = kv_cache.key_cache[layer_idx][
        :,
        :,
        prompt_len:total_len,
        :,
    ]

    engine_v = kv_cache.value_cache[layer_idx][
        :,
        :,
        prompt_len:total_len,
        :,
    ]

    print()
    print("=" * 100)
    print(f"LAYER {layer_idx} ENGINE CACHE")
    print("=" * 100)

    engine_k_diff = (
        engine_k.float()
        - cached_k_rope.float()
    ).abs()

    engine_v_diff = (
        engine_v.float()
        - cached_v.float()
    ).abs()

    print(
        "K max:",
        engine_k_diff.max().item(),
    )

    print(
        "K mean:",
        engine_k_diff.mean().item(),
    )

    print(
        "V max:",
        engine_v_diff.max().item(),
    )

    print(
        "V mean:",
        engine_v_diff.mean().item(),
    )

    # ============================================================
    # 10. FULL SEQUENCE Q/K/V
    # ============================================================

    full_layer_h = full_forward[layer_idx][
        "hidden_states"
    ]

    raw_q_all = attn.q_proj(full_layer_h)
    raw_k_all = attn.k_proj(full_layer_h)
    raw_v_all = attn.v_proj(full_layer_h)

    raw_q_all = raw_q_all.view(
        1,
        total_len,
        num_heads,
        head_dim,
    ).transpose(1, 2)

    raw_k_all = raw_k_all.view(
        1,
        total_len,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    raw_v_all = raw_v_all.view(
        1,
        total_len,
        num_kv_heads,
        head_dim,
    ).transpose(1, 2)

    full_position_ids = torch.arange(
        total_len,
        device=adapter.device,
        dtype=torch.long,
    ).unsqueeze(0)

    cos_full, sin_full = rotary_emb(
        raw_k_all,
        full_position_ids,
    )

    full_q_all_rope, full_k_all_rope = (
        apply_rotary_pos_emb(
            raw_q_all,
            raw_k_all,
            cos_full,
            sin_full,
        )
    )

    # ============================================================
    # 11. ATTENTION SCORE DIAGNOSTIC
    # ============================================================

    # GQA:
    #
    # Q : [B, H_q, Q, D]
    # K : [B, H_kv, K, D]
    #
    # Repeat K to H_q.

    repeat_factor = (
        num_heads // num_kv_heads
    )

    cached_k_attn = (
        kv_cache.key_cache[layer_idx][
            :, :, :total_len, :
        ]
        .repeat_interleave(
            repeat_factor,
            dim=1,
        )
    )

    full_k_attn = (
        full_k_all_rope
        .repeat_interleave(
            repeat_factor,
            dim=1,
        )
    )

    # Cached query is only the last token.
    cached_q_last = cached_q_rope

    # Full query corresponding to last token.
    full_q_last = full_q_all_rope[:, :, -1:, :]

    scale = 1.0 / (head_dim ** 0.5)

    cached_scores = torch.matmul(
        cached_q_last.float(),
        cached_k_attn.float().transpose(-1, -2),
    ) * scale

    full_scores = torch.matmul(
        full_q_last.float(),
        full_k_attn.float().transpose(-1, -2),
    ) * scale

    score_diff = (
        cached_scores
        - full_scores
    ).abs()

    print()
    print("=" * 100)
    print(f"LAYER {layer_idx} ATTENTION SCORES")
    print("=" * 100)

    print(
        "cached scores:",
        tuple(cached_scores.shape),
    )

    print(
        "full scores:",
        tuple(full_scores.shape),
    )

    print(
        "score max diff:",
        score_diff.max().item(),
    )

    print(
        "score mean diff:",
        score_diff.mean().item(),
    )

    # ============================================================
    # 12. SOFTMAX DIAGNOSTIC
    # ============================================================

    cached_probs = torch.softmax(
        cached_scores,
        dim=-1,
    )

    full_probs = torch.softmax(
        full_scores,
        dim=-1,
    )

    prob_diff = (
        cached_probs
        - full_probs
    ).abs()

    print()
    print("=" * 100)
    print(f"LAYER {layer_idx} ATTENTION PROBABILITIES")
    print("=" * 100)

    print(
        "cached probs:",
        cached_probs,
    )

    print(
        "full probs:",
        full_probs,
    )

    print(
        "prob max diff:",
        prob_diff.max().item(),
    )

    print(
        "prob mean diff:",
        prob_diff.mean().item(),
    )

    # ============================================================
    # 13. ATTENTION OUTPUT
    # ============================================================

    cached_v_attn = (
        kv_cache.value_cache[layer_idx][
            :, :, :total_len, :
        ]
        .repeat_interleave(
            repeat_factor,
            dim=1,
        )
    )

    full_v_attn = (
        raw_v_all
        .repeat_interleave(
            repeat_factor,
            dim=1,
        )
    )

    cached_attn_output = torch.matmul(
        cached_probs,
        cached_v_attn.float(),
    )

    full_attn_output = torch.matmul(
        full_probs,
        full_v_attn.float(),
    )

    attention_output_diff = (
        cached_attn_output
        - full_attn_output
    ).abs()

    print()
    print("=" * 100)
    print(f"LAYER {layer_idx} ATTENTION OUTPUT")
    print("=" * 100)

    print(
        "cached:",
        tuple(cached_attn_output.shape),
    )

    print(
        "full:",
        tuple(full_attn_output.shape),
    )

    print(
        "max diff:",
        attention_output_diff.max().item(),
    )

    print(
        "mean diff:",
        attention_output_diff.mean().item(),
    )

    # ============================================================
    # 14. FINAL OUTPUT
    # ============================================================

    cached_logits = cached_output.logits[:, -1, :]
    full_logits = full_output.logits[:, -1, :]

    logits_diff = (
        cached_logits.float()
        - full_logits.float()
    ).abs()

    cached_token = torch.argmax(
        cached_logits,
        dim=-1,
    )

    full_token = torch.argmax(
        full_logits,
        dim=-1,
    )

    print()
    print("=" * 100)
    print("FINAL OUTPUT")
    print("=" * 100)

    print(
        "logits max diff:",
        logits_diff.max().item(),
    )

    print(
        "logits mean diff:",
        logits_diff.mean().item(),
    )

    print(
        "cached token:",
        cached_token.tolist(),
    )

    print(
        "full token:",
        full_token.tolist(),
    )

    print(
        "same token:",
        torch.equal(
            cached_token,
            full_token,
        ),
    )

    # ============================================================
    # 15. DIAGNOSTIC ASSERTION
    # ============================================================

    # Do NOT require exact equality for BF16 attention.
    #
    # This test is diagnostic: it should tell us whether the
    # divergence originates in attention rather than the cache.

    assert first_bad_layer is not None



def test_mlp_shape_numerical_equivalence():
    import torch

    # IMPORTANT:
    # Do NOT import ModelAdapter here.
    # Use the ModelAdapter already imported by this test file.

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

    # ============================================================
    # 1. GET THE TOKEN THAT WAS USED IN YOUR PREVIOUS TEST
    # ============================================================

    with torch.no_grad():
        prefill = model(
            input_ids=input_ids,
            use_cache=False,
        )

    first_token = torch.argmax(
        prefill.logits[:, -1, :],
        dim=-1,
    )

    full_input_ids = torch.cat(
        [
            input_ids,
            first_token.unsqueeze(0),
        ],
        dim=1,
    )

    # ============================================================
    # 2. CAPTURE LAYER 0 MLP INPUT
    # ============================================================

    layer = model.model.layers[0]

    captured = {}

    def mlp_hook(module, inputs):
        captured["hidden_states"] = (
            inputs[0]
            .detach()
            .clone()
        )

    handle = layer.mlp.register_forward_pre_hook(
        mlp_hook
    )

    try:
        with torch.no_grad():
            model(
                input_ids=full_input_ids,
                use_cache=False,
            )
    finally:
        handle.remove()

    full_hidden = captured["hidden_states"]

    # Shape:
    #
    # full_hidden:
    #     [1, 6, 896]
    #
    # We isolate the final token:
    #
    # x_last:
    #     [1, 1, 896]

    x_last = full_hidden[:, -1:, :]

    # ============================================================
    # 3. RUN THE SAME MLP TWO WAYS
    # ============================================================

    with torch.no_grad():

        # MLP sees all 6 tokens
        full_mlp_output = layer.mlp(
            full_hidden
        )

        # MLP sees ONLY the final token
        single_mlp_output = layer.mlp(
            x_last
        )

    # Extract final token from full computation.
    full_last = full_mlp_output[:, -1:, :]

    # ============================================================
    # 4. COMPARE
    # ============================================================

    diff = (
        full_last.float()
        - single_mlp_output.float()
    ).abs()

    print()
    print("=" * 100)
    print("MLP SHAPE NUMERICAL EQUIVALENCE")
    print("=" * 100)

    print(
        "full hidden:",
        tuple(full_hidden.shape),
    )

    print(
        "single hidden:",
        tuple(x_last.shape),
    )

    print(
        "full MLP output:",
        tuple(full_mlp_output.shape),
    )

    print(
        "single MLP output:",
        tuple(single_mlp_output.shape),
    )

    print()
    print(
        "max diff:",
        diff.max().item(),
    )

    print(
        "mean diff:",
        diff.mean().item(),
    )

    print(
        "exact:",
        torch.equal(
            full_last,
            single_mlp_output,
        ),
    )

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    if torch.equal(full_last, single_mlp_output):
        print("MLP is exactly shape-independent.")
    else:
        print(
            "MLP produces different results when evaluated "
            "with S=6 versus S=1."
        )


def test_mlp_internal_numerical_divergence():
    import torch

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

    with torch.no_grad():
        prefill = model(
            input_ids=input_ids,
            use_cache=False,
        )

    first_token = torch.argmax(
        prefill.logits[:, -1, :],
        dim=-1,
    )

    full_input_ids = torch.cat(
        [
            input_ids,
            first_token.unsqueeze(0),
        ],
        dim=1,
    )

    # ------------------------------------------------------------
    # Capture layer-0 MLP input
    # ------------------------------------------------------------

    layer = model.model.layers[0]

    captured = {}

    def hook(module, inputs):
        captured["x"] = inputs[0].detach().clone()

    handle = layer.mlp.register_forward_pre_hook(hook)

    try:
        with torch.no_grad():
            model(
                input_ids=full_input_ids,
                use_cache=False,
            )
    finally:
        handle.remove()

    x_full = captured["x"]
    x_single = x_full[:, -1:, :]

    mlp = layer.mlp

    # ------------------------------------------------------------
    # Qwen2 MLP components
    # ------------------------------------------------------------

    with torch.no_grad():

        gate_full = mlp.gate_proj(x_full)
        gate_single = mlp.gate_proj(x_single)

        up_full = mlp.up_proj(x_full)
        up_single = mlp.up_proj(x_single)

        gate_full_last = gate_full[:, -1:, :]
        up_full_last = up_full[:, -1:, :]

        # Activation
        act_full = mlp.act_fn(gate_full)
        act_single = mlp.act_fn(gate_single)

        act_full_last = act_full[:, -1:, :]

        # Elementwise product
        product_full = act_full * up_full
        product_single = act_single * up_single

        product_full_last = product_full[:, -1:, :]

        # Down projection
        down_full = mlp.down_proj(product_full)
        down_single = mlp.down_proj(product_single)

        down_full_last = down_full[:, -1:, :]

    # ------------------------------------------------------------
    # Comparison helper
    # ------------------------------------------------------------

    def compare(name, a, b):
        diff = (
            a.float() - b.float()
        ).abs()

        print(
            f"{name:20s}"
            f" max={diff.max().item():.10f}"
            f" mean={diff.mean().item():.10f}"
            f" exact={torch.equal(a, b)}"
        )

    print()
    print("=" * 100)
    print("MLP INTERNAL NUMERICAL DIVERGENCE")
    print("=" * 100)

    print()

    compare(
        "gate_proj",
        gate_full_last,
        gate_single,
    )

    compare(
        "up_proj",
        up_full_last,
        up_single,
    )

    compare(
        "activation",
        act_full_last,
        act_single,
    )

    compare(
        "SiLU * up",
        product_full_last,
        product_single,
    )

    compare(
        "down_proj",
        down_full_last,
        down_single,
    )

    print()
    print("=" * 100)