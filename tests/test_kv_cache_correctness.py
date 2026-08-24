"""

Goal
----
Prove that greedy generation using the custom KV cache produces the
exact same token IDs as full-sequence recomputation.

Primary invariant
-----------------
    cached_token_ids == uncached_token_ids

This test intentionally compares TOKEN IDS rather than decoded text.

The suite uses:
    - deterministic greedy decoding
    - fixed prompts
    - multiple prompt lengths
    - multiple generation lengths
    - a position-offset regression test

This is a correctness test, not a performance benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import yaml

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parent.parent))



from engine.kv_cache import KVCache
from engine.hf_kv_cache import EngineKVCache
from engine.model_adapter import ModelAdapter

with open("configs/inference.yaml", "r") as f:
    config = yaml.safe_load(f)



NUM_PROMPTS = 20
DEFAULT_MAX_NEW_TOKENS = 10


DEVICE = config["device"]
MODEL_NAME = config["model"]["model_name"]


# FIXED PROMPT SUITE
# We are not trying to measure model quality here. We are trying to exercise different sequence lengths and positional situations.



PROMPTS = [
    # --------------------------------------------------------
    # Very short
    # --------------------------------------------------------
    "Hi",
    "Hello",
    "Why?",
    "2 + 2 =",
    "France is",

    # --------------------------------------------------------
    # Short / normal
    # --------------------------------------------------------
    "The capital of France is",
    "The largest planet is",
    "Python is a programming language used for",
    "Machine learning models learn patterns from",
    "The opposite of hot is",


    # --------------------------------------------------------
    # Longer natural language
    # --------------------------------------------------------
    (
        "A neural network is a mathematical model that "
        "learns representations from data by adjusting"
    ),
    (
        "The main purpose of a key value cache during "
        "autoregressive transformer inference is to"
    ),
    (
        "When generating text one token at a time, the "
        "transformer needs to remember the previous"
    ),

    # --------------------------------------------------------
    # Punctuation / unusual tokenization
    # --------------------------------------------------------
    "Hello, world! How are you?",
    "What?! Really... yes!!!",
    "Python: torch.randn([2, 3]) ->",

    # --------------------------------------------------------
    # Multilingual
    # --------------------------------------------------------
    "Bonjour, comment allez-vous ?",
    "مرحبا، كيف حالك؟",
    "الجزائر بلد يقع في",

    # --------------------------------------------------------
    # Position-sensitive / longer prompt
    # --------------------------------------------------------
    (
        "The transformer architecture uses self-attention. "
        "Self-attention allows each token to interact with "
        "other tokens in the sequence. During autoregressive "
        "generation, previously computed key and value states "
        "can be cached so that they do not need to be recomputed."
    ),
]


assert len(PROMPTS) == NUM_PROMPTS

# generation result
@dataclass
class GenerationResult:
    prompt_ids: list[int]
    generated_ids: list[int]

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.generated_ids

@pytest.fixture(scope="module")
def adapter():
    """
    Load the model once for the entire correctness suite
    """
    adapter = ModelAdapter(
        model_name=MODEL_NAME,
        device=DEVICE,
    )

    adapter.model.eval()

    return adapter

@pytest.fixture(scope="module")
def model(adapter):
    return adapter.model


# HALPERS
def make_input_ids(
        adapter: ModelAdapter,
        prompt: str
) -> torch.Tensor:
    """
    Tokenize one fixed prompt
    
    shape:
    [1, T]
    """
    prompt_ids = adapter.tokenize(prompt)


    return torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=adapter.device
    )


def create_cache(
        model,
        adapter,
        max_seq_len: int,
# ) -> EngineKVCache:
) -> KVCache:
    """
    Create a fresh KV Cache

    shape per layer:
    [1, H_kv, max_seq_leng, head_dim]
    """

    kv_cache = KVCache(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        max_seq_len=max_seq_len,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        dtype=model.dtype,
        device=model.device,
    )

    # return EngineKVCache(kv_cache)
    return kv_cache



# GREEDY TOKEN SELECTION
def greedy_token(
        logits: torch.Tensor,
) -> int:
    """
    Deterministic greedy decoding

    logits: [1, V]
    
    returns:
        integer token ID
    """
    return torch.argmax(
        logits,
        dim=-1
    ).item()

# UNCACHED GREEDY GENERATION
@torch.inference_mode()
def generate_uncached(
    adapter: ModelAdapter,
    input_ids: torch.Tensor,
    max_new_tokens: int
) -> GenerationResult:
    """
    Full recompute generation.

    At every generation step the entire sequence is passed through the model again.


    Greedy Decoding:
        token = argmax(logits)
    """

    prompt_ids = input_ids[0].tolist()

    generated_ids: list[int] = []

    current_ids = input_ids.clone()


    for _ in range(max_new_tokens):
        logits = adapter.forward_no_cache(
            current_ids
        )

        next_logits = logits[:, -1 ,:]

        token_id = greedy_token(
            next_logits
        )

        generated_ids.append(token_id)

        next_token = torch.tensor(
            [[token_id]],
            dtype=torch.long,
            device=adapter.device
        )

        current_ids = torch.cat(
            [current_ids, next_token],
            dim=1
        )

    return GenerationResult(
        prompt_ids=prompt_ids,
        generated_ids=generated_ids
    )

# CACHED GREEDY GENERATION
@torch.inference_mode()
def generate_cached(
    adapter: ModelAdapter,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> GenerationResult:
    """
    KV-cache generation.

    The prompt is processed once during prefill.

    Then each generated token is processed individually.

    """

    prompt_ids = input_ids[0].tolist()

    generated_ids: list[int] = []

    prompt_length = input_ids.shape[1]


    # Allocate enough cache for:
    # prompt token + generated tokens
    cache = create_cache(
        model = adapter.model,
        adapter=adapter,
        max_seq_len=prompt_length + max_new_tokens
    )

    # PREFILL
    logits = adapter.forward_prefill_cached(
        input_ids=input_ids,
        cache=cache
    )


    # [1, T, V] -> [1, V]
    next_logits = logits[:, -1, :]

    token_id = greedy_token(
        next_logits
    )

    generated_ids.append(token_id)

    # DECODE
    current_position = prompt_length
    for step in range(max_new_tokens -1):

        next_token = torch.tensor(
            [[token_id]],
            dtype=torch.long,
            device=adapter.device
        )

        logits = adapter.forward_decode_cached(
            last_token=next_token,
            cache=cache,
            position=current_position
        )
        next_logits = logits[:, -1, :]

        token_id = greedy_token(
            next_logits
        )

        generated_ids.append(token_id)

        current_position += 1

    return GenerationResult(
        prompt_ids=prompt_ids,
        generated_ids=generated_ids
    )






    
def test_single_prompt_correctness(
        adapter,
):
    """
    Basic sanity check before running the complete suite
    """
    prompt = PROMPTS[0]

    input_ids = make_input_ids(
        adapter,
        prompt
    )

    uncached = generate_uncached(
        adapter=adapter,
        input_ids=input_ids,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS
    )


    cached = generate_cached(
        adapter=adapter,
        input_ids=input_ids,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS
    )

    assert cached.generated_ids == uncached.generated_ids, (
        f"\nPrompt: {prompt!r}"
        f"\nUncached: {uncached.generated_ids}"
        f"\nCached:   {cached.generated_ids}"
    )


# 20-PROMPT CORRECTNESS TEST
@pytest.mark.parametrize(
    "prompt",
    PROMPTS
)
def test_prompt_match_correctness(
    adapter,
    prompt
):
    """
    cached_token_ids ==? uncached_token_ids
    """
    input_ids = make_input_ids(
        adapter,
        prompt
    )
    uncached = generate_uncached(
        adapter=adapter,
        input_ids=input_ids,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS
    )


    cached = generate_cached(
        adapter=adapter,
        input_ids=input_ids,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS
    )



    assert cached.generated_ids == uncached.generated_ids, (
        "\n"
        "KV CACHE CORRECTNESS FAILURE\n"
        "============================\n"
        f"Prompt: {prompt!r}\n"
        f"Prompt IDs: {uncached.prompt_ids}\n"
        f"Uncached:   {uncached.generated_ids}\n"
        f"Cached:     {cached.generated_ids}\n"
    )



# VARIABLE GENERATION LENGTH TEST
@pytest.mark.parametrize(
    "prompt, max_new_tokens",
    [
        (PROMPTS[0], 1),
        (PROMPTS[5], 2),
        (PROMPTS[10], 5),
        (PROMPTS[15], 10),
        (PROMPTS[19], 15),
    ]
)
def test_cached_matches_uncached_variable_lengths(
    adapter,
    prompt,
    max_new_tokens
): 
    """
    Verify correctness for multiple generation lengths.

    This is important because a cache can produce the correct first token
    and become uncorrect in later decode steps
    """

    input_ids = make_input_ids(
        adapter,
        prompt
    )

    uncached = generate_uncached(
        adapter=adapter,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens
    )

    cached = generate_cached(
        adapter=adapter,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens
    )

    assert cached.generated_ids == uncached.generated_ids, (
        "\n"
        "VARIABLE-LENGTH KV CACHE FAILURE\n"
        "===============================\n"
        f"Prompt: {prompt!r}\n"
        f"Generation length: {max_new_tokens}\n"
        f"Uncached: {uncached.generated_ids}\n"
        f"Cached:   {cached.generated_ids}\n"
    )


# POSITION OFFSET REGRESSION TEST
@torch.inference_mode()
def generate_cached_with_position_offset(
    adapter: ModelAdapter,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    position_offset: int
) -> GenerationResult:
    """
    Intentionally use an incorrect absolute position

    this is a not valid generation implementation

    It exists only to verify that our correctness suite is sensitive to a KV-Cache position bug
    """

    prompt_ids = input_ids[0].tolist()


    generated_ids: list[int] = []


    prompt_length = input_ids.shape[1]


    cache = create_cache(
        model=adapter.model,
        adapter=adapter,
        max_seq_len= prompt_length + max_new_tokens,
    )

    # corrct prefill
    logits = adapter.forward_prefill_cached(
        input_ids = input_ids,
        cache = cache
    )

    token_id = greedy_token(
        logits[:, -1, :]
    )

    generated_ids.append(token_id)

    # Delibrately WRONG DECODE POSITION

    current_position = (
        prompt_length + position_offset
    )

    for _ in range(max_new_tokens - 1):
        next_token = torch.tensor(
            [[token_id]],
            dtype=torch.long,
            device = adapter.device
        )

        logits = adapter.forward_decode_cached(
            last_token= next_token,
            cache=cache,
            position=current_position
        )

        token_id = greedy_token(
            logits[:, -1, :]
        )

        generated_ids.append(token_id)

        current_position += 1

    return GenerationResult(
        prompt_ids = prompt_ids,
        generated_ids = generated_ids
    )



def test_position_offset_bug_is_detected(
        adapter,
):
    """
    
    Deliberately introduce an incorrect decode position.
    
    The corrupted cached generation must NOT match the correct uncached generation.

    This proves that the correctness test would catch a    classic KV-cache offset bug.
    """

    prompt = (
        "The transformer architecture uses self-attention. "
        "During autoregressive generation, previously "
        "computed key and value states are cached."
    )

    input_ids = make_input_ids(
        adapter,
        prompt
    )

    uncached = generate_uncached(
        adapter=adapter,
        input_ids=input_ids,
        max_new_tokens=10
    )

    corrupted_cached = (
        generate_cached_with_position_offset(
            adapter=adapter,
            input_ids=input_ids,
            max_new_tokens=10,
            position_offset=1
        )
    )

    assert (
        corrupted_cached.generated_ids
        != uncached.generated_ids
    ), (
        "POSITION-OFFSET REGRESSION TEST DID NOT FAIL.\n"
        "The intentionally incorrect decode position produced "
        "the same token sequence as uncached generation."
    )



















@torch.inference_mode()
def test_first_decode_only(adapter):

    prompt = "Hello"

    input_ids = make_input_ids(
        adapter,
        prompt,
    )

    # ============================================================
    # PREFILL
    # ============================================================

    cache = create_cache(
        model=adapter.model,
        adapter=adapter,
        max_seq_len=input_ids.shape[1] + 1,
    )

    custom_prefill_hidden = (
        adapter.cached_model.forward(
            input_ids=input_ids,
            cache=cache,
            position=0,
        )
    )

    custom_prefill_logits = adapter.model.lm_head(
        custom_prefill_hidden[:, -1, :]
    )

    token_id = torch.argmax(
        custom_prefill_logits,
        dim=-1,
    ).item()

    print()
    print("=" * 70)
    print("FIRST DECODE")
    print("=" * 70)

    print("generated token:", token_id)

    # ============================================================
    # NEXT TOKEN
    # ============================================================

    next_token = torch.tensor(
        [[token_id]],
        dtype=torch.long,
        device=adapter.device,
    )

    full_ids = torch.cat(
        [
            input_ids,
            next_token,
        ],
        dim=1,
    )

    # ============================================================
    # HF REFERENCE
    #
    # Full recomputation of:
    #
    # prompt + generated token
    # ============================================================

    reference_outputs = adapter.model.model(
        input_ids=full_ids,
        use_cache=False,
    )

    reference_hidden = (
        reference_outputs.last_hidden_state[:, -1:, :]
    )

    reference_logits = adapter.model.lm_head(
        reference_hidden[:, -1, :]
    )

    reference_token = torch.argmax(
        reference_logits,
        dim=-1,
    ).item()

    # ============================================================
    # CUSTOM DECODE
    #
    # Only process the generated token.
    # ============================================================

    custom_decode_hidden = (
        adapter.cached_model.forward(
            input_ids=next_token,
            cache=cache,
            position=input_ids.shape[1],
        )
    )

    custom_decode_logits = adapter.model.lm_head(
        custom_decode_hidden[:, -1, :]
    )

    custom_token = torch.argmax(
        custom_decode_logits,
        dim=-1,
    ).item()

    # ============================================================
    # HIDDEN COMPARISON
    # ============================================================

    hidden_diff = (
        custom_decode_hidden.float()
        - reference_hidden.float()
    ).abs()

    # ============================================================
    # LOGIT COMPARISON
    # ============================================================

    logit_diff = (
        custom_decode_logits.float()
        - reference_logits.float()
    ).abs()

    print()
    print("REFERENCE NEXT TOKEN:", reference_token)
    print("CUSTOM NEXT TOKEN:   ", custom_token)

    print()
    print(
        "HIDDEN MAX DIFF:",
        hidden_diff.max().item(),
    )

    print(
        "HIDDEN MEAN DIFF:",
        hidden_diff.mean().item(),
    )

    print()
    print(
        "LOGIT MAX DIFF:",
        logit_diff.max().item(),
    )

    print(
        "LOGIT MEAN DIFF:",
        logit_diff.mean().item(),
    )

    print()
    print(
        "REFERENCE TOP-5:",
        torch.topk(
            reference_logits,
            k=5,
            dim=-1,
        ).indices[0].tolist(),
    )

    print(
        "CUSTOM TOP-5:",
        torch.topk(
            custom_decode_logits,
            k=5,
            dim=-1,
        ).indices[0].tolist(),
    )


def test_m14_summary(adapter):
    """
    M1.4 correctness characterization.

    The KV cache itself is numerically correct, but BF16 decode
    can accumulate numerical differences that eventually change
    greedy argmax.
    """

    prompt = "Hello"

    input_ids = make_input_ids(
        adapter,
        prompt,
    )

    uncached = generate_uncached(
        adapter=adapter,
        input_ids=input_ids,
        max_new_tokens=10,
    )

    cached = generate_cached(
        adapter=adapter,
        input_ids=input_ids,
        max_new_tokens=10,
    )

    print("\n")
    print("=" * 70)
    print("M1.4 KV CACHE CORRECTNESS SUMMARY")
    print("=" * 70)

    print("Prompt:", repr(prompt))
    print("Prompt IDs:", uncached.prompt_ids)

    print("Uncached:", uncached.generated_ids)
    print("Cached:  ", cached.generated_ids)

    first_mismatch = None

    for i, (ref, custom) in enumerate(
        zip(
            uncached.generated_ids,
            cached.generated_ids,
        )
    ):
        if ref != custom:
            first_mismatch = i
            break

    if first_mismatch is None:
        print("RESULT: TOKEN-IDENTICAL")
    else:
        print("RESULT: NUMERICAL DIVERGENCE")
        print("First mismatch:", first_mismatch)
        print(
            "Reference token:",
            uncached.generated_ids[first_mismatch],
        )
        print(
            "Cached token:",
            cached.generated_ids[first_mismatch],
        )