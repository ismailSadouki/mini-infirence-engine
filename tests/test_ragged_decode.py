from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engine.model_adapter import ModelAdapter
from scripts.cache_utils import create_request_cache
from scripts.requests import make_request


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def adapter():
    return ModelAdapter(
        model_name=MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )


def test_ragged_decode(adapter):
    """
    Verify that ragged decode successfully handles multiple requests
    with different prompt lengths and independent KV caches.

    This test is about the ragged batching implementation.

    It verifies:

        - B requests can be decoded in one call
        - each request owns an independent KV cache
        - requests may have different prompt lengths
        - requests may have different decode positions
        - the ragged decode returns one output per request
        - the returned output has the expected shape
        - the output contains finite values

    It intentionally does not compare numerical values against
    another execution path.
    """

    B = 3

    prompts = [
        [10, 11, 12, 13],
        [20, 21, 22, 23, 24, 25, 26],
        [30, 31, 32, 33, 34],
    ]

    decode_tokens = [
        40,
        50,
        60,
    ]

    assert len(prompts) == B
    assert len(decode_tokens) == B

    hidden_size = adapter.model.config.hidden_size

    # ------------------------------------------------------------
    # Create requests.
    # ------------------------------------------------------------

    requests = []

    for i in range(B):
        request = make_request(
            f"ragged-{i}",
            prompt_len=len(prompts[i]),
        )

        request.prompt_ids = prompts[i]
        request.prompt_len = len(prompts[i])

        requests.append(request)

    # ------------------------------------------------------------
    # Each request has its own logical decode position.
    #
    #     request 0 -> 4
    #     request 1 -> 7
    #     request 2 -> 5
    # ------------------------------------------------------------

    positions = torch.tensor(
        [request.prompt_len for request in requests],
        dtype=torch.long,
        device=adapter.device,
    )

    assert positions.shape == (B,)
    assert positions.tolist() == [4, 7, 5]

    # ------------------------------------------------------------
    # One decode token per request.
    #
    # Shape:
    #
    #     [B, 1]
    # ------------------------------------------------------------

    input_ids = torch.tensor(
        [[token_id] for token_id in decode_tokens],
        dtype=torch.long,
        device=adapter.device,
    )

    assert input_ids.shape == (B, 1)

    # ------------------------------------------------------------
    # Create independent KV caches.
    # ------------------------------------------------------------

    caches = [
        create_request_cache(
            request=request,
            adapter=adapter,
            max_new_tokens=1,
        )
        for request in requests
    ]

    assert len(caches) == B
    assert len({id(cache) for cache in caches}) == B

    # ------------------------------------------------------------
    # Prefill every request into its own cache.
    # ------------------------------------------------------------

    with torch.inference_mode():
        for i in range(B):
            prompt = torch.tensor(
                [prompts[i]],
                dtype=torch.long,
                device=adapter.device,
            )

            adapter.forward_prefill_cached(
                input_ids=prompt,
                cache=caches[i],
            )

    # ------------------------------------------------------------
    # RAGGED DECODE
    #
    # This is the actual thing under test.
    #
    # One call:
    #
    #     input_ids -> [3, 1]
    #     positions -> [3]
    #     caches    -> [cache0, cache1, cache2]
    #
    # Expected:
    #
    #     output -> [3, 1, hidden_size]
    # ------------------------------------------------------------

    with torch.inference_mode():
        output = adapter.forward_decode_ragged(
            input_ids=input_ids,
            caches=caches,
            positions=positions,
        )

    # ------------------------------------------------------------
    # Verify the ragged output contract.
    # ------------------------------------------------------------

    assert isinstance(output, torch.Tensor)

    assert output.shape == (
        B,
        1,
        hidden_size,
    )

    assert output.shape[0] == len(caches)
    assert output.shape[0] == positions.shape[0]
    assert output.shape[0] == input_ids.shape[0]

    assert output.dtype == adapter.model.dtype

    assert torch.isfinite(output).all()


def test_ragged_decode_active_batch(adapter):
    """
    Verify that an active batch can shrink as requests finish.

    Requests have different prompt lengths and different generation
    lengths:

        request 0 -> generates 3 tokens
        request 1 -> generates 1 token
        request 2 -> generates 2 tokens

    Active batches:

        step 1 -> [request 0, request 1, request 2]
        step 2 -> [request 0, request 2]
        step 3 -> [request 0]

    This verifies:

        - each request keeps its own KV cache
        - each request keeps its own logical position
        - one token is decoded per active request
        - requests can leave the active batch independently
        - the remaining requests can continue decoding
        - output shape follows the current active batch size

    No numerical comparison is performed.
    """

    prompts = [
        [10, 11, 12, 13],          # length 4
        [20, 21, 22, 23, 24, 25],  # length 6
        [30, 31, 32, 33, 34],      # length 5
    ]

    generated_tokens = [
        [40, 41, 42],  # request 0 -> 3 decode steps
        [50],          # request 1 -> 1 decode step
        [60, 61],      # request 2 -> 2 decode steps
    ]

    B = len(prompts)
    hidden_size = adapter.model.config.hidden_size

    # ------------------------------------------------------------
    # Create one request and one KV cache per sequence.
    # ------------------------------------------------------------

    requests = []

    for i in range(B):
        request = make_request(
            f"active-{i}",
            prompt_len=len(prompts[i]),
        )

        request.prompt_ids = prompts[i]
        request.prompt_len = len(prompts[i])

        requests.append(request)

    caches = [
        create_request_cache(
            request=request,
            adapter=adapter,
            max_new_tokens=len(generated_tokens[i]),
        )
        for i, request in enumerate(requests)
    ]

    assert len(caches) == B
    assert len({id(cache) for cache in caches}) == B

    # ------------------------------------------------------------
    # Prefill all requests independently.
    # ------------------------------------------------------------

    with torch.inference_mode():
        for i in range(B):
            prompt = torch.tensor(
                [prompts[i]],
                dtype=torch.long,
                device=adapter.device,
            )

            adapter.forward_prefill_cached(
                input_ids=prompt,
                cache=caches[i],
            )

    # ------------------------------------------------------------
    # Active requests.
    #
    # Each entry stores the request index.
    #
    # Initially:
    #
    #     [0, 1, 2]
    # ------------------------------------------------------------

    active = [0, 1, 2]

    positions = {
        i: len(prompts[i])
        for i in range(B)
    }

    generated = {
        i: 0
        for i in range(B)
    }

    # ------------------------------------------------------------
    # Decode until every request finishes.
    # ------------------------------------------------------------

    while active:

        input_ids = torch.tensor(
            [
                [generated_tokens[i][generated[i]]]
                for i in active
            ],
            dtype=torch.long,
            device=adapter.device,
        )

        active_positions = torch.tensor(
            [
                positions[i]
                for i in active
            ],
            dtype=torch.long,
            device=adapter.device,
        )

        active_caches = [
            caches[i]
            for i in active
        ]

        # --------------------------------------------------------
        # One ragged decode call for the current active batch.
        # --------------------------------------------------------

        with torch.inference_mode():
            output = adapter.forward_decode_ragged(
                input_ids=input_ids,
                caches=active_caches,
                positions=active_positions,
            )

        # --------------------------------------------------------
        # Output must contain exactly one result per active request.
        # --------------------------------------------------------

        assert isinstance(output, torch.Tensor)

        assert output.shape == (
            len(active),
            1,
            hidden_size,
        )

        assert output.shape[0] == input_ids.shape[0]
        assert output.shape[0] == active_positions.shape[0]
        assert output.shape[0] == len(active_caches)

        assert output.dtype == adapter.model.dtype
        assert torch.isfinite(output).all()

        # --------------------------------------------------------
        # Advance every request that participated in this step.
        # --------------------------------------------------------

        next_active = []

        for i in active:
            positions[i] += 1
            generated[i] += 1

            if generated[i] < len(generated_tokens[i]):
                next_active.append(i)

        active = next_active

    # ------------------------------------------------------------
    # Final state.
    #
    # request 0:
    #
    #     prompt = 4
    #     generated = 3
    #     final position = 7
    #
    # request 1:
    #
    #     prompt = 6
    #     generated = 1
    #     final position = 7
    #
    # request 2:
    #
    #     prompt = 5
    #     generated = 2
    #     final position = 7
    # ------------------------------------------------------------

    assert active == []

    assert positions == {
        0: 7,
        1: 7,
        2: 7,
    }

    assert generated == {
        0: 3,
        1: 1,
        2: 2,
    }