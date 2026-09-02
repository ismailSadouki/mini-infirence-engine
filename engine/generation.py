from __future__ import annotations


from pathlib import Path
import sys

import yaml

from engine.kv_cache import KVCache

sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import torch
from .model_adapter import ModelAdapter
from .types import (
    GenerationOutput,
    GenerationRequest,
    RequestState,
)






def generate_prefill_decode_cached(
        adapter: ModelAdapter,
        request: GenerationRequest
) -> GenerationOutput:
    # prepare prompt
    if request.prompt_ids is not None:
        prompt_ids = request.prompt_ids

    elif request.prompt_text is not None:
        prompt_ids = adapter.tokenize(
            request.prompt_text
        )

    else:
        raise ValueError(
            "Either prompt_text or prompt_ids must be provided."
        )

    if len(prompt_ids) == 0 :
        raise ValueError(
            "prompt cannot be empty"
        )

    # request state
    state = RequestState(
        request_id = request.request_id,
        prompt_ids = prompt_ids,
        arrival_time = request.arrival_time
    )


    with open("configs/inference.yaml", "r") as f:
        config = yaml.safe_load(f)

        


    # Cache
    cache = KVCache(
        num_layers=config["kv_cache"]["num_layers"],
        num_kv_heads=config["kv_cache"]["num_kv_heads"],
        max_seq_len=config["kv_cache"]["max_seq_len"],
        head_dim=config["kv_cache"]["head_dim"],
        dtype=adapter.dtype,
        device=adapter.device,
    )

    # PREFILL
    #========================================
    state.prefill_start_time = time.perf_counter()
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=adapter.device
    )

    logits = adapter.forward_prefill_cached(
        input_ids,
        cache=cache
    )

    prefill_tokens = len(prompt_ids)

    # the logits at the final prompt position
    # predict the first generated tokens
    next_token = adapter.sample_next_token(
        logits[0, -1],
        request.sampling
    )
    now = time.perf_counter()


    state.generated_ids.append(next_token)
    state.token_timestamps.append(now)

    state.first_token_time = now

    # DECODE
    #========================================
    state.position = len(prompt_ids)

    decode_tokens = 0
    for _ in range(
        request.max_new_tokens - 1
    ):
        last_token = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=adapter.device
        )

        logits = adapter.forward_decode_cached(
            last_token = last_token,
            cache=cache,
            position=state.position
        )

        next_token = adapter.sample_next_token(
            logits[0, -1],
            request.sampling
        )

        now = time.perf_counter()

        state.generated_ids.append(next_token)
        state.token_timestamps.append(now)
        decode_tokens += 1
        state.position += 1


    state.finished = True
    state.finish_reason = "length"
    state.finish_time = time.perf_counter()
    ttft = (
        state.first_token_time
        - state.arrival_time
    )

    itl = [
        state.token_timestamps[i]
        - state.token_timestamps[i - 1]
        for i in range(
            1,
            len(state.token_timestamps),
        )
    ]

    text = adapter.decode(
        state.generated_ids
    )


    return GenerationOutput(
        request_id=request.request_id,
        output_ids=state.generated_ids,
        text=text,
        ttft=ttft,
        itl=itl,
        finish_reason=state.finish_reason,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
    )



def generate_prefill_decode(
        adapter: ModelAdapter,
        request: GenerationRequest
) -> GenerationOutput:
    # prepare prompt
    if request.prompt_ids is not None:
        prompt_ids = request.prompt_ids

    elif request.prompt_text is not None:
        prompt_ids = adapter.tokenize(
            request.prompt_text
        )

    else:
        raise ValueError(
            "Either prompt_text or prompt_ids must be provided."
        )

    if len(prompt_ids) == 0 :
        raise ValueError(
            "prompt cannot be empty"
        )

    # request state
    state = RequestState(
        request_id = request.request_id,
        prompt_ids = prompt_ids,
        arrival_time = request.arrival_time
    )




    # PREFILL
    #========================================
    state.prefill_start_time = time.perf_counter()
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=adapter.device
    )

    logits, cache = adapter.forward_prefill(
        input_ids
    )

    prefill_tokens = len(prompt_ids)

    # the logits at the final prompt position
    # predict the first generated tokens
    next_token = adapter.sample_next_token(
        logits[0, -1],
        request.sampling
    )
    now = time.perf_counter()


    state.generated_ids.append(next_token)
    state.token_timestamps.append(now)

    state.first_token_time = now

    # DECODE
    #========================================
    state.position = len(prompt_ids)

    decode_tokens = 0
    for _ in range(
        request.max_new_tokens - 1
    ):
        last_token = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=adapter.device
        )

        logits, cache = adapter.forward_decode(
            last_token = last_token,
            cache=cache,
            position=state.position
        )

        next_token = adapter.sample_next_token(
            logits[0, -1],
            request.sampling
        )

        now = time.perf_counter()

        state.generated_ids.append(next_token)
        state.token_timestamps.append(now)
        decode_tokens += 1
        state.position += 1


    state.finished = True
    state.finish_reason = "length"
    state.finish_time = time.perf_counter()
    ttft = (
        state.first_token_time
        - state.arrival_time
    )

    itl = [
        state.token_timestamps[i]
        - state.token_timestamps[i - 1]
        for i in range(
            1,
            len(state.token_timestamps),
        )
    ]

    text = adapter.decode(
        state.generated_ids
    )


    return GenerationOutput(
        request_id=request.request_id,
        output_ids=state.generated_ids,
        text=text,
        ttft=ttft,
        itl=itl,
        finish_reason=state.finish_reason,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
    )


def generate_no_cache(
        adapter: ModelAdapter,
        request: GenerationRequest
) -> GenerationOutput:
    if request.prompt_ids is not None:
        prompt_ids = request.prompt_ids
    elif request.prompt_text is not None:
        prompt_ids = adapter.tokenize(
            request.prompt_text
        )
    else:
        raise ValueError(
            "request must have either prompt_text or prompt_ids"
        )

    state = RequestState(
        request_id = request.request_id,
        prompt_ids = prompt_ids,
        arrival_time = request.arrival_time
    )

    state.prefill_start_time = time.perf_counter()
    current_ids = list(prompt_ids)

    for _ in range(request.max_new_tokens):
        input_ids = torch.tensor(
            [current_ids],
            dtype=torch.long,
            device=adapter.device
        )
        logits = adapter.forward_no_cache(
            input_ids
        )

        next_token = adapter.sample_next_token(
            logits[0, -1],
            request.sampling
        )

        current_ids.append(next_token)
        state.generated_ids.append(next_token)

        now = time.perf_counter()
        state.token_timestamps.append(now)

        if state.first_token_time is None:
            state.first_token_time = now

    state.finished = True
    state.finish_reason = "length"
    state.finish_time = time.perf_counter()


    ttft = (
        state.first_token_time - state.arrival_time
    )

    itl = [
        state.token_timestamps[i] - state.token_timestamps[i-1]
        for i in range(
            1, len(state.token_timestamps)
        )
    ]


    text = adapter.decode(
        state.generated_ids
    )

    return GenerationOutput(
        request_id = request.request_id,
        output_ids = state.generated_ids,
        text = text,
        ttft = ttft,
        itl = itl,
        finish_reason = state.finish_reason
    )