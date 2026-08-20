from __future__ import annotations


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import torch
from .model_adapter import ModelAdapter
from .types import (
    GenerationOutput,
    GenerationRequest,
    RequestState,
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