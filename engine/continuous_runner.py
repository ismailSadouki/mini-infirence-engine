from __future__ import annotations


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))






import time
import torch


from engine.kv_cache import KVCache
from scripts.cache_utils import create_request_cache
from engine.batch_builder import build_decode_batch

from engine.scheduler import ContinuousScheduler
from engine.types import RequestState, SamplingConfig





def prefill_request(
        request: RequestState,
        adapter,
        max_new_tokens: int,
        sampling_config: SamplingConfig
):
    request.prefill_start_time = time.perf_counter()

    input_ids = torch.tensor(
        [request.prompt_ids],
        dtype=torch.long,
        device=adapter.device
    )


    cache = create_request_cache(request, adapter, max_new_tokens)

    request.cache_handle = cache

    logits = adapter.forward_prefill_cached(
        input_ids=input_ids,
        cache=cache
    )

    token_id = adapter.sample_next_token(
        logits[:, -1, :].squeeze(0),
        config=sampling_config
    )

    now = time.perf_counter()

    request.generated_ids.append(token_id)
    request.generated_count = 1

    request.current_pos = request.prompt_len + 1

    request.first_token_time = now
    request.token_timestamps.append(now)

    if token_id == adapter.tokenizer.eos_token_id:
        request.finished = True
        request.finished_reason = "eos"
        request.finish_time = now

    elif request.generated_count >= max_new_tokens:
        request.finished = True
        request.finished_reason = "length"
        request.finish_time = now
    



    

def run_continuous(
        scheduler: ContinuousScheduler,
        adapter,
        max_new_tokens,
        sampling_config: SamplingConfig,
) -> list[RequestState]:
    completed = []

    while scheduler.waiting or scheduler.active:
        # scheduler will 
        # 1. evicts finished requests
        # 2. admits waiting requests
        # 3 . returns currently active requests
        finished, active_requests = scheduler.step()

        completed.extend(finished)

        # process each active request for ONE token
        for request in active_requests:

            # newly admitted request - prefill
            if request.cache_handle is None:

                prefill_request(
                    request=request,
                    adapter=adapter,
                    max_new_tokens=max_new_tokens,
                    sampling_config=sampling_config,
                )

                continue

        decode_requests = [
            request
            for request in active_requests
            if request.cache_handle is not None
            and not request.finished
        ]

        if not decode_requests:
            continue

        # Build one regged decode batch
        batch = build_decode_batch(
            requests=decode_requests,
            device=adapter.device
        )
        # one model forward for the entire active batch
        logits = adapter.forward_decode_ragged(
            input_ids=batch.input_ids,
            caches=[
                request.cache_handle
                for request in batch.requests
            ],
            positions=batch.positions
        )
        # Sample + Update each request
        token_time = time.perf_counter()

        for batch_idx, request in enumerate(batch.requests):
            token_id = adapter.sample_next_token(
                logits[batch_idx, -1, :],
                config=sampling_config
            )


            request.generated_ids.append(token_id)
            request.generated_count += 1
            request.current_pos += 1
            request.token_timestamps.append(token_time)

            if token_id == adapter.tokenizer.eos_token_id:

                request.finished = True
                request.finished_reason = "eos"
                request.finish_time = token_time

            elif request.generated_count >= max_new_tokens:

                request.finished = True
                request.finished_reason = "length"
                request.finish_time = token_time
            



            # if request.finished:
            #     continue

            # # first decode step
            # last_token = torch.tensor(
            #     [[request.generated_ids[-1]]],
            #     dtype=torch.long,
            #     device=adapter.device,
            # )  


            # logits = adapter.forward_decode_cached(
            #     last_token=last_token,
            #     cache=request.cache_handle,
            #     position=request.current_pos
            # )

            # token_id = adapter.sample_next_token(
            #     logits[:, -1, :].squeeze(0),
            #     config=sampling_config
            # )


            # token_time = time.perf_counter()


            # # update request state
            # request.generated_ids.append(token_id)
            # request.generated_count += 1
            # request.current_pos += 1
            # request.token_timestamps.append(token_time)

            # # finish conditions
            # if token_id == adapter.tokenizer.eos_token_id:
            #     request.finished = True
            #     request.finished_reason = "eos"
            #     request.finish_time = time.perf_counter()
            # elif (
            #     request.generated_count
            #     >= max_new_tokens
            # ):
            #     request.finished = True
            #     request.finished_reason = "length"
            #     request.finish_time = token_time

    return completed


