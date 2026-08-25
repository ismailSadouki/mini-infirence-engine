from dataclasses import dataclass
import time



from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engine.kv_cache import KVCache
from engine.types import RequestState, SamplingConfig



@dataclass
class StaticRequestResult:
    request_id: str

    ttft: float
    itl: float
    total_latency: float

    generated_tokens: int
    finish_reason: str


def create_request_cache(request, adapter, max_new_tokens):
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


def run_static_batch(
        requests: list[RequestState],
        batch_size: int,
        adapter,
        max_new_tokens: int,
        sampling_config: SamplingConfig

) ->  list[StaticRequestResult]:

    
    if batch_size <= 0:
        raise ValueError(
            "batch size must be positive"
        )

    results = []

    # Fixed batches
    for batch_start in range(0, len(requests), batch_size):

        batch = requests[
            batch_start: batch_start + batch_size
        ]

        print(
            f"Running static batch: "
            f"{[request.request_id for request in batch]}"
        )


        # PREFILL
        for request in batch:
            request.prefill_start_time = time.perf_counter()

            # For now, use the existing single-request path.
            # M2.2 is establishing scheduler semantics,
            # not ragged batched KV-cache execution yet.
            input_ids = adapter.to_tensor(request.prompt_ids)


            cache = create_request_cache(
                request,
                adapter,
                max_new_tokens
            )

            logits = adapter.forward_prefill_cached(
                input_ids=input_ids,
                cache=cache
            )

            request.cache_handle = cache 


            token_id = adapter.sample_next_token(
                logits[:, -1, :].squeeze(0),
                sampling_config
            )


            request.generated_ids.append(token_id)
            request.generated_count = 1
            request.current_pos = request.prompt_len

            request.first_token_time = time.perf_counter()
            request.token_timestamps.append(
                request.first_token_time
            )

            if token_id == adapter.tokenizer.eos_token_id:
                request.finished = True
                request.finished_reason = "eos"
            elif request.generated_count == max_new_tokens:
                request.finished = True
                request.finished_reason = "length"

        # DECODE
        while not all(request.finished for request in batch):
            for request in batch:

                if request.finished:
                    continue


                last_token = torch.tensor(
                    [[request.generated_ids[-1]]],
                    dtype=torch.long,
                    device=adapter.device,
                )

                decode_start = time.perf_counter()


                logits = adapter.forward_decode_cached(
                    last_token=last_token,
                    cache=request.cache_handle,
                    position=request.current_pos
                )


                decode_end = time.perf_counter()

                request.token_timestamps.append(decode_end)

                # select next token
                token_id = adapter.sample_next_token(
                    logits[:, -1, :].squeeze(0),
                    sampling_config
                )

                request.generated_ids.append(token_id)

                request.generated_count += 1
                request.current_pos += 1

                if token_id == adapter.tokenizer.eos_token_id:

                    request.finished = True
                    request.finished_reason = "eos"

                elif request.generated_count >= max_new_tokens:

                    request.finished = True
                    request.finished_reason = "length"


        # batch finished
        for request in batch:

            request.finish_time = time.perf_counter()
            

            ttft = (
                request.first_token_time
                - request.prefill_start_time
            )

            total_latency = (
                request.finish_time
                - request.arrival_time
            )

            if len(request.token_timestamps) > 1:
                intervals = [
                    t2 - t1
                    for t1, t2 in zip(
                        request.token_timestamps,
                        request.token_timestamps[1:]
                    )
                ]

                itl = sum(intervals) / len(intervals)


            else:
                itl  = 0.0

            results.append(
                StaticRequestResult(
                   request_id=request.request_id,
                   ttft=ttft,
                   itl=itl,
                   total_latency=total_latency,
                   generated_tokens=request.generated_count,
                   finish_reason=request.finished_reason 
                )
            )

    return results

            