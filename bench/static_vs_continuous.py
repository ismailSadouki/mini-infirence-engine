from __future__ import annotations


from pathlib import Path
from sched import scheduler
import sys


sys.path.append(str(Path(__file__).resolve().parent.parent))




import statistics
import time
from dataclasses import dataclass

import torch

from engine.continuous_runner import run_continuous
from engine.model_adapter import ModelAdapter
from engine.scheduler import ContinuousScheduler, SchedulerConfig
from engine.types import RequestState, SamplingConfig
from scripts.cache_utils import create_request_cache




MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_BATCH_SIZE = 2


WORKLOAD = [
    {
        "request_id": "r1",
        "prompt_len": 32,
        "max_new_tokens": 8,
    },
    {
        "request_id": "r2",
        "prompt_len": 32,
        "max_new_tokens": 32,
    },
    {
        "request_id": "r3",
        "prompt_len": 32,
        "max_new_tokens": 64,
    },
    {
        "request_id": "r4",
        "prompt_len": 32,
        "max_new_tokens": 16,
    },
    {
        "request_id": "r5",
        "prompt_len": 32,
        "max_new_tokens": 48,
    },
    {
        "request_id": "r6",
        "prompt_len": 32,
        "max_new_tokens": 8,
    },
]

@dataclass
class BenchmarkResult:
    name: str
    elapsed: float
    output_tokens: int
    throughput: float
    ttft_p50: float
    ttft_p95: float
    itl_p50: float
    itl_p95: float
    total_latency_p50: float
    total_latency_p95: float

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]


    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return (
        values[lower]
        + weight * (values[upper] - values[lower])
    )


def build_workload() -> list[RequestState]:
    requests = []


    for item in WORKLOAD:
        request = RequestState(
            request_id=item["request_id"],
            prompt_ids=list(range(item["prompt_len"]))
        )

        request.max_new_tokens = item["max_new_tokens"]

        requests.append(request)

    return requests



def synchronize_cuda() -> None:

    if torch.cuda.is_available():
        torch.cuda.synchronize()

def compute_result(
        name: str,
        requests: list[RequestState],
        elapsed: float
) -> BenchmarkResult:

    output_tokens = sum(
        request.generated_count
        for request in requests
    )

    throughput = (
        output_tokens / elapsed
        if elapsed > 0
        else 0.0
    )

    ttfts = [
        request.first_token_time - request.prefill_start_time
        for request in requests
        if request.first_token_time is not None
        and request.prefill_start_time is not None
    ]

    itls = []

    for request in requests:
        timestamps = request.token_timestamps

        for previous, current in zip(
            timestamps,
            timestamps[1:]
        ): 
            itls.append(current - previous)

    total_latencies = [
        request.finish_time - request.arrival_time
        for request in requests
        if request.finish_time is not None
    ]


    return BenchmarkResult(
        name=name,
        elapsed=elapsed,
        output_tokens=output_tokens,
        throughput=throughput,
        ttft_p50=percentile(ttfts, 0.50),
        ttft_p95=percentile(ttfts, 0.95),
        itl_p50=percentile(itls, 0.50),
        itl_p95=percentile(itls, 0.95),
        total_latency_p50=percentile(
            total_latencies,
            0.50,
        ),
        total_latency_p95=percentile(
            total_latencies,
            0.95,
        ),
    )


def run_static(
        adapter: ModelAdapter,
        requests: list[RequestState],
        sampling_config: SamplingConfig
) -> None:
    for start in range(
        0,
        len(requests),
        MAX_BATCH_SIZE
    ):
        batch = requests[
            start: start + MAX_BATCH_SIZE
        ]
        # Create independent KVCache for each requast

        caches = []
        for request in batch:
            cache = create_request_cache(
                request=request,
                adapter=adapter,
                max_new_tokens=request.max_new_tokens
            )

            request.cache_handle = cache
            caches.append(cache)


        # PREFILL
        for request, cache in zip(batch, caches):
            request.prefill_start_time = (time.perf_counter())
            input_ids = torch.tensor(
                [request.prompt_ids],
                dtype=torch.long,
                device=adapter.device
            )

            with torch.inference_mode():
                logits = (
                    adapter.forward_prefill_cached(
                        input_ids=input_ids,
                        cache=cache
                    )
                )

            # First generated token
            token_id = adapter.sample_next_token(
                logits[0, -1, :], # why 0 in here?
                sampling_config
            )

            now = time.perf_counter()

            request.generated_ids.append(
                token_id
            )

            request.generated_count = 1

            request.current_pos = (
                request.prompt_len + 1
            )

            request.first_token_time = now

            request.token_timestamps.append(
                now
            )


            # Finished?

            if (
                token_id
                == adapter.tokenizer.eos_token_id
            ):

                request.finished = True
                request.finished_reason = "eos"
                request.finish_time = now

            elif (
                request.generated_count
                >= request.max_new_tokens
            ):

                request.finished = True
                request.finished_reason = "length"
                request.finish_time = now



        # Decode remaining tokens
        # We keep iterating over this SAME batch.
        # Finished requests are skipped.
        # They are NOT replaced by waiting requests.

        while not all(
            request.finished
            for request in batch
        ):
            for request, cache in zip(
                batch,
                caches,
            ):
                if request.finished:
                    continue

                # one token
                last_token = torch.tensor(
                    [[request.generated_ids[-1]]],
                    dtype=torch.long,
                    device=adapter.device
                )

                with torch.inference_mode():
                    logits = (
                        adapter.forward_decode_cached(
                            last_token = last_token,
                            cache=cache,
                            position=request.current_pos
                        )
                    )


                # sample
                token_id = adapter.sample_next_token(
                    logits[0, -1, :],
                    sampling_config
                )

                now = time.perf_counter()


                request.generated_ids.append(
                    token_id
                )

                request.generated_count += 1

                request.current_pos += 1

                request.token_timestamps.append(
                    now
                )
                # Finished?
                if (
                    token_id
                    == adapter.tokenizer.eos_token_id
                ):

                    request.finished = True
                    request.finished_reason = "eos"
                    request.finish_time = now

                elif (
                    request.generated_count
                    >= request.max_new_tokens
                ):

                    request.finished = True
                    request.finished_reason = "length"
                    request.finish_time = now





def run_continuous_benchmark(
        adapter: ModelAdapter,
        requests: list[RequestState],
        sampling_config: SamplingConfig
) -> None:
    scheduler = ContinuousScheduler(
        SchedulerConfig(
            max_batch_size=MAX_BATCH_SIZE,
            max_total_active_tokens=10000,
            max_waiting=len(requests)
        )
    )

    # submit exactly the same requests
    for request in requests:
        scheduler.submit(request)


    run_continuous(
        scheduler=scheduler,
        adapter=adapter,
        sampling_config=sampling_config
    )



def print_result(
    result: BenchmarkResult,
) -> None:

    print()

    print("=" * 72)

    print(result.name)

    print("=" * 72)

    print(
        f"Elapsed time:          "
        f"{result.elapsed:.4f} s"
    )

    print(
        f"Output tokens:         "
        f"{result.output_tokens}"
    )

    print(
        f"Throughput:            "
        f"{result.throughput:.2f} tokens/s"
    )

    print()

    print(
        f"TTFT p50:              "
        f"{result.ttft_p50 * 1000:.2f} ms"
    )

    print(
        f"TTFT p95:              "
        f"{result.ttft_p95 * 1000:.2f} ms"
    )

    print()

    print(
        f"ITL p50:               "
        f"{result.itl_p50 * 1000:.2f} ms"
    )

    print(
        f"ITL p95:               "
        f"{result.itl_p95 * 1000:.2f} ms"
    )

    print()

    print(
        f"Total latency p50:     "
        f"{result.total_latency_p50 * 1000:.2f} ms"
    )

    print(
        f"Total latency p95:     "
        f"{result.total_latency_p95 * 1000:.2f} ms"
    )


def print_comparison(
    static_result: BenchmarkResult,
    continuous_result: BenchmarkResult,
) -> None:

    print()

    print("=" * 72)
    print("STATIC VS CONTINUOUS")
    print("=" * 72)

    print(
        f"{'Metric':<25}"
        f"{'Static':>15}"
        f"{'Continuous':>15}"
    )

    print("-" * 55)

    print(
        f"{'Throughput (tok/s)':<25}"
        f"{static_result.throughput:>15.2f}"
        f"{continuous_result.throughput:>15.2f}"
    )

    print(
        f"{'TTFT p50 (ms)':<25}"
        f"{static_result.ttft_p50 * 1000:>15.2f}"
        f"{continuous_result.ttft_p50 * 1000:>15.2f}"
    )

    print(
        f"{'TTFT p95 (ms)':<25}"
        f"{static_result.ttft_p95 * 1000:>15.2f}"
        f"{continuous_result.ttft_p95 * 1000:>15.2f}"
    )

    print(
        f"{'ITL p50 (ms)':<25}"
        f"{static_result.itl_p50 * 1000:>15.2f}"
        f"{continuous_result.itl_p50 * 1000:>15.2f}"
    )

    print(
        f"{'ITL p95 (ms)':<25}"
        f"{static_result.itl_p95 * 1000:>15.2f}"
        f"{continuous_result.itl_p95 * 1000:>15.2f}"
    )



    print(
        f"{'Total latency p50 (ms)':<25}"
        f"{static_result.total_latency_p50 * 1000:>15.2f}"
        f"{continuous_result.total_latency_p50 * 1000:>15.2f}"
    )

    print(
        f"{'Total latency p95 (ms)':<25}"
        f"{static_result.total_latency_p95 * 1000:>15.2f}"
        f"{continuous_result.total_latency_p95 * 1000:>15.2f}"
    )



def main() -> None:
    adapter = ModelAdapter(
        model_name=MODEL_NAME,
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    sampling_config = SamplingConfig(
        temperature=0.0,
        greedy=True,
    )


    print()

    print("Static vs Continuous Benchmark")

    print()

    print("Workload:")

    for item in WORKLOAD:

        print(
            f"  {item['request_id']}: "
            f"prompt={item['prompt_len']} "
            f"max_new_tokens={item['max_new_tokens']}"
        )

    print()

    print(
        f"max_batch_size = "
        f"{MAX_BATCH_SIZE}"
    )
    # STATIC
    # ============================================================

    static_requests = build_workload()

    synchronize_cuda()

    static_start = time.perf_counter()

    run_static(
        adapter=adapter,
        requests=static_requests,
        sampling_config=sampling_config,
    )

    synchronize_cuda()

    static_elapsed = (
        time.perf_counter()
        - static_start
    )


    static_result = compute_result(
        name="STATIC BATCHING",
        requests=static_requests,
        elapsed=static_elapsed,
    )
    # CONTINUOUS
    # ============================================================

    continuous_requests = build_workload()

    synchronize_cuda()

    continuous_start = time.perf_counter()

    run_continuous_benchmark(
        adapter=adapter,
        requests=continuous_requests,
        sampling_config=sampling_config,
    )

    synchronize_cuda()

    continuous_elapsed = (
        time.perf_counter()
        - continuous_start
    )

    continuous_result = compute_result(
        name="CONTINUOUS BATCHING",
        requests=continuous_requests,
        elapsed=continuous_elapsed,
    )


    # Print results

    print_result(static_result)

    print_result(continuous_result)

    print_comparison(
        static_result,
        continuous_result,
    )


if __name__ == "__main__":
    main()