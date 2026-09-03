from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from engine.continuous_runner import run_continuous
from engine.scheduler import (
    ContinuousScheduler,
    SchedulerConfig,
)
from engine.model_adapter import ModelAdapter
from engine.types import RequestState, SamplingConfig

from bench.workload import (
    WorkloadSpec,
    generate_requests,
    load_workload,
)
from bench.metrics import aggregate_metrics


def get_torch_dtype(dtype: str) -> torch.dtype:
    """
    Convert workload dtype string into torch.dtype.
    """
    dtype = dtype.replace("torch.", "")

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype not in mapping:
        raise ValueError(
            f"Unsupported dtype: {dtype}"
        )

    return mapping[dtype]


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_peak_gpu_memory_mb(
    device: torch.device,
) -> float | None:
    if device.type != "cuda":
        return None

    peak_bytes = torch.cuda.max_memory_allocated(device)

    return peak_bytes / (1024 ** 2)

def create_scheduler(
        spec: WorkloadSpec,
        max_total_active_tokens: int
) -> ContinuousScheduler:

    return ContinuousScheduler(
        SchedulerConfig(
            max_batch_size=spec.concurrency,
            max_total_active_tokens=max_total_active_tokens,
            max_waiting=spec.num_requests
        )
    )


def request_to_state(request, arrival_time: float) -> RequestState:
    return RequestState(
        request_id=request.request_id,
        prompt_ids=request.prompt_ids,
        max_new_tokens=request.max_new_tokens,
        arrival_time=arrival_time,
    )
# one continuous run
def run_one(
        spec: WorkloadSpec,
        adapter: ModelAdapter,
        max_total_active_tokens: int
):
    """
    Execute one complete continuous-batching run
    """
    run_start = time.perf_counter()

    requests = generate_requests(spec)

    states = [
        request_to_state(
            request,
            arrival_time=run_start,
        )
        for request in requests
    ]
    scheduler = create_scheduler(
        spec=spec,
        max_total_active_tokens=max_total_active_tokens
    )


    for state in states:
        scheduler.submit(state)

    synchronize_cuda(adapter.device)


    start_time = time.perf_counter()

    completed = run_continuous(
        scheduler=scheduler,
        adapter=adapter,
        sampling_config=SamplingConfig(
            greedy=True
        )
    )

    synchronize_cuda(adapter.device)

    end_time = time.perf_counter()

    wall_time = end_time - start_time

    return completed, wall_time


def run_warmup(
        spec: WorkloadSpec,
        adapter: ModelAdapter,
        max_total_active_tokens: int
) -> None:
    if spec.warmup <= 0:
        return

    print()
    print(
        f"Warmup runs: {spec.warmup}"
    )

    for i in range(spec.warmup):
        run_one(
            spec=spec,
            adapter=adapter,
            max_total_active_tokens=max_total_active_tokens
        )

        print(
            f" warmup {i + 1}/{spec.warmup}"
        )

    print('Warmup excluded from metrics.')


def run_benchmark(
        workload_path: str | Path,
        max_total_active_tokens: int
): 
    spec = load_workload(workload_path)


    dtype = get_torch_dtype(
        spec.dtype
    )

    print("=" * 72)
    print("CONTINUOUS BATCHING BENCHMARK")
    print("=" * 72)

    print(
        f"workload              : {spec.name}"
    )

    print(
        f"model                 : {spec.model}"
    )

    print(
        f"dtype                 : {spec.dtype}"
    )

    print(
        f"device                : {spec.device}"
    )

    print(
        f"requests              : {spec.num_requests}"
    )

    print(
        f"concurrency           : {spec.concurrency}"
    )

    print(
        f"max active tokens     : "
        f"{max_total_active_tokens}"
    )

    print(
        f"warmup                : {spec.warmup}"
    )

    print(
        f"repetitions           : {spec.repetitions}"
    )

    print("=" * 72)



    print()
    print("Loading model...")

    adapter = ModelAdapter(
        model_name=spec.model,
        device=spec.device,
        dtype=dtype
    )

    print("Model loaded.")


    run_warmup(
        spec=spec,
        adapter=adapter,
        max_total_active_tokens=max_total_active_tokens
    )


    if adapter.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            adapter.device
        )


    # Measured repetitions
    measured_states = []

    total_wall_time = 0.0

    print()
    print(
        f"Measured repetitions: "
        f"{spec.repetitions}"
    )

    for repetition in range(spec.repetitions):
        print()
        print(
            f"--- Repetition "
            f"{repetition + 1}/"
            f"{spec.repetitions} ---"
        )

        completed, wall_time = run_one(
            spec=spec,
            adapter=adapter,
            max_total_active_tokens=max_total_active_tokens
        )

        measured_states.extend(
            completed
        )

        total_wall_time += wall_time

        print(
            f"completed requests : "
            f"{len(completed)}"
        )

        print(
            f"wall time          : "
            f"{wall_time:.4f} s"
        )


    peak_gpu_memory_mb = get_peak_gpu_memory_mb(
        adapter.device
    )
    # Aggregate metrics



    metrics = aggregate_metrics(
        measured_states,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
    )

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    print(
        f"requests            : "
        f"{len(metrics.requests)}"
    )

    print(
        f"output tokens       : "
        f"{metrics.output_tokens}"
    )

    print()


    print(
        f"TTFT p50             : "
        f"{metrics.ttft_p50:.6f} s"
    )

    print(
        f"TTFT p95             : "
        f"{metrics.ttft_p95:.6f} s"
    )

    if metrics.itl_p50 is not None:

        print(
            f"ITL p50              : "
            f"{metrics.itl_p50:.6f} s"
        )

        print(
            f"ITL p95              : "
            f"{metrics.itl_p95:.6f} s"
        )

    else:

        print(
            "ITL                  : N/A"
        )

    print()

    print(
        f"total wall time      : "
        f"{total_wall_time:.4f} s"
    )

    print(
        f"throughput           : "
        f"{metrics.throughput:.2f} tokens/s"
    )

    print(
        f"peak GPU memory     : "
        f"{metrics.peak_gpu_memory_mb:.2f} MB"
        if metrics.peak_gpu_memory_mb is not None
        else "peak GPU memory     : N/A"
    )

    print("=" * 72)

    return metrics



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark continuous batching "
            "in the mini inference engine."
        )
    )

    parser.add_argument(
        "workload",
        type=str,
        help="Path to workload YAML",
    )

    parser.add_argument(
        "--max-total-active-tokens",
        type=int,
        default=32768,
        help=(
            "Maximum number of active tokens "
            "allowed by the scheduler."
        ),
    )

    args = parser.parse_args()

    run_benchmark(
        workload_path=args.workload,
        max_total_active_tokens=args.max_total_active_tokens,
    )


if __name__ == "__main__":
    main()