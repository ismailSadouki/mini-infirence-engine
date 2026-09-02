
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))



from engine.types import RequestState


@dataclass
class RequestMetrics:
    request_id: str
    ttft: float
    itl: list[float]
    total_latency: float
    output_tokens: int


@dataclass
class AggregateMetrics:
    requests: list[RequestMetrics]

    ttft_p50: float
    ttft_p95: float

    itl_p50: float | None
    itl_p95: float | None

    total_latency_p50: float
    total_latency_p95: float

    output_tokens: int
    throughput: float


def percentile(
    values: list[float],
    p: float,
) -> float:
    """
    Compute a percentile using linear interpolation.

    p must be between 0 and 1.
    """

    if not values:
        raise ValueError(
            "Cannot compute percentile of empty list"
        )

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            "Percentile must be between 0 and 1"
        )
    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p

    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    weight = position - lower

    return (
        values[lower]
        + weight * (
            values[upper] - values[lower]
        )
    )

def compute_request_metrics(
    state: RequestState,
) -> RequestMetrics:
    """
    Compute metrics for one completed request.
    """

    if state.first_token_time is None:
        raise ValueError(
            f"{state.request_id} has no first_token_time"
        )

    if state.finish_time is None:
        raise ValueError(
            f"{state.request_id} has no finish_time"
        )

    if not state.token_timestamps:
        raise ValueError(
            f"{state.request_id} has no token timestamps"
        )


    # TTFT:
    # request arrival -> first generated token
    ttft = (
        state.first_token_time
        - state.arrival_time
    )

    # ITL:
    # time between consecutive generated tokens
    itl = [
        state.token_timestamps[i]
        - state.token_timestamps[i - 1]
        for i in range(
            1,
            len(state.token_timestamps),
        )
    ]

    # Total request latency:
    # request arrival -> request completion
    total_latency = (
        state.finish_time
        - state.arrival_time
    )

    output_tokens = len(
        state.generated_ids
    )

    return RequestMetrics(
        request_id=state.request_id,
        ttft=ttft,
        itl=itl,
        total_latency=total_latency,
        output_tokens=output_tokens,
    )
def aggregate_metrics(
    states: list[RequestState],
) -> AggregateMetrics:
    """
    Aggregate metrics across completed requests.

    Throughput:

        total output tokens
        -------------------
        benchmark wall time

    where benchmark wall time is:

        latest finish time - earliest arrival time
    """

    if not states:
        raise ValueError(
            "Cannot aggregate empty request list"
        )

    requests = [
        compute_request_metrics(state)
        for state in states
    ]


    ttft_values = [
        request.ttft
        for request in requests
    ]

    itl_values = [
        interval
        for request in requests
        for interval in request.itl
    ]

    latency_values = [
        request.total_latency
        for request in requests
    ]

    output_tokens = sum(
        request.output_tokens
        for request in requests
    )

    # Benchmark wall time


    arrival_times = [
        state.arrival_time
        for state in states
    ]

    finish_times = [
        state.finish_time
        for state in states
        if state.finish_time is not None
    ]

    start_time = min(arrival_times)
    end_time = max(finish_times)

    wall_time = end_time - start_time

    if wall_time <= 0:
        raise ValueError(
            "Benchmark wall time must be positive"
        )

    # Throughput


    throughput = (
        output_tokens / wall_time
    )

    return AggregateMetrics(
        requests=requests,

        ttft_p50=percentile(
            ttft_values,
            0.50,
        ),

        ttft_p95=percentile(
            ttft_values,
            0.95,
        ),
        itl_p50=(
            percentile(itl_values, 0.50)
            if itl_values
            else None
        ),

        itl_p95=(
            percentile(itl_values, 0.95)
            if itl_values
            else None
        ),

        total_latency_p50=percentile(
            latency_values,
            0.50,
        ),

        total_latency_p95=percentile(
            latency_values,
            0.95,
        ),

        output_tokens=output_tokens,

        throughput=throughput,
    )