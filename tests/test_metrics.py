from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from bench.metrics import (
    compute_request_metrics,
    aggregate_metrics,
)
from engine.types import RequestState


def make_state(
    request_id,
    arrival,
    token_times,
    finish,
):
    state = RequestState(
        request_id=request_id,
        prompt_ids=[0, 0],
        arrival_time=arrival,
    )

    state.generated_ids = list(
        range(len(token_times))
    )

    state.token_timestamps = token_times

    state.first_token_time = token_times[0]

    state.finish_time = finish

    state.finished = True

    return state


def test_ttft():

    state = make_state(
        request_id="req-0",
        arrival=0.0,
        token_times=[
            1.0,
            1.5,
            2.0,
        ],
        finish=2.5,
    )

    metrics = compute_request_metrics(state)

    assert metrics.ttft == 1.0


def test_itl():

    state = make_state(
        request_id="req-0",
        arrival=0.0,
        token_times=[
            1.0,
            1.5,
            2.5,
            4.0,
        ],
        finish=4.5,
    )

    metrics = compute_request_metrics(state)

    assert metrics.itl == [
        0.5,
        1.0,
        1.5,
    ]
def test_total_latency():

    state = make_state(
        request_id="req-0",
        arrival=2.0,
        token_times=[
            3.0,
            4.0,
        ],
        finish=7.0,
    )

    metrics = compute_request_metrics(state)

    assert metrics.total_latency == 5.0

def test_p50_p95():

    states = [
        make_state(
            "req-0",
            arrival=0.0,
            token_times=[1.0, 1.5],
            finish=2.0,
        ),
        make_state(
            "req-1",
            arrival=0.0,
            token_times=[2.0, 3.0],
            finish=4.0,
        ),
        make_state(
            "req-2",
            arrival=0.0,
            token_times=[4.0, 6.0],
            finish=8.0,
        ),
    ]

    metrics = aggregate_metrics(states)

    assert metrics.ttft_p50 == 2.0
    assert metrics.ttft_p95 == 3.8


def test_throughput():

    states = [
        make_state(
            "req-0",
            arrival=0.0,
            token_times=[
                1.0,
                1.5,
                2.0,
                2.5,
            ],
            finish=5.0,
        ),
        make_state(
            "req-1",
            arrival=0.0,
            token_times=[
                1.0,
                1.5,
                2.0,
                2.5,
            ],
            finish=5.0,
        ),
    ]

    metrics = aggregate_metrics(states)

    assert metrics.output_tokens == 8

    # 8 tokens / (5 - 0) seconds
    assert metrics.throughput == 1.6