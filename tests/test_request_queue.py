

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))




import time

from engine.request_queue import (
    RequestState,
    WaitingQueue,
    ActiveSet,
)


def make_request(
    request_id: str,
    prompt_len: int = 4,
) -> RequestState:
    return RequestState(
        request_id=request_id,
        prompt_ids=list(range(prompt_len)),
    )


def test_enqueue_request():
    queue = WaitingQueue(max_waiting=10)

    request = make_request("req-1")

    queue.enqueue(request)

    assert len(queue) == 1
    assert queue.peek().request_id == "req-1"



def test_enqueue_preserves_fifo_order():
    queue = WaitingQueue(max_waiting=10)

    queue.enqueue(make_request("req-1"))
    queue.enqueue(make_request("req-2"))
    queue.enqueue(make_request("req-3"))

    assert queue._queue.popleft().request_id == "req-1"
    assert queue._queue.popleft().request_id == "req-2"
    assert queue._queue.popleft().request_id == "req-3"


def test_admit_moves_request_from_waiting_to_active():
    waiting = WaitingQueue(max_waiting=10)
    active = ActiveSet(max_batch_size=2)

    request = make_request("req-1")

    waiting.enqueue(request)

    admitted = active.admit(request)

    assert admitted is True
    assert len(active) == 1
    assert active.get("req-1") is request



def test_finished_request_is_evicted():
    active = ActiveSet(max_batch_size=2)

    request = make_request("req-1")

    active.admit(request)

    request.finished = True

    removed = active.evict_finished()

    assert removed == [request]
    assert len(active) == 0



def test_current_position_is_tracked_per_request():
    request_1 = make_request("req-1", prompt_len=10)
    request_2 = make_request("req-2", prompt_len=20)

    request_1.position = 10
    request_2.position = 20

    assert request_1.position == 10
    assert request_2.position == 20





def test_max_batch_size_is_enforced():
    active = ActiveSet(max_batch_size=2)

    assert active.admit(make_request("req-1")) is True
    assert active.admit(make_request("req-2")) is True
    assert active.admit(make_request("req-3")) is False

    assert len(active) == 2



def test_finished_request_has_finish_reason():
    active = ActiveSet(max_batch_size=2)

    request = make_request("req-1")

    active.admit(request)

    request.finished = True
    request.finish_reason = "eos"

    removed = active.evict_finished()

    assert removed[0].finish_reason == "eos"



def test_different_requests_keep_independent_positions():
    request_1 = make_request("req-1", prompt_len=5)
    request_2 = make_request("req-2", prompt_len=8)

    request_1.position = 5
    request_2.position = 8

    request_1.position += 1

    assert request_1.position == 6
    assert request_2.position == 8