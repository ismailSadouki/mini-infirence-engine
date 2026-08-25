

from pathlib import Path
import sys

import torch



sys.path.append(str(Path(__file__).resolve().parent.parent))

from engine.continuous_runner import run_continuous
from engine.scheduler import ContinuousScheduler, SchedulerConfig
from engine.types import SamplingConfig

from scripts.requests import make_request



def test_scheduler_admits_fcfs():
    scheduler = ContinuousScheduler(
        SchedulerConfig(
            max_batch_size=2,
            max_total_active_tokens=100,
            max_waiting=10
        )
    )

    r1 = make_request("r1")
    r2 = make_request("r2")
    r3 = make_request("r3")

    scheduler.submit(r1)
    scheduler.submit(r2)
    scheduler.submit(r3)

    _, active = scheduler.step()


    

    
    assert [r.request_id for r in active] == [
        "r1",
        "r2"
    ]

    assert len(scheduler.waiting) == 1


def test_finished_request_is_evicted_and_replaced():
    scheduler = ContinuousScheduler(
        SchedulerConfig(
            max_batch_size=2,
            max_total_active_tokens=100,
            max_waiting=10
        )
    )

    r1 = make_request("r1")
    r2 = make_request("r2")
    r3 = make_request("r3")

    scheduler.submit(r1)
    scheduler.submit(r2)
    scheduler.submit(r3)

    scheduler.step()



    assert len(scheduler.active) == 2

    r1.finished = True
    r1.finished_reason = "eos"

    
    _, active = scheduler.step()



    assert [r.request_id for r in active] == [
        "r2",
        "r3",
    ]


def test_scheduler_respects_max_batch_size():
    scheduler = ContinuousScheduler(
        SchedulerConfig(
            max_batch_size=2,
            max_total_active_tokens=100,
            max_waiting=10,
        )
    )

    requests = [
        make_request("r1"),
        make_request("r2"),
        make_request("r3"),
    ]

    for request in requests:
        scheduler.submit(request)

    _, active = scheduler.step()

    assert len(active) == 2
    assert len(scheduler.waiting) == 1




def test_scheduler_respects_max_total_active_tokens():
    scheduler = ContinuousScheduler(
        SchedulerConfig(
            max_batch_size=10,
            max_total_active_tokens=10,
            max_waiting=10,
        )
    )

    r1 = make_request("r1", prompt_len=6)
    r2 = make_request("r2", prompt_len=5)

    scheduler.submit(r1)
    scheduler.submit(r2)

    _, active = scheduler.step()

    assert [r.request_id for r in active] == ["r1"]
    assert len(scheduler.waiting) == 1


def test_scheduler_allows_exact_total_active_token_limit():
    scheduler = ContinuousScheduler(
        SchedulerConfig(
            max_batch_size=10,
            max_total_active_tokens=10,
            max_waiting=10,
        )
    )

    r1 = make_request("r1", prompt_len=6)
    r2 = make_request("r2", prompt_len=4)

    scheduler.submit(r1)
    scheduler.submit(r2)

    _, active = scheduler.step()

    assert [r.request_id for r in active] == [
        "r1",
        "r2",
    ]

    assert len(scheduler.waiting) == 0







# Continuous runner tests
class FakeTokenizer:
    eos_token_id = 999


class FakeModel:
    class Config:
        num_hidden_layers = 1
        num_key_value_heads = 1
        hidden_size = 4
        num_attention_heads = 1

    config = Config()


class FakeAdapter:

    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()

        self.decode_calls = 0

    def forward_prefill_cached(self, input_ids, cache):
        return torch.zeros(
            1,
            input_ids.shape[1],
            10,
        )

    def forward_decode_cached(
        self,
        last_token,
        cache,
        position,
    ):
        self.decode_calls += 1

        return torch.zeros(
            1,
            1,
            10,
        )

    def sample_next_token(self, logits, config):
        if self.decode_calls == 1:
            return self.tokenizer.eos_token_id

        return 1

class NonEosFakeAdapter(FakeAdapter):

    def sample_next_token(self, logits, config):
        return 1

# request finishes → next request admitted
def test_finished_request_is_replaced_by_waiting_request():
    adapter = FakeAdapter()

    scheduler = ContinuousScheduler(
        SchedulerConfig(
            max_batch_size=2,
            max_total_active_tokens=100,
            max_waiting=10,
        )
    )

    requests = [
        make_request("r1"),
        make_request("r2"),
        make_request("r3"),
    ]

    for request in requests:
        scheduler.submit(request)

    completed = run_continuous(
        scheduler=scheduler,
        adapter=adapter,
        max_new_tokens=3,
        sampling_config=SamplingConfig(),
    )

    assert [r.request_id for r in completed] == [
        "r1",
        "r2",
        "r3",
    ]

    assert all(r.finished for r in requests)





def test_ttft_is_recorded_once():
    adapter = FakeAdapter()

    scheduler = ContinuousScheduler(
        SchedulerConfig(
            max_batch_size=1,
            max_total_active_tokens=100,
            max_waiting=10,
        )
    )

    request = make_request("r1")

    scheduler.submit(request)

    run_continuous(
        scheduler=scheduler,
        adapter=adapter,
        max_new_tokens=3,
        sampling_config=SamplingConfig(),
    )

    assert request.first_token_time is not None

    assert request.prefill_start_time is not None

    assert (
        request.first_token_time
        >= request.prefill_start_time
    )

def test_request_finishes_exactly_at_max_new_tokens():
    adapter = NonEosFakeAdapter()

    scheduler = ContinuousScheduler(
        SchedulerConfig(
            max_batch_size=2,
            max_total_active_tokens=100,
            max_waiting=10,
        )
    )

    request = make_request("r1")

    scheduler.submit(request)

    completed = run_continuous(
        scheduler=scheduler,
        adapter=adapter,
        max_new_tokens=3,
        sampling_config=SamplingConfig(),
    )

    assert len(completed) == 1

    assert request.finished
    assert request.finished_reason == "length"

    assert request.generated_count == 3
    assert len(request.generated_ids) == 3


def test_token_timestamps_are_recorded_for_each_generated_token():
    adapter = NonEosFakeAdapter()

    scheduler = ContinuousScheduler(
        SchedulerConfig(
            max_batch_size=1,
            max_total_active_tokens=100,
            max_waiting=10,
        )
    )

    request = make_request("r1")

    scheduler.submit(request)

    run_continuous(
        scheduler=scheduler,
        adapter=adapter,
        max_new_tokens=4,
        sampling_config=SamplingConfig(),
    )

    assert len(request.token_timestamps) == 4

    assert all(
        t2 >= t1
        for t1, t2 in zip(
            request.token_timestamps,
            request.token_timestamps[1:],
        )
    )