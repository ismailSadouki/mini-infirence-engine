from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch

from engine.types import RequestState, SamplingConfig
from engine.static_runner import run_static_batch


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

    def to_tensor(self, token_ids):
        return torch.tensor(
            [token_ids],
            dtype=torch.long,
        )

    def forward_prefill_cached(
        self,
        input_ids,
        cache,
    ):
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
        return torch.zeros(
            1,
            1,
            10,
        )

    def sample_next_token(
        self,
        logits,
        config,
    ):
        return 1


def make_request(
    request_id,
    prompt_len=4,
):
    return RequestState(
        request_id=request_id,
        prompt_ids=list(range(prompt_len)),
    )


def test_static_batch_processes_requests():
    adapter = FakeAdapter()

    requests = [
        make_request("req-1"),
        make_request("req-2"),
    ]

    results = run_static_batch(
        requests=requests,
        batch_size=2,
        adapter=adapter,
        max_new_tokens=3,
        sampling_config=SamplingConfig(),
    )

    assert len(results) == 2

    assert requests[0].finished
    assert requests[1].finished

    assert results[0].generated_tokens == 3
    assert results[1].generated_tokens == 3


def test_static_batch_respects_batch_size():
    adapter = FakeAdapter()

    requests = [
        make_request("req-1"),
        make_request("req-2"),
        make_request("req-3"),
        make_request("req-4"),
        make_request("req-5"),
    ]

    results = run_static_batch(
        requests=requests,
        batch_size=2,
        adapter=adapter,
        max_new_tokens=2,
        sampling_config=SamplingConfig(),
    )

    assert len(results) == 5

    assert all(
        request.finished
        for request in requests
    )