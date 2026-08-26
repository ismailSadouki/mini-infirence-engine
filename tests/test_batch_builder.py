import torch

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engine.batch_builder import build_decode_batch
from engine.types import RequestState


def make_request(
    request_id,
    prompt_len,
    generated_ids,
):
    request = RequestState(
        request_id=request_id,
        prompt_ids=list(range(prompt_len)),
    )

    request.generated_ids = generated_ids
    request.generated_count = len(generated_ids)
    request.current_pos = prompt_len + len(generated_ids)

    return request


def test_decode_batch_preserves_per_request_positions():

    r1 = make_request(
        "r1",
        prompt_len=4,
        generated_ids=[10, 20, 30],
    )

    r2 = make_request(
        "r2",
        prompt_len=8,
        generated_ids=[50],
    )

    r3 = make_request(
        "r3",
        prompt_len=3,
        generated_ids=[70, 80, 90, 100],
    )

    batch = build_decode_batch(
        requests=[r1, r2, r3],
        device=torch.device("cpu"),
    )

    assert batch.input_ids.shape == (3, 1)

    assert batch.input_ids.tolist() == [
        [30],
        [50],
        [100],
    ]

    assert batch.positions.tolist() == [
        7,
        9,
        7,
    ]