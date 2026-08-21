

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch

from engine.generation import generate_prefill_decode
from engine.types import (
    GenerationRequest,
    SamplingConfig,
)


class RecordingAdapter:

    def __init__(self):
        self.prefill_shapes = []
        self.decode_shapes = []

        self.device = "cpu"

    def tokenize(self, text):
        return [1, 2, 3, 4]

    def decode(self, token_ids):
        return "test"

    def forward_prefill(self, input_ids):
        self.prefill_shapes.append(
            tuple(input_ids.shape)
        )

        logits = torch.zeros(
            1,
            input_ids.shape[1],
            10,
        )

        return logits, None

    def forward_decode(
        self,
        last_token,
        cache,
        position,
    ):
        self.decode_shapes.append(
            tuple(last_token.shape)
        )

        logits = torch.zeros(
            1,
            1,
            10,
        )

        return logits, cache

    def sample_next_token(
        self,
        logits,
        config,
    ):
        return 1


def test_prefill_decode_structure():

    adapter = RecordingAdapter()

    request = GenerationRequest(
        request_id="m1-2-test",
        prompt_text="hello",
        max_new_tokens=5,
        sampling=SamplingConfig(
            greedy=True,
            temperature=0,
            seed=42,
        ),
    )

    output = generate_prefill_decode(
        adapter,
        request,
    )

    # -------------------------
    # Prefill
    # -------------------------

    assert adapter.prefill_shapes == [
        (1, 4)
    ]

    # -------------------------
    # Decode
    # -------------------------

    assert len(adapter.decode_shapes) == 4

    assert all(
        shape == (1, 1)
        for shape in adapter.decode_shapes
    )

    # -------------------------
    # Output
    # -------------------------

    assert len(output.output_ids) == 5

    assert output.prefill_tokens == 4

    assert output.decode_tokens == 4