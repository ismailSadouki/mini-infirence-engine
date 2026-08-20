

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engine.generation import generate_no_cache
from engine.model_adapter import ModelAdapter
from engine.types import (
    GenerationRequest,
    SamplingConfig,
)



MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def test_single_greedy_generation():
    adapter = ModelAdapter(
        model_name=MODEL_NAME,
        device="cuda"
    )

    request = GenerationRequest(
        request_id="test-001",
        prompt_text="The capital of France is",
        max_new_tokens=10,
        sampling=SamplingConfig(
            greedy=True,
            temperature=0.0,
            seed=42,
        ),
    )

    output = generate_no_cache(adapter, request)
    print("output:", output)


    assert output.request_id == "test-001"

    assert len(output.output_ids) == 10


    assert isinstance(
        output.text,
        str,
    )

    assert output.ttft is not None
    assert len(output.itl) == 9
    assert output.finish_reason == "length"
    