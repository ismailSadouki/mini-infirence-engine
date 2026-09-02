from __future__ import annotations
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


from dataclasses import dataclass
from pathlib import Path
import random
import time

import yaml

from engine.types import (
    GenerationRequest,
    SamplingConfig,
)


@dataclass
class WorkloadSpec:
    """
    Reproducible benchmark workload specification.
    """

    name: str

    model: str
    dtype: str
    device: str

    num_requests: int
    concurrency: int

    arrival_pattern: str

    prompt_length: dict
    output_length: dict

    warmup: int
    repetitions: int

    seed: int


def load_workload(
    path: str | Path,
) -> WorkloadSpec:

    path = Path(path)

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return WorkloadSpec(
        name=config["name"],

        model=config["model"],
        dtype=config["dtype"],
        device=config["device"],

        num_requests=config["num_requests"],
        concurrency=config["concurrency"],

        arrival_pattern=config["arrival_pattern"],

        prompt_length=config["prompt_length"],
        output_length=config["output_length"],

        warmup=config["warmup"],
        repetitions=config["repetitions"],

        seed=config["seed"],
    )



def sample_distribution(
    distribution: dict,
    rng: random.Random,
) -> int:

    kind = distribution["distribution"]

    if kind == "fixed":

        return distribution["value"]

    if kind == "uniform":

        return rng.randint(
            distribution["min"],
            distribution["max"],
        )

    if kind == "empirical":

        values = distribution["values"]

        return rng.choice(values)

    raise ValueError(
        f"Unknown distribution: {kind}"
    )



def generate_requests(
    spec: WorkloadSpec,
) -> list[GenerationRequest]:

    rng = random.Random(spec.seed)

    requests = []

    arrival_time = 0.0

    for i in range(spec.num_requests):

        prompt_len = sample_distribution(
            spec.prompt_length,
            rng,
        )

        output_len = sample_distribution(
            spec.output_length,
            rng,
        )
        request = GenerationRequest(
            request_id=f"req-{i:05d}",

            # The actual benchmark can convert these
            # lengths into token IDs later.
            prompt_ids=[0] * prompt_len,

            max_new_tokens=output_len,

            sampling=SamplingConfig(
                seed=spec.seed + i,
                greedy=True,
            ),

            arrival_time=arrival_time,
        )

        requests.append(request)

    return requests