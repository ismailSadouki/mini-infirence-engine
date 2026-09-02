from pathlib import Path
import random

import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from bench.workload import (
    WorkloadSpec,
    generate_requests,
    load_workload,
    sample_distribution,
)


# Helpers

def make_test_spec(num_requests=8):
    return WorkloadSpec(
        name="test",

        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        device="cuda",

        num_requests=num_requests,
        concurrency=num_requests,

        arrival_pattern="burst",

        prompt_length={
            "distribution": "fixed",
            "value": 128,
        },

        output_length={
            "distribution": "fixed",
            "value": 64,
        },

        warmup=1,
        repetitions=2,

        seed=2026,
    )


# Distribution tests

def test_fixed_distribution():

    distribution = {
        "distribution": "fixed",
        "value": 128,
    }

    rng = random.Random(2026)

    values = [
        sample_distribution(distribution, rng)
        for _ in range(100)
    ]

    assert values == [128] * 100


def test_uniform_distribution():

    distribution = {
        "distribution": "uniform",
        "min": 16,
        "max": 64,
    }

    rng = random.Random(2026)

    values = [
        sample_distribution(distribution, rng)
        for _ in range(1000)
    ]

    assert all(
        16 <= value <= 64
        for value in values
    )


def test_empirical_distribution():

    distribution = {
        "distribution": "empirical",
        "values": [32, 64, 128],
    }

    rng = random.Random(2026)

    values = [
        sample_distribution(distribution, rng)
        for _ in range(1000)
    ]

    assert all(
        value in [32, 64, 128]
        for value in values
    )


def test_unknown_distribution():

    distribution = {
        "distribution": "unknown",
    }

    rng = random.Random(2026)

    with pytest.raises(ValueError):
        sample_distribution(distribution, rng)


# Request generation

def test_request_count():

    spec = make_test_spec(
        num_requests=8
    )

    requests = generate_requests(spec)

    assert len(requests) == 8


def test_request_ids():

    spec = make_test_spec(
        num_requests=4
    )

    requests = generate_requests(spec)

    assert [
        request.request_id
        for request in requests
    ] == [
        "req-00000",
        "req-00001",
        "req-00002",
        "req-00003",
    ]


def test_fixed_lengths():

    spec = make_test_spec(
        num_requests=8
    )

    requests = generate_requests(spec)

    for request in requests:

        assert request.prompt_ids is not None
        assert len(request.prompt_ids) == 128

        assert request.max_new_tokens == 64


def test_request_arrival_times():

    spec = make_test_spec(
        num_requests=8
    )

    requests = generate_requests(spec)

    assert all(
        request.arrival_time == 0.0
        for request in requests
    )


# Determinism

def test_generation_is_deterministic():

    spec = make_test_spec(
        num_requests=100
    )

    requests_a = generate_requests(spec)
    requests_b = generate_requests(spec)

    data_a = [
        (
            request.request_id,
            request.prompt_ids,
            request.max_new_tokens,
            request.arrival_time,
        )
        for request in requests_a
    ]

    data_b = [
        (
            request.request_id,
            request.prompt_ids,
            request.max_new_tokens,
            request.arrival_time,
        )
        for request in requests_b
    ]

    assert data_a == data_b


def test_different_seed_changes_workload():

    spec_a = WorkloadSpec(
        name="test",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        device="cuda",

        num_requests=100,
        concurrency=100,

        arrival_pattern="burst",

        prompt_length={
            "distribution": "uniform",
            "min": 8,
            "max": 128,
        },

        output_length={
            "distribution": "uniform",
            "min": 16,
            "max": 128,
        },

        warmup=1,
        repetitions=2,

        seed=2026,
    )

    spec_b = WorkloadSpec(
        name="test",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        device="cuda",

        num_requests=100,
        concurrency=100,

        arrival_pattern="burst",

        prompt_length={
            "distribution": "uniform",
            "min": 8,
            "max": 128,
        },

        output_length={
            "distribution": "uniform",
            "min": 16,
            "max": 128,
        },

        warmup=1,
        repetitions=2,

        seed=9999,
    )

    requests_a = generate_requests(spec_a)
    requests_b = generate_requests(spec_b)

    lengths_a = [
        (
            len(request.prompt_ids),
            request.max_new_tokens,
        )
        for request in requests_a
    ]

    lengths_b = [
        (
            len(request.prompt_ids),
            request.max_new_tokens,
        )
        for request in requests_b
    ]

    assert lengths_a != lengths_b


# Distribution properties

def test_uniform_workload_stays_within_bounds():

    spec = WorkloadSpec(
        name="uniform-test",

        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        device="cuda",

        num_requests=1000,
        concurrency=32,

        arrival_pattern="burst",

        prompt_length={
            "distribution": "uniform",
            "min": 32,
            "max": 256,
        },

        output_length={
            "distribution": "uniform",
            "min": 16,
            "max": 128,
        },

        warmup=1,
        repetitions=2,

        seed=2026,
    )

    requests = generate_requests(spec)

    prompt_lengths = [
        len(request.prompt_ids)
        for request in requests
    ]

    output_lengths = [
        request.max_new_tokens
        for request in requests
    ]

    assert all(
        32 <= length <= 256
        for length in prompt_lengths
    )

    assert all(
        16 <= length <= 128
        for length in output_lengths
    )


def test_empirical_workload_uses_only_configured_values():

    prompt_values = [
        32,
        64,
        128,
        256,
    ]

    output_values = [
        16,
        32,
        64,
    ]

    spec = WorkloadSpec(
        name="empirical-test",

        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        device="cuda",

        num_requests=1000,
        concurrency=32,

        arrival_pattern="burst",

        prompt_length={
            "distribution": "empirical",
            "values": prompt_values,
        },

        output_length={
            "distribution": "empirical",
            "values": output_values,
        },

        warmup=1,
        repetitions=2,

        seed=2026,
    )

    requests = generate_requests(spec)

    prompt_lengths = {
        len(request.prompt_ids)
        for request in requests
    }

    output_lengths = {
        request.max_new_tokens
        for request in requests
    }

    assert prompt_lengths <= set(prompt_values)
    assert output_lengths <= set(output_values)


# ============================================================
# YAML loading
# ============================================================

def test_load_workload(tmp_path):

    workload_file = tmp_path / "test.yaml"

    workload_file.write_text(
        """
name: test

model: Qwen/Qwen2.5-0.5B-Instruct
dtype: bfloat16
device: cuda

num_requests: 8
concurrency: 8

arrival_pattern: burst

prompt_length:
  distribution: fixed
  value: 128

output_length:
  distribution: fixed
  value: 64

warmup: 2
repetitions: 10

seed: 2026
"""
    )

    spec = load_workload(workload_file)

    assert spec.name == "test"
    assert spec.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert spec.dtype == "bfloat16"
    assert spec.device == "cuda"

    assert spec.num_requests == 8
    assert spec.concurrency == 8

    assert spec.arrival_pattern == "burst"

    assert spec.prompt_length["distribution"] == "fixed"
    assert spec.prompt_length["value"] == 128

    assert spec.output_length["distribution"] == "fixed"
    assert spec.output_length["value"] == 64

    assert spec.warmup == 2
    assert spec.repetitions == 10
    assert spec.seed == 2026


def test_loaded_workload_generates_requests(
    tmp_path,
):

    workload_file = tmp_path / "test.yaml"

    workload_file.write_text(
        """
name: test

model: Qwen/Qwen2.5-0.5B-Instruct
dtype: bfloat16
device: cuda

num_requests: 16
concurrency: 8

arrival_pattern: burst

prompt_length:
  distribution: uniform
  min: 32
  max: 128

output_length:
  distribution: uniform
  min: 16
  max: 64

warmup: 1
repetitions: 5

seed: 2026
"""
    )

    spec = load_workload(workload_file)

    requests = generate_requests(spec)

    assert len(requests) == 16

    for request in requests:
        assert 32 <= len(request.prompt_ids) <= 128
        assert 16 <= request.max_new_tokens <= 64
        assert request.arrival_time == 0.0