from __future__ import annotations

import statistics
import time
from pathlib import Path
import sys

import csv
import json
import subprocess
from datetime import datetime
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from engine.kv_cache import KVCache
from engine.model_adapter import ModelAdapter




# workload
WORKLOAD_CONFIG = "configs/workload_single.yaml"


def load_workload(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)



def create_cache(
        model,
        adapter,
        max_seq_len: int,
# ) -> EngineKVCache:
) -> KVCache:
    """
    Create a fresh KV Cache

    shape per layer:
    [1, H_kv, max_seq_leng, head_dim]
    """

    kv_cache = KVCache(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        max_seq_len=max_seq_len,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        dtype=model.dtype,
        device=model.device,
    )

    # return EngineKVCache(kv_cache)
    return kv_cache



# GREEDY TOKEN SELECTION


        
# TIMING HELPERS
def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def percentile_ms(values: list[float], percentile: float) -> float:
    return (
        statistics.quantiles(
            values,
            n=100,
            method="inclusive",
        )[int(percentile) - 1]
        * 1000
    )


def mean_ms(values: list[float]) -> float:
    return statistics.mean(values) * 1000


def median_ms(values: list[float]) -> float:
    return statistics.median(values) * 1000

# INPUT
def make_prompt(
        adapter: ModelAdapter,
        target_tokens: int
) -> torch.Tensor:
    """
    Construct a deterministic prompt with exactly target_tokens tokens.
    """
    text = (
        "The transformer architecture is a neural network architecture "
        "based on self attention and feed forward networks. "
        "During autoregressive inference, previously computed key and "
        "value states can be reused through a KV cache. "
    )

    token_ids = adapter.tokenize(text)

    # Repeat until sufficientlly long
    while len(token_ids) < target_tokens:
        token_ids += token_ids

    token_ids = token_ids[:target_tokens]

    return torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=adapter.device
    )


# UNCACHED GENERATION
@torch.inference_mode()
def run_uncached(
    adapter: ModelAdapter,
    input_ids: torch.Tensor,
    max_new_tokens: int,
): 
    current_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        logits = adapter.forward_no_cache(
            current_ids
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )


        current_ids = torch.cat(
            [current_ids, next_token],
            dim=1
        )



# CACHED GENERATION
@torch.inference_mode()
def run_cached(
    adapter: ModelAdapter,
    input_ids: torch.Tensor,
    max_new_tokens: int,
):
    prompt_length = input_ids.shape[1]

    cache = create_cache(
        model=adapter.model,
        adapter=adapter,
        max_seq_len=prompt_length + max_new_tokens,
    )

    # PREFILL / TTFT

    sync(adapter.device)

    prefill_start = time.perf_counter()

    logits = adapter.forward_prefill_cached(
        input_ids=input_ids,
        cache=cache,
    )

    sync(adapter.device)

    prefill_time = time.perf_counter() - prefill_start

    # First generated token
    token_id = torch.argmax(
        logits[:, -1, :],
        dim=-1,
    ).item()

    # TTFT = prompt processing + first token availability
    ttft = prefill_time

    # DECODE
    current_position = prompt_length

    decode_times = []

    generated_tokens = 1

    for _ in range(max_new_tokens - 1):

        next_token = torch.tensor(
            [[token_id]],
            dtype=torch.long,
            device=adapter.device,
        )

        sync(adapter.device)

        decode_start = time.perf_counter()

        logits = adapter.forward_decode_cached(
            last_token=next_token,
            cache=cache,
            position=current_position,
        )

        sync(adapter.device)

        decode_time = time.perf_counter() - decode_start

        decode_times.append(decode_time)

        token_id = torch.argmax(
            logits[:, -1, :],
            dim=-1,
        ).item()

        current_position += 1
        generated_tokens += 1

    decode_total = sum(decode_times)

    total_time = prefill_time + decode_total

    itl = (
        decode_total / len(decode_times)
        if decode_times
        else 0.0
    )

    generation_throughput = (
        generated_tokens / total_time
        if total_time > 0
        else 0.0
    )

    return {
        "ttft": ttft,
        "prefill": prefill_time,
        "decode_total": decode_total,
        "itl": itl,
        "total": total_time,
        "generation_throughput": generation_throughput,
        "generated_tokens": generated_tokens,
    }

# MAIN BENCHMARK
def main():

    # LOAD WORKLOAD
    workload_config = load_workload(
        WORKLOAD_CONFIG
    )

    workload = workload_config["workload"]
    hardware = workload_config["hardware"]
    model_config = workload_config["model"]
    request = workload_config["request"]
    decoding = workload_config["decoding"]
    benchmark = workload_config["benchmark"]
    decoding_strategy = decoding["strategy"]
    temperature = decoding["temperature"]
    top_p = decoding["top_p"]
    seed = decoding["seed"]

    model_name = model_config["name"]
    device = hardware["device"]

    prompt_length = request["prompt"]["tokens"]
    max_new_tokens = request["output"]["new_tokens"]

    warmup_runs = benchmark["warmup_requests"]
    measurement_runs = benchmark["measured_requests"]




    print("=" * 70)
    print("KV CACHE SPEED BENCHMARK")
    print("=" * 70)

    print(f"Model:          {model_name}")
    print(f"Device:         {device}")
    print(f"Prompt length:  {prompt_length} tokens")
    print(f"Output length:  {max_new_tokens} tokens")
    print(f"Warmup runs:    {warmup_runs}")
    print(f"Measured runs:  {measurement_runs}")
    print(f"Decoding:       {decoding_strategy}")
    print(f"Temperature:    {temperature}")
    print(f"Top-p:          {top_p}")
    print(f"Seed:            {seed}")


    dtype_name = model_config["dtype"]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    dtype = dtype_map[dtype_name]

    adapter = ModelAdapter(
        model_name = model_name,
        device=device,
        dtype=dtype,
    )


    adapter.model.eval()

    input_ids = make_prompt(
        adapter,
        prompt_length
    )

    
    print()
    print(f"Actual prompt length: {input_ids.shape[1]}")

    # WARMUP
    print()
    print("Warming up...")


    for _ in range(warmup_runs):
        run_uncached(
            adapter,
            input_ids,
            max_new_tokens
        )

        run_cached(
            adapter,
            input_ids,
            max_new_tokens
        )

    # UNCACHED
    uncached_times = []

    for i in range(measurement_runs):
        sync(adapter.device)
        start = time.perf_counter()

        run_uncached(
            adapter,
            input_ids,
            max_new_tokens
        )

        sync(adapter.device)
        elapsed = time.perf_counter() - start

        uncached_times.append(elapsed)

        print(
            f"uncached run {i + 1}: "
            f"{elapsed * 1000:.2f} ms"
        )

    # CACHED
    results = []

    for i in range(measurement_runs):

        result = run_cached(
            adapter,
            input_ids,
            max_new_tokens,
        )

        result["request_id"] = i

        results.append(result)

        print(
            f"request {i + 1:02d}: "
            f"TTFT={result['ttft'] * 1000:.2f} ms | "
            f"prefill={result['prefill'] * 1000:.2f} ms | "
            f"ITL={result['itl'] * 1000:.2f} ms | "
            f"total={result['total'] * 1000:.2f} ms | "
            f"tok/s={result['generation_throughput']:.2f}"
        )
    # RESULTS
    ttft_values = [
        r["ttft"]
        for r in results
    ]

    prefill_values = [
        r["prefill"]
        for r in results
    ]

    decode_values = [
        r["decode_total"]
        for r in results
    ]

    itl_values = [
        r["itl"]
        for r in results
    ]

    total_values = [
        r["total"]
        for r in results
    ]

    throughput_values = [
        r["generation_throughput"]
        for r in results
    ]

    print()
    print("=" * 70)
    print("CACHED RESULTS")
    print("=" * 70)

    print()
    print("TTFT")
    print(f"  mean: {mean_ms(ttft_values):.2f} ms")
    print(f"  p50:  {median_ms(ttft_values):.2f} ms")
    print(f"  p95:  {percentile_ms(ttft_values, 95):.2f} ms")

    print()
    print("PREFILL")
    print(f"  mean: {mean_ms(prefill_values):.2f} ms")
    print(f"  p50:  {median_ms(prefill_values):.2f} ms")
    print(f"  p95:  {percentile_ms(prefill_values, 95):.2f} ms")

    print()
    print("DECODE")
    print(f"  mean: {mean_ms(decode_values):.2f} ms")
    print(f"  p50:  {median_ms(decode_values):.2f} ms")
    print(f"  p95:  {percentile_ms(decode_values, 95):.2f} ms")

    print()
    print("ITL")
    print(f"  mean: {mean_ms(itl_values):.2f} ms")
    print(f"  p50:  {median_ms(itl_values):.2f} ms")
    print(f"  p95:  {percentile_ms(itl_values, 95):.2f} ms")

    print()
    print("TOTAL GENERATION LATENCY")
    print(f"  mean: {mean_ms(total_values):.2f} ms")
    print(f"  p50:  {median_ms(total_values):.2f} ms")
    print(f"  p95:  {percentile_ms(total_values, 95):.2f} ms")

    print()
    print("GENERATION THROUGHPUT")
    print(f"  mean: {statistics.mean(throughput_values):.2f} tokens/s")
    print(f"  p50:  {statistics.median(throughput_values):.2f} tokens/s")





    uncached_ms = {
        "mean": mean_ms(uncached_times),
        "p50": median_ms(uncached_times),
        "p95": percentile_ms(uncached_times, 95),
    }

    cached_ms = {
        "mean": mean_ms(total_values),
        "p50": median_ms(total_values),
        "p95": percentile_ms(total_values, 95),
    }

    speedup = {
        "mean": uncached_ms["mean"] / cached_ms["mean"],
        "p50": uncached_ms["p50"] / cached_ms["p50"],
        "p95": uncached_ms["p95"] / cached_ms["p95"],
    }
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        git_commit = "unknown"

    print(f"Git commit: {git_commit}")

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    report_path = report_dir / "kv_cache_speedup.md"

    report = f"""# KV Cache Speedup

    ## Workload

    - Model: `{model_name}`
    ...

    - Model: `{model_name}`
    - Device: `{device}`
    - Dtype: `{adapter.dtype}`
    - Prompt length: `{input_ids.shape[1]}` tokens
    - Output length: `{max_new_tokens}` tokens
    - Decoding strategy: `{decoding_strategy}`
    - Temperature: `{temperature}`
    - Top-p: `{top_p}`
    - Seed: `{seed}`
    - Warmup requests: `{warmup_runs}`
    - Measured requests: `{measurement_runs}`

    ## Latency

    | Metric | Uncached | Cached | Speedup |
    |---|---:|---:|---:|
    | Mean | {uncached_ms["mean"]:.2f} ms | {cached_ms["mean"]:.2f} ms | {speedup["mean"]:.2f}x |
    | p50 | {uncached_ms["p50"]:.2f} ms | {cached_ms["p50"]:.2f} ms | {speedup["p50"]:.2f}x |
    | p95 | {uncached_ms["p95"]:.2f} ms | {cached_ms["p95"]:.2f} ms | {speedup["p95"]:.2f}x |

    ## Cached inference breakdown

    | Metric | Mean | p50 | p95 |
    |---|---:|---:|---:|
    | TTFT | {mean_ms(ttft_values):.2f} ms | {median_ms(ttft_values):.2f} ms | {percentile_ms(ttft_values, 95):.2f} ms |
    | Prefill | {mean_ms(prefill_values):.2f} ms | {median_ms(prefill_values):.2f} ms | {percentile_ms(prefill_values, 95):.2f} ms |
    | Decode total | {mean_ms(decode_values):.2f} ms | {median_ms(decode_values):.2f} ms | {percentile_ms(decode_values, 95):.2f} ms |
    | ITL | {mean_ms(itl_values):.2f} ms | {median_ms(itl_values):.2f} ms | {percentile_ms(itl_values, 95):.2f} ms |
    | Total | {mean_ms(total_values):.2f} ms | {median_ms(total_values):.2f} ms | {percentile_ms(total_values, 95):.2f} ms |

    

    ## Interpretation

    The cached implementation avoids recomputing the full prefix during autoregressive
    decoding. The benchmark therefore compares full-sequence recomputation against
    prefill followed by single-token decode using the KV cache.

    The decode loop contains `{max_new_tokens - 1}` cache-based decode steps because
    the first generated token is produced directly from the prefill logits.

    ## Correctness note


    Numerical correctness is validated separately from this performance benchmark.
    This benchmark measures the latency and throughput of the current KV-cache
    implementation and does not itself establish numerical equivalence with the
    reference implementation.

    ## Reproducibility

    - Git commit: `{git_commit}`
    - Timestamp: `{datetime.now().isoformat()}`
    - Raw measurements: `results/kv_cache_speed_raw.csv`
    - Workload: `results/workload_single.yaml`
    """

    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report saved to: {report_path}")




    # -----------------
    print()
    print("=" * 70)
    print("KV CACHE SPEEDUP")
    print("=" * 70)

    print(f"Uncached mean: {uncached_ms['mean']:.2f} ms")
    print(f"Cached mean:   {cached_ms['mean']:.2f} ms")
    print(f"Speedup:       {speedup['mean']:.2f}x")

    print()
    print(f"Uncached p50:  {uncached_ms['p50']:.2f} ms")
    print(f"Cached p50:    {cached_ms['p50']:.2f} ms")
    print(f"Speedup p50:   {speedup['p50']:.2f}x")

    print()
    print(f"Uncached p95:  {uncached_ms['p95']:.2f} ms")
    print(f"Cached p95:    {cached_ms['p95']:.2f} ms")
    print(f"Speedup p95:   {speedup['p95']:.2f}x")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    raw_path = results_dir / "kv_cache_speed_raw.csv"

    with open(raw_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_id",
                "ttft",
                "prefill",
                "decode_total",
                "itl",
                "total",
                "generation_throughput",
                "generated_tokens",
                "uncached_total",
            ],
        )

        writer.writeheader()

        for i, result in enumerate(results):
            row = dict(result)
            row["uncached_total"] = uncached_times[i]

            writer.writerow(row)
    print()
    print(f"Raw measurements saved to: {raw_path}")
    workload_output = results_dir / "workload_single.yaml"

    with open(workload_output, "w") as f:
        yaml.safe_dump(
            workload_config,
            f,
            sort_keys=False,
        )

    print(f"Workload configuration saved to: {workload_output}")



    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "model": model_name,
        "device": device,
        "dtype": str(adapter.dtype),
        "prompt_tokens": input_ids.shape[1],
        "output_tokens": max_new_tokens,
        "decoding_strategy": decoding_strategy,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }

    metadata_path = results_dir / "kv_cache_speed_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()