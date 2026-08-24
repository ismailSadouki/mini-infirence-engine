# KV Cache Speedup

    ## Workload

    - Model: `Qwen/Qwen2.5-0.5B-Instruct`
    ...

    - Model: `Qwen/Qwen2.5-0.5B-Instruct`
    - Device: `cuda`
    - Dtype: `torch.bfloat16`
    - Prompt length: `512` tokens
    - Output length: `128` tokens
    - Decoding strategy: `greedy`
    - Temperature: `0.0`
    - Top-p: `1.0`
    - Seed: `42`
    - Warmup requests: `5`
    - Measured requests: `20`

    ## Latency

    | Metric | Uncached | Cached | Speedup |
    |---|---:|---:|---:|
    | Mean | 8457.85 ms | 1809.89 ms | 4.67x |
    | p50 | 8460.42 ms | 1817.14 ms | 4.66x |
    | p95 | 8563.06 ms | 1848.89 ms | 4.63x |

    ## Cached inference breakdown

    | Metric | Mean | p50 | p95 |
    |---|---:|---:|---:|
    | TTFT | 69.61 ms | 69.50 ms | 70.11 ms |
    | Prefill | 69.61 ms | 69.50 ms | 70.11 ms |
    | Decode total | 1740.28 ms | 1747.42 ms | 1779.37 ms |
    | ITL | 13.70 ms | 13.76 ms | 14.01 ms |
    | Total | 1809.89 ms | 1817.14 ms | 1848.89 ms |

    ## Interpretation

    The cached implementation avoids recomputing the full prefix during autoregressive
    decoding. The benchmark therefore compares full-sequence recomputation against
    prefill followed by single-token decode using the KV cache.

    The decode loop contains `127` cache-based decode steps because
    the first generated token is produced directly from the prefill logits.

    ## Correctness note


    Numerical correctness is validated separately from this performance benchmark.
    This benchmark measures the latency and throughput of the current KV-cache
    implementation and does not itself establish numerical equivalence with the
    reference implementation.

    ## Reproducibility

    - Git commit: `cd74f7c8485a8b3dc2bdca8ba63e06f1dd35cdd4`
    - Timestamp: `2026-08-24T19:03:43.949157`
    - Raw measurements: `results/kv_cache_speed_raw.csv`
    - Workload: `results/workload_single.yaml`
    