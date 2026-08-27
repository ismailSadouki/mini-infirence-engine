## Static vs Continuous Throughput Benchmark

### Run

```bash
python bench/static_vs_continuous.py
```

### What this command does

Runs a benchmark comparing **static batching** against **continuous batching** under the **same request workload**.

The purpose of this experiment is to measure how replacing finished requests dynamically affects:

* generation throughput
* TTFT
* ITL
* total request latency
* tail latency

The workload intentionally uses **different output lengths** because continuous batching provides the largest scheduling benefit when requests finish at different times.

### Workload

The benchmark uses six requests with the same prompt length but different generation lengths:

```text
Request    Prompt    Max new tokens
------------------------------------
r1         32        8
r2         32        32
r3         32        64
r4         32        16
r5         32        48
r6         32        8
```

```text
max_batch_size = 2
```

Total requested output:

```text
176 tokens
```

Both schedulers process exactly the same six requests.

### Static batching

Static batching divides the workload into fixed batches of at most:

```text
max_batch_size = 2
```

For example:

```text
Batch 1: r1, r2
Batch 2: r3, r4
Batch 3: r5, r6
```

Once a batch starts, its membership does not change.

If `r1` finishes after 8 tokens while `r2` still needs 32 tokens, the slot belonging to `r1` remains inactive until the batch completes.

This creates GPU under-utilization when requests have different output lengths.

The static benchmark uses the existing cached model execution:

```text
Prefill:
    forward_prefill_cached()

Decode:
    forward_decode_cached()
```

### Continuous batching

Continuous batching maintains:

```text
waiting requests
        ↓
    scheduler
        ↓
active requests
```

At every scheduling step:

1. Finished requests are evicted.
2. Waiting requests are admitted.
3. All currently active requests generate one token.
4. Newly available slots can immediately be assigned to waiting requests.

Therefore, when a short request finishes, another request can enter the active batch instead of leaving the slot idle.

### Metrics

The benchmark reports:

#### Throughput

```text
output tokens / elapsed time
```

Measured in:

```text
tokens/s
```

#### TTFT

Time from the request's prefill start until its first generated token.

Reported as:

```text
p50
p95
```

#### ITL

Inter-token latency between consecutive generated tokens.

Reported as:

```text
p50
p95
```

#### Total latency

Time from request arrival until request completion.

Reported as:

```text
p50
p95
```

### Current benchmark result

Representative runs produced results in approximately this range:

```text
Metric                            Static     Continuous
-------------------------------------------------------
Throughput (tok/s)                 46.35          86.78
TTFT p50 (ms)                      19.95          19.83
TTFT p95 (ms)                     427.68          20.73
ITL p50 (ms)                       18.39          21.04
ITL p95 (ms)                       36.67          21.63
Total latency p50 (ms)           2335.60        1330.48
Total latency p95 (ms)           3615.60        1960.53
```

### Interpretation

The main result is the throughput improvement:

```text
Static:       46.35 tokens/s
Continuous:   86.78 tokens/s
```

Continuous batching achieved approximately:

```text
1.88x
```

the static throughput in this run.

The most important scheduling effect is visible in tail latency.

TTFT p95 decreased from approximately:

```text
427.68 ms
```

to:

```text
20.73 ms
```

This happens because static batching can force requests to wait for previous batches to finish, while continuous batching can admit waiting requests as soon as active slots become available.

ITL is slightly higher under continuous batching:

```text
Static:       18.39 ms
Continuous:   21.04 ms
```

This is **not a contradiction**.

Continuous batching is not expected to make the computation of each individual decode step intrinsically faster. Its main benefit is **better utilization of the available batch slots over the lifetime of the workload**.

In other words:

```text
Static batching
    ↓
fixed batch membership
    ↓
finished requests leave idle slots
    ↓
lower utilization
    ↓
lower throughput


Continuous batching
    ↓
finished requests are removed
    ↓
waiting requests replace them
    ↓
more active requests over time
    ↓
higher utilization
    ↓
higher throughput
```

Therefore, the key M2.5 result is not that continuous batching must reduce ITL.

The key result is:

```text
continuous batching
        →
better utilization under variable output lengths
        →
higher workload-level throughput
        →
lower waiting/tail latency
```

### Why output-length variance matters

If every request generated exactly the same number of tokens, static batching would waste much less capacity.

For example:

```text
r1 → 64 tokens
r2 → 64 tokens
```

Both requests remain active for approximately the same number of decode steps.

But with:

```text
r1 → 8 tokens
r2 → 64 tokens
```

static batching eventually has an inactive slot after `r1` finishes.

Continuous batching can use that slot for another waiting request.

This is why M2.5 deliberately uses:

```text
8, 32, 64, 16, 48, 8
```

as the output-length distribution.
