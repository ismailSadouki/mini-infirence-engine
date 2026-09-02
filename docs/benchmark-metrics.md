# Benchmark Metrics & Measurement


## Goal

Implementation and validation benchmark metrics for the continuous batching engine:

* Time to First Token (TTFT)
* Inter-Token Latency (ITL)
* Total request latency
* Throughput
* p50 / p95 percentiles
* Warmup exclusion
* Repeated benchmark runs

The benchmark uses the existing continuous scheduler and ragged decode path.

---

## Benchmark Command

We ran the batch-8 workload with a maximum active-token budget of 32,768:

```bash
python bench/runner.py \
    configs/workloads/batch8.yaml \
    --max-total-active-tokens 32768
```

---

## Workload

```text
workload              : batch8
model                 : Qwen/Qwen2.5-0.5B-Instruct
dtype                 : bfloat16
device                : cuda
requests              : 8
concurrency           : 8
max active tokens     : 32768
warmup                : 2
repetitions           : 10
```

Each measured repetition processes 8 requests using continuous batching.

---

## Benchmark Procedure

For each run:

<!-- 1. Generate a fresh set of requests.
2. Convert workload requests into `RequestState`.
3. Create a fresh scheduler.
4. Admit requests through the continuous scheduler.
5. Prefill newly admitted requests.
6. Decode active requests using ragged batching.
7. Record timestamps for generated tokens.
8. Repeat for all requests.
9. Discard warmup results.
10. Aggregate metrics over the measured repetitions. -->

Two warmup runs were performed before the ten measured repetitions.

```text
Warmup runs: 2
Measured repetitions: 10
```

Warmup results were excluded from the final metrics.

---

## Metrics

### TTFT

Time from request arrival until the first generated token:

```text
TTFT = first_token_time - arrival_time
```

### ITL

Time between consecutive generated tokens:

```text
ITL_i = token_timestamp_i - token_timestamp_(i-1)
```

### Total Latency

Time from request arrival until generation finishes:

```text
total_latency = finish_time - arrival_time
```

### Throughput

Aggregate output-token throughput:

```text
throughput = total_output_tokens / wall_time
```

p50 and p95 are reported for TTFT, ITL, and total latency.

---



## Results

the benchmark produced consistent measurements.

### Repetition Wall Times

```text
Repetition 1  : 3.1091 s
Repetition 2  : 3.1546 s
Repetition 3  : 3.1684 s
Repetition 4  : 3.1329 s
Repetition 5  : 3.1282 s
Repetition 6  : 3.1303 s
Repetition 7  : 3.1382 s
Repetition 8  : 3.1490 s
Repetition 9  : 3.1500 s
Repetition 10 : 3.1308 s
```

All repetitions completed successfully:

```text
completed requests : 8
```

### Aggregate Results

```text
requests            : 80
output tokens       : 5120

TTFT p50            : 0.096641 s
TTFT p95            : 0.169214 s

ITL p50             : 0.047100 s
ITL p95             : 0.048007 s

total wall time     : 31.3916 s
throughput          : 163.10 tokens/s
```

---

## Interpretation

The benchmark is producing valid and stable measurements.

The decode latency was particularly consistent:

```text
ITL p50 : 47.10 ms
ITL p95 : 48.01 ms
```

The difference between p50 and p95 is small, indicating relatively stable per-token decode latency for this workload.

The aggregate throughput was:

```text
163.10 tokens/s
```

across the 8 concurrent requests.

The ten repetitions also showed relatively little variation, with wall times between approximately:

```text
3.11 s — 3.17 s
```
