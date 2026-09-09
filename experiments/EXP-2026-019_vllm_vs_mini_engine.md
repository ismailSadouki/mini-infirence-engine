cd ~/mini-inference-engine
# EXP-2026-019 — Mini Engine vs vLLM on T4

## Objective

Comparing the mini inference engine against vLLM under a controlled workload and explaining the observed performance gap.

The comparison uses:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Hardware: NVIDIA Tesla T4
- Precision: BF16
- Prompt length: 128 tokens
- Output length: 64 tokens
- Arrival pattern: burst
- Sampling: deterministic / greedy
- Concurrency: 1, 8, 32

The mini-engine benchmark uses the existing workload specifications:

- `configs/workloads/batch1.yaml`
- `configs/workloads/batch8.yaml`
- `configs/workloads/batch32.yaml`

Each workload uses 2 warmup runs and 10 measured repetitions.

## Mini Engine Results

| Concurrency | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | Throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 61.08 ms | 91.94 ms | 27.15 ms | 40.82 ms | 33.31 tok/s |
| 8 | 275.74 ms | 468.25 ms | 102.39 ms | 142.38 ms | 71.29 tok/s |
| 32 | 1038.24 ms | 1862.93 ms | 341.99 ms | 461.24 ms | 84.40 tok/s |

Throughput improves with concurrency, but scaling begins to flatten:

- 1 → 8: 2.14×
- 8 → 32: 1.18×
- 1 → 32: 2.53×

The mini engine therefore benefits from batching, but its throughput saturates relatively early.

## vLLM Results

vLLM was evaluated on the same Tesla T4 using the same model, precision, prompt length, output length, and concurrency levels.

The client benchmark submitted requests concurrently using asynchronous requests.

| Concurrency | End-to-end p50 | End-to-end p95 | Throughput |
|---:|---:|---:|---:|
| 1 | 378.04 ms | 388.01 ms | 169.36 tok/s |
| 8 | 432.29 ms | 493.79 ms | 1150.91 tok/s |
| 32 | 736.72 ms | 879.87 ms | 2544.58 tok/s |

These latency measurements are end-to-end HTTP request latency and should not be compared directly with the mini engine's TTFT or ITL measurements.

## Throughput Comparison

| Concurrency | Mini Engine | vLLM | vLLM / Mini |
|---:|---:|---:|---:|
| 1 | 33.31 tok/s | 169.36 tok/s | 5.08× |
| 8 | 71.29 tok/s | 1150.91 tok/s | 16.15× |
| 32 | 84.40 tok/s | 2544.58 tok/s | 30.15× |

The performance gap grows substantially with concurrency.

At concurrency 1, vLLM achieves approximately 5.1× the mini-engine throughput.

At concurrency 32, vLLM achieves approximately 30.2× the mini-engine throughput.

This indicates that the primary difference is not simply single-request execution speed. vLLM is much more effective at converting increasing concurrency into GPU throughput.

## vLLM Runtime Evidence

During the benchmark environment inspection:

- `vllm:num_preemptions_total = 0`
- `vllm:prefix_cache_hits_total = 0`
- `vllm:prompt_tokens_total = 4160`
- `vllm:generation_tokens_total = 26624`

The generation counter is cumulative across requests made to the server and was therefore not used directly to calculate the controlled benchmark throughput.

The zero prefix-cache hit count is important: prefix caching did not contribute to the measured performance gap for this workload.

The zero preemption count also indicates that the observed benchmark did not experience KV-cache preemption.

## Gap Analysis

The measured throughput gap is consistent with several architectural differences.

### 1. Optimized GPU kernels

The mini engine relies heavily on ordinary PyTorch tensor operations.

vLLM uses optimized implementations and attention backends designed specifically for LLM inference.

This reduces kernel overhead and improves GPU utilization.

### 2. Optimized attention execution

The mini engine implements attention explicitly using tensor operations.

vLLM uses optimized attention implementations designed for efficient KV-cache access and batched inference.

This becomes increasingly important as concurrency grows.

### 3. More efficient scheduling

The mini engine implements continuous/ragged batching as an educational implementation.

vLLM has a production-oriented scheduler that dynamically schedules tokens under a token budget and coordinates scheduling with KV-cache management.

The increasing throughput gap at higher concurrency is consistent with this difference.

### 4. Lower runtime and Python overhead

The mini engine performs substantial orchestration through Python and general PyTorch operations.

vLLM is engineered to minimize CPU-side and kernel-launch overhead during serving.

This becomes increasingly significant when many requests are active simultaneously.

### 5. Production KV-cache management

The mini engine implements paged KV-cache concepts as part of the project.

vLLM has a mature block-based KV-cache manager and block pool integrated directly with scheduling.

The mini implementation demonstrates the concept, while vLLM is optimized around it as a production serving system.

### 6. Prefix caching was not responsible for this result

The workload did not contain repeated prefixes that produced cache hits.

The observed metric was:

`prefix_cache_hits_total = 0`

Therefore prefix caching cannot explain the measured throughput advantage in this experiment.

### 7. CUDA graphs and other production optimizations

vLLM contains additional GPU execution optimizations that are outside the scope of the mini engine.

These include CUDA graph execution and other production-level execution optimizations.

These were not separately isolated in this experiment, so they should be considered plausible contributors rather than individually measured causes.

## Interpretation

The experiment demonstrates that implementing the high-level architecture of an inference engine is not sufficient to reproduce the performance of a production serving system.

The mini engine implements:

- KV caching
- continuous batching
- ragged batches
- paged KV-cache management
- token-level scheduling

Nevertheless, vLLM achieves substantially higher throughput, especially as concurrency increases.

The key observation is the different scaling behavior.

The mini engine increases from 33.31 tok/s at concurrency 1 to 84.40 tok/s at concurrency 32.

vLLM increases from 169.36 tok/s to 2544.58 tok/s over the same concurrency range.

Therefore the main lesson is not simply that "vLLM is faster." The experiment shows that production inference performance depends heavily on the quality of the GPU execution path, scheduling, memory management, and runtime overhead.

## Limitations

The comparison has several limitations.

1. Mini-engine TTFT/ITL and vLLM HTTP latency are different measurements and are not directly comparable.
2. The benchmark was performed on a single Tesla T4 and should not be generalized to other GPUs.
3. Individual contributors such as attention kernels, CUDA graphs, and scheduler overhead were not isolated experimentally.
4. GPU memory measurements from the mini-engine benchmark are not considered reliable for this comparison because the reported values were unexpectedly identical to previous measurements on different hardware.
5. The vLLM benchmark used an HTTP client and therefore includes network/client/runtime overhead, although the server was local to the Colab environment.

## Conclusion

The experiment validates the mini engine's architectural direction while showing why a production inference engine such as vLLM is substantially faster.

The mini engine successfully reproduces several important serving concepts, but vLLM's optimized execution path and production-grade scheduling allow it to exploit increasing concurrency much more effectively.

The measured throughput advantage increased from approximately 5× at concurrency 1 to approximately 30× at concurrency 32.

The gap itself is therefore the main result of M5.4: it identifies the parts of an inference stack that matter after the basic architecture is already correct.