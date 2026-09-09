# Why vLLM Wins — Mini Inference Engine Gap Analysis

## Summary

The mini inference engine and vLLM were benchmarked on the same Tesla T4 using:

- `Qwen/Qwen2.5-0.5B-Instruct`
- BF16
- 128-token prompts
- 64 generated tokens
- burst workload
- deterministic decoding
- concurrency 1, 8, and 32

The mini engine implements KV caching, continuous batching, ragged batches, paged KV-cache management, and token-level scheduling.

Despite these similarities, vLLM achieved substantially higher throughput.

| Concurrency | Mini Engine | vLLM | vLLM / Mini |
|---:|---:|---:|---:|
| 1 | 33.31 tok/s | 169.36 tok/s | 5.08× |
| 8 | 71.29 tok/s | 1150.91 tok/s | 16.15× |
| 32 | 84.40 tok/s | 2544.58 tok/s | 30.15× |

The most important result is not the absolute speedup. It is the scaling behavior.

The mini engine starts to flatten as concurrency increases, while vLLM continues to extract significantly more throughput from the GPU.

---

## Where the Gap Comes From

### 1. GPU kernels

The mini engine primarily uses standard PyTorch operations.

This is useful for understanding the computation, but it does not provide the same level of kernel optimization as a production inference engine.

vLLM uses optimized GPU execution paths designed specifically for LLM serving.

This reduces memory traffic, kernel-launch overhead, and unnecessary intermediate operations.


---

### 2. Attention implementation

Attention is particularly important during decode because the model repeatedly reads the KV cache.

The mini engine implements the attention computation explicitly.

vLLM uses optimized attention backends designed around efficient KV-cache access and batched execution.

As concurrency grows, this difference becomes increasingly important.


---

### 3. Scheduling

The mini engine implements continuous batching as an educational scheduler.

vLLM uses a production scheduler that operates around token budgets and integrates scheduling with KV-cache allocation and request state.

This allows vLLM to keep the GPU supplied with useful work as requests arrive and progress at different rates.

The throughput results strongly suggest that scheduling and execution efficiency become more important at higher concurrency.


---

### 4. Python and runtime overhead

The mini engine performs a significant amount of orchestration in Python.

That is appropriate for a learning implementation, but Python-side scheduling and repeated PyTorch operations introduce overhead.

A production engine minimizes this overhead and moves more of the critical execution path into optimized runtime and GPU operations.

This difference is expected to become more visible when serving many requests.


---

### 5. KV-cache management

The mini engine implements:

- fixed-size KV blocks
- a free-list
- logical-to-physical block mappings
- paged KV attention

These reproduce the important concepts behind production KV-cache systems.

However, vLLM integrates KV-cache management directly with scheduling and execution and has a substantially more mature implementation.

The important lesson is that having the same architectural idea does not imply having the same implementation efficiency.


---

### 6. Prefix caching

Prefix caching is often mentioned as a reason vLLM can be faster.

It is **not** the explanation for this experiment.

The vLLM metrics reported:

```text
prefix_cache_hits_total = 0
```

Therefore there were no prefix-cache hits contributing to the measured workload.

This is an important limitation because it prevents us from incorrectly attributing the throughput gap to prefix reuse.

---

### 7. Preemption

The vLLM metrics reported:

```
num_preemptions_total = 0
```

Therefore the benchmark did not experience KV-cache preemption.

Preemption therefore did not introduce an observed performance penalty in this run.


---

### 8. CUDA graphs and additional production optimizations

vLLM contains additional execution optimizations that are outside the scope of the mini engine, including CUDA graph execution and specialized GPU execution paths.

These optimizations can reduce CPU launch overhead and improve execution efficiency.

However, this experiment did not isolate CUDA graphs as an individual factor.

Therefore they should be treated as plausible contributors rather than measured causes.



---

### What the Experiment Actually Proves

The experiment directly measures a large throughput gap.

It also shows that the gap increases with concurrency:

```
Concurrency 1
vLLM ≈ 5.1× mini

Concurrency 8
vLLM ≈ 16.2× mini

Concurrency 32
vLLM ≈ 30.2× mini
```
This is strong evidence that the difference is not simply a constant single-request overhead.

The production engine is substantially better at exploiting concurrent work.

---

### What It Does Not Prove

This benchmark does not independently measure the contribution of:

- attention kernels
- CUDA graphs
- scheduler implementation
- Python overhead
- KV-cache implementation

Therefore it would be incorrect to claim that one specific optimization accounts for a particular percentage of the speedup.

The correct conclusion is that the measured gap is consistent with the combined effect of these architectural and implementation differences.


---

### The Main Engineering Lesson

Building a serving engine has several layers.

At the first layer, we need the correct architecture:

```
Requests
   ↓
Scheduler
   ↓
Prefill / Decode
   ↓
KV Cache
   ↓
Batching
   ↓
GPU
```

The mini engine now demonstrates these concepts.

But production performance requires another layer:

```
Correct architecture
        ↓
Efficient memory management
        ↓
Efficient scheduling
        ↓
Optimized kernels
        ↓
Low runtime overhead
        ↓
High GPU utilization
```

That second layer is where vLLM has a major advantage.

The purpose of the mini engine was therefore not to beat vLLM.

The purpose was to understand why it does not.


---

# Conclusion

The benchmark shows that the mini engine successfully reproduces several core ideas behind modern LLM serving, but reproducing the architecture alone is not enough to reproduce production performance.

On the same Tesla T4 and the same 0.5B model, vLLM achieved:

- 5.08× the mini-engine throughput at concurrency 1
- 16.15× at concurrency 8
- 30.15× at concurrency 32

The increasing gap demonstrates the importance of optimized GPU execution and production serving infrastructure.

Prefix caching did not explain the result because the benchmark recorded zero prefix-cache hits.

The most defensible explanation is therefore a combination of optimized attention/GPU kernels, more mature scheduling and KV-cache integration, lower runtime overhead, and other production execution optimizations.
