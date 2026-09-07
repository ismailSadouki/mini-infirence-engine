# vLLM Source Mapping

## Production vLLM vs Mini Inference Engine

**vLLM version:** 0.28.0
**Architecture studied:** vLLM V1

---

## 1. Scheduler Mapping

| Mini Engine         | vLLM V1                              |
| ------------------- | ------------------------------------ |
| Request queue       | `waiting` / `running` request queues |
| Scheduler           | `vllm/v1/core/sched/scheduler.py`    |
| Continuous batching | `Scheduler.schedule()`               |
| Token budget        | `max_num_scheduled_tokens`           |
| Running requests    | `self.running`                       |
| Waiting requests    | `self.waiting`                       |
| New requests        | `scheduled_new_reqs`                 |
| Resumed requests    | `scheduled_resumed_reqs`             |
| Preemption          | `preempted_reqs`                     |
| Scheduled tokens    | `num_scheduled_tokens`               |

### Important difference

My mini scheduler explicitly thinks in terms of **prefill vs decode**.

vLLM V1 does not require the scheduler to have separate scheduling phases.

Instead, each request tracks how many tokens have already been computed:

```text
num_computed_tokens
        ↓
num_tokens_with_spec
        ↓
schedule more tokens
```

This lets the same scheduler handle:

* prefill
* decode
* chunked prefill
* prefix caching
* speculative decoding

This is more general than my mini scheduler.

---

## 2. KV Cache Mapping

My mini engine:

```text
BlockPool
    ↓
BlockTable
    ↓
KV cache tensors
```

vLLM V1:

```text
KVCacheManager
      ↓
KVCacheCoordinator
      ↓
BlockPool
      ↓
GPU KV cache blocks
```

Relevant files:

```text
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/block_pool.py
```

### BlockPool

My `BlockPool` manages:

* allocation
* freeing
* reusable blocks

vLLM's `BlockPool` also manages these, but adds substantially more functionality.

It maintains:

```text
blocks
free_block_queue
cached_block_hash_to_block
cached_block_hashes_by_block
```

The free-block queue provides an eviction order for blocks.

---

## 3. Block Table Mapping

My mini engine uses a block table to map:

```text
logical token blocks
        ↓
physical KV blocks
```

vLLM similarly represents allocated KV blocks and returns them through:

```text
KVCacheBlocks
```

The scheduler does not directly manipulate the internal block structures.

Instead:

```text
Scheduler
    ↓
KVCacheManager
    ↓
KVCacheBlocks
    ↓
BlockPool
```

This separation is cleaner than exposing the block pool directly to the scheduler.

---

## 4. Prefix Caching

This is one of the biggest differences.

My mini engine:

```text
request
   ↓
block table
   ↓
physical KV blocks
```

vLLM additionally hashes completed blocks:

```text
tokens
   ↓
block hash
   ↓
cached_block_hash_to_block
   ↓
reusable KV block
```

`BlockPool` contains a `BlockHashToBlockMap` specifically for prefix caching.

Therefore vLLM can reuse KV blocks from previous requests with the same prefix.

---

## 5. Production Features Not Implemented in Mini Engine

vLLM adds many capabilities beyond my implementation:

* Prefix caching
* Request preemption
* KV-cache block eviction
* KV swapping / KV transfer
* Chunked prefill
* Speculative decoding
* CUDA Graph support
* Multi-GPU / tensor parallelism
* Pipeline parallelism
* Context parallelism
* Multimodal scheduling
* Structured-output constraints
* Remote KV connectors
* KV-cache metrics/events
* Hybrid attention / Mamba cache management
* Optimized attention backends and kernels

The mini engine intentionally implements only the core concepts needed to understand serving.

---

## 6. Attention Backend

The scheduler and block manager decide **which KV blocks belong to a request**.

The attention backend is responsible for efficiently using those blocks during attention.

Conceptually:

```text
Scheduler
    ↓
Block allocation
    ↓
Block table
    ↓
Attention backend
    ↓
Paged attention kernel
    ↓
KV blocks
```

The production implementation moves the expensive attention work into optimized GPU kernels rather than Python-level tensor operations.

---

## 7. Mini Engine vs vLLM

### Mini Engine

```text
Request
   ↓
Scheduler
   ↓
BlockTable
   ↓
BlockPool
   ↓
KV Cache
   ↓
Attention
```

Designed for:

* learning
* correctness
* understanding KV caching
* understanding continuous batching
* understanding paged KV memory

### vLLM

```text
Request
   ↓
V1 Scheduler
   ↓
KVCacheManager
   ↓
KVCacheCoordinator
   ↓
BlockPool
   ↓
Block Tables / KV blocks
   ↓
Attention Backend
   ↓
Optimized GPU kernels
```

Designed for:

* production serving
* high throughput
* efficient GPU utilization
* large numbers of concurrent requests
* advanced caching and scheduling policies

---

## 8. Conclusion

**What is the main difference between my mini inference engine and vLLM?**

> I implemented the core serving concepts myself: continuous batching, prefill/decode separation, KV caching, block pools, block tables, and paged KV memory. vLLM V1, has the same ideas but are generalized and integrated into a much more sophisticated scheduler and KV-cache manager. vLLM additionally handles preemption, prefix caching, speculative decoding, KV transfer, multi-GPU execution, CUDA graphs, and optimized attention kernels.




