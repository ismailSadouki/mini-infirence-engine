# Mini Inference Engine

A from-scratch LLM inference engine focused on the core mechanisms behind modern high-throughput serving systems. Implements prefill/decode execution, KV caching, continuous batching, paged KV memory, block-table management, and a reproducible inference benchmark harness in PyTorch. The project then compares the implementation against vLLM and evaluates BF16, GPTQ, and AWQ serving for an Algerian Darija DPO model.

## Why This Project?

Most LLM projects focus on **training models** or calling existing inference frameworks.

This project focuses on a different question:

> **What actually happens between an incoming generation request and the tokens produced by an LLM server?**

The engine is built incrementally from first principles before studying the corresponding production mechanisms in vLLM.

The project emphasizes **correctness, reproducibility, and measured performance**, rather than presenting unexplained throughput numbers.

---

## What It Implements

### Core Inference

* Model adapter and generation interface
* Explicit prefill/decode execution paths
* Pre-allocated KV cache
* Cached autoregressive generation
* Deterministic sampling
* Per-request generation state

### Scheduling

* Request queue
* Static batching baseline
* FCFS continuous batching
* Token-level request admission and eviction
* Ragged active batches

### Memory Management

* Contiguous KV-cache baseline
* Fixed-size KV block pool
* Free-list allocation
* Logical-to-physical block tables
* Paged KV attention
* KV memory utilization and fragmentation analysis

### Benchmarking

Measures:

* Time To First Token (TTFT)
* Inter-Token Latency (ITL)
* Time Per Output Token (TPOT)
* Throughput
* p50 latency
* p95 latency
* Error and timeout rates
* Latency-throughput curves

Every reported benchmark includes a complete workload specification.

---

## Architecture

```text
                    +-----------------+
                    |  Generation API |
                    +--------+--------+
                             |
                             v
                    +-----------------+
                    | Request Manager |
                    +--------+--------+
                             |
                    +--------v--------+
                    |    Scheduler    |
                    +--------+--------+
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
          +-------------+         +-------------+
          |   Prefill   |         |    Decode   |
          +------+------+         +------+------+
                 |                       |
                 +-----------+-----------+
                             |
                             v
                      +-------------+
                      |   KV Cache  |
                      +------+------+ 
                             |
                     +-------v--------+
                     |  Block Manager |
                     +-------+--------+
                             |
                     +-------v--------+
                     |  Block Table  |
                     +-------+--------+
                             |
                     +-------v--------+
                     | Paged Attention|
                     +----------------+
```

---

## Prefill and Decode

The engine explicitly separates the two phases of autoregressive generation.

For a prompt of length $T$, prefill processes:

$$
X \in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

and populates the KV cache.

During decode, each iteration processes the newly generated token:

$$
X_t \in \mathbb{R}^{B \times 1 \times d_{\text{model}}}
$$

while reusing previously computed keys and values.

For layer $l$, the cached tensors have the conceptual shape:

$$
K_l, V_l
\in
\mathbb{R}^{B \times H \times T \times d_h}
$$

where:

$$
d_{\text{model}} = H d_h
$$

This separation allows the project to measure prefill and decode behavior independently.

---

## KV Cache

Without caching, autoregressive generation repeatedly recomputes the keys and values for previous tokens.

With caching, previously computed:

$$
K_{1:t-1}, V_{1:t-1}
$$

are retained and only:

$$
K_t, V_t
$$

are computed for the new token.

The implementation validates this optimization through deterministic cached-vs-uncached generation tests.

The correctness requirement is:

$$
y^{\text{cached}}_{1:T}
=======================

y^{\text{uncached}}_{1:T}
$$

Where appropriate, the implementation also compares cached and uncached logits within numerical tolerance.

---

## Continuous Batching

The project first implements static batching as a baseline.

It then introduces token-level continuous batching:

```text
Waiting requests
       |
       v
   Scheduler
       |
       v
 Active batch
       |
       v
  Decode step
       |
   +---+---+
   |       |
   v       v
Finished  Active
   |       |
  Evict    |
   |       |
   +---+---+
       |
       v
Admit waiting requests
```

This makes it possible to measure how request-length variability affects GPU utilization, throughput, and latency.

---

## Paged KV Memory

The project implements a simplified paged KV-memory system inspired by the ideas behind PagedAttention.

Instead of requiring each sequence to occupy one contiguous KV allocation, KV memory is divided into fixed-size blocks.

A logical sequence:

```text
Logical blocks:
[0, 1, 2, 3]
```

can be mapped to arbitrary physical blocks:

```text
Physical blocks:
[7, 2, 13, 4]
```

through a block table:

$$
\text{BlockTable}_i[j] = p
$$

Attention then gathers the required keys and values through this logical-to-physical mapping.

The project measures the resulting difference in memory utilization and concurrency capacity compared with contiguous KV allocation.

---

# Benchmarking Philosophy

A central rule of the project is:

> **No benchmark number is reported without a workload specification.**

Inference performance depends on the complete experimental setup.

Each benchmark records:

```text
Hardware
Model
Model revision
Dtype / quantization
Prompt-length distribution
Output-length distribution
Concurrency
Request arrival pattern
Sampling configuration
Warmup requests
Measured requests
Software environment
Git commit
```

A throughput result should therefore be interpreted as:

$$
\text{Throughput}
=================

f(
\text{engine},
\text{hardware},
\text{model},
\text{workload}
)
$$

This prevents isolated numbers such as `500 tokens/sec` from being presented without the conditions that produced them.

---

# Experimental Workloads

The benchmark suite progressively introduces more realistic serving conditions.

### S1 — Single Request

```text
Concurrency: 1
Prompt length: fixed
Output length: fixed
Deterministic decoding
```

Used for:

* KV-cache correctness
* prefill/decode timing
* cached vs uncached comparison

### S2 — Variable-Length Requests

Introduces different prompt and generation lengths.

Used for:

* batching behavior
* scheduler behavior
* KV memory utilization

### S3 — Concurrent Requests

Introduces multiple simultaneous requests.

Used for:

* static vs continuous batching
* throughput
* TTFT
* ITL

### S4 — Saturation

Concurrency is progressively increased to identify the point where throughput stops scaling and latency begins increasing sharply.

---

# Metrics

## TTFT

Time To First Token:

$$
\text{TTFT}
===========

## t_{\text{first token}}

t_{\text{request arrival}}
$$

## ITL

For consecutive generated tokens:

$$
\text{ITL}_i
============

t_i - t_{i-1}
$$

## Throughput

Output-token throughput:

$$
\text{Throughput}
=================

\frac{
N_{\text{output tokens}}
}{
T_{\text{wall}}
}
$$

reported in output tokens/sec.

## Percentiles

Latency distributions are reported using:

$$
p50
===

\text{percentile}(x, 50)
$$

and:

$$
p95
===

\text{percentile}(x, 95)
$$

Mean latency may also be reported, but it is not treated as a sufficient characterization of serving performance.

---

# Mini Engine → vLLM

After implementing the core mechanisms from scratch, the project moves to vLLM.

The comparison is performed under controlled workloads:

```text
Same model
Same hardware
Same dtype
Same workload
        |
   +----+----+
   |         |
   v         v
Mini Engine  vLLM
   |         |
   +----+----+
        |
        v
Performance comparison
        |
        v
Gap decomposition
```

The objective is not simply to show that vLLM is faster.

The project investigates **why** the production system is faster, including areas such as:

* optimized GPU kernels
* scheduling
* KV-cache management
* CUDA graphs
* batching
* memory management
* framework/runtime overhead

Measured effects are distinguished from architectural inferences that cannot be isolated experimentally.

---

# Quantized Serving

The final stage applies the inference stack to an Algerian Darija DPO model.

The serving comparison evaluates:

* BF16
* GPTQ
* AWQ

against a common evaluation and serving workload.

The resulting trade-off is studied across:

$$
\text{quality}
\leftrightarrow
\text{memory}
\leftrightarrow
\text{latency}
$$

Quality is evaluated using AlgerianMMLU rather than assuming that quantization preserves model quality for free.

The selected model is exposed through a streaming API:

```text
vLLM
  |
  v
FastAPI
  |
  v
SSE streaming
  |
  v
Load testing
```

---

# Repository Structure

```text
mini-inference-engine/
├── engine/
│   ├── __init__.py
│   ├── model_adapter.py
│   ├── request.py
│   ├── generation.py
│   ├── kv_cache.py
│   ├── scheduler.py
│   ├── block_pool.py
│   ├── block_table.py
│   └── attention.py
│
├── bench/
│   ├── workloads.py
│   ├── generator.py
│   ├── metrics.py
│   ├── runner.py
│   └── plots.py
│
├── tests/
│   ├── test_generation.py
│   ├── test_kv_cache.py
│   ├── test_scheduler.py
│   ├── test_block_pool.py
│   ├── test_block_table.py
│   └── test_paged_attention.py
│
├── serve/
│   ├── fastapi_app.py
│   └── locustfile.py
│
├── configs/
│   ├── workload_single.yaml
│   ├── workload_batch.yaml
│   └── workload_saturation.yaml
│
├── experiments/
│   └── results/
│
├── docs/
│   ├── architecture.md
│   ├── environment.md
│   ├── workload_contract.md
│   ├── inference_glossary.md
│   └── kv_cache_bug_playbook.md
│
├── scripts/
│   ├── benchmark.py
│   └── plot_results.py
│
├── pyproject.toml
└── README.md
```

---

# Evidence

The project treats implementation and evidence as separate deliverables.

| Component           | Implementation           | Evidence                      |
| ------------------- | ------------------------ | ----------------------------- |
| KV cache            | Cache allocation/update  | Cached = uncached             |
| Prefill/decode      | Separate execution paths | Phase timing                  |
| Static batching     | Batch executor           | Baseline throughput           |
| Continuous batching | FCFS scheduler           | Throughput/latency comparison |
| Paged KV            | Block pool + block table | Memory utilization            |
| Paged attention     | KV gathering             | Correctness tests             |
| Benchmarking        | Metrics harness          | Reproducible curves           |
| vLLM                | Production deployment    | Load-test results             |
| Quantization        | GPTQ/AWQ                 | Quality/memory/latency table  |

---

# Project Outcome

The final artifact demonstrates an end-to-end understanding of modern LLM inference:

$$
\boxed{
\text{Request}
\rightarrow
\text{Scheduling}
\rightarrow
\text{Prefill}
\rightarrow
\text{KV Cache}
\rightarrow
\text{Decode}
\rightarrow
\text{Batching}
\rightarrow
\text{Paged Memory}
\rightarrow
\text{Serving}
}
$$

Rather than treating inference frameworks as black boxes, the project builds simplified versions of their fundamental mechanisms, validates them experimentally, and then connects those mechanisms to a production serving system.

## Status

```text
M0  Scope, workload contract, environment        ⬜
M1  KV cache and generation                      ⬜
M2  Continuous batching                          ⬜
M3  Paged KV memory                              ⬜
M4  Benchmark harness                            ⬜
M5  vLLM deployment and gap analysis             ⬜
M6  Quantized Darija model serving               ⬜
```
