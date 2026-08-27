# Mini Inference Engine

A from-scratch LLM inference engine focused on the core mechanisms behind modern high-throughput serving systems. Implements prefill/decode execution, KV caching, continuous batching, paged KV memory, block-table management, and a reproducible inference benchmark harness in PyTorch. The project then compares the implementation against vLLM and evaluates BF16, GPTQ, and AWQ serving for an Algerian Darija DPO model.

## Why This Project?

Most LLM projects focus on **training models** or using existing inference frameworks.

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

# Architecture

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

# Prefill and Decode

The engine explicitly separates the two phases of autoregressive generation.

For a prompt of length $T$, prefill processes:

$$
X \in R^{B \times T \times d_{model}}
$$

and populates the KV cache.

During decode, each iteration processes the newly generated token:

$$
X_t \in R^{B \times 1 \times d_{model}}
$$

while reusing previously computed keys and values.

For layer $l$, the cached tensors have the conceptual shape:

$$
K_l, V_l \in R^{B \times H \times T \times d_h}
$$

where:

$$
d_{model} = H d_h
$$

This separation allows the project to measure prefill and decode behavior independently.

---

# KV Cache

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
y^{cached}*{1:T} = y^{uncached}*{1:T}
$$

Where appropriate, the implementation also compares cached and uncached logits within numerical tolerance.






---



# Continuous Batching

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

Each decode iteration processes one token for every active request. When a request finishes, it is evicted from the active set and a waiting request can be admitted into the newly available slot.

This allows the engine to maintain a higher level of active batch capacity when requests have different output lengths.

The implementation is evaluated against the static batching baseline using workloads with variable generation lengths, measuring:

* Throughput
* TTFT
* ITL
* p50 latency
* p95 latency

The benchmark demonstrates how continuous batching can improve workload-level throughput and reduce tail latency under variable request lengths.



**For More Informations:**

[EXP-2026-014 - Static vs Continuous Batching](experiments/EXP-2026-014.md)

[Continuous Batching Documentation](docs/static_vs_continuous.md)

**Related Papers:**

[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)


---

# Paged KV Memory

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
BlockTable_i[j] = p
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
Throughput = f(engine, hardware, model, workload)
$$

This prevents isolated numbers such as `500 tokens/sec` from being presented without the conditions that produced them.

---

# Experimental Workloads

The benchmark suite progressively introduces more realistic serving conditions.

## S1 — Single Request

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

## S2 — Variable-Length Requests

Introduces different prompt and generation lengths.

Used for:

* batching behavior
* scheduler behavior
* KV memory utilization

## S3 — Concurrent Requests

Introduces multiple simultaneous requests.

Used for:

* static vs continuous batching
* throughput
* TTFT
* ITL

## S4 — Saturation

Concurrency is progressively increased to identify the point where throughput stops scaling and latency begins increasing sharply.

---

# Metrics

## TTFT

Time To First Token:

$$
TTFT = t_{first} - t_{request}
$$

where:

* $t_{first}$ is the timestamp of the first generated token
* $t_{request}$ is the request arrival timestamp

## ITL

For consecutive generated tokens:

$$
ITL_i = t_i - t_{i-1}
$$

## TPOT

Time Per Output Token:

$$
TPOT = \frac{T_{decode}}{N_{output}}
$$

where:

* $T_{decode}$ is the decode time
* $N_{output}$ is the number of generated tokens

## Throughput

Output-token throughput:

$$
Throughput = \frac{N_{output}}{T_{wall}}
$$

reported in output tokens/sec.

## Percentiles

For a latency distribution $x$:

$$
p50 = percentile(x, 50)
$$

$$
p95 = percentile(x, 95)
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
quality \leftrightarrow memory \leftrightarrow latency
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
Request
\rightarrow
Scheduling
\rightarrow
Prefill
\rightarrow
KV\ Cache
\rightarrow
Decode
\rightarrow
Batching
\rightarrow
Paged\ Memory
\rightarrow
Serving
$$

Rather than treating inference frameworks as black boxes, the project builds simplified versions of their fundamental mechanisms, validates them experimentally, and then connects those mechanisms to a production serving system.

---

# Status

```text
M0  Scope, workload contract, environment        ⬜
M1  KV cache and generation                      ⬜
M2  Continuous batching                          ⬜
M3  Paged KV memory                              ⬜
M4  Benchmark harness                            ⬜
M5  vLLM deployment and gap analysis             ⬜
M6  Quantized Darija model serving               ⬜
```
