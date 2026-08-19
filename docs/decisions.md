# Engineering Decisions

## AUG-19 — Mini engine before vLLM

### Decision

Build the mini inference engine before using vLLM as the production reference.

### Reason

The purpose of the project is to understand the mechanisms behind
modern LLM inference rather than treating an inference framework as
a black box.

The mini engine will implement:

- KV caching
- prefill/decode
- continuous batching
- paged KV memory
- block tables
- benchmarking

vLLM will then be used as a production reference for comparison.

### Principle

The goal is not to beat vLLM.

The goal is to explain why the mini engine differs from vLLM.


---


# The fundamental rule

<mark>No inference performance number is considered valid unless the workload and execution environment that produced it are recorded.</mark>

For every future result, there should be answers to:

```txt
What GPU?
What driver?
What CUDA/PyTorch?
What model?
What revision?
What dtype?
What quantization?
What prompt lengths?
What output lengths?
What concurrency?
What batching policy?
What sampling settings?
How many warmups?
How many measured requests?
What Git commit?
```