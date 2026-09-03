# KV-Cache Bug Playbook

## 1. Purpose

This document describes how to debug correctness problems in the mini inference engine, with a focus on KV-cache and decode-time bugs.

The main principle is:

> Do not debug from the final generated token alone. Find the first point where cached and uncached execution diverge.

---

## 2. Cached vs Uncached Correctness

The primary correctness check compares two execution paths:

1. Uncached generation
2. Cached generation

Given the same prompt and sampling configuration, both paths should produce the same generated token sequence under greedy decoding.

The correctness test suite covers:

- multiple prompt types
- different prompt lengths
- variable generation lengths
- position-offset behavior
- non-ASCII text

Run the complete correctness suite with:

```bash
pytest -q tests/test_kv_cache_correctness.py
```

A passing result means the cached decode path is functionally equivalent to the uncached reference for the tested workloads.


## 3. Debugging Order

When cached and uncached generation produce different tokens, trace the computation in this order:

```txt
Prompt
  ↓
Tokenization
  ↓
Prefill
  ↓
KV cache contents
  ↓
Decode position
  ↓
Position IDs / RoPE
  ↓
K/V cache update
  ↓
K/V cache read
  ↓
Attention
  ↓
Logits
  ↓
Selected token
  ↓
Next decode position
```

Do not immediately modify the cache implementation.


First identify the first divergence.

## 4. KV Position / Offset Bugs

A KV cache stores keys and values associated with specific sequence positions.

During decode, the newly generated token must use its correct absolute position.

For a request whose current position is p:
```txt
current token position = p

valid KV prefix:
[0, 1, 2, ..., p]
```

Common mistakes include:

- starting decode positions from zero
- confusing local decode position with absolute sequence position
- writing a token to the wrong cache slot
- reading fewer or more cached positions than expected
- applying RoPE using the wrong position
- using one request's position for another request
    
**Symptoms**

Typical symptoms include:

- first generated token differs
- cached generation diverges after several tokens
- different prompts fail at different decode steps
- output appears plausible but does not match the uncached reference
    
**Debugging**

Compare:

```txt
uncached position
cached position
```

for every decode step.

Then compare the corresponding:

```txt
RoPE position
cache write position
cache read length
```

## 5. Cache Mask Bugs

Attention must only attend to valid positions.

For a token at position p, the valid cached sequence is:

```
[0, ..., p]
```

A cache/mask bug can cause attention to:

- attend to uninitialized cache entries
- attend beyond the current sequence
- exclude valid previous tokens
- use another request's tokens
    
**Symptoms**

- divergence immediately after enabling caching
- unstable or nonsensical generated tokens
- errors that depend on sequence length
    
**Debugging**

Inspect:

```
cache length
attention key length
attention mask
query position
```

and verify that they describe the same logical sequence.

## 6. Paged KV / Block Bugs

With paged KV caching, logical sequence positions are mapped to physical blocks.

The important distinction is:

```
logical position
      ↓
logical block
      ↓
physical block
      ↓
physical cache slot
```

Common bugs include:

- incorrect logical-to-physical mapping
- writing to the wrong physical block
- reading the wrong block
- incorrect block offset
- stale blocks
- blocks not being released
- accidentally sharing blocks between requests
    
**Debugging**

For a failing request, inspect:

```
request
logical position
logical block
physical block
block offset
```

Then verify that the physical location corresponds to the expected logical token.

## 7. Streaming Buffer Bugs

Streaming introduces another source of correctness problems.

Verify that:

```
generated token
    ↓
request state
    ↓
streaming buffer
    ↓
consumer
```

does not:

- drop tokens
- duplicate tokens
- reorder tokens
- expose stale tokens

When debugging generation, compare the internal generated_ids with the streamed tokens.


## 8. Numerical Difference vs Logical Bug

Not every numerical difference means the implementation is incorrect.

Different tensor execution shapes can produce small floating-point differences.

For example:
```
[1, 1, hidden]
```
and

```
[1, T, hidden]
```

may execute through different kernels.

This is especially relevant for lower-precision execution such as BF16.

Therefore distinguish between:

**Numerical difference**

Small floating-point differences that may accumulate across layers but do not change the selected token.

**Logical difference**

A difference caused by incorrect:

- positions
- RoPE
- cache contents
- cache length
- attention masking
- block mapping
- request state

The debugging priority is to find the first logical divergence.
