# KV Cache Layout

The inference engine stores key/value states in pre-allocated tensors. Each transformer layer owns one key cache and one value cache.

For a transformer with $L$ layers:

$$
K_\ell, V_\ell
\in
\mathbb{R}^{B \times H_{kv} \times T_{\max} \times D}
\qquad
\ell \in {0,\ldots,L-1}
$$

where:

* $B$ is the batch size.
* $H_{kv}$ is the number of key/value heads.
* $T_{\max}$ is the maximum sequence length allocated for the cache.
* $D$ is the attention head dimension.

The implementation currently allocates one request:

$$
B=1
$$

so every layer contains tensors with shape:

```text
[1, H_kv, T_max, D]
```

The cache is represented as:

```python
key_cache[layer]
value_cache[layer]
```

with:

```python
key_cache[layer].shape == (1, H_kv, T_max, D)
value_cache[layer].shape == (1, H_kv, T_max, D)
```

For the Qwen2 model used by the engine:

```text
layers    = 24
KV heads  = 2
head_dim  = 64
```

so, for example, with `max_seq_len=16`:

```text
[1, 2, 16, 64]
```

for every layer.

## Sequence Dimension

The third dimension is the logical sequence-position dimension:

$$
K_\ell[:, :, t, :]
$$

contains the key vector for token position $t$, and:

$$
V_\ell[:, :, t, :]
$$

contains its corresponding value vector.

The cache is allocated once for the maximum supported sequence length:

$$
T_{\max}
$$

and is not resized during generation.

Only the prefix corresponding to already processed tokens is valid.

After processing $T$ tokens:

$$
K_\ell^{valid}
==============

K_\ell[:, :, 0:T, :]
$$

$$
V_\ell^{valid}
==============

V_\ell[:, :, 0:T, :]
$$

The remaining positions are unused capacity.

## Cache Update

New K/V states are written directly into their logical sequence positions.

For new tensors:

$$
K_{new},V_{new}
\in
\mathbb{R}^{1 \times H_{kv} \times T_{new} \times D}
$$

and starting position $p$, the update performs:

$$
K_\ell[:, :, p:p+T_{new}, :]
\leftarrow K_{new}
$$

$$
V_\ell[:, :, p:p+T_{new}, :]
\leftarrow V_{new}
$$

The implementation therefore uses indexed assignment rather than concatenating tensors:

```python
self.key_cache[layer][
    :, :, start_position:end_position, :
] = key

self.value_cache[layer][
    :, :, start_position:end_position, :
] = value
```

There is no `torch.cat()` during decoding and the underlying cache allocation remains unchanged.

For autoregressive decoding, normally:

$$
T_{new}=1
$$

so a new token is written to exactly one sequence position.

For example, after a five-token prompt:

```text
positions:  0  1  2  3  4  5  6  ...
            └───────┘  └────────────
             valid       unused
```

The next token is written at:

```text
position = 5
```

using:

```python
start_position=5
```

## Prefix Reads

Attention does not read the entire allocated tensor. It reads only the valid prefix:

```python
cache.read_prefix(
    layer=layer,
    length=sequence_length,
)
```

which returns:

```python
key_cache[layer][:, :, :length, :]
value_cache[layer][:, :, :length, :]
```

Therefore, if the current sequence length is $T$:

$$
K_{read},V_{read}
\in
\mathbb{R}^{1 \times H_{kv} \times T \times D}
$$

The allocated capacity $T_{\max}$ is independent from the number of valid tokens $T$.

## Position Semantics

`start_position` represents the **logical position of the first token being written**.

It is not an index relative to the current forward call.

For a prompt of length $T_p$:

$$
p=0
$$

during prefill, and the cache contains:

$$
[0,T_p)
$$

After prefill, the first decode token has logical position:

$$
p=T_p
$$

The following decode token has:

$$
p=T_p+1
$$

and so forth.

Consequently, cache positions and model positional encoding must use the same logical sequence positions.

## RoPE and Cached Keys

Qwen2 applies rotary positional embeddings to the query and key states before the key is stored.

For a token at logical position $p$:

$$
Q_p,K_p
\rightarrow
\operatorname{RoPE}(Q_p,K_p,p)
$$

The resulting rotated key is what enters the cache:

$$
K_{\ell,\text{cache}}[:, :, p, :]
=================================

\operatorname{RoPE}(K_p,p)
$$

This is important during decoding because the position does not restart at zero when the model receives a single token.

For example, after a prompt of length $5$, the decode call contains one token but its position is:

$$
p=5
$$

not:

$$
p=0
$$

The cache therefore preserves the positional representation that would have been produced if the complete sequence had been processed in one forward pass.

## Attention and GQA

The cache stores only the key/value heads:

$$
H_{kv}
$$

rather than duplicating them for every query head.

Qwen2 uses grouped-query attention (GQA), where:

$$
H_q > H_{kv}
$$

The cached tensors therefore have:

$$
K,V
\in
\mathbb{R}^{1 \times H_{kv} \times T \times D}
$$

while attention operates with:

$$
Q
\in
\mathbb{R}^{1 \times H_q \times T_q \times D}
$$

Before computing attention, cached K/V states are expanded across the KV groups:

```python
cached_key = cached_key.repeat_interleave(
    num_key_value_groups,
    dim=1,
)

cached_value = cached_value.repeat_interleave(
    num_key_value_groups,
    dim=1,
)
```

After expansion:

$$
K,V
\in
\mathbb{R}^{1 \times H_q \times T_{kv} \times D}
$$

The cache itself remains in the compact $H_{kv}$ representation.

## Cache Length

The cache allocation has a fixed capacity:

$$
T_{\max}
$$

while the current valid sequence length is dynamic:

$$
0 \le T \le T_{\max}
$$

A write beginning at position $p$ with $T_{new}$ tokens is valid only if:

$$
p \ge 0
$$

and:

$$
p+T_{new}\le T_{\max}
$$

An update that exceeds the allocated capacity is rejected rather than silently truncating the data.

Similarly, `read_prefix()` rejects lengths outside:

$$
[0,T_{\max}]
$$

## Example

Assume:

```text
H_kv      = 2
D         = 64
T_max     = 16
```

The cache for one layer is:

$$
K,V
\in
\mathbb{R}^{1\times2\times16\times64}
$$

A five-token prefill produces:

$$
K_{new},V_{new}
\in
\mathbb{R}^{1\times2\times5\times64}
$$

and writes them to:

```text
positions 0:5
```

The first decode token produces:

$$
K_{new},V_{new}
\in
\mathbb{R}^{1\times2\times1\times64}
$$

and writes them to:

```text
position 5
```

The next decode token writes to:

```text
position 6
```

After these operations, the valid cache region is:

```text
positions 0:7
```

with capacity remaining for:

```text
positions 7:16
```

No tensor reallocation is required.

## Implementation Contract

The cache implementation guarantees the following:

1. Every transformer layer has independently allocated K and V tensors.
2. K and V use the layout `[B, H_kv, T_max, D]`.
3. New states are written at explicit logical positions.
4. Existing cached states are preserved when new positions are written.
5. Reads return only the requested valid prefix.
6. K/V tensors are never grown with `torch.cat()` during decoding.
7. Invalid layer indices are rejected.
8. Invalid tensor ranks are rejected.
9. Batch sizes other than `1` are rejected.
10. Incorrect KV-head counts are rejected.
11. Incorrect head dimensions are rejected.
12. Key/value shape mismatches are rejected.
13. Writes beyond `T_max` are rejected.
14. RoPE uses the same logical positions used by cache writes.

This layout provides a fixed memory region for each layer while allowing the valid sequence length to grow through indexed writes during autoregressive generation.









## Validation and Current Limitations

The cache implementation has been validated at several levels.

### Cache storage

The pre-allocated tensors are verified to have the expected layout:

$$
[1,H_{kv},T_{\max},D]
$$

Tests verify that:

* all transformer layers receive a K cache and a V cache;
* prefill writes the expected sequence range;
* decode writes new K/V states at the next position;
* previously written positions remain unchanged;
* invalid tensor shapes are rejected;
* writes exceeding `max_seq_len` are rejected;
* invalid prefix lengths are rejected.

### Hugging Face integration

The custom `EngineKVCache` implements the required Hugging Face cache interface and forwards K/V updates into the pre-allocated `KVCache`.

A model-level test verifies that:

$$
T_p \rightarrow T_p+1
$$

when a decode token is appended after prefill.

The observed cache transitions are:

```text
prefill: cache length = prompt length
decode:  cache length = prompt length + 1
```

### Cached vs. uncached generation

Cached generation was compared against full-sequence uncached generation.

The test used:

```text
prompt:
"The capital of France is"

generated tokens:
10
```

The cached and uncached generation produced the same token sequence for this test:

```text
cached generated == full generated
```

However, this test does **not establish numerical equivalence** of the model computation.

The measured difference for the single-token cached decode was:

```text
HIDDEN MAX DIFF : 1.0
HIDDEN MEAN DIFF: 0.055908203125

LOGITS MAX DIFF:  0.173828125
LOGITS MEAN DIFF: 0.02734375
```

while both computations selected the same next token:

```text
CACHED TOKEN:    13
UNCACHED TOKEN:  13
```

The current assertion checks only:

$$
\arg\max(z_{cached})
====================

\arg\max(z_{uncached})
$$

It does not check:

$$
z_{cached}\approx z_{uncached}
$$

Therefore, cached and uncached computation should currently be considered **generation-equivalent for the tested workload**, rather than numerically equivalent.

The source of the observed numerical difference has not yet been isolated. Possible areas for further investigation include positional handling, attention-mask construction, Hugging Face cache integration, and differences in the execution path between cached and full-sequence attention.

### Scope limitations

The current cache implementation is intentionally limited to a single request:

$$
B=1
$$

It does not yet provide request-level memory management.

In particular:

```python
free(request_id)
```

is currently a placeholder.

The cache therefore should not yet be considered a multi-request or continuous-batching cache.

The current layout also stores one fixed allocation per transformer layer:

```text
key_cache[layer]
value_cache[layer]
```

rather than a shared pool capable of dynamically assigning sequence regions to multiple requests.

These capabilities belong to the later batching/paging design and are not required by the current single-request implementation.

### Current correctness status

The implementation has established that:

* pre-allocation works;
* indexed K/V updates work;
* prefix reads work;
* cache length advances correctly;
* logical decode positions are propagated;
* RoPE receives the decode position;
* the Hugging Face model can use the custom cache;
* cached generation produced the same tokens as uncached generation in the current test.

The implementation has **not** yet established strict numerical equivalence between cached and uncached forward passes.
