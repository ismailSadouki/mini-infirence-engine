# KV Cache Layout

The inference engine stores key/value states in **pre-allocated tensors**. Each transformer layer owns one key cache and one value cache.

For a transformer with (L) layers:

$$
K_\ell,V_\ell
\in
\mathbb{R}^{B\times H_{kv}\times T_{\max}\times D},
\qquad
\ell\in{0,\ldots,L-1}
$$

where:

* (B) is the batch size.
* (H_{kv}) is the number of key/value heads.
* (T_{\max}) is the maximum allocated sequence length.
* (D) is the attention head dimension.

The current implementation supports a **single request**:

$$
B=1
$$

Therefore, every layer owns:

```text
[1, H_kv, T_max, D]
```

represented by:

```python
key_cache[layer]
value_cache[layer]
```

For the Qwen2 model used by the engine:

```text
layers    = 24
KV heads  = 2
head_dim  = 64
```

For example, with:

```text
max_seq_len = 16
```

each layer contains:

```text
key_cache[layer]   -> [1, 2, 16, 64]
value_cache[layer] -> [1, 2, 16, 64]
```

---

# Sequence Dimension

The third dimension is the **logical sequence-position dimension**.

For layer (\ell):

$$
K_\ell[:,:,t,:]
$$

contains the key corresponding to logical token position (t), while:

$$
V_\ell[:,:,t,:]
$$

contains its corresponding value.

The cache is allocated once with capacity:

$$
T_{\max}
$$

and is not resized during autoregressive decoding.

If (T) tokens have currently been processed, only:

$$
K_\ell[:,:,0:T,:]
$$

and

$$
V_\ell[:,:,0:T,:]
$$

contain valid states.

The remaining region:

$$
[T,T_{\max})
$$

is unused capacity.

For example:

```text
positions:

0  1  2  3  4  5  6  7  8  ...  15
|-----------|-------------------------|
   valid              unused
```

after five tokens:

```text
0  1  2  3  4  5  6  ...
|-----------|------------|
   valid        unused
```

---

# Cache Update

New K/V states have shape:

$$
K_{\text{new}},V_{\text{new}}
\in
\mathbb{R}^{1\times H_{kv}\times T_{\text{new}}\times D}
$$

and are written beginning at logical position (p).

The update is:

$$
K_\ell[:,:,p:p+T_{\text{new}},:]
\leftarrow
K_{\text{new}}
$$

$$
V_\ell[:,:,p:p+T_{\text{new}},:]
\leftarrow
V_{\text{new}}
$$

The implementation uses indexed assignment:

```python
self.key_cache[layer][
    :, :, start_position:end_position, :
] = key

self.value_cache[layer][
    :, :, start_position:end_position, :
] = value
```

There is therefore no:

```python
torch.cat(...)
```

during decoding.

The underlying cache allocation remains unchanged.

For autoregressive decoding:

$$
T_{\text{new}}=1
$$

so one new token occupies exactly one new cache position.

---

# Prefill and Decode Positions

Suppose the prompt contains (T_p=5) tokens.

During prefill:

$$
p=0
$$

and the five tokens occupy:

$$
[0,5)
$$

so:

```text
position:  0  1  2  3  4
            └───────┘
              prompt
```

The cache length becomes:

```text
5
```

The next token must therefore be written at:

$$
p=5
$$

After that decode step:

```text
position:  0  1  2  3  4  5
            └─────────────┘
                 valid
```

and the cache length becomes:

```text
6
```

The following token would use:

$$
p=6
$$

This distinction is important:

> `start_position` is the **logical sequence position** of the first token being written. It is not an index relative to the current forward call.

---

# Prefix Reads

Attention should only read the valid portion of the cache.

Conceptually:

```python
key, value = cache.read_prefix(
    layer=layer,
    length=sequence_length,
)
```

which corresponds to:

```python
key_cache[layer][:, :, :length, :]
value_cache[layer][:, :, :length, :]
```

Therefore, for current sequence length (T):

$$
K_{\text{read}},V_{\text{read}}
\in
\mathbb{R}^{1\times H_{kv}\times T\times D}
$$

The important distinction is:

$$
T \leq T_{\max}
$$

where (T) is the **current valid length**, while (T_{\max}) is the **allocated capacity**.

---

# Position Semantics and RoPE

The logical cache position must agree with the model's positional encoding.

Qwen2 applies rotary positional embeddings to query and key states.

For a token at logical position (p):

$$
(Q_p,K_p)
\rightarrow
\operatorname{RoPE}(Q_p,K_p,p)
$$

The rotated key is then stored in the cache:

$$
K_{\ell,\text{cache}}[:,:,p,:]
==============================

\operatorname{RoPE}(K_p,p)
$$

This is essential during decoding.

After a five-token prompt, the next token is processed with:

$$
p=5
$$

not:

$$
p=0
$$

even though the decode forward call contains only one token.

Therefore:

```text
prompt:
positions 0 1 2 3 4

decode:
position 5
```

rather than:

```text
decode:
position 0   # incorrect
```

The cache and positional encoding consequently operate in the same logical coordinate system.

---

# Grouped-Query Attention

The cache stores only the (H_{kv}) key/value heads.

Qwen2 uses grouped-query attention (GQA), where:

$$
H_q > H_{kv}
$$

Therefore:

$$
Q
\in
\mathbb{R}^{1\times H_q\times T_q\times D}
$$

while the compact cached states are:

$$
K,V
\in
\mathbb{R}^{1\times H_{kv}\times T_{kv}\times D}
$$

Before attention, K/V are expanded to the number of query heads.

Conceptually:

```python
key = key.repeat_interleave(
    num_key_value_groups,
    dim=1,
)

value = value.repeat_interleave(
    num_key_value_groups,
    dim=1,
)
```

After GQA expansion:

$$
K,V
\in
\mathbb{R}^{1\times H_q\times T_{kv}\times D}
$$

The important point is that **the cache itself remains compact**:

$$
[1,H_{kv},T_{\max},D]
$$

The expansion happens only when preparing the tensors for attention.

---

# Cache Length and Bounds

The cache has fixed capacity:

$$
T_{\max}
$$

while its valid length changes during generation:

$$
0\leq T\leq T_{\max}
$$

For a write beginning at (p) with (T_{\text{new}}) tokens:

$$
0\leq p
$$

and:

$$
p+T_{\text{new}}\leq T_{\max}
$$

must hold.

Otherwise the update is rejected.

For example, if:

```text
max_seq_len = 6
```

and the current cache length is:

```text
6
```

then attempting to write another token would require:

$$
6:7
$$

which exceeds:

$$
T_{\max}=6
$$

and correctly raises a cache-overflow error.

This behavior was encountered during validation and confirmed that the cache does not silently write beyond its allocated memory.

---

# Example

Assume:

```text
H_kv = 2
D    = 64
T_max = 16
```

Then each layer has:

$$
K,V
\in
\mathbb{R}^{1\times2\times16\times64}
$$

A five-token prompt produces:

$$
K_{\text{new}},V_{\text{new}}
\in
\mathbb{R}^{1\times2\times5\times64}
$$

and writes:

```text
positions 0:5
```

The cache length becomes:

```text
5
```

The first decode token produces:

$$
K_{\text{new}},V_{\text{new}}
\in
\mathbb{R}^{1\times2\times1\times64}
$$

and writes:

```text
position 5
```

The cache length becomes:

```text
6
```

The next token writes to:

```text
position 6
```

After two decode steps, the valid region is:

```text
positions 0:7
```

while:

```text
positions 7:16
```

remain unused.

No cache reallocation is required.

---

# Implementation Contract

The current cache implementation guarantees:

1. Every transformer layer owns an independent K cache.

2. Every transformer layer owns an independent V cache.

3. K/V use the layout:

$$
   [B,H_{kv},T_{\max},D]
$$

4. The current implementation supports (B=1).

5. New states are written at explicit logical positions.

6. Existing cache positions are preserved.

7. Reads return only the valid prefix.

8. Decoding does not use `torch.cat()` to grow K/V.

9. Invalid layer indices are rejected.

10. Invalid tensor ranks are rejected.

11. Unsupported batch sizes are rejected.

12. Incorrect KV-head counts are rejected.

13. Incorrect head dimensions are rejected.

14. Key/value shape mismatches are rejected.

15. Writes beyond (T_{\max}) are rejected.

16. Invalid prefix lengths are rejected.

17. RoPE uses the same logical positions as cache writes.

18. The Hugging Face `EngineKVCache` interface forwards model K/V updates into the engine's pre-allocated `KVCache`.

---

# Validation

The implementation has been validated at several levels.

## 1. Raw V Cache Equivalence

For layer 0, cached and uncached V states were compared position-by-position.

The observed result was:

```text
cached V:   [1, 2, 6, 64]
uncached V: [1, 2, 6, 64]

MAX DIFF : 0.0
MEAN DIFF: 0.0
```

Every tested position had:

```text
max = 0
mean = 0
```

This establishes exact equality for the tested V states.

---

## 2. GQA Expansion

After GQA expansion:

```text
cached:   [1, 14, 6, 64]
uncached: [1, 14, 6, 64]
```

The resulting attention context:

$$
A V
$$

was also exactly equal:

```text
cached:   [1, 14, 1, 64]
uncached: [1, 14, 1, 64]

MAX DIFF : 0.0
MEAN DIFF: 0.0
```

---

## 3. Attention Output Projection

The output projection was also exactly equal:

```text
cached:   [1, 1, 896]
uncached: [1, 1, 896]

MAX DIFF : 0.0
MEAN DIFF: 0.0
```

Therefore, for the tested layer-0 attention path:

$$
AV
\rightarrow
W_O(AV)
$$

was numerically identical between cached and uncached execution.

---

# Layer-0 MLP Investigation

The initial BF16 experiment showed a difference at the complete MLP output:

```text
cached:   [1, 1, 896]
uncached: [1, 1, 896]

MAX DIFF : 0.00390625
MEAN DIFF: 0.00021693324379157275
```

This initially suggested that the MLP might be responsible for the numerical discrepancy.

A more detailed decomposition was then performed.

The layer-0 MLP is:

```python
def forward(self, x):
    down_proj = self.down_proj(
        self.act_fn(self.gate_proj(x))
        * self.up_proj(x)
    )
    return down_proj
```

with:

```text
gate_proj: Linear(896 -> 4864)
up_proj:   Linear(896 -> 4864)
down_proj: Linear(4864 -> 896)
activation: SiLU
```

The final-token MLP inputs were exactly equal:

$$
x_{\text{cached}}=x_{\text{uncached}}
$$

with:

```text
MAX DIFF : 0.0
MEAN DIFF: 0.0
```

The individual MLP components were then checked.

### Gate projection

The gate projection was initially measured at:

```text
MAX DIFF : 4.76837158203125e-07
MEAN DIFF: 9.803395595309183e-11
```

This was extremely small and consistent with floating-point numerical behavior.

After isolating the **same final-token inputs** and comparing the corresponding projections directly, the result was:

```text
MAX DIFF : 0.0
MEAN DIFF: 0.0
```

### Up projection

```text
MAX DIFF : 0.0
MEAN DIFF: 0.0
```

### SwiGLU intermediate

```text
MAX DIFF : 0.0
MEAN DIFF: 0.0
```

### Down projection

```text
MAX DIFF : 0.0
MEAN DIFF: 0.0
```

### Direct MLP

When the exact same final-token MLP input was supplied directly to the MLP:

```text
DIRECT CACHED MLP vs DIRECT UNCACHED MLP

MAX DIFF : 0.0
MEAN DIFF: 0.0
```

Furthermore, the manual decomposition:

$$
\operatorname{down_proj}
\left(
\operatorname{SiLU}(\operatorname{gate_proj}(x))
\odot
\operatorname{up_proj}(x)
\right)
$$

matched the direct MLP exactly:

```text
MAX DIFF : 0.0
MEAN DIFF: 0.0
```

This was an important finding.

### Conclusion from the MLP investigation

The BF16 MLP discrepancy was **not reproduced when the exact same final-token tensor was supplied directly to the MLP**.

Therefore, the earlier:

```text
MAX DIFF : 0.00390625
```

should not be interpreted as evidence that the mathematical MLP computation itself differs between cached and uncached execution.

The isolated MLP computation is numerically identical for the tested final-token input.

---

# BF16 Layer-0 Observation

The complete layer-0 MLP output captured during the model forward still showed:

```text
MAX DIFF : 0.00390625
MEAN DIFF: 0.00021693324379157275
```

However, the direct MLP test showed:

```text
DIRECT CACHED MLP vs DIRECT UNCACHED MLP

MAX DIFF : 0.0
MEAN DIFF: 0.0
```

and the exact same input was confirmed:

```text
MLP INPUT

MAX DIFF : 0.0
MEAN DIFF: 0.0
```

This means the observed BF16 discrepancy cannot simply be attributed to:

```text
different MLP inputs
```

or:

```text
different gate projection
```

or:

```text
different up projection
```

or:

```text
different SwiGLU computation
```

or:

```text
different down projection
```

when those components are isolated using the same final-token tensor.

This substantially narrows the investigation.

---

# FP32 Validation

Because BF16 has relatively coarse representational precision, the experiment was repeated with the model in FP32.

The model was successfully loaded on the RTX 3050:

```text
dtype:  torch.float32
device: cuda:0
```

with approximately:

```text
allocated: 2929.87 MB
reserved:  3014.00 MB
```

after the test.

The same cached-vs-uncached experiment was performed.

The final logits had shape:

```text
cached logits:   [1, 151936]
uncached logits: [1, 151936]
```

The measured difference was:

$$
\boxed{
\max_i
\left|
z_{\text{cached},i}
-------------------

z_{\text{uncached},i}
\right|
=======

1.239776611328125\times10^{-5}
}
$$

and:

$$
\boxed{
\operatorname{mean}*i
\left|
z*{\text{cached},i}
-------------------

z_{\text{uncached},i}
\right|
=======

1.8670865529202274\times10^{-6}
}
$$

The logits themselves were approximately:

```text
cached max:   20.463897705078125
cached min:  -11.957040786743164

uncached max:  20.463897705078125
uncached min: -11.95704460144043
```

Most importantly:

```text
cached token  : 13
uncached token: 13

TOKEN MATCH: True
```

Thus FP32 substantially reduces the numerical discrepancy compared with the BF16 observation.

---

# Cached vs. Uncached Numerical Equivalence

The validation should therefore distinguish between **exact numerical equality**, **floating-point numerical closeness**, and **generation equivalence**.

For the tested workload:

### BF16

The complete model-level comparison exhibited a nonzero difference:

```text
MAX DIFF ≈ 1.7e-1
MEAN DIFF ≈ 2.7e-2
```

for the final logits in the earlier BF16 end-to-end test.

The selected token was nevertheless identical:

```text
cached   -> 13
uncached -> 13
```

At the isolated layer-0 MLP level, the apparent BF16 discrepancy was investigated further and the direct MLP computation was shown to be exactly equal when supplied with the same final-token input.

### FP32

The end-to-end final-logit difference was much smaller:

$$
\boxed{
\max |\Delta z|
\approx
1.24\times10^{-5}
}
$$

with:

$$
\boxed{
\operatorname{mean}|\Delta z|
\approx
1.87\times10^{-6}
}
$$

and:

```text
argmax(cached logits)   = 13
argmax(uncached logits) = 13
```

Therefore, the current evidence supports:

$$
z_{\text{cached}}
\approx
z_{\text{uncached}}
$$

rather than:

$$
z_{\text{cached}}
=================

z_{\text{uncached}}
$$

exactly.

---

# Important Interpretation

The cache itself is not currently showing evidence of corrupting the stored K/V states.

In particular, the validation established:

$$
V_{\text{cached}}=V_{\text{uncached}}
$$

for the tested layer and positions, and the attention context and output projection were also exactly equal in the tested layer-0 experiment.

Likewise, the layer-0 MLP was shown to produce exactly the same result when the same final-token input is passed directly through the MLP.

Therefore, the remaining end-to-end numerical difference should **not** be described simply as:

> "The KV cache produces incorrect MLP values."

That conclusion is not supported by the component-level tests.

The remaining discrepancy is more appropriately treated as a **model-level floating-point execution-path difference**, with FP32 reducing it from the much larger BF16 observation to approximately:

$$
1.24\times10^{-5}
$$

for the tested final logits.

Potential areas for further investigation include:

* differences in the full-sequence versus incremental attention execution path;
* attention-mask construction;
* positional-embedding execution;
* accumulation/reduction order;
* Hugging Face cache integration;
* differences between processing (T+1) tokens together and processing (T) tokens followed by one token;
* numerical effects specific to BF16.

The current evidence does **not** justify attributing the residual FP32 difference to a cache-layout error.

---

# Generation Equivalence

The strongest functional test currently performed is token-level equivalence.

For the tested prompt:

```text
"The capital of France is"
```

the cached and uncached paths selected the same token:

```text
cached token   : 13
uncached token : 13
```

The relevant criterion is:

$$
\arg\max_i z_{\text{cached},i}
==============================

\arg\max_i z_{\text{uncached},i}
$$

which held for the tested workload.

However, this is weaker than numerical equivalence.

Generation equivalence means:

$$
\operatorname{argmax}(z_c)
==========================

\operatorname{argmax}(z_u)
$$

whereas numerical equivalence would require something such as:

$$
\max_i |z_{c,i}-z_{u,i}|
\leq\epsilon
$$

for a specified tolerance (\epsilon).

Therefore, the current implementation should be described as having **validated cached-vs-uncached generation behavior**, with strong numerical agreement in FP32, rather than claiming exact model-wide numerical equivalence.

---

# Current Correctness Status

The current implementation has established that:

* the K/V cache is pre-allocated;
* the cache layout is `[B, H_kv, T_max, D]`;
* the implementation currently supports `B=1`;
* K/V states are written using explicit logical positions;
* prefill writes positions `[0,T_p)`;
* decode writes the next logical position;
* cache length advances correctly;
* cache overflow is detected;
* prefix reads operate on the valid sequence region;
* RoPE receives the logical decode position;
* GQA expansion operates correctly for the tested layer;
* cached and uncached V states were exactly equal in the tested comparison;
* cached and uncached attention context were exactly equal in the tested comparison;
* the attention output projection was exactly equal in the tested comparison;
* layer-0 MLP inputs were exactly equal for the final token;
* isolated layer-0 MLP computation was exactly equal when given the same input;
* the Hugging Face model successfully consumes the custom `EngineKVCache`;
* cached and uncached generation selected the same token for the tested workload;
* FP32 reduces the end-to-end final-logit discrepancy to approximately (1.24\times10^{-5}).

The implementation has **not** established exact bit-for-bit numerical equality for the complete cached and uncached model forward passes.

The most accurate current statement is therefore:

$$
\boxed{
\text{KV cache functionality is validated, and cached generation is functionally equivalent on the tested workload.}
}
$$

For FP32, the measured final-logit discrepancy is:

$$
\boxed{
\max |\Delta z|
===============

1.2397766\times10^{-5}
}
$$

with identical selected tokens.

---

# Scope Limitations

The current cache is intentionally a **single-request cache**:

$$
B=1
$$

It does not yet implement request-level memory management.

For example:

```python
free(request_id)
```

is not yet a complete operation.

The current architecture therefore should not yet be described as a multi-request continuous-batching cache.

The current design allocates:

```text
key_cache[layer]
value_cache[layer]
```

independently for every transformer layer.

It does not yet provide a shared memory pool capable of dynamically assigning physical cache blocks to multiple requests.

Those capabilities belong to the later **paged KV-cache / continuous-batching** stage.

---

# Summary

The current cache design is:

$$
\boxed{
K_\ell,V_\ell
\in
\mathbb{R}^{1\times H_{kv}\times T_{\max}\times D}
}
$$

with logical sequence positions determining where new states are written.

For a five-token prompt:

```text
prefill:
positions 0:5

decode token 1:
position 5

decode token 2:
position 6
```

The cache grows logically without reallocating physically:

$$
T:5\rightarrow6\rightarrow7\rightarrow\cdots
$$

while the underlying allocation remains:

$$
T_{\max}.
$$

The component-level validation shows that the tested K/V, GQA attention context, output projection, and isolated MLP computations agree exactly.

The remaining end-to-end numerical discrepancy is small in FP32:

$$
\boxed{
\max |\Delta z|
\approx1.24\times10^{-5}
}
$$

and the cached and uncached paths produce the same selected token.

Thus the current result is **not "the cache is numerically wrong."** The evidence instead shows a functioning cache with cached/uncached generation agreement and a remaining floating-point execution-path discrepancy that warrants further investigation if strict numerical equivalence is a project requirement.