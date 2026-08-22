# KV Cache Layout

For each transformer layer, the KV cache uses:

$$
K,V \in \mathbb{R}^{B \times H_{kv} \times T_{\max} \times D}
$$

where:

- $B$ = batch size / number of sequences
- $H_{kv}$ = number of key/value heads
- $T_{\max}$ = maximum cached sequence length
- $D$ = head dimension

The cache is pre-allocated to $T_{\max}$.

A decode step writes the new key/value vectors at the current logical position.

For a single request:

$$
B=1
$$

The valid cache prefix after processing $t$ tokens is:

$$
K[:, :, :t, :] \,\,\,\,positions 0 ... t-1

$$ 

and:

$$
V[:, :, :t, :]
$$

The implementation does not use `torch.cat()` to grow the cache.