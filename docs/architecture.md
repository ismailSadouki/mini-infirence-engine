



## KV Block Size

The paged KV memory implementation uses a block size of:

```text
block_size = 16 tokens
```

A block stores the key and value tensors corresponding to 16 sequence positions.

The block size is a trade-off between internal fragmentation and block management overhead.

For a sequence using \(T\) tokens, the number of required blocks is:

$$
N_{\text{blocks}} =
\left\lceil
\frac{T}{B}
\right\rceil
$$

where \(B\) is the block size.

The allocated token capacity is therefore:

$$
C = N_{\text{blocks}}B
$$

and the internal unused capacity is:

$$
W = C-T
$$

A smaller block size reduces this unused capacity but increases the number of blocks and therefore the amount of block-table metadata and allocation management.

For this project, \(B=16\) is used because it provides a simple balance between the two effects while making the reduction in internal fragmentation visible in the experiments.