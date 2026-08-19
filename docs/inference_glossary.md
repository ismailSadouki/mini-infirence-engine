
At minimum define:

**TTFT**

$$
TTFT=t_{\text{first token}}-t_{\text{request arrival}}
$$

**ITL**

For generated tokens (1,\ldots,T):

$$
ITL_i=t_i-t_{i-1}
$$

**TPOT**

Token generation time excluding the initial prompt-to-first-token delay, depending on your exact measurement convention.

**Throughput**

$$
\text{throughput}
=================

\frac{N_{\text{generated tokens}}}
{T_{\text{wall}}}
$$

**p50**

$$
p50=\operatorname{percentile}(x,50)
$$

**p95**

$$
p95=\operatorname{percentile}(x,95)
$$

Also explicitly define whether your throughput is:

```text
output tokens/sec
```

or:

```text
input + output tokens/sec
```

For this project, I'd use **output tokens/sec** for generation throughput and report the convention clearly.
