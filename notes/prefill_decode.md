# Prefill vs Decode

## Prefill

Prefill processes the complete prompt.

For an input with batch size $$B$$ and sequence length $$T$$, the input has shape:

$$
[B, T]
$$

The prompt tokens can be processed in parallel.

The prefill phase produces the logits needed to generate the first output token and, in the KV-cache implementation, initializes the key-value cache.

Prefill is commonly compute-bound because it exposes substantial parallelism to the GPU.

## Decode

Decode generates one new token at a time for each active sequence.

The decode input has shape:

$$
[B, 1]
$$

The new token attends to the previous context through the KV cache.

Decode is commonly memory-bandwidth-bound because each step must access the previously stored KV states while processing only a small number of new tokens.

## Latency

TTFT measures the time from request arrival until the first generated token.

ITL measures the time between consecutive generated tokens.

Therefore:

- TTFT is strongly affected by prefill.
- ITL is strongly affected by decode.

M1.2 separates these phases at the engine interface.

M1.3 will implement the actual KV cache required for correct cached decoding.