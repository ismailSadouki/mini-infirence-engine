## KV Cache Speed Benchmark

### Run

```bash
python benchmarks/kv_cache_speed.py
```

### What this command does

Runs a benchmark comparing **uncached autoregressive generation** against **KV-cache generation**.

The workload is loaded from:

```text
configs/workload_single.yaml
```

The benchmark:

1. Loads the configured model and dtype.
2. Creates a deterministic prompt with the configured number of tokens.
3. Runs warmup requests.
4. Measures **uncached generation**, where the full sequence is recomputed at every generation step.
5. Measures **cached generation**, using:

   * prefill
   * KV cache
   * single-token cached decoding
6. Reports:

   * TTFT — time to first token
   * Prefill latency
   * ITL — inter-token latency
   * Total generation latency
   * Generation throughput
   * KV-cache speedup
7. Saves the raw measurements, workload configuration, metadata, and benchmark report.

### Current benchmark result

For the current workload:

```text
Model:          Qwen/Qwen2.5-0.5B-Instruct
Prompt:         512 tokens
Output:         128 tokens
Decoding:       greedy
Warmup:         5
Measurements:   20
```

The benchmark measured approximately:

```text
Uncached mean:       8457.85 ms
Cached mean:         1809.89 ms
KV-cache speedup:    4.67x

TTFT:                69.61 ms
ITL:                 13.70 ms/token
Throughput:          70.74 tokens/s
```

### Output files

```text
results/kv_cache_speed_raw.csv
results/workload_single.yaml
results/kv_cache_speed_metadata.json
reports/kv_cache_speedup.md
```

The important result is that the current KV-cache implementation gives approximately **4.67× lower mean generation latency** than full-sequence recomputation for this workload.
