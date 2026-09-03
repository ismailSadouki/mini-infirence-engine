# vLLM on NVIDIA T4

## Environment

- GPU: NVIDIA Tesla T4
- VRAM: 15360 MiB
- Compute capability: 7.5
- vLLM: 0.28.0
- PyTorch: 2.13.0+cu130
- CUDA available: True
- Tensor parallelism: 1

## Serving model

- Model: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Model size: 7B class
- Quantization: AWQ
- Precision: 4-bit weights
- Max model length: 2048

## Launch command

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --tensor-parallel-size 1 \
    --max-model-len 2048
```

**OpenAI-compatible endpoint**

```
OpenAI-compatible endpoint
```

Available endpoints used:

```
GET  /v1/models
POST /v1/chat/completions
```

**Model endpoint test**

```
curl http://localhost:8000/v1/models
```

Result:

```
Qwen/Qwen2.5-7B-Instruct-AWQ
max_model_len: 2048
```

## Smoke test

Request:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    messages=[
        {
            "role": "user",
            "content": "Explain what a KV cache is in one sentence."
        }
    ],
    max_tokens=50,
)

print(response.choices[0].message.content)
```

Response:

```
A KV cache is a type of cache that stores data using key-value pairs, allowing for rapid lookups based on the key.
```

**GPU memory**

Before serving:

```
Used: 3 MiB
Free: 14910 MiB
```

After model loading and smoke request:

```
Used: 14389 MiB
Free: 524 MiB
```
