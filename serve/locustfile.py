import os
import random

from locust import HttpUser, between, task


MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

PROMPTS = [
    # Very short
    "Hi",
    "Hello",
    "Why?",
    "2 + 2 =",
    "France is",

    # Short / normal
    "The capital of France is",
    "The largest planet is",
    "Python is a programming language used for",
    "Machine learning models learn patterns from",
    "The opposite of hot is",

    # Longer natural language
    (
        "A neural network is a mathematical model that "
        "learns representations from data by adjusting"
    ),
    (
        "The main purpose of a key value cache during "
        "autoregressive transformer inference is to"
    ),
    (
        "When generating text one token at a time, the "
        "transformer needs to remember the previous"
    ),

    # Punctuation / unusual tokenization
    "Hello, world! How are you?",
    "What?! Really... yes!!!",
    "Python: torch.randn([2, 3]) ->",

    # Multilingual
    "Bonjour, comment allez-vous ?",
    "مرحبا، كيف حالك؟",
    "الجزائر بلد يقع في",

    # Position-sensitive / longer
    (
        "The transformer architecture uses self-attention. "
        "Self-attention allows each token to interact with "
        "other tokens in the sequence. During autoregressive "
        "generation, previously computed key and value states "
        "can be cached so that they do not need to be recomputed."
    ),
]


class VLLMUser(HttpUser):

    # Small pause between completed requests
    wait_time = between(0.5, 1.0)

    def on_start(self):
        self.model = os.getenv("VLLM_MODEL", MODEL)

    @task
    def chat_completion(self):

        prompt = random.choice(PROMPTS)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": 10,
            "temperature": 0.0,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            name="/v1/chat/completions",
            timeout=120,
            catch_response=True,
        ) as response:

            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return

            try:
                data = response.json()

                if not data.get("choices"):
                    response.failure(
                        "Response contains no choices"
                    )
                    return

            except Exception as exc:
                response.failure(
                    f"Invalid JSON response: {exc}"
                )
                return