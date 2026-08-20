from __future__ import annotations


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .types import SamplingConfig



class ModelAdapter:
    """
    Thin interface between the inference engine and the underlying model.
    """
    def __init__(
            self,
            model_name: str,
            device: str = "cuda",
            dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device)


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)


        self.model.eval()
        


    def tokenize(self, text: str) -> list[int]:
        """
            conver text to token ids
        """
        encoded = self.tokenizer(
            text, 
            return_tensors="pt",
        )

        return encoded["input_ids"][0].tolist()

    def decode(self, token_ids: list[int]) -> str:
        """
            Convert token IDs -> text.
        """

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )


    @torch.inference_mode()
    def forward_no_cache(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Run the model without using a KV cache.

        input_ids:
            [B, T]

        returns:
            logits [B, T, V]
        """
        input_ids = input_ids.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            use_cache=False
        )


        return outputs.logits

    def sample_next_token(
            self,
            logits: torch.Tensor,
            config: SamplingConfig,
    ) -> int:
        """
        Select the next token from logits.

        logits:
            [V]
        """
        if config.seed is not None:
            torch.manual_seed(config.seed)
            random.seed(config.seed)

        if config.greedy or config.temperature == 0.0:
            return torch.argmax(logits).item()

        logits = logits / config.temperature

        if config.top_k is not None:
            logits = self._apply_top_k(
                logits,
                config.top_k
            )
        if config.top_p < 1.0:
            logits = self._apply_top_p(
                logits,
                config.top_p
            )

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        token = torch.multinomial(
            probabilities,
            num_samples = 1
        )

        return token.item()

    @staticmethod
    def _apply_top_k(
        logits: torch.Tensor,
        top_k: int,

    ) -> torch.Tensor:
        top_k = min(top_k, logits.size(-1))

        values, _ = torch.topk(
            logits,
            top_k
        )

        threshold = values[-1]
        return torch.where(
            logits < threshold,
            torch.full_like(logits, float("-inf")),
            logits
        )


    @staticmethod
    def _apply_top_p(
        logits: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True
        )
        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        cumulative = torch.cumsum(
            probabilities,
            dim=-1
        )


        remove = cumulative > top_p

        # keep at least one tokne
        remove[0] = False
        sorted_logits[remove] = float("-inf")

        filtered = torch.full_like(logits, float("-inf"))

        filtered.scatter_(
            0,
            sorted_indices,
            sorted_logits
        )


        return filtered