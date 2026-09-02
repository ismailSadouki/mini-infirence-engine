from __future__ import annotations


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .types import SamplingConfig
from engine.qwen2_cached import CachedQwen2Model



MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"




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
        self.dtype = dtype


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.cached_model = CachedQwen2Model(
            self.model.model,
            self.model.config
        )

        self.model.eval()
        
        
    def to_tensor(self, token_ids: list[int]) -> torch.Tensor:
        return torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

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


    @torch.inference_mode()
    def forward_prefill(
        self,
        input_ids: torch.Tensor
    ):
        """
        Process the entire prompt.

        input_ids:
            [B, T]

        Returns:
            logits:
                [B, T, V]

            cache:
                placeholder for M1.3
        """
        input_ids = input_ids.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            use_cache=False
        )

        return outputs.logits, None


    @torch.inference_mode()
    def forward_decode(
        self,
        last_token: torch.Tensor,
        cache,
        position: int
    ):
        """
        Process exactly one token during decode.

        last_token:
            [B, 1]

        cache:
            KV cache placeholder for M1.3

        position:
            absolute position of the token
        """
        if last_token.ndim != 2:
            raise ValueError(
                f"last_token must have shape [B, 1], "
                f"got {tuple(last_token.shape)}"
            )
        if last_token.shape[1] != 1:
            raise ValueError(
                "forward_decode() must receive exactly one token."
            )
        


        last_token = last_token.to(self.device)
        # M1.2 placeholder.
        #
        # This is NOT yet a correct cached decode.
        # M1.3 will replace this with actual KV-cache usage.
        outputs = self.model(
            input_ids=last_token,
            use_cache=False
        )

        return outputs.logits, cache


    @torch.inference_mode()
    def forward_prefill_cached(
        self,
        input_ids,
        cache,
    ):
        hidden_states = self.cached_model.forward(
            input_ids=input_ids,
            cache=cache,
            position=0
        )
        logits = self.model.lm_head(
            hidden_states
        )


        return logits


    @torch.inference_mode()
    def forward_decode_cached(
        self,
        last_token,
        cache,
        position,
    ):
        if last_token.shape[1] != 1:
            raise ValueError(
                "Decode must receive exactly one token"
            )

        hidden_states  = self.cached_model.forward(
            input_ids=last_token,
            cache=cache,
            position=position
        )

        logits = self.model.lm_head(
            hidden_states
        )


        return logits

    @torch.inference_mode()
    def forward_decode_hidden_cached(
        self,
        input_ids,
        cache,
        position,
    ):
        input_ids = input_ids.to(self.device)

        return self.cached_model.forward_decode(
            input_ids=input_ids,
            caches=[cache],
            positions=torch.tensor(
                [position],
                dtype=torch.long,
                device=self.device,
            ),
        )
    @torch.inference_mode()
    def forward_decode_ragged(
        self,
        input_ids,
        caches,
        positions,
    ):
        """
        Ragged decode
        
        input_ids: [B, 1]
        caches: list of B independent KV caches
        positions: [B]
        
        Returns:
            [B, 1, hidden_size]
        """
        return self.cached_model.forward_decode(
            input_ids=input_ids,
            caches=caches,
            positions=positions,
        )



    @torch.inference_mode()
    def forward_prefill_paged(
        self,
        input_ids,
        cache,
        seq_id
    ):
        hidden_states = self.cached_model.forward_paged(
            input_ids=input_ids,
            cache=cache,
            seq_id=seq_id,
            position=0
        )
        logits = self.model.lm_head(hidden_states)

        return logits


    @torch.inference_mode()
    def forward_decode_paged(
        self,
        last_token,
        cache,
        seq_id,
        position,
    ):
        hidden_states = self.cached_model.forward_paged(
            input_ids=last_token,
            cache=cache,
            seq_id=seq_id,
            position=position,
        )

        logits = self.model.lm_head(hidden_states)

        return logits