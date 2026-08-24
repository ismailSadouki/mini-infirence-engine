
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch


import torch.nn.functional as F

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model,
    apply_rotary_pos_emb,
)

from engine.kv_cache import KVCache


class CachedQwen2Attention:
    """
    Thin wrapper around a Qwen2 attention module.

    The actual Qwen2 projection weights are reused.
    Our KVCache stores the projected K/V tensors.
    """

    def __init__(self, attention, config, layer_idx:int):

        self.attention = attention
        self.layer_idx = layer_idx

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )

        self.head_dim = attention.head_dim
        self.scaling = attention.scaling

        

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings,
            cache: KVCache,
            position: int
    ): 
        """
        hidden_states:
            [B, T, hidden_size]

        During prefill:
            T = prompt length

        During decode:
            T = 1
        """
        B, T, _ = hidden_states.shape

        # Q/K/V projections
        query_states = self.attention.q_proj(hidden_states)
        key_states = self.attention.k_proj(hidden_states)
        value_states = self.attention.v_proj(hidden_states)


        # [B, T, H*D] -> [B, H, T, D]
        query_states = query_states.view(
            B,
            T,
            self.num_attention_heads,
            self.head_dim
        ).transpose(1, 2)

        key_states = key_states.view(
            B,
            T,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        value_states = value_states.view(
            B,
            T,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)


        # Apply RoPE BEFORE storing K.
        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin
        )

        # storing the newly produced K/V
        cache.update(
            layer=self.layer_idx,
            key=key_states,
            value=value_states,
            start_position=position
        )

        total_length = position + T

        cached_key, cached_value = cache.read_prefix(
            layer=self.layer_idx,
            length=total_length,
        )

        # GQA
        # [B, H_kv, T, D] -> [B, H_q, T, D]
        cached_key = cached_key.repeat_interleave(
            self.num_key_value_groups,
            dim=1
        )
        cached_value = cached_value.repeat_interleave(
            self.num_key_value_groups,
            dim=1
        )

        


        # Attention scores
        # Q: [B, H, T_query, D]
        # K: [B, H, T_cache, D]
        # Result: [B, H, T_query, T_cache]
        
        scores = torch.matmul(
            query_states,
            cached_key.transpose(-1, -2)
        ) * self.scaling

        # casual masking
        # For decode T=1, the query is allowed to attend to the entire prefix.
        #
        # For prefill, token i can only attend to positions <= i.
        query_positions = torch.arange(
            position,
            position + T,
            device= hidden_states.device
        )

        key_positions = torch.arange(
            total_length,
            device=hidden_states.device
        )

        casual_mask = (
            key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        )


        scores = scores.masked_fill(
            ~casual_mask,
            torch.finfo(scores.dtype).min
        )


        attention_weights = F.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)

        output = torch.matmul(
            attention_weights,
            cached_value
        )

        # [B, H, T, D] -> [B, T, H*D]
        
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            B,
            T,
            self.num_attention_heads * self.head_dim
        )

        output = self.attention.o_proj(output)

        return output


class CachedQwen2Model:
    def __init__(self, model: Qwen2Model, config):
        self.model = model

        self.cached_attentions = [
            CachedQwen2Attention(
                layer.self_attn,
                layer_idx=i,
                config=config
            ) 
            for i, layer in enumerate(model.layers)
        ]


    def forward(
            self,
            input_ids: torch.Tensor,
            cache: KVCache,
            position: int
    ) :
        """
        input_ids:
            [B, T]

        position:
            logical starting position.
        """
        inputs_embeds = self.model.embed_tokens(
            input_ids
        )

        B, T, _ = inputs_embeds.shape

        position_ids = torch.arange(
            position,
            position + T,
            dtype=torch.long,
            device=input_ids.device
        ).unsqueeze(0)

        # Generate RoPE embeddings using the exact Qwen2 implementation.
        position_embeddings = self.model.rotary_emb(
            inputs_embeds,
            position_ids
        )

        hidden_states = inputs_embeds

        for layer_idx, layer in enumerate(self.model.layers):
            residual = hidden_states

            hidden_states = layer.input_layernorm(hidden_states)

            attention_outputs = self.cached_attentions[
                layer_idx
            ].forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache=cache,
                position=position
            )
            hidden_states = (
                residual + attention_outputs
            )

            residual = hidden_states

            hidden_states = layer.post_attention_layernorm(
                hidden_states
            )

            hidden_states = layer.mlp(
                hidden_states
            )

            hidden_states = (
                residual + hidden_states
            )

        hidden_states = self.model.norm(hidden_states)

        return hidden_states