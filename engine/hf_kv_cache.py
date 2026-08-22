import torch 
from transformers.cache_utils import Cache


class EngineKVCache(Cache):
    def __init__(
            self,
            kv_cache
    ):

        super().__init__(
            layers=[],
        )

        self.kv_cache = kv_cache

        # Number of sequence tokens currently stored.
        self._seen_tokens = 0

        # Number of tokens written for each layer.
        self._layer_seq_lengths = [
            0
            for _ in range(kv_cache.num_layers)
        ]
    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs=None,
    ):
        batch_size = key_states.shape[0]
        num_tokens = key_states.shape[2]

        if batch_size != 1:
            raise ValueError(
                "M1.3 cache currently supports only batch size 1."
            )

        start_position = self._layer_seq_lengths[layer_idx]


        self.kv_cache.update(
            layer=layer_idx,
            key=key_states,
            value=value_states,
            start_position=start_position,
        )

        self._layer_seq_lengths[layer_idx] += num_tokens

        self._seen_tokens = max(
            self._layer_seq_lengths
        )

        return (
            self.kv_cache.key_cache[layer_idx][
                :, :, :self._layer_seq_lengths[layer_idx], :
            ],
            self.kv_cache.value_cache[layer_idx][
                :, :, :self._layer_seq_lengths[layer_idx], :
            ],
        )

    def get_seq_length(self, layer_idx=0):
        return self._layer_seq_lengths[layer_idx]

    def get_max_cache_shape(self):
        return self.kv_cache.max_seq_len
    def get_mask_sizes(
        self,
        query_length: int,
        layer_idx: int,
    ):
        kv_length = (
            self._layer_seq_lengths[layer_idx]
            + query_length
        )



        return kv_length, 0