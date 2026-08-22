import torch


class KVCache:
    def __init__(
            self,
            num_layers: int,
            num_kv_heads: int,
            max_seq_len: int,
            head_dim: int,
            dtype: torch.dtype,
            device: torch.device
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device


        # self.key_cache[layer] has [1, H_kv, T_max, D]
        self.key_cache = [
            torch.empty(
                1, # batch size
                num_kv_heads,
                max_seq_len,
                head_dim,
                dtype=dtype,
                device=device
            ) 
            for _ in range(num_layers)
        ]

        self.value_cache = [
            torch.empty(
                1,
                num_kv_heads,
                max_seq_len,
                head_dim,
                dtype=dtype,
                device=device
            )
            for _ in range(num_layers)
        ]

    def update(
            self,
            layer: int,
            key: torch.Tensor,
            value: torch.Tensor,
            start_position: int
    ):
        if not 0 <= layer < self.num_layers:
            raise IndexError(
                f"Invalid layer: {layer}"
            )


        if key.shape != value.shape:
            raise ValueError(
                f"Key/value shapes must match. "
                f"Got {tuple(key.shape)} and {tuple(value.shape)}"
            )

        if key.ndim != 4:
            raise ValueError(
                f"Expected [B, H_kv, T, D], "
                f"got {tuple(key.shape)}"
            )

        if value.ndim != 4:
            raise ValueError(
                f"Expected [B, H_kv, T, D], "
                f"got {tuple(value.shape)}"
            )


        batch_size, num_heads, seq_len, head_dim = key.shape

        if batch_size != 1:
            raise ValueError(
                "M1.3 supports only batch size 1."
            )

        if num_heads != self.num_kv_heads:
            raise ValueError(
                f"Expected {self.num_kv_heads} KV heads, "
                f"got {num_heads}"
            )

        if head_dim != self.head_dim:
            raise ValueError(
                f"Expected head_dim={self.head_dim}, "
                f"got {head_dim}"
            )



        
        end_position = start_position + seq_len
        if end_position > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: "
                f"trying to write positions "
                f"{start_position}:{end_position}, "
                f"max_seq_len={self.max_seq_len}"
            )


        self.key_cache[layer][:, :, start_position:end_position, :] = key
        self.value_cache[layer][:, :, start_position:end_position, :] = value







    def read_prefix(
            self,
            layer: int,
            length: int,
    ): 
        if not 0 <= length <= self.max_seq_len:
            raise ValueError(
                f"Invalid cache length: {length}"
            )
        return (
            self.key_cache[layer][:, :, :length, :],
            self.value_cache[layer][:, :, :length, :]
        )
    def free(self, request_id):
        """Placeholder for request-aware cache management.

        Request-specific allocation/freeing will be implemented
        when continuous batching is introduced.
        """
        pass


