from __future__ import annotations

import torch

from pathlib import Path
import sys



sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)



from engine.block_pool import BlockPool
from engine.block_table import BlockTable


class PagedKVCache:
    def __init__(
            self,
            block_pool: BlockPool,
            block_table: BlockTable
    ):
        self.block_pool = block_pool
        self.block_table = block_table

    def write_kv(
            self,
            seq_id: str,
            pos: int,
            layer: int,
            key: torch.Tensor,
            value: torch.Tensor
    ) -> None:
        if pos < 0:
            raise ValueError(
                "pos must be non-negative"
            )

        if not 0 <= layer < self.block_pool.num_layers:
            raise ValueError(
                "layer out of range"
            )

        expected_shape = (
            self.block_pool.num_kv_heads,
            self.block_pool.head_dim
        )

        if key.shape != expected_shape:
            raise ValueError(
                f"key must have shape {expected_shape}, "
                f"got {tuple(key.shape)}"
            )

        if value.shape != expected_shape:
            raise ValueError(
                f"value must have shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )

        self.block_table.ensure_capacity(
            seq_id=seq_id,
            next_pos=pos
        )

        block_id, offset = self.block_table.translate(
            seq_id=seq_id,
            logical_pos=pos
        )

        self.block_pool.key_blocks[
            layer,
            block_id,
            :,
            offset,
            :,
        ] = key
        self.block_pool.value_blocks[
            layer,
            block_id,
            :,
            offset,
            :,
        ] = value

    def gather_kv_for_sequence(
            self,
            seq_id: str,
            layer: int,
            length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if length < 0:
            raise ValueError(
                "length must be non-negative"
            )

        if not 0 <= layer < self.block_pool.num_layers:
            raise ValueError(
                "layer out of range"
            )

        keys = []

        values = []

        for pos in range(length):
            block_id, offset = (
                self.block_table.translate(
                    seq_id=seq_id,
                    logical_pos=pos
                )
            )

            key = self.block_pool.key_blocks[
                layer,
                block_id,
                :,
                offset,
                :
            ]

            value = self.block_pool.value_blocks[
                layer,
                block_id,
                :,
                offset,
                :,
            ]

            keys.append(key)
            values.append(value)


        if length == 0:
            shape = (
                0,
                self.block_pool.num_kv_heads,
                self.block_pool.head_dim
            )


            return (
                torch.empty(
                    shape,
                    dtype=self.block_pool.dtype,
                    device=self.block_pool.device
                ),
                torch.empty(
                    shape,
                    dtype=self.block_pool.dtype,
                    device =self.block_pool.device
                )
            )



        return (
            torch.stack(keys, dim=0),
            torch.stack(values, dim=0)
        )