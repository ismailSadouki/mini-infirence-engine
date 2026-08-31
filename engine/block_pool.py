from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch




@dataclass
class SequenceAllocation:
    seq_id: str
    block_ids: list[int]


class BlockPool:
    """
    Fixed-size physical KV-cache block pool.

    Each physical block stores KV values for BLOCK_SIZE tokens.

    The pool owns the actual K/V storage and a free-list describing
    which physical blocks are currently available.
    """
    def __init__(
            self,
            num_blocks: int,
            block_size: int,
            num_layers: int,
            num_kv_heads: int,
            head_dim: int,
            dtype: torch.dtype,
            device: torch.device | str
    ):
        if num_blocks <= 0:
            raise ValueError(
                "num_block must be positive"
            )

        if block_size <= 0:
            raise ValueError(
                "block_size must be positive"
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive"
            )

        if num_kv_heads <= 0:
            raise ValueError(
                "num_kv_heads must be positive"
            )

        if head_dim <= 0:
            raise ValueError(
                "head_dim must be positive"
            )

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        # Physical K/V memory
        # [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
        self.key_blocks = torch.empty(
            (
                num_layers,
                num_blocks,
                num_kv_heads,
                block_size,
                head_dim,
            ),
            dtype=dtype,
            device=self.device
        )

        self.value_blocks = torch.empty(
            (
                num_layers,
                num_blocks,
                num_kv_heads,
                block_size,
                head_dim,
            ),
            dtype=dtype,
            device=self.device
        )

        # physical blocks that are currently available
        self._free_blocks: list[int] = list(range(num_blocks))
        # seq metadata
        self._sequences: dict[
            str,
            SequenceAllocation
        ] = {}


    def allocate_block(self) -> int:
        """
        Allocate one physical block.

        Returns:
            Physical block ID.
        """
        if not self._free_blocks:
            raise RuntimeError(
                "KV block pool exhausted: "
                "no free blocks available"
            )

        return self._free_blocks.pop()



    def free_block(self, block_id: int) -> None:
        """
        Return one physical block to the free list
        """

        if not 0 <= block_id < self.num_blocks:
            raise ValueError(
                f"Invalid block_id: {block_id}"
            )

        if block_id in self._free_blocks:
            raise ValueError(
                f"Block {block_id} is already free"
            )

        self._free_blocks.append(block_id)


    def allocate_sequence(
            self,
            seq_id: str,
            num_blocks: int
    ) -> list[int]:
        """
        Allocate physical blocks for one sequence.

        The returned list contains physical block IDs.

        This method does not create a logical block table yet.
        """


        if seq_id in self._sequences:
            raise ValueError(
                f"Sequence {seq_id} already allocated"
            )

        if num_blocks <= 0:
            raise ValueError(
                "num_blocks must be positive"
            )

        if num_blocks > self.free_count:
            raise RuntimeError(
                f"KV block pool exhausted: "
                f"requested {num_blocks} blocks, "
                f"but only {self.free_count} are free"
            )

        block_ids = [
            self.allocate_block()
            for _ in range(num_blocks)
        ]

        self._sequences[seq_id] = SequenceAllocation(
            seq_id=seq_id,
            block_ids=block_ids
        )

        return block_ids

    def free_sequence(self, seq_id: str) -> None:
        """
        free physical block owned by a sequence
        """

        if seq_id not in self._sequences:
            raise KeyError(
                seq_id
            )

        allocation = self._sequences.pop(seq_id)

        for block_id in allocation.block_ids:
            self.free_block(block_id)

    @property
    def free_count(self) -> int:
        return len(self._free_blocks)

    @property
    def allocated_count(self) -> int:
        return self.num_blocks - self.free_count

    @property
    def free_blocks(self) -> list[int]:
        return list(self._free_blocks)

    @property
    def allocated_sequences(self) -> dict[
        str,
        SequenceAllocation,
    ]:
        return dict(self._sequences)


    