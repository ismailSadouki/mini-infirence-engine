from __future__ import annotations

from dataclasses import dataclass


from pathlib import Path
import sys



sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)



from engine.block_pool import BlockPool



@dataclass 
class SequenceBlockTable:
    seq_id: str
    physical_blocks: list[int]

class BlockTable:
    def __init__(
            self,
            block_pool: BlockPool,
            block_size: int
    ):
        if block_size <= 0:
            raise ValueError(
                "block_size must be positive"
            )

        self.block_pool = block_pool
        self.block_size = block_size


        self.tables: dict[str, SequenceBlockTable] = {}


    def ensure_capacity(
            self,
            seq_id: str,
            next_pos: int
    ) -> None:
        if next_pos < 0:
            raise ValueError(
                "next_pos must be non-negative"
            )

        if seq_id not in self.tables:
            self.tables[seq_id] = SequenceBlockTable(
                seq_id=seq_id,
                physical_blocks=[]
            )

        table = self.tables[seq_id]
        required_block = next_pos // self.block_size

        while len(table.physical_blocks) <= required_block:
            physical_block = self.block_pool.allocate_block()


            table.physical_blocks.append(physical_block)


    def translate(
            self,
            seq_id: str,
            logical_pos: int
    ) -> tuple[int, int]:
        if logical_pos < 0:
            raise ValueError(
                "logical_pos must be non-negative"
            )

        if seq_id not in self.tables:
            raise KeyError(seq_id)

        logical_block = (
            logical_pos // self.block_size
        )

        offset = (
            logical_pos % self.block_size
        )

        table = self.tables[seq_id]

        if logical_block >= len(
            table.physical_blocks
        ):
            raise IndexError(
                "logical position has no allocated block"
            )

        physical_block = table.physical_blocks[logical_block]

        return physical_block, offset


    def free_sequence(
            self,
            seq_id: str
    ) -> None:
        if seq_id not in self.tables:
            raise KeyError(seq_id)

        table = self.tables[seq_id]

        for physical_block in table.physical_blocks:
            self.block_pool.free_block(physical_block)

        del self.tables[seq_id]

    def read_sequence_blocks(
            self,
            seq_id: str,
            length: int
    ) -> list[tuple[int, int]]:
        if length < 0:
            raise ValueError(
                'length must be non-negative'
            )

        return [
            self.translate(
                seq_id,
                pos
            )
            for pos in range(length)
        ]