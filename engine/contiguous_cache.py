from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContiguousAllocation:
    request_id: str
    reserved_tokens: int
    used_tokens: int = 0


    @property
    def wasted_tokens(self) -> int:
        return self.reserved_tokens - self.used_tokens


class ContiguousKVCache:
    def __init__(self, max_seq_len: int):
        if max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be positive"
            )

        self.max_seq_len = max_seq_len
        self.allocations: dict[
            str,
            ContiguousAllocation
        ] = {}

    def allocate(
            self,
            request_id: str,
            used_tokens: int
    ) -> ContiguousAllocation:
        if request_id in self.allocations:
            raise ValueError(
                f"Request {request_id} already allocated"
            )

        if not 0 <= used_tokens <= self.max_seq_len:
            raise ValueError(
                "used_tokens must satisfy "
                "0 <= used_tokens <= max_seq_len"
            )

        allocation = ContiguousAllocation(
            request_id=request_id,
            reserved_tokens=self.max_seq_len,
            used_tokens=used_tokens,
        )


        self.allocations[request_id] = allocation

        return allocation


    def free(self, request_id: str) -> None:
        if request_id not in self.allocations:
            raise KeyError(request_id)


        del self.allocations[request_id]


    @property
    def reserved_tokens(self) -> int:
        return sum(
            allocation.reserved_tokens
            for allocation in self.allocations.values()
        )
    
    @property
    def used_tokens(self) -> int:
        return sum(
            allocation.used_tokens
            for allocation in self.allocations.values()
        )


    @property
    def wasted_tokens(self) -> int:
        return (
            self.reserved_tokens - self.used_tokens
        )


    @property
    def waste_ratio(self) -> float:
        
        if self.reserved_tokens == 0:
            return 0.0

        return (
            self.wasted_tokens
            / self.reserved_tokens
        )
