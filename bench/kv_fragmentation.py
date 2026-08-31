from __future__ import annotations

from dataclasses import dataclass



from pathlib import Path
import sys



sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)



from engine.contiguous_cache import ContiguousKVCache


MAX_SEQ_LENS = [
    512,
    1024,
    2048,
]



ACTIVE_LENGTHS = [
    64,
    128,
    256,
    384,
]


@dataclass
class FragmentationResult:
    max_seq_len: int
    num_requests: int
    reserved_tokens: int
    used_tokens: int
    wasted_tokens: int
    waste_ratio: float


def run_fragmentation_experiment(
    max_seq_len: int,
    active_lengths: list[int],
) -> FragmentationResult:

    cache = ContiguousKVCache(
        max_seq_len=max_seq_len
    )

    for request_idx, used_tokens in enumerate(
        active_lengths
    ):
        cache.allocate(
            request_id=f"r{request_idx + 1}",
            used_tokens=used_tokens,
        )

    return FragmentationResult(
        max_seq_len=max_seq_len,
        num_requests=len(active_lengths),
        reserved_tokens=cache.reserved_tokens,
        used_tokens=cache.used_tokens,
        wasted_tokens=cache.wasted_tokens,
        waste_ratio=cache.waste_ratio,
    )


def print_result(result: FragmentationResult) -> None:

    print(
        f"{result.max_seq_len:<15}"
        f"{result.num_requests:<12}"
        f"{result.reserved_tokens:<18}"
        f"{result.used_tokens:<15}"
        f"{result.wasted_tokens:<15}"
        f"{result.waste_ratio * 100:>10.2f}%"
    )


def main() -> None:

    print()
    print("Contiguous KV Cache Fragmentation")
    print("=" * 80)

    print()
    print("Active sequence lengths:")
    print(ACTIVE_LENGTHS)

    print()
    print(
        f"{'max_seq_len':<15}"
        f"{'requests':<12}"
        f"{'reserved':<18}"
        f"{'used':<15}"
        f"{'wasted':<15}"
        f"{'waste ratio':>12}"
    )

    print("-" * 80)

    for max_seq_len in MAX_SEQ_LENS:

        result = run_fragmentation_experiment(
            max_seq_len=max_seq_len,
            active_lengths=ACTIVE_LENGTHS,
        )

        print_result(result)


if __name__ == "__main__":
    main()