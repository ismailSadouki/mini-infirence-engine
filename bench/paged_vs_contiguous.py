from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import yaml



CONFIG_PATH = "configs/inference.yaml"

BLOCK_SIZES = [16, 32, 64]

# Workload length distributions.
# These are prompt/current sequence lengths in tokens.
import random


def make_workloads(
    num_sequences=1000,
    seed=2026,
):
    rng = random.Random(seed)

    many_short = [
        rng.randint(8, 64)
        for _ in range(num_sequences)
    ]

    mixed = [
        rng.choice(
            [
                rng.randint(8, 64),
                rng.randint(65, 256),
                rng.randint(257, 768),
                rng.randint(769, 1536),
                rng.randint(1537, 2048),
            ]
        )
        for _ in range(num_sequences)
    ]

    many_long = [
        rng.randint(512, 2048)
        for _ in range(num_sequences)
    ]

    return {
        "many_short": many_short,
        "mixed": mixed,
        "many_long": many_long,
    }

DISTRIBUTIONS = make_workloads()

# Fixed KV-memory budget.
MEMORY_BUDGET_GIB = 1.0



with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

MAX_SEQ_LEN = 2048


@dataclass
class KVConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int


@dataclass
class WorkloadResult:
    distribution: str
    num_sequences: int
    total_tokens: int

    contiguous_reserved_bytes: int
    contiguous_used_bytes: int

    paged_reserved_bytes: int
    paged_used_bytes: int

    block_size: int


def kv_bytes_per_token(kv: KVConfig) -> int:
    """
    Bytes required for ONE token of KV cache.
    """

    return (
        2
        * kv.num_layers
        * kv.num_kv_heads
        * kv.head_dim
        * kv.dtype_bytes
    )



def contiguous_used_bytes(
    lengths: list[int],
    kv: KVConfig,
) -> int:
    """
    Actual KV bytes required by the tokens.
    """

    return sum(lengths) * kv_bytes_per_token(kv)


def contiguous_reserved_bytes(
    lengths: list[int],
    kv: KVConfig,
    max_seq_len: int,
) -> int:
    """
    Contiguous per-sequence allocation.

    Every sequence gets:

        max_seq_len

    positions regardless of its actual length.
    """

    per_sequence = (
        max_seq_len
        * kv_bytes_per_token(kv)
    )

    return len(lengths) * per_sequence

def paged_used_bytes(
    lengths: list[int],
    kv: KVConfig,
) -> int:
    """
    Actual KV bytes required by real tokens.

    Same as contiguous used bytes.

    Paged KV does NOT reduce the actual bytes required
    to store real K/V tokens.
    """

    return sum(lengths) * kv_bytes_per_token(kv)



def paged_reserved_bytes(
    lengths: list[int],
    kv: KVConfig,
    block_size: int,
) -> int:
    """
    Paged allocation.

    Each sequence gets:

        ceil(length / block_size)

    blocks.

    The final block may be partially empty.
    """

    allocated_blocks = sum(
        math.ceil(length / block_size)
        for length in lengths
    )

    return (
        allocated_blocks
        * block_size
        * kv_bytes_per_token(kv)
    )


# Capacity calculations
def max_contiguous_sequences(
    kv: KVConfig,
    memory_budget_bytes: int,
    max_seq_len: int,
) -> int:
    """
    Maximum number of sequences when each sequence
    reserves a full max_seq_len KV buffer.
    """

    per_sequence = (
        max_seq_len
        * kv_bytes_per_token(kv)
    )

    return memory_budget_bytes // per_sequence




def average_paged_bytes_per_sequence(
    lengths: list[int],
    kv: KVConfig,
    block_size: int,
) -> float:

    total = sum(
        math.ceil(length / block_size)
        * block_size
        * kv_bytes_per_token(kv)
        for length in lengths
    )

    return total / len(lengths)


def estimate_paged_capacity(
    lengths: list[int],
    kv: KVConfig,
    memory_budget_bytes: int,
    block_size: int,
) -> int:

    avg_bytes = average_paged_bytes_per_sequence(
        lengths,
        kv,
        block_size,
    )

    return int(
        memory_budget_bytes // avg_bytes
    )


def mib(value: int) -> float:
    return value / (1024 ** 2)


def gib(value: int) -> float:
    return value / (1024 ** 3)



def main():


    kv = KVConfig(
        num_layers=config["kv_cache"]["num_layers"],
        num_kv_heads=config["kv_cache"]["num_kv_heads"],
        head_dim=config["kv_cache"]["head_dim"],
        dtype_bytes=2,  # BF16
    )

    budget = int(
        MEMORY_BUDGET_GIB
        * (1024 ** 3)
    )

    print("=" * 72)
    print("PAGED VS CONTIGUOUS KV MEMORY EXPERIMENT")
    print("=" * 72)

    print()
    print(f"KV bytes/token : {kv_bytes_per_token(kv):,}")
    print(f"max_seq_len    : {MAX_SEQ_LEN}")
    print(f"memory budget  : {MEMORY_BUDGET_GIB:.2f} GiB")
    print()

    # Experiment 1:
        # Workload distributions
    for name, lengths in DISTRIBUTIONS.items():

        print("-" * 72)
        print(f"WORKLOAD: {name}")
        print("-" * 72)

        total_tokens = sum(lengths)

        contiguous_reserved = (
            contiguous_reserved_bytes(
                lengths,
                kv,
                MAX_SEQ_LEN,
            )
        )

        contiguous_used = (
            contiguous_used_bytes(
                lengths,
                kv,
            )
        )


        print(f"Sequences       : {len(lengths)}")
        print(f"Total tokens    : {total_tokens}")
        print(
            f"Contig reserved : "
            f"{mib(contiguous_reserved):.2f} MiB"
        )
        print(
            f"Contig used     : "
            f"{mib(contiguous_used):.2f} MiB"
        )

        print()

        for block_size in BLOCK_SIZES:

            paged_reserved = (
                paged_reserved_bytes(
                    lengths,
                    kv,
                    block_size,
                )
            )

            paged_used = (
                paged_used_bytes(
                    lengths,
                    kv,
                )
            )

            savings = (
                1
                - paged_reserved
                / contiguous_reserved
            )

            print(
                f"Block {block_size:>2}: "
                f"reserved={mib(paged_reserved):8.2f} MiB  "
                f"used={mib(paged_used):8.2f} MiB  "
                f"savings={savings * 100:6.2f}%"
            )

        print()


    # Experiment 2:
        # Capacity under fixed budget
    print("=" * 72)
    print("CONCURRENT SEQUENCE CAPACITY")
    print("=" * 72)

    contiguous_capacity = (
        max_contiguous_sequences(
            kv=kv,
            memory_budget_bytes=budget,
            max_seq_len=MAX_SEQ_LEN,
        )
    )

    print(
        f"\nContiguous capacity: "
        f"{contiguous_capacity} sequences"
    )

    capacity_results = {}



    for name, lengths in DISTRIBUTIONS.items():

        capacity_results[name] = {}

        print()
        print(f"{name}:")

        for block_size in BLOCK_SIZES:

            capacity = estimate_paged_capacity(
                lengths=lengths,
                kv=kv,
                memory_budget_bytes=budget,
                block_size=block_size,
            )

            capacity_results[name][block_size] = capacity

            capacity_gain = capacity / contiguous_capacity

            print(
                f"  block_size={block_size:>2}: "
                f"{capacity} sequences "
                f"({capacity_gain:.2f}x contiguous)"
            )



    # Experiment 3:
    # Fragmentation / block overhead
    print()
    print("=" * 72)
    print("BLOCK-SIZE TRADE-OFF")
    print("=" * 72)

    for name, lengths in DISTRIBUTIONS.items():

        print()
        print(name)

        actual = contiguous_used_bytes(
            lengths,
            kv,
        )

        for block_size in BLOCK_SIZES:

            reserved = paged_reserved_bytes(
                lengths,
                kv,
                block_size,
            )

            waste = reserved - actual

            waste_ratio = (
                waste / reserved
                if reserved > 0
                else 0
            )

            print(
                f"  block={block_size:>2}: "
                f"waste={mib(waste):8.2f} MiB "
                f"({waste_ratio * 100:6.2f}%)"
            )
    # Chart

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = range(len(DISTRIBUTIONS))

    for block_size in BLOCK_SIZES:

        values = [
            capacity_results[name][block_size]
            for name in DISTRIBUTIONS
        ]
        ax.plot(
            list(x),
            values,
            marker="o",
            label=f"Block {block_size}",
        )

    ax.axhline(
        contiguous_capacity,
        linestyle="--",
        label="Contiguous",
    )

    ax.set_xticks(
        list(x),
        list(DISTRIBUTIONS.keys()),
    )

    ax.set_ylabel(
        "Concurrent sequences"
    )

    ax.set_xlabel(
        "Workload distribution"
    )
    ax.set_title(
        "Paged vs Contiguous KV Capacity"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = (
        "bench/paged_vs_contiguous_capacity.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    print()
    print(
        f"Chart saved to: {output_path}"
    )


if __name__ == "__main__":
    main()