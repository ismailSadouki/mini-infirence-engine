from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from bench.runner import run_benchmark

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the mini-engine latency-vs-throughput "
            "benchmark at batch 1, 8, and 32."
        )
    )

    parser.add_argument(
        "--max-total-active-tokens",
        type=int,
        default=32768,
        help=(
            "Maximum number of active tokens "
            "allowed by the scheduler."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bench/latency_throughput_curve.png",
    )

    args = parser.parse_args()

    workloads = [
        (
            1,
            "configs/workloads/batch1.yaml",
        ),
        (
            8,
            "configs/workloads/batch8.yaml",
        ),
        (
            32,
            "configs/workloads/batch32.yaml",
        ),
    ]

    results = []

    print()
    print("=" * 80)
    print("M4.3 · LATENCY VS THROUGHPUT")
    print("=" * 80)


    for batch_size, workload_path in workloads:
        print()
        print(
            f"\n{'#' * 80}"
        )
        print(
            f"BATCH / CONCURRENCY = {batch_size}"
        )
        print(
            f"{'#' * 80}"
        )
        metrics = run_benchmark(
            workload_path=workload_path,
            max_total_active_tokens=(
                args.max_total_active_tokens
            ),
        )
        results.append(
            {
                "batch": batch_size,
                "ttft_p50": metrics.ttft_p50,
                "ttft_p95": metrics.ttft_p95,
                "itl_p50": metrics.itl_p50,
                "itl_p95": metrics.itl_p95,
                "throughput": metrics.throughput,
                "peak_gpu_memory_mb": metrics.peak_gpu_memory_mb,
            }
        )

    # Final comparison table

    print()
    print()
    print("=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)

    print(
        f"{'Batch':>8}"
        f"{'TTFT p50':>14}"
        f"{'TTFT p95':>14}"
        f"{'ITL p50':>14}"
        f"{'ITL p95':>14}"
        f"{'Throughput':>14}"
        f"{'peak_gpu_memory_mb':>18}"
    )

    print("-" * 100)

    for result in results:

        print(
            f"{result['batch']:>8}"
            f"{result['ttft_p50'] * 1000:>13.2f} ms"
            f"{result['ttft_p95'] * 1000:>13.2f} ms"
            f"{result['itl_p50'] * 1000:>13.2f} ms"
            f"{result['itl_p95'] * 1000:>13.2f} ms"
            f"{result['throughput']:>18.2f} tok/s"
            f"{result['peak_gpu_memory_mb']:>18.2f} MB"
        )

    print("=" * 100)

    # Plot


    import matplotlib.pyplot as plt

    throughputs = [
        result["throughput"]
        for result in results
    ]

    ttft_p50 = [
        result["ttft_p50"] * 1000
        for result in results
    ]

    ttft_p95 = [
        result["ttft_p95"] * 1000
        for result in results
    ]

    batches = [
        result["batch"]
        for result in results
    ]

    plt.figure(figsize=(9, 6))


    plt.plot(
        throughputs,
        ttft_p50,
        marker="o",
        label="TTFT p50",
    )

    plt.plot(
        throughputs,
        ttft_p95,
        marker="o",
        label="TTFT p95",
    )

    for x, y, batch in zip(
        throughputs,
        ttft_p50,
        batches,
    ):
        plt.annotate(
            f"batch={batch}",
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
        )


    for x, y, batch in zip(
        throughputs,
        ttft_p95,
        batches,
    ):
        plt.annotate(
            f"batch={batch}",
            (x, y),
            xytext=(6, -14),
            textcoords="offset points",
        )

    plt.xlabel(
        "Throughput (output tokens/s)"
    )

    plt.ylabel(
        "TTFT (ms)"
    )

    plt.title(
        "Mini Inference Engine: Latency vs Throughput"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print()
    print(
        f"Chart saved to: {output_path}"
    )

if __name__ == "__main__":
    main()
