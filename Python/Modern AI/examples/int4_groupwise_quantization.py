"""Demonstrate symmetric groupwise INT4 weight quantization.

The example calculates quantized codes, dequantized weights, output error, and a
theoretical storage estimate. Python integers are not actually packed into 4 bits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizedGroup:
    """Hold signed INT4-like codes and one floating-point scale."""

    values: tuple[int, ...]
    scale: float


def quantize_group(values: list[float]) -> QuantizedGroup:
    """Quantize one group to the symmetric range -7 through 7."""

    max_abs = max((abs(value) for value in values), default=0.0)
    if max_abs == 0.0:
        return QuantizedGroup(tuple(0 for _ in values), 1.0)

    scale = max_abs / 7.0
    quantized = tuple(
        max(-7, min(7, round(value / scale))) for value in values
    )
    return QuantizedGroup(quantized, scale)


def quantize_matrix(
    matrix: list[list[float]],
    group_size: int,
) -> list[list[QuantizedGroup]]:
    """Split every row into groups and quantize each group independently."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return [
        [
            quantize_group(row[start : start + group_size])
            for start in range(0, len(row), group_size)
        ]
        for row in matrix
    ]


def dequantize_matrix(
    matrix: list[list[QuantizedGroup]],
) -> list[list[float]]:
    """Reconstruct approximate floating-point weights from codes and scales."""

    reconstructed: list[list[float]] = []
    for row in matrix:
        values: list[float] = []
        for group in row:
            values.extend(code * group.scale for code in group.values)
        reconstructed.append(values)
    return reconstructed


def matrix_vector_product(
    matrix: list[list[float]],
    vector: list[float],
) -> list[float]:
    """Multiply a small dense matrix by one vector."""

    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("each matrix row must match the vector length")
    return [sum(weight * value for weight, value in zip(row, vector)) for row in matrix]


def mean_absolute_error(left: list[float], right: list[float]) -> float:
    """Return the mean absolute difference between two equally sized lists."""

    if len(left) != len(right):
        raise ValueError("inputs must have equal lengths")
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def storage_bits(
    quantized: list[list[QuantizedGroup]],
) -> tuple[int, int]:
    """Estimate FP32 and packed INT4-plus-FP32-scale storage."""

    weight_count = sum(
        len(group.values) for row in quantized for group in row
    )
    group_count = sum(len(row) for row in quantized)
    fp32_bits = weight_count * 32
    int4_bits = weight_count * 4 + group_count * 32
    return fp32_bits, int4_bits


def main() -> None:
    """Quantize a tiny linear layer and compare its outputs."""

    weights = [
        [0.90, -0.40, 0.15, 0.70, -1.20, 0.35, 0.05, 0.80],
        [-0.25, 0.55, 1.10, -0.75, 0.45, -0.10, 0.65, -0.95],
    ]
    inputs = [0.80, -0.30, 0.50, 1.00, -0.20, 0.40, 0.70, -0.60]
    group_size = 4

    quantized = quantize_matrix(weights, group_size)
    reconstructed = dequantize_matrix(quantized)

    original_output = matrix_vector_product(weights, inputs)
    quantized_output = matrix_vector_product(reconstructed, inputs)
    fp32_bits, int4_bits = storage_bits(quantized)

    print(f"Teaching group size: {group_size}")
    for row_index, row in enumerate(quantized):
        print(f"Row {row_index} quantized groups:")
        for group_index, group in enumerate(row):
            scale = f"{group.scale:.6f}"
            print(f"  group {group_index}: codes={group.values}, scale={scale}")

    print("Original output:     ", [round(value, 6) for value in original_output])
    print("Reconstructed output:", [round(value, 6) for value in quantized_output])
    error = mean_absolute_error(original_output, quantized_output)
    print(f"Output mean absolute error: {error:.6f}")
    print(f"Theoretical FP32 storage: {fp32_bits} bits")
    print(f"Packed INT4 + FP32 scales: {int4_bits} bits")
    print(f"Estimated storage ratio: {int4_bits / fp32_bits:.2%}")


if __name__ == "__main__":
    main()
