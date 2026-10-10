"""Compare dense attention with blockwise online softmax, without GPU claims."""

from __future__ import annotations

import math
import random

Matrix = list[list[float]]


def validate_inputs(queries: Matrix, keys: Matrix, values: Matrix) -> tuple[int, int]:
    """Validate finite, rectangular, equally long self-attention inputs."""
    if not queries or len(queries) != len(keys) or len(keys) != len(values):
        raise ValueError("This example requires non-empty, equally long Q, K, and V.")
    key_dimension, value_dimension = len(keys[0]), len(values[0])
    if key_dimension == 0 or value_dimension == 0:
        raise ValueError("Vector dimensions must be positive.")
    if any(len(row) != key_dimension for row in queries + keys):
        raise ValueError("Q and K must have matching, consistent dimensions.")
    if any(len(row) != value_dimension for row in values):
        raise ValueError("V must have a consistent dimension.")
    if not all(math.isfinite(value) for row in queries + keys + values for value in row):
        raise ValueError("All coordinates must be finite.")
    return key_dimension, value_dimension


def scaled_score(query: list[float], key: list[float], scale: float) -> float:
    """Calculate one finite scaled dot product."""
    score = math.fsum(a * b for a, b in zip(query, key)) / scale
    if not math.isfinite(score):
        raise ValueError("The dot product overflowed; reduce input magnitudes.")
    return score


def dense_attention(queries: Matrix, keys: Matrix, values: Matrix, *, causal: bool) -> Matrix:
    """Materialize an N-by-N score matrix as a small reference implementation."""
    key_dimension, value_dimension = validate_inputs(queries, keys, values)
    scale = math.sqrt(key_dimension)
    scores = [
        [
            scaled_score(query, key, scale) if not causal or key_index <= query_index
            else -math.inf
            for key_index, key in enumerate(keys)
        ]
        for query_index, query in enumerate(queries)
    ]

    outputs: Matrix = []
    for row in scores:
        largest = max(row)
        exponentials = [math.exp(score - largest) for score in row]
        denominator = math.fsum(exponentials)
        outputs.append([
            math.fsum(weight * value[column] for weight, value in zip(exponentials, values))
            / denominator
            for column in range(value_dimension)
        ])
    return outputs


def online_attention(
    queries: Matrix,
    keys: Matrix,
    values: Matrix,
    *,
    block_size: int,
    causal: bool,
) -> Matrix:
    """Maintain a running maximum, normalizer, and weighted-value numerator."""
    key_dimension, value_dimension = validate_inputs(queries, keys, values)
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    scale = math.sqrt(key_dimension)
    outputs: Matrix = []

    for query_index, query in enumerate(queries):
        largest = -math.inf
        denominator = 0.0
        numerator = [0.0] * value_dimension
        visible_count = query_index + 1 if causal else len(keys)

        for start in range(0, visible_count, block_size):
            stop = min(start + block_size, visible_count)
            block_scores = [scaled_score(query, keys[index], scale) for index in range(start, stop)]
            new_largest = max(largest, max(block_scores))
            # Recenter all earlier contributions when a later block raises the maximum.
            correction = math.exp(largest - new_largest)
            block_weights = [math.exp(score - new_largest) for score in block_scores]
            denominator = correction * denominator + math.fsum(block_weights)
            numerator = [
                correction * numerator[column]
                + math.fsum(
                    weight * values[index][column]
                    for index, weight in zip(range(start, stop), block_weights)
                )
                for column in range(value_dimension)
            ]
            largest = new_largest

        outputs.append([value / denominator for value in numerator])
    return outputs


def maximum_error(left: Matrix, right: Matrix) -> float:
    """Compare identically shaped outputs."""
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right)):
        raise ValueError("Output shapes must match.")
    return max(abs(a - b) for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b))


def main() -> None:
    rng = random.Random(7)
    token_count = 8
    queries = [[rng.uniform(-2, 2) for _ in range(4)] for _ in range(token_count)]
    keys = [[rng.uniform(-2, 2) for _ in range(4)] for _ in range(token_count)]
    values = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(token_count)]

    for causal in (False, True):
        reference = dense_attention(queries, keys, values, causal=causal)
        for block_size in (1, 3, 8, 16):
            output = online_attention(queries, keys, values, block_size=block_size, causal=causal)
            error = maximum_error(reference, output)
            assert error < 1e-12
            print(f"causal={causal}, block_size={block_size:2d}, maximum error={error:.3e}")

    large_queries = [[1000.0], [1000.0]]
    large_keys = [[1000.0], [1001.0]]
    large_values = [[1.0], [2.0]]
    stable_output = online_attention(
        large_queries, large_keys, large_values, block_size=1, causal=False,
    )
    assert maximum_error(stable_output, [[2.0], [2.0]]) < 1e-12
    print("Large-logit stability check passed.")
    print(f"Dense reference stores {token_count ** 2} score entries.")
    print("Online version keeps one score block plus running accumulators per query.")
    print("This is a CPU formula demonstration, not a FlashAttention performance benchmark.")


if __name__ == "__main__":
    main()
