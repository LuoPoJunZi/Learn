"""Compare naive autoregressive decoding with a tiny key/value cache."""

from __future__ import annotations

import math
from dataclasses import dataclass

Vector = list[float]
Matrix = list[Vector]

QUERY_WEIGHT: Matrix = [
    [0.8, 0.2, 0.0],
    [0.1, 0.7, 0.2],
    [0.0, 0.3, 0.9],
]
KEY_WEIGHT: Matrix = [
    [0.6, 0.1, 0.2],
    [0.2, 0.8, 0.0],
    [0.1, 0.2, 0.7],
]
VALUE_WEIGHT: Matrix = [
    [0.7, 0.0, 0.3],
    [0.2, 0.9, 0.1],
    [0.1, 0.2, 0.8],
]


@dataclass
class ProjectionCounts:
    """Track how often each projection is evaluated."""

    query: int = 0
    key: int = 0
    value: int = 0


def dot(left: Vector, right: Vector) -> float:
    """Return a vector dot product."""
    return sum(a * b for a, b in zip(left, right))


def project(vector: Vector, weight: Matrix) -> Vector:
    """Apply a small row-vector matrix projection."""
    if not weight or any(len(row) != len(weight) for row in weight):
        raise ValueError("This demo expects a non-empty square weight matrix.")
    if len(vector) != len(weight):
        raise ValueError("The vector and matrix dimensions must match.")
    return [
        sum(vector[row] * weight[row][column] for row in range(len(vector)))
        for column in range(len(vector))
    ]


def softmax(values: Vector) -> Vector:
    """Return stable softmax weights."""
    largest = max(values)
    exponentials = [math.exp(value - largest) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def attend(query: Vector, keys: Matrix, values: Matrix) -> Vector:
    """Attend from one decoding query to the available prefix."""
    scale = math.sqrt(len(query))
    weights = softmax([dot(query, key) / scale for key in keys])
    return [
        sum(weight * values[index][column] for index, weight in enumerate(weights))
        for column in range(len(values[0]))
    ]


def decode_without_cache(embeddings: Matrix) -> tuple[Matrix, ProjectionCounts]:
    """Recompute all prefix keys and values at every decoding step."""
    counts = ProjectionCounts()
    outputs: Matrix = []

    for end in range(1, len(embeddings) + 1):
        prefix = embeddings[:end]
        query = project(prefix[-1], QUERY_WEIGHT)
        counts.query += 1

        keys = []
        values = []
        for embedding in prefix:
            keys.append(project(embedding, KEY_WEIGHT))
            values.append(project(embedding, VALUE_WEIGHT))
            counts.key += 1
            counts.value += 1

        outputs.append(attend(query, keys, values))

    return outputs, counts


def decode_with_cache(embeddings: Matrix) -> tuple[Matrix, ProjectionCounts]:
    """Project each new key and value once, then reuse the cached prefix."""
    counts = ProjectionCounts()
    cached_keys: Matrix = []
    cached_values: Matrix = []
    outputs: Matrix = []

    for embedding in embeddings:
        query = project(embedding, QUERY_WEIGHT)
        new_key = project(embedding, KEY_WEIGHT)
        new_value = project(embedding, VALUE_WEIGHT)
        counts.query += 1
        counts.key += 1
        counts.value += 1

        cached_keys.append(new_key)
        cached_values.append(new_value)
        outputs.append(attend(query, cached_keys, cached_values))

    return outputs, counts


def outputs_match(left: Matrix, right: Matrix, tolerance: float = 1e-12) -> bool:
    """Check that both decoding strategies produce the same toy outputs."""
    return all(
        abs(a - b) <= tolerance
        for left_row, right_row in zip(left, right)
        for a, b in zip(left_row, right_row)
    )


def main() -> int:
    tokens = ["the", "cache", "reuses", "past", "states"]
    embeddings = [
        [1.0, 0.1, 0.0],
        [0.3, 1.0, 0.2],
        [0.1, 0.4, 1.0],
        [0.8, 0.5, 0.2],
        [0.2, 0.9, 0.7],
    ]

    naive_outputs, naive_counts = decode_without_cache(embeddings)
    cached_outputs, cached_counts = decode_with_cache(embeddings)

    print("Autoregressive tokens:", tokens)
    print("Outputs match:", outputs_match(naive_outputs, cached_outputs))
    print("\nProjection counts")
    print("  without cache:", naive_counts)
    print("  with cache:   ", cached_counts)

    avoided = (
        naive_counts.key
        + naive_counts.value
        - cached_counts.key
        - cached_counts.value
    )
    print("Avoided repeated K/V projections:", avoided)
    print("\nFinal context vector:")
    print(" ", [round(value, 4) for value in cached_outputs[-1]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
