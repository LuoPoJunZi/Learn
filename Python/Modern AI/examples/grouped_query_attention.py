"""Explain grouped-query attention with plain Python lists."""

from __future__ import annotations

import math

Vector = list[float]
Head = list[Vector]
Heads = list[Head]


def dot(left: Vector, right: Vector) -> float:
    """Return the dot product of two equally sized vectors."""
    if len(left) != len(right):
        raise ValueError("Vectors must have the same length.")
    return sum(a * b for a, b in zip(left, right))


def softmax(values: Vector) -> Vector:
    """Convert scores into stable probability-like weights."""
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        raise ValueError("Softmax needs at least one finite value.")

    largest = max(finite_values)
    exponentials = [
        math.exp(value - largest) if math.isfinite(value) else 0.0
        for value in values
    ]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def attend(
    queries: Head,
    keys: Head,
    values: Head,
    *,
    causal: bool,
) -> tuple[Head, Head]:
    """Apply one query head to one shared key/value head."""
    if not queries or not keys or len(keys) != len(values):
        raise ValueError("Queries, keys, and values must be non-empty and aligned.")

    key_dimension = len(keys[0])
    value_dimension = len(values[0])
    if any(len(vector) != key_dimension for vector in queries + keys):
        raise ValueError("Query and key dimensions must match.")
    if any(len(vector) != value_dimension for vector in values):
        raise ValueError("All value vectors must have the same dimension.")

    scale = math.sqrt(key_dimension)
    outputs: Head = []
    all_weights: Head = []

    for query_index, query in enumerate(queries):
        scores = []
        for key_index, key in enumerate(keys):
            score = dot(query, key) / scale
            if causal and key_index > query_index:
                score = -math.inf
            scores.append(score)

        weights = softmax(scores)
        output = [
            sum(weight * values[index][column] for index, weight in enumerate(weights))
            for column in range(value_dimension)
        ]
        outputs.append(output)
        all_weights.append(weights)

    return outputs, all_weights


def grouped_query_attention(
    query_heads: Heads,
    key_heads: Heads,
    value_heads: Heads,
    *,
    causal: bool = True,
) -> tuple[Heads, Heads, list[int]]:
    """Map several query heads to each key/value head."""
    if not query_heads or not key_heads or len(key_heads) != len(value_heads):
        raise ValueError("Query and key/value heads must be non-empty and aligned.")
    if len(query_heads) % len(key_heads) != 0:
        raise ValueError("The query-head count must be divisible by the KV-head count.")

    group_size = len(query_heads) // len(key_heads)
    outputs: Heads = []
    weights: Heads = []
    mappings: list[int] = []

    for query_head_index, queries in enumerate(query_heads):
        kv_head_index = query_head_index // group_size
        head_output, head_weights = attend(
            queries,
            key_heads[kv_head_index],
            value_heads[kv_head_index],
            causal=causal,
        )
        outputs.append(head_output)
        weights.append(head_weights)
        mappings.append(kv_head_index)

    return outputs, weights, mappings


def main() -> int:
    tokens = ["learn", "modern", "ai"]
    query_heads = [
        [[1.0, 0.1], [0.3, 1.0], [0.9, 0.8]],
        [[0.8, 0.2], [0.1, 1.1], [0.7, 1.0]],
        [[0.2, 1.0], [1.0, 0.3], [0.8, 0.9]],
        [[0.1, 0.9], [1.1, 0.2], [1.0, 0.7]],
    ]
    key_heads = [
        [[1.0, 0.0], [0.2, 1.0], [0.8, 0.7]],
        [[0.4, 0.8], [1.0, 0.1], [0.3, 1.0]],
    ]
    value_heads = [
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        [[0.3, 0.9], [0.9, 0.2], [0.5, 1.0]],
    ]

    outputs, weights, mappings = grouped_query_attention(
        query_heads,
        key_heads,
        value_heads,
    )

    print("Tokens:", tokens)
    print("Query heads:", len(query_heads))
    print("Key/value heads:", len(key_heads))
    print("Query -> KV mapping:", mappings)

    last_token = len(tokens) - 1
    for head_index, kv_head_index in enumerate(mappings):
        rounded_weights = [round(value, 3) for value in weights[head_index][last_token]]
        rounded_output = [round(value, 3) for value in outputs[head_index][last_token]]
        print(
            f"Q head {head_index} uses KV head {kv_head_index}: "
            f"weights={rounded_weights}, output={rounded_output}"
        )

    token_count = len(tokens)
    head_dimension = len(key_heads[0][0])
    mha_cache_values = token_count * len(query_heads) * head_dimension * 2
    gqa_cache_values = token_count * len(key_heads) * head_dimension * 2
    print("\nCached K/V scalar count for ordinary MHA:", mha_cache_values)
    print("Cached K/V scalar count for this GQA setup:", gqa_cache_values)
    print("Cache ratio:", f"{gqa_cache_values / mha_cache_values:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
