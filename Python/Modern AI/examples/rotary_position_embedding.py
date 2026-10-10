"""Explain RoPE through paired rotations and relative-position identities."""

from __future__ import annotations

import math


def rotary_embedding(
    vector: list[float],
    position: int,
    *,
    base: float = 10000.0,
) -> list[float]:
    """Rotate adjacent coordinate pairs at position-dependent frequencies."""
    if not vector or len(vector) % 2:
        raise ValueError("RoPE requires a non-empty, even-dimensional vector.")
    if not math.isfinite(base) or base <= 0:
        raise ValueError("base must be finite and positive.")
    if not isinstance(position, int):
        raise ValueError("position must be an integer.")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Vector coordinates must be finite.")

    dimension = len(vector)
    rotated: list[float] = []
    for start in range(0, dimension, 2):
        frequency = base ** (-start / dimension)
        angle = position * frequency
        cosine, sine = math.cos(angle), math.sin(angle)
        left, right = vector[start : start + 2]
        rotated.extend([left * cosine - right * sine, left * sine + right * cosine])
    return rotated


def dot(left: list[float], right: list[float]) -> float:
    """Return the dot product without silently truncating mismatched vectors."""
    if len(left) != len(right):
        raise ValueError("Vectors must have equal dimensions.")
    return math.fsum(a * b for a, b in zip(left, right))


def main() -> None:
    query = [1.0, 0.5, -0.2, 0.8]
    key = [0.3, 1.0, 0.7, -0.4]
    query_position, key_position = 2, 5
    shift = 10

    rotated_query = rotary_embedding(query, query_position)
    rotated_key = rotary_embedding(key, key_position)
    score = dot(rotated_query, rotated_key)
    shifted_score = dot(
        rotary_embedding(query, query_position + shift),
        rotary_embedding(key, key_position + shift),
    )
    relative_score = dot(query, rotary_embedding(key, key_position - query_position))
    norm_error = abs(dot(query, query) - dot(rotated_query, rotated_query))

    assert norm_error < 1e-12
    assert math.isclose(score, shifted_score, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(score, relative_score, rel_tol=1e-12, abs_tol=1e-12)

    print("Original query:", query)
    print("Rotated query:", [round(value, 6) for value in rotated_query])
    print(f"Squared-norm error: {norm_error:.3e}")
    print(f"Score at positions ({query_position}, {key_position}): {score:.6f}")
    print(f"Score after shifting both positions by {shift}: {shifted_score:.6f}")
    print(f"Score using only the relative offset: {relative_score:.6f}")
    print("\nRelative offset -> rotated dot product (before attention scaling)")
    for offset in range(6):
        print(f"  {offset}: {dot(query, rotary_embedding(key, offset)):.6f}")
    print("Identity checks passed. No model training is performed.")


if __name__ == "__main__":
    main()
