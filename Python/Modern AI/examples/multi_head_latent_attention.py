"""Demonstrate Multi-Head Latent Attention (MLA) KV cache compression."""

from __future__ import annotations

import math
from dataclasses import dataclass

Vector = list[float]
Matrix = list[Vector]


def _check_finite_vector(vector: Vector, name: str) -> None:
    if not vector:
        raise ValueError(f"{name} must not be empty.")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite numbers.")


def _check_matrix(matrix: Matrix, expected_rows: int, expected_cols: int, name: str) -> None:
    if len(matrix) != expected_rows:
        raise ValueError(f"{name} expected {expected_rows} rows, got {len(matrix)}.")
    for row in matrix:
        if len(row) != expected_cols:
            raise ValueError(f"{name} expected {expected_cols} columns, got {len(row)}.")
        if any(not math.isfinite(value) for value in row):
            raise ValueError(f"{name} must contain only finite numbers.")


def matvec(matrix: Matrix, vector: Vector) -> Vector:
    """Multiply a 2D matrix by a 1D vector."""
    if not matrix or len(matrix[0]) != len(vector):
        raise ValueError("Matrix columns must match vector length.")
    return [sum(weight * x for weight, x in zip(row, vector)) for row in matrix]


def dot(left: Vector, right: Vector) -> float:
    """Compute the dot product of two equal-length vectors."""
    if len(left) != len(right):
        raise ValueError("Vectors must have equal length.")
    return sum(a * b for a, b in zip(left, right))


def softmax(scores: Vector) -> Vector:
    """Compute numerically stable softmax probabilities."""
    _check_finite_vector(scores, "scores")
    peak = max(scores)
    exps = [math.exp(value - peak) for value in scores]
    total = sum(exps)
    return [value / total for value in exps]


@dataclass(frozen=True)
class MLAConfig:
    """Configuration for a tiny teaching Multi-Head Latent Attention block."""

    d_model: int
    num_heads: int
    head_dim: int
    kv_latent_dim: int

    def __post_init__(self) -> None:
        dims = (self.d_model, self.num_heads, self.head_dim, self.kv_latent_dim)
        if any(dim <= 0 for dim in dims):
            raise ValueError("All MLA dimensions must be positive integers.")
        if self.kv_latent_dim >= 2 * self.num_heads * self.head_dim:
            raise ValueError(
                "kv_latent_dim should be smaller than full K+V dimension to compress cache."
            )


@dataclass(frozen=True)
class MLAWeights:
    """Projection matrices for down-projecting KV into latent space and up-projecting per head."""

    w_q: list[Matrix]  # [num_heads][head_dim][d_model]
    w_dkv: Matrix      # [kv_latent_dim][d_model]
    w_uk: list[Matrix] # [num_heads][head_dim][kv_latent_dim]
    w_uv: list[Matrix] # [num_heads][head_dim][kv_latent_dim]

    def validate(self, config: MLAConfig) -> None:
        if len(self.w_q) != config.num_heads:
            raise ValueError("w_q head count must match config.num_heads.")
        if len(self.w_uk) != config.num_heads or len(self.w_uv) != config.num_heads:
            raise ValueError("w_uk and w_uv head counts must match config.num_heads.")
        _check_matrix(self.w_dkv, config.kv_latent_dim, config.d_model, "w_dkv")
        for head in range(config.num_heads):
            _check_matrix(self.w_q[head], config.head_dim, config.d_model, f"w_q[{head}]")
            _check_matrix(self.w_uk[head], config.head_dim, config.kv_latent_dim, f"w_uk[{head}]")
            _check_matrix(self.w_uv[head], config.head_dim, config.kv_latent_dim, f"w_uv[{head}]")


def compress_to_kv_latent(tokens: Matrix, weights: MLAWeights, config: MLAConfig) -> Matrix:
    """Compress each token representation into a shared low-dimensional KV latent vector."""
    if not tokens:
        raise ValueError("tokens must not be empty.")
    weights.validate(config)
    latents: Matrix = []
    for index, token in enumerate(tokens):
        _check_finite_vector(token, f"tokens[{index}]")
        if len(token) != config.d_model:
            raise ValueError(f"tokens[{index}] length must match d_model={config.d_model}.")
        latents.append(matvec(weights.w_dkv, token))
    return latents


def mla_causal_forward(
    tokens: Matrix,
    weights: MLAWeights,
    config: MLAConfig,
) -> tuple[Matrix, Matrix, list[Matrix]]:
    """Run causal MLA forward pass using compressed latent KV cache.

    Returns:
        (concatenated_outputs, kv_latent_cache, per_head_attention_weights)
    """
    kv_latent_cache = compress_to_kv_latent(tokens, weights, config)
    seq_len = len(tokens)
    scale = 1.0 / math.sqrt(config.head_dim)

    per_head_outputs: list[Matrix] = []
    per_head_weights: list[Matrix] = []

    for head in range(config.num_heads):
        queries = [matvec(weights.w_q[head], token) for token in tokens]
        # Reconstruct K and V on the fly from the compressed latent cache.
        keys = [matvec(weights.w_uk[head], latent) for latent in kv_latent_cache]
        values = [matvec(weights.w_uv[head], latent) for latent in kv_latent_cache]

        head_out: Matrix = []
        head_attn: Matrix = []
        for pos in range(seq_len):
            raw_scores = [dot(queries[pos], keys[j]) * scale for j in range(pos + 1)]
            attn_probs = softmax(raw_scores)
            padded_probs = attn_probs + [0.0] * (seq_len - pos - 1)
            head_attn.append(padded_probs)

            context = [
                sum(attn_probs[j] * values[j][dim] for j in range(pos + 1))
                for dim in range(config.head_dim)
            ]
            head_out.append(context)

        per_head_outputs.append(head_out)
        per_head_weights.append(head_attn)

    concatenated: Matrix = []
    for pos in range(seq_len):
        row: Vector = []
        for head in range(config.num_heads):
            row.extend(per_head_outputs[head][pos])
        concatenated.append(row)

    return concatenated, kv_latent_cache, per_head_weights


def compare_kv_cache_scalars(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    gqa_kv_heads: int,
    kv_latent_dim: int,
) -> dict[str, int]:
    """Compare total scalar values cached across one layer for MHA, GQA, and MLA."""
    if seq_len <= 0 or num_heads <= 0 or head_dim <= 0 or gqa_kv_heads <= 0 or kv_latent_dim <= 0:
        raise ValueError("All cache comparison arguments must be positive integers.")
    if num_heads % gqa_kv_heads != 0:
        raise ValueError("num_heads must be divisible by gqa_kv_heads.")
    return {
        "MHA": seq_len * 2 * num_heads * head_dim,
        "GQA": seq_len * 2 * gqa_kv_heads * head_dim,
        "MLA": seq_len * kv_latent_dim,
    }


def build_demo_setup() -> tuple[MLAConfig, MLAWeights, Matrix]:
    """Create deterministic toy weights and input tokens for classroom inspection."""
    config = MLAConfig(d_model=6, num_heads=4, head_dim=2, kv_latent_dim=2)
    w_dkv = [
        [0.4, 0.2, -0.1, 0.3, 0.1, 0.2],
        [-0.2, 0.3, 0.5, 0.1, 0.2, -0.1],
    ]
    w_q = [
        [[0.5, 0.1, 0.0, 0.2, -0.1, 0.3], [0.0, 0.4, 0.2, -0.1, 0.3, 0.1]],
        [[0.2, -0.2, 0.5, 0.1, 0.0, 0.4], [0.3, 0.1, 0.1, 0.4, -0.2, 0.2]],
        [[-0.1, 0.3, 0.2, 0.5, 0.2, 0.0], [0.2, 0.0, -0.2, 0.3, 0.4, 0.1]],
        [[0.1, 0.2, -0.1, 0.0, 0.5, 0.3], [0.4, -0.1, 0.3, 0.2, 0.1, 0.0]],
    ]
    w_uk = [
        [[0.8, 0.2], [-0.1, 0.9]],
        [[0.3, -0.6], [0.7, 0.4]],
        [[-0.5, 0.5], [0.4, 0.6]],
        [[0.6, 0.3], [-0.4, 0.7]],
    ]
    w_uv = [
        [[0.6, -0.2], [0.3, 0.7]],
        [[0.2, 0.8], [-0.3, 0.5]],
        [[0.7, 0.1], [0.2, -0.6]],
        [[-0.2, 0.6], [0.8, 0.3]],
    ]
    tokens = [
        [1.0, 0.5, 0.2, 0.1, 0.3, 0.4],
        [0.2, 1.0, 0.6, 0.3, 0.1, 0.2],
        [0.1, 0.3, 0.9, 0.8, 0.4, 0.1],
    ]
    return config, MLAWeights(w_q=w_q, w_dkv=w_dkv, w_uk=w_uk, w_uv=w_uv), tokens


def main() -> int:
    config, weights, tokens = build_demo_setup()
    outputs, latent_cache, attn_weights = mla_causal_forward(tokens, weights, config)

    print("Multi-Head Latent Attention (MLA) teaching demo")
    print(
        f"Config: d_model={config.d_model}, num_heads={config.num_heads}, "
        f"head_dim={config.head_dim}, kv_latent_dim={config.kv_latent_dim}\n"
    )
    print("Compressed KV latent cache (stored per token instead of full K and V):")
    for idx, latent in enumerate(latent_cache):
        print(f"  token {idx}: {[round(v, 4) for v in latent]}")

    print("\nHead 0 causal attention weights:")
    for idx, row in enumerate(attn_weights[0]):
        print(f"  pos {idx}: {[round(v, 4) for v in row]}")

    print("\nConcatenated multi-head outputs (first 4 dims per token):")
    for idx, out in enumerate(outputs):
        print(f"  pos {idx}: {[round(v, 4) for v in out[:4]]} ... (len={len(out)})")

    # Realistic scale cache comparison (e.g. 4096 tokens, 32 heads, head_dim=128)
    toy_cache = compare_kv_cache_scalars(
        seq_len=len(tokens),
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        gqa_kv_heads=2,
        kv_latent_dim=config.kv_latent_dim,
    )
    large_cache = compare_kv_cache_scalars(
        seq_len=4096,
        num_heads=32,
        head_dim=128,
        gqa_kv_heads=8,
        kv_latent_dim=512,
    )
    print("\nKV cache scalar count (toy setup):", toy_cache)
    print("KV cache scalar count (seq=4096, 32 heads, d_h=128, latent=512):", large_cache)
    compression_ratio = large_cache["MHA"] / large_cache["MLA"]
    print(f"MLA vs MHA cache compression ratio: {compression_ratio:.1f}x smaller")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
