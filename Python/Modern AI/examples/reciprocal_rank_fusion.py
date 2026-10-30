"""Fuse ranked document IDs without comparing incompatible retrieval scores."""

from __future__ import annotations

import math
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: float = 60.0,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Use one-based ranks; reject duplicates within each retriever's list."""
    if isinstance(k, bool) or not math.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive")
    resolved_weights = [1.0] * len(rankings) if weights is None else list(weights)
    if len(resolved_weights) != len(rankings):
        raise ValueError("provide exactly one weight per ranking")
    if any(
        isinstance(weight, bool) or not math.isfinite(weight) or weight < 0
        for weight in resolved_weights
    ):
        raise ValueError("weights must be finite and non-negative")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, resolved_weights):
        seen: set[str] = set()
        for rank, document_id in enumerate(ranking, start=1):
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("document IDs must be non-empty strings")
            if document_id in seen:
                raise ValueError("a ranking must not contain duplicate document IDs")
            seen.add(document_id)
            if weight == 0:
                continue
            contribution = weight / (k + rank)
            total = scores.get(document_id, 0.0) + contribution
            if not math.isfinite(total):
                raise ValueError("fused score overflow")
            scores[document_id] = total
    # Document ID is an explicit, reproducible tie-breaker, not a relevance signal.
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def recall_at_k(results: Sequence[str], relevant: set[str], cutoff: int) -> float:
    """Measure unique relevant IDs found in a result prefix; require known relevance."""
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff <= 0:
        raise ValueError("cutoff must be a positive integer")
    if not relevant:
        raise ValueError("recall requires a non-empty relevance set")
    return len(set(results[:cutoff]) & relevant) / len(relevant)


def main() -> int:
    # These lists are hand-authored outputs, not a real keyword/embedding engine.
    keyword = ["exact-api", "hybrid-guide", "legacy-note"]
    semantic = ["semantic-guide", "hybrid-guide", "exact-api"]
    relevant = {"exact-api", "hybrid-guide", "semantic-guide"}
    fused = reciprocal_rank_fusion([keyword, semantic])
    print("keyword:", keyword)
    print("semantic (fixture):", semantic)
    for document_id, score in fused:
        print(f"{document_id:16s} {score:.6f}")
    print("Recall@2 keyword:", recall_at_k(keyword, relevant, 2))
    print("Recall@2 fused:", recall_at_k([item[0] for item in fused], relevant, 2))
    print("Fixture scores are not proof of real-world retrieval quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
