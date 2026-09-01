"""Demonstrate greedy speculative decoding with a tiny deterministic model.

The draft model proposes several tokens. The target model verifies each proposal
block conceptually in one pass. This script models control flow, not tensor work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


Predictor = Callable[[list[str]], str]
EOS = "<eos>"

TARGET_TRANSITIONS = {
    "start": "learn",
    "learn": "modern",
    "modern": "AI",
    "AI": "by",
    "by": "understanding",
    "understanding": "principles",
    "principles": ".",
    ".": EOS,
}

DRAFT_TRANSITIONS = {
    "start": "learn",
    "learn": "modern",
    "modern": "AI",
    "AI": "through",
    "through": "practice",
    "practice": ".",
    "by": "understanding",
    "understanding": "principles",
    "principles": ".",
    ".": EOS,
}


@dataclass
class DecodeStats:
    """Collect the amount of target verification and draft work."""

    target_calls: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0


def transition_predictor(table: dict[str, str]) -> Predictor:
    """Build a deterministic next-token predictor from a transition table."""

    def predict(context: list[str]) -> str:
        return table.get(context[-1], EOS)

    return predict


def greedy_decode(
    prompt: list[str],
    target: Predictor,
    max_new_tokens: int,
) -> tuple[list[str], DecodeStats]:
    """Generate one token per target-model call."""

    context = prompt.copy()
    output: list[str] = []
    stats = DecodeStats()

    while len(output) < max_new_tokens:
        token = target(context)
        stats.target_calls += 1
        if token == EOS:
            break
        output.append(token)
        context.append(token)

    return output, stats


def make_draft(
    context: list[str],
    draft: Predictor,
    block_size: int,
) -> list[str]:
    """Ask the small draft model to propose up to ``block_size`` tokens."""

    proposal_context = context.copy()
    proposal: list[str] = []
    for _ in range(block_size):
        token = draft(proposal_context)
        proposal.append(token)
        if token == EOS:
            break
        proposal_context.append(token)
    return proposal


def verify_draft(
    context: list[str],
    proposal: list[str],
    target: Predictor,
) -> list[str]:
    """Represent one target pass that returns logits for a whole draft block."""

    verification_context = context.copy()
    expected: list[str] = []
    for candidate in proposal:
        expected.append(target(verification_context))
        if candidate == EOS:
            break
        verification_context.append(candidate)
    return expected


def speculative_greedy_decode(
    prompt: list[str],
    draft: Predictor,
    target: Predictor,
    max_new_tokens: int,
    block_size: int,
) -> tuple[list[str], DecodeStats]:
    """Accept matching draft tokens and correct the first mismatch."""

    context = prompt.copy()
    output: list[str] = []
    stats = DecodeStats()
    finished = False

    while len(output) < max_new_tokens and not finished:
        proposal = make_draft(context, draft, block_size)
        stats.draft_tokens += len(proposal)

        expected = verify_draft(context, proposal, target)
        stats.target_calls += 1

        for candidate, target_token in zip(proposal, expected):
            matched = candidate == target_token
            token = candidate if matched else target_token

            if matched:
                stats.accepted_tokens += 1
            else:
                stats.rejected_tokens += 1

            if token == EOS:
                finished = True
                break

            output.append(token)
            context.append(token)

            if not matched or len(output) >= max_new_tokens:
                break

    return output, stats


def main() -> None:
    """Compare normal greedy decoding with its speculative counterpart."""

    target = transition_predictor(TARGET_TRANSITIONS)
    draft = transition_predictor(DRAFT_TRANSITIONS)
    prompt = ["start"]

    baseline, baseline_stats = greedy_decode(prompt, target, max_new_tokens=12)
    speculative, speculative_stats = speculative_greedy_decode(
        prompt,
        draft,
        target,
        max_new_tokens=12,
        block_size=3,
    )

    assert speculative == baseline

    print("Baseline output:   ", " ".join(baseline))
    print("Speculative output:", " ".join(speculative))
    print("Outputs match:     ", speculative == baseline)
    print("Baseline target calls:", baseline_stats.target_calls)
    print("Speculative target verification calls:", speculative_stats.target_calls)
    print("Draft tokens proposed:", speculative_stats.draft_tokens)
    print("Accepted draft tokens:", speculative_stats.accepted_tokens)
    print("Rejected draft tokens:", speculative_stats.rejected_tokens)


if __name__ == "__main__":
    main()
