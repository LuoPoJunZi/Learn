"""Demonstrate Group Relative Policy Optimization (GRPO) without a critic model."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GroupSample:
    """One sampled completion in a GRPO rollout group for the same prompt."""

    label: str
    reward: float
    old_log_prob: float
    ref_log_prob: float


def compute_group_advantages(rewards: list[float], *, eps: float = 1e-8) -> list[float]:
    """Normalize rewards within a rollout group to obtain relative advantages.

    If all rewards in the group are identical, return 0.0 for every sample so
    no NaN or spurious gradient is produced.
    """
    if not rewards:
        raise ValueError("rewards group must not be empty.")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be a positive finite float.")
    if any(not math.isfinite(value) for value in rewards):
        raise ValueError("All rewards must be finite numbers.")

    mean_reward = sum(rewards) / len(rewards)
    variance = sum((value - mean_reward) ** 2 for value in rewards) / len(rewards)
    if variance == 0.0:
        return [0.0 for _ in rewards]

    std_reward = math.sqrt(variance)
    return [(value - mean_reward) / (std_reward + eps) for value in rewards]


def schulman_kl_approx(current_log_prob: float, ref_log_prob: float) -> float:
    """Non-negative KL divergence estimator: r - log(r) - 1 where r = pi_ref / pi_theta."""
    if not (math.isfinite(current_log_prob) and math.isfinite(ref_log_prob)):
        raise ValueError("Log probabilities must be finite.")
    log_ratio = ref_log_prob - current_log_prob
    return math.exp(log_ratio) - log_ratio - 1.0


def grpo_objective(
    samples: list[GroupSample],
    current_log_probs: list[float],
    *,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[float, list[float], list[float]]:
    """Compute the mean GRPO surrogate objective across one rollout group.

    Returns:
        (mean_objective, advantages, per_sample_kl)
    """
    if not samples:
        raise ValueError("samples must not be empty.")
    if len(samples) != len(current_log_probs):
        raise ValueError("current_log_probs must match samples length.")
    if clip_eps <= 0.0 or not math.isfinite(clip_eps):
        raise ValueError("clip_eps must be positive and finite.")
    if kl_beta < 0.0 or not math.isfinite(kl_beta):
        raise ValueError("kl_beta must be non-negative and finite.")

    advantages = compute_group_advantages([sample.reward for sample in samples])
    terms: list[float] = []
    kl_values: list[float] = []

    for sample, cur_lp, advantage in zip(samples, current_log_probs, advantages):
        if not (
            math.isfinite(cur_lp)
            and math.isfinite(sample.old_log_prob)
            and math.isfinite(sample.ref_log_prob)
        ):
            raise ValueError("All log probabilities must be finite.")

        ratio = math.exp(cur_lp - sample.old_log_prob)
        clipped_ratio = min(max(ratio, 1.0 - clip_eps), 1.0 + clip_eps)
        surrogate = min(ratio * advantage, clipped_ratio * advantage)
        kl_penalty = schulman_kl_approx(cur_lp, sample.ref_log_prob)
        terms.append(surrogate - kl_beta * kl_penalty)
        kl_values.append(kl_penalty)

    return sum(terms) / len(terms), advantages, kl_values


def log_softmax(logits: list[float]) -> list[float]:
    """Convert raw logits over discrete candidate completions into log-probabilities."""
    if not logits:
        raise ValueError("logits must not be empty.")
    if any(not math.isfinite(value) for value in logits):
        raise ValueError("logits must be finite.")
    peak = max(logits)
    shifted = [value - peak for value in logits]
    log_denom = math.log(sum(math.exp(value) for value in shifted))
    return [value - log_denom for value in shifted]


def run_toy_grpo_training(
    *,
    steps: int = 10,
    lr: float = 0.4,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[list[float], list[float], list[float]]:
    """Run finite-difference gradient ascent on a toy 4-candidate rollout group."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if lr <= 0.0 or not math.isfinite(lr):
        raise ValueError("lr must be positive and finite.")

    # Initial reference/old logits over 4 candidate completions for one math prompt.
    base_logits = [0.0, 0.2, 0.1, -0.1]
    base_log_probs = log_softmax(base_logits)
    samples = [
        GroupSample(
            "correct_with_clear_steps",
            reward=1.0,
            old_log_prob=base_log_probs[0],
            ref_log_prob=base_log_probs[0],
        ),
        GroupSample(
            "format_ok_wrong_number",
            reward=0.2,
            old_log_prob=base_log_probs[1],
            ref_log_prob=base_log_probs[1],
        ),
        GroupSample(
            "wrong_logic_wrong_answer",
            reward=0.0,
            old_log_prob=base_log_probs[2],
            ref_log_prob=base_log_probs[2],
        ),
        GroupSample(
            "correct_concise_steps",
            reward=1.0,
            old_log_prob=base_log_probs[3],
            ref_log_prob=base_log_probs[3],
        ),
    ]

    logits = list(base_logits)
    objective_history: list[float] = []
    delta = 1e-5

    for _ in range(steps):
        cur_lp = log_softmax(logits)
        obj, _, _ = grpo_objective(samples, cur_lp, clip_eps=clip_eps, kl_beta=kl_beta)
        objective_history.append(obj)

        grads: list[float] = []
        for idx in range(len(logits)):
            perturbed = list(logits)
            perturbed[idx] += delta
            obj_plus, _, _ = grpo_objective(
                samples, log_softmax(perturbed), clip_eps=clip_eps, kl_beta=kl_beta
            )
            grads.append((obj_plus - obj) / delta)

        logits = [value + lr * grad for value, grad in zip(logits, grads)]

    initial_probs = [math.exp(lp) for lp in base_log_probs]
    final_probs = [math.exp(lp) for lp in log_softmax(logits)]
    return initial_probs, final_probs, objective_history


def main() -> int:
    print("Group Relative Policy Optimization (GRPO) teaching demo\n")
    rewards = [1.0, 0.2, 0.0, 1.0]
    advantages = compute_group_advantages(rewards)
    print("Step 1: Group-relative advantage normalization (no Critic model needed)")
    for idx, (r, adv) in enumerate(zip(rewards, advantages)):
        print(f"  candidate {idx}: reward={r:.2f} -> relative advantage={adv:+.4f}")

    zero_var = compute_group_advantages([1.0, 1.0, 1.0])
    print("  zero-variance edge case [1.0, 1.0, 1.0] ->", zero_var)

    initial_probs, final_probs, history = run_toy_grpo_training(steps=10, lr=0.4)
    labels = [
        "correct_with_clear_steps (r=1.0)",
        "format_ok_wrong_number   (r=0.2)",
        "wrong_logic_wrong_answer (r=0.0)",
        "correct_concise_steps    (r=1.0)",
    ]
    print("\nStep 2: Toy policy probability shift after 10 GRPO steps")
    for label, p0, p1 in zip(labels, initial_probs, final_probs):
        print(f"  {label}: {p0 * 100:5.1f}% -> {p1 * 100:5.1f}%")

    print(f"\nSurrogate objective: start={history[0]:+.4f} -> end={history[-1]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
