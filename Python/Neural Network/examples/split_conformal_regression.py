"""Calibrate regression intervals with held-out absolute residuals."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

Samples = list[tuple[float, float]]


@dataclass(frozen=True)
class LinearModel:
    """Store a fitted baseline that can later be replaced by a neural regressor."""

    slope: float
    intercept: float

    def predict(self, value: float) -> float:
        return self.slope * value + self.intercept


def make_samples(count: int, rng: random.Random, *, noise_scale: float = 0.6) -> Samples:
    """Generate independent synthetic regression observations."""
    if count <= 0 or not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("count must be positive and noise_scale finite and non-negative.")
    samples: Samples = []
    for _ in range(count):
        value = rng.uniform(-2.0, 2.0)
        target = 2.0 * value + 1.0 + rng.gauss(0.0, noise_scale)
        samples.append((value, target))
    return samples


def fit_linear_model(samples: Samples) -> LinearModel:
    """Fit ordinary least squares using only the training split."""
    if len(samples) < 2 or not all(math.isfinite(v) for pair in samples for v in pair):
        raise ValueError("At least two finite training observations are required.")
    mean_x = math.fsum(value for value, _ in samples) / len(samples)
    mean_y = math.fsum(target for _, target in samples) / len(samples)
    variance = math.fsum((value - mean_x) ** 2 for value, _ in samples)
    if variance <= 0:
        raise ValueError("Training inputs must not all be identical.")
    covariance = math.fsum((value - mean_x) * (target - mean_y) for value, target in samples)
    slope = covariance / variance
    return LinearModel(slope, mean_y - slope * mean_x)


def conformal_radius(scores: list[float], alpha: float) -> float:
    """Select ceil((n+1)*(1-alpha)); an unavailable rank means infinite radius."""
    if not scores or not 0.0 < alpha < 1.0:
        raise ValueError("Calibration scores must be non-empty and 0 < alpha < 1.")
    if any(not math.isfinite(score) or score < 0 for score in scores):
        raise ValueError("Absolute residual scores must be finite and non-negative.")
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha))
    if rank > len(scores):
        return math.inf
    return sorted(scores)[rank - 1]


def calibrate(model: LinearModel, samples: Samples, alpha: float) -> float:
    """Compute calibration residuals without refitting the model."""
    if not all(math.isfinite(value) for pair in samples for value in pair):
        raise ValueError("Calibration observations must be finite.")
    scores = [abs(target - model.predict(value)) for value, target in samples]
    return conformal_radius(scores, alpha)


def coverage(model: LinearModel, samples: Samples, radius: float) -> float:
    """Measure empirical coverage on an independent evaluation split."""
    if not samples or math.isnan(radius) or radius < 0:
        raise ValueError("Evaluation samples must be non-empty and radius non-negative.")
    hits = sum(abs(target - model.predict(value)) <= radius for value, target in samples)
    return hits / len(samples)


def main() -> None:
    rng = random.Random(42)
    training = make_samples(200, rng)
    calibration = make_samples(120, rng)
    evaluation = make_samples(1000, rng)
    shifted_evaluation = make_samples(1000, rng, noise_scale=1.8)
    model = fit_linear_model(training)

    assert conformal_radius([0.1, 0.2, 0.3, 0.4], 0.4) == 0.3
    assert math.isinf(conformal_radius([0.1, 0.2, 0.3], 0.1))
    print(f"Training-only fit: y = {model.slope:.4f} * x + {model.intercept:.4f}")
    print("Separate splits: train=200, calibration=120, evaluation=1000")

    for alpha in (0.20, 0.10, 0.05):
        radius = calibrate(model, calibration, alpha)
        observed = coverage(model, evaluation, radius)
        shifted = coverage(model, shifted_evaluation, radius)
        print(
            f"target={1 - alpha:.0%}, radius={radius:.4f}, width={2 * radius:.4f}, "
            f"observed={observed:.1%}, increased-noise coverage={shifted:.1%}"
        )

    radius = calibrate(model, calibration, 0.1)
    print("\nInput -> prediction and calibrated interval")
    for value in (-1.0, 0.0, 1.0):
        prediction = model.predict(value)
        print(f"  {value:+.1f} -> {prediction:.4f} [{prediction - radius:.4f}, "
              f"{prediction + radius:.4f}]")
    print("Rank checks passed; tiny calibration sets may require an infinite interval.")
    print("Coverage is marginal under exchangeability, not a per-input guarantee.")


if __name__ == "__main__":
    main()
