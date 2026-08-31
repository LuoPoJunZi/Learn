"""Demonstrate top-k sparse mixture-of-experts routing."""

from __future__ import annotations

import math
from dataclasses import dataclass

Vector = list[float]
Matrix = list[Vector]


@dataclass(frozen=True)
class Expert:
    """A tiny linear expert used only for routing intuition."""

    name: str
    weight: Matrix
    bias: Vector

    def forward(self, inputs: Vector) -> Vector:
        """Return a linear transformation of one token vector."""
        if len(self.weight) != len(self.bias):
            raise ValueError("Each expert output needs a matching bias value.")
        if any(len(row) != len(inputs) for row in self.weight):
            raise ValueError("Expert input dimensions must match the token vector.")
        return [
            sum(value * coefficient for value, coefficient in zip(inputs, row))
            + bias
            for row, bias in zip(self.weight, self.bias)
        ]

    @property
    def parameter_count(self) -> int:
        """Count this toy expert's weights and biases."""
        return sum(len(row) for row in self.weight) + len(self.bias)


def dot(left: Vector, right: Vector) -> float:
    """Return a vector dot product."""
    if len(left) != len(right):
        raise ValueError("Vectors must have the same length.")
    return sum(a * b for a, b in zip(left, right))


def softmax(values: Vector) -> Vector:
    """Convert router logits into probabilities."""
    largest = max(values)
    exponentials = [math.exp(value - largest) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def route_token(
    token: Vector,
    router_weights: Matrix,
    experts: list[Expert],
    *,
    top_k: int,
) -> tuple[Vector, list[tuple[str, float]], Vector]:
    """Route a token to the top-k experts and combine their outputs."""
    if len(router_weights) != len(experts):
        raise ValueError("The router needs one score vector per expert.")
    if not 1 <= top_k <= len(experts):
        raise ValueError("top_k must select at least one available expert.")

    router_logits = [dot(token, weights) for weights in router_weights]
    router_probabilities = softmax(router_logits)
    selected_indices = sorted(
        range(len(experts)),
        key=lambda index: router_probabilities[index],
        reverse=True,
    )[:top_k]

    selected_total = sum(router_probabilities[index] for index in selected_indices)
    selected_weights = [
        router_probabilities[index] / selected_total for index in selected_indices
    ]
    expert_outputs = [experts[index].forward(token) for index in selected_indices]

    mixed_output = [
        sum(
            weight * output[column]
            for weight, output in zip(selected_weights, expert_outputs)
        )
        for column in range(len(expert_outputs[0]))
    ]
    selected = [
        (experts[index].name, weight)
        for index, weight in zip(selected_indices, selected_weights)
    ]
    return mixed_output, selected, router_probabilities


def main() -> int:
    experts = [
        Expert("pattern", [[1.0, 0.2, 0.0], [0.1, 0.8, 0.1]], [0.0, 0.1]),
        Expert("sequence", [[0.2, 1.0, 0.2], [0.0, 0.3, 0.9]], [0.1, 0.0]),
        Expert("numeric", [[0.1, 0.1, 1.1], [0.8, 0.1, 0.3]], [0.0, -0.1]),
        Expert("general", [[0.6, 0.5, 0.4], [0.4, 0.6, 0.5]], [0.0, 0.0]),
    ]
    router_weights = [
        [1.2, 0.1, 0.0],
        [0.1, 1.1, 0.2],
        [0.0, 0.2, 1.3],
        [0.5, 0.5, 0.5],
    ]
    tokens = {
        "visual token": [1.0, 0.2, 0.1],
        "ordered token": [0.1, 1.0, 0.3],
        "number token": [0.1, 0.2, 1.0],
    }
    top_k = 2

    print(f"Sparse routing: top-{top_k} of {len(experts)} experts\n")
    for token_name, token in tokens.items():
        output, selected, probabilities = route_token(
            token,
            router_weights,
            experts,
            top_k=top_k,
        )
        choices = ", ".join(
            f"{name}={weight:.3f}" for name, weight in selected
        )
        print(token_name)
        print("  all router probabilities:", [round(value, 3) for value in probabilities])
        print("  active experts:", choices)
        print("  mixed output:", [round(value, 3) for value in output])

    all_expert_parameters = sum(expert.parameter_count for expert in experts)
    active_expert_parameters = sum(
        expert.parameter_count for expert in experts[:top_k]
    )
    print("\nToy expert parameters stored:", all_expert_parameters)
    print("Toy expert parameters active per token:", active_expert_parameters)
    print("The router is evaluated for every token and is not included above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
