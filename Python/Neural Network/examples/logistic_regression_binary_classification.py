"""Logistic regression for binary classification using pure Python.

This example classifies whether a point is above a simple boundary.

Target rule:

    if x1 + x2 >= 1.0 -> 1
    else -> 0

Logistic regression outputs a probability-like value between 0 and 1.
"""

import math


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def predict_probability(inputs, weights, bias):
    total = bias
    for x, w in zip(inputs, weights):
        total += x * w
    return sigmoid(total)


def train(samples, learning_rate=0.5, epochs=1000):
    weights = [0.0, 0.0]
    bias = 0.0

    for epoch in range(epochs):
        total_loss = 0.0

        for inputs, target in samples:
            probability = predict_probability(inputs, weights, bias)
            error = probability - target
            total_loss += error * error

            for index in range(len(weights)):
                weights[index] -= learning_rate * error * inputs[index]
            bias -= learning_rate * error

        if (epoch + 1) % 200 == 0:
            print(f"epoch={epoch + 1}, loss={total_loss:.6f}, weights={weights}, bias={bias:.3f}")

    return weights, bias


def main():
    samples = [
        ([0.0, 0.0], 0),
        ([0.2, 0.2], 0),
        ([0.4, 0.4], 0),
        ([0.6, 0.5], 1),
        ([0.8, 0.4], 1),
        ([1.0, 0.2], 1),
    ]

    weights, bias = train(samples)

    print("\nFinal predictions:")
    for inputs, target in samples:
        probability = predict_probability(inputs, weights, bias)
        predicted = 1 if probability >= 0.5 else 0
        print(f"{inputs} -> probability={probability:.3f}, predicted={predicted}, target={target}")


if __name__ == "__main__":
    main()

