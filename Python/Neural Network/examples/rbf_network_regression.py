"""A small radial basis function (RBF) network for regression.

RBF networks use distance to centers as features. This example keeps the
centers fixed and only trains the output weights with gradient descent.

Target function:

    y = sin(x)
"""

import math


def rbf(x, center, width):
    distance = x - center
    return math.exp(-(distance * distance) / (2 * width * width))


def features(x, centers, width):
    return [rbf(x, center, width) for center in centers]


def predict(x, centers, width, weights, bias):
    total = bias
    for feature, weight in zip(features(x, centers, width), weights):
        total += feature * weight
    return total


def train(samples, centers, width=1.0, learning_rate=0.05, epochs=1000):
    weights = [0.0 for _ in centers]
    bias = 0.0

    for epoch in range(epochs):
        total_loss = 0.0

        for x, target in samples:
            current_features = features(x, centers, width)
            output = bias + sum(feature * weight for feature, weight in zip(current_features, weights))
            error = output - target
            total_loss += error * error

            for index in range(len(weights)):
                weights[index] -= learning_rate * error * current_features[index]
            bias -= learning_rate * error

        if (epoch + 1) % 250 == 0:
            print(f"epoch={epoch + 1}, loss={total_loss:.6f}")

    return weights, bias


def main():
    samples = [
        (-3.0, math.sin(-3.0)),
        (-2.0, math.sin(-2.0)),
        (-1.0, math.sin(-1.0)),
        (0.0, math.sin(0.0)),
        (1.0, math.sin(1.0)),
        (2.0, math.sin(2.0)),
        (3.0, math.sin(3.0)),
    ]

    centers = [-3.0, -1.5, 0.0, 1.5, 3.0]
    width = 1.0
    weights, bias = train(samples, centers, width)

    print("\nPredictions:")
    for x, target in samples:
        output = predict(x, centers, width, weights, bias)
        print(f"x={x:+.1f}, predicted={output:+.3f}, target={target:+.3f}")


if __name__ == "__main__":
    main()

