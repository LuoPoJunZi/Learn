"""Overfitting and L2 regularization demo using pure Python.

This example fits a high-degree polynomial model to a tiny noisy dataset.

The model has more capacity than the data really needs. Without
regularization, it can fit the training samples very closely but may
perform worse on validation samples. L2 regularization discourages
large weights and often improves generalization.
"""


def make_features(x, degree):
    scaled_x = x / 2
    return [scaled_x**power for power in range(degree + 1)]


def predict(x, weights):
    return sum(feature * weight for feature, weight in zip(make_features(x, len(weights) - 1), weights))


def mse(samples, weights):
    total = 0.0
    for x, target in samples:
        error = predict(x, weights) - target
        total += error * error
    return total / len(samples)


def train(samples, degree=7, learning_rate=0.003, epochs=8000, l2_strength=0.0):
    weights = [0.0 for _ in range(degree + 1)]

    for _ in range(epochs):
        gradients = [0.0 for _ in weights]

        for x, target in samples:
            features = make_features(x, degree)
            error = sum(feature * weight for feature, weight in zip(features, weights)) - target

            for index, feature in enumerate(features):
                gradients[index] += 2 * error * feature

        for index in range(len(weights)):
            gradients[index] /= len(samples)
            if index != 0:
                gradients[index] += 2 * l2_strength * weights[index]
            weights[index] -= learning_rate * gradients[index]

    return weights


def main():
    train_samples = [
        (-2.0, 5.1),
        (-1.0, 1.2),
        (0.0, 1.0),
        (1.0, 3.2),
        (2.0, 5.2),
    ]
    validation_samples = [
        (-1.5, 3.25),
        (-0.5, 1.25),
        (0.5, 1.25),
        (1.5, 3.25),
    ]

    plain_weights = train(train_samples, l2_strength=0.0)
    regularized_weights = train(train_samples, l2_strength=0.1)

    print("Model comparison:")
    print(
        f"plain       train_mse={mse(train_samples, plain_weights):.4f}, "
        f"validation_mse={mse(validation_samples, plain_weights):.4f}"
    )
    print(
        f"regularized train_mse={mse(train_samples, regularized_weights):.4f}, "
        f"validation_mse={mse(validation_samples, regularized_weights):.4f}"
    )

    print("\nPredictions on validation samples:")
    for x, target in validation_samples:
        plain = predict(x, plain_weights)
        regularized = predict(x, regularized_weights)
        print(f"x={x:+.1f}, target={target:.2f}, plain={plain:.2f}, regularized={regularized:.2f}")


if __name__ == "__main__":
    main()
