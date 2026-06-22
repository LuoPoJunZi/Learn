"""A tiny perceptron example for binary classification.

This script trains a perceptron to learn the AND logic gate:

    0 AND 0 -> 0
    0 AND 1 -> 0
    1 AND 0 -> 0
    1 AND 1 -> 1

It uses only the Python standard library so beginners can run it directly.
"""


def step(value):
    """Return 1 if value is non-negative, otherwise return 0."""
    return 1 if value >= 0 else 0


def predict(inputs, weights, bias):
    total = bias
    for x, w in zip(inputs, weights):
        total += x * w
    return step(total)


def train_perceptron(samples, learning_rate=0.1, epochs=20):
    weights = [0.0, 0.0]
    bias = 0.0

    for epoch in range(epochs):
        total_error = 0

        for inputs, target in samples:
            output = predict(inputs, weights, bias)
            error = target - output
            total_error += abs(error)

            for index in range(len(weights)):
                weights[index] += learning_rate * error * inputs[index]
            bias += learning_rate * error

        print(f"epoch={epoch + 1:02d}, error={total_error}, weights={weights}, bias={bias:.2f}")

        if total_error == 0:
            break

    return weights, bias


def main():
    samples = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1),
    ]

    weights, bias = train_perceptron(samples)

    print("\nFinal predictions:")
    for inputs, target in samples:
        output = predict(inputs, weights, bias)
        print(f"{inputs} -> predicted={output}, target={target}")


if __name__ == "__main__":
    main()

