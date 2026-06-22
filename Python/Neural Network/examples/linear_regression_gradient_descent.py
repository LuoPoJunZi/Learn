"""Linear regression with gradient descent using pure Python.

This is not a neural network with hidden layers, but it teaches the same
training loop idea used in neural networks:

1. Make a prediction.
2. Measure the error.
3. Compute gradients.
4. Update parameters.

The target pattern is roughly:

    y = 2 * x + 1
"""


def predict(x, weight, bias):
    return weight * x + bias


def mean_squared_error(samples, weight, bias):
    total = 0.0
    for x, target in samples:
        error = target - predict(x, weight, bias)
        total += error * error
    return total / len(samples)


def train(samples, learning_rate=0.01, epochs=800):
    weight = 0.0
    bias = 0.0

    for epoch in range(epochs):
        weight_gradient = 0.0
        bias_gradient = 0.0

        for x, target in samples:
            output = predict(x, weight, bias)
            error = output - target
            weight_gradient += 2 * error * x
            bias_gradient += 2 * error

        weight_gradient /= len(samples)
        bias_gradient /= len(samples)

        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

        if (epoch + 1) % 100 == 0:
            loss = mean_squared_error(samples, weight, bias)
            print(f"epoch={epoch + 1:03d}, loss={loss:.6f}, weight={weight:.3f}, bias={bias:.3f}")

    return weight, bias


def main():
    samples = [
        (0.0, 1.0),
        (1.0, 3.0),
        (2.0, 5.0),
        (3.0, 7.0),
        (4.0, 9.0),
    ]

    weight, bias = train(samples)

    print("\nFinal model:")
    print(f"y = {weight:.3f} * x + {bias:.3f}")

    print("\nPredictions:")
    for x, target in samples:
        output = predict(x, weight, bias)
        print(f"x={x:.1f}, predicted={output:.3f}, target={target:.1f}")


if __name__ == "__main__":
    main()

