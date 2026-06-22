"""A tiny time-series prediction example using a sliding window.

This script predicts the next value from the previous three values.
It uses a simple linear model trained with gradient descent.

This is not an LSTM, but it teaches the data preparation idea used by
many sequence models:

    [x(t-3), x(t-2), x(t-1)] -> x(t)
"""


def make_windows(series, window_size):
    samples = []
    for index in range(len(series) - window_size):
        inputs = series[index : index + window_size]
        target = series[index + window_size]
        samples.append((inputs, target))
    return samples


def predict(inputs, weights, bias):
    total = bias
    for x, w in zip(inputs, weights):
        total += x * w
    return total


def train(samples, learning_rate=0.001, epochs=2000):
    weights = [0.0 for _ in samples[0][0]]
    bias = 0.0

    for epoch in range(epochs):
        total_loss = 0.0

        for inputs, target in samples:
            output = predict(inputs, weights, bias)
            error = output - target
            total_loss += error * error

            for index in range(len(weights)):
                weights[index] -= learning_rate * error * inputs[index]
            bias -= learning_rate * error

        if (epoch + 1) % 500 == 0:
            print(f"epoch={epoch + 1}, loss={total_loss:.6f}, weights={weights}, bias={bias:.3f}")

    return weights, bias


def main():
    series = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    window_size = 3
    samples = make_windows(series, window_size)

    weights, bias = train(samples)

    print("\nPredictions:")
    for inputs, target in samples:
        output = predict(inputs, weights, bias)
        print(f"{inputs} -> predicted={output:.2f}, target={target}")

    next_inputs = series[-window_size:]
    next_value = predict(next_inputs, weights, bias)
    print(f"\nNext prediction from {next_inputs}: {next_value:.2f}")


if __name__ == "__main__":
    main()

