"""A small neural network that learns XOR using pure Python.

XOR cannot be solved by a single linear perceptron. This example uses:

- 2 input neurons
- 2 hidden neurons
- 1 output neuron

The math is intentionally written out for learning purposes.
"""

import math
import random


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def sigmoid_derivative(output):
    return output * (1.0 - output)


def main():
    random.seed(1)

    samples = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    learning_rate = 0.5
    epochs = 5000

    hidden_weights = [
        [random.uniform(-1, 1), random.uniform(-1, 1)],
        [random.uniform(-1, 1), random.uniform(-1, 1)],
    ]
    hidden_biases = [random.uniform(-1, 1), random.uniform(-1, 1)]

    output_weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
    output_bias = random.uniform(-1, 1)

    for epoch in range(epochs):
        total_loss = 0.0

        for inputs, target in samples:
            hidden_outputs = []
            for neuron_index in range(2):
                total = hidden_biases[neuron_index]
                total += inputs[0] * hidden_weights[neuron_index][0]
                total += inputs[1] * hidden_weights[neuron_index][1]
                hidden_outputs.append(sigmoid(total))

            output_total = output_bias
            output_total += hidden_outputs[0] * output_weights[0]
            output_total += hidden_outputs[1] * output_weights[1]
            prediction = sigmoid(output_total)

            error = target - prediction
            total_loss += error * error

            output_delta = error * sigmoid_derivative(prediction)

            hidden_deltas = []
            for neuron_index in range(2):
                hidden_error = output_delta * output_weights[neuron_index]
                hidden_deltas.append(hidden_error * sigmoid_derivative(hidden_outputs[neuron_index]))

            for neuron_index in range(2):
                output_weights[neuron_index] += learning_rate * output_delta * hidden_outputs[neuron_index]
            output_bias += learning_rate * output_delta

            for neuron_index in range(2):
                hidden_weights[neuron_index][0] += learning_rate * hidden_deltas[neuron_index] * inputs[0]
                hidden_weights[neuron_index][1] += learning_rate * hidden_deltas[neuron_index] * inputs[1]
                hidden_biases[neuron_index] += learning_rate * hidden_deltas[neuron_index]

        if (epoch + 1) % 1000 == 0:
            print(f"epoch={epoch + 1}, loss={total_loss:.6f}")

    print("\nFinal predictions:")
    for inputs, target in samples:
        hidden_outputs = []
        for neuron_index in range(2):
            total = hidden_biases[neuron_index]
            total += inputs[0] * hidden_weights[neuron_index][0]
            total += inputs[1] * hidden_weights[neuron_index][1]
            hidden_outputs.append(sigmoid(total))

        output_total = output_bias
        output_total += hidden_outputs[0] * output_weights[0]
        output_total += hidden_outputs[1] * output_weights[1]
        prediction = sigmoid(output_total)

        print(f"{inputs} -> predicted={prediction:.3f}, rounded={round(prediction)}, target={int(target)}")


if __name__ == "__main__":
    main()

