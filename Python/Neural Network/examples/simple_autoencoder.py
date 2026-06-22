"""A tiny autoencoder-style example using pure Python.

An autoencoder learns to reconstruct its input. This toy version uses:

- 3 input values
- 2 hidden values
- 3 output values

The hidden layer acts like a compressed representation.
"""

import math
import random


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def sigmoid_derivative(output):
    return output * (1.0 - output)


def forward(inputs, encoder_weights, encoder_biases, decoder_weights, decoder_biases):
    hidden = []
    for hidden_index in range(2):
        total = encoder_biases[hidden_index]
        for input_index in range(3):
            total += inputs[input_index] * encoder_weights[hidden_index][input_index]
        hidden.append(sigmoid(total))

    outputs = []
    for output_index in range(3):
        total = decoder_biases[output_index]
        for hidden_index in range(2):
            total += hidden[hidden_index] * decoder_weights[output_index][hidden_index]
        outputs.append(sigmoid(total))

    return hidden, outputs


def main():
    random.seed(3)

    samples = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    encoder_weights = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(2)]
    encoder_biases = [random.uniform(-1, 1) for _ in range(2)]
    decoder_weights = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(3)]
    decoder_biases = [random.uniform(-1, 1) for _ in range(3)]

    learning_rate = 0.8
    epochs = 8000

    for epoch in range(epochs):
        total_loss = 0.0

        for inputs in samples:
            hidden, outputs = forward(inputs, encoder_weights, encoder_biases, decoder_weights, decoder_biases)

            output_deltas = []
            for output_index in range(3):
                error = outputs[output_index] - inputs[output_index]
                total_loss += error * error
                output_deltas.append(error * sigmoid_derivative(outputs[output_index]))

            hidden_deltas = []
            for hidden_index in range(2):
                error = 0.0
                for output_index in range(3):
                    error += output_deltas[output_index] * decoder_weights[output_index][hidden_index]
                hidden_deltas.append(error * sigmoid_derivative(hidden[hidden_index]))

            for output_index in range(3):
                for hidden_index in range(2):
                    decoder_weights[output_index][hidden_index] -= learning_rate * output_deltas[output_index] * hidden[hidden_index]
                decoder_biases[output_index] -= learning_rate * output_deltas[output_index]

            for hidden_index in range(2):
                for input_index in range(3):
                    encoder_weights[hidden_index][input_index] -= learning_rate * hidden_deltas[hidden_index] * inputs[input_index]
                encoder_biases[hidden_index] -= learning_rate * hidden_deltas[hidden_index]

        if (epoch + 1) % 2000 == 0:
            print(f"epoch={epoch + 1}, loss={total_loss:.6f}")

    print("\nReconstruction results:")
    for inputs in samples:
        hidden, outputs = forward(inputs, encoder_weights, encoder_biases, decoder_weights, decoder_biases)
        rounded_outputs = [round(value) for value in outputs]
        print(
            f"input={inputs}, hidden={[round(value, 3) for value in hidden]}, "
            f"output={[round(value, 3) for value in outputs]}, rounded={rounded_outputs}"
        )


if __name__ == "__main__":
    main()

