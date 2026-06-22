"""A tiny recurrent neural network forward-pass demo.

This example does not train an RNN. It focuses on the key idea:

    the hidden state carries information from earlier time steps.

The sequence is processed one value at a time. The final hidden state is
then used to classify whether the sequence is mostly increasing.
"""

import math


def tanh(value):
    return math.tanh(value)


def rnn_step(x, hidden, input_weight, hidden_weight, bias):
    return tanh(input_weight * x + hidden_weight * hidden + bias)


def classify_sequence(sequence):
    hidden = 0.0
    input_weight = 0.8
    hidden_weight = 0.6
    bias = 0.0

    print(f"sequence={sequence}")
    for index, value in enumerate(sequence):
        hidden = rnn_step(value, hidden, input_weight, hidden_weight, bias)
        print(f"  step={index + 1}, input={value:+.1f}, hidden={hidden:+.3f}")

    label = "increasing-like" if hidden > 0 else "decreasing-like"
    print(f"  final label: {label}\n")


def main():
    # First differences: positive means rising, negative means falling.
    increasing_differences = [0.2, 0.4, 0.3, 0.5]
    decreasing_differences = [-0.2, -0.4, -0.3, -0.5]
    mixed_differences = [0.5, -0.1, 0.3, -0.2]

    classify_sequence(increasing_differences)
    classify_sequence(decreasing_differences)
    classify_sequence(mixed_differences)


if __name__ == "__main__":
    main()

