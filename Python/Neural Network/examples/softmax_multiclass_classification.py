"""Softmax regression for a tiny three-class classification problem.

This example shows how multi-class classification differs from binary
classification:

- binary classification often uses sigmoid
- multi-class classification often uses softmax

The dataset is tiny and hand-written for learning purposes.
"""

import math


def softmax(scores):
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    total = sum(exp_scores)
    return [value / total for value in exp_scores]


def predict_probabilities(inputs, weights, biases):
    scores = []
    for class_index in range(len(weights)):
        score = biases[class_index]
        for x, w in zip(inputs, weights[class_index]):
            score += x * w
        scores.append(score)
    return softmax(scores)


def argmax(values):
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def train(samples, class_count=3, learning_rate=0.2, epochs=1200):
    feature_count = len(samples[0][0])
    weights = [[0.0 for _ in range(feature_count)] for _ in range(class_count)]
    biases = [0.0 for _ in range(class_count)]

    for epoch in range(epochs):
        total_loss = 0.0

        for inputs, target_class in samples:
            probabilities = predict_probabilities(inputs, weights, biases)

            for class_index in range(class_count):
                target = 1.0 if class_index == target_class else 0.0
                error = probabilities[class_index] - target

                if target == 1.0:
                    total_loss -= math.log(max(probabilities[class_index], 1e-12))

                for feature_index in range(feature_count):
                    weights[class_index][feature_index] -= learning_rate * error * inputs[feature_index]
                biases[class_index] -= learning_rate * error

        if (epoch + 1) % 300 == 0:
            print(f"epoch={epoch + 1}, loss={total_loss:.6f}")

    return weights, biases


def main():
    # Three simple classes in a 2D feature space.
    samples = [
        ([1.0, 0.0], 0),
        ([0.9, 0.1], 0),
        ([0.0, 1.0], 1),
        ([0.2, 0.8], 1),
        ([0.8, 0.8], 2),
        ([1.0, 1.0], 2),
    ]

    class_names = ["A", "B", "C"]
    weights, biases = train(samples, class_count=len(class_names))

    print("\nFinal predictions:")
    for inputs, target_class in samples:
        probabilities = predict_probabilities(inputs, weights, biases)
        predicted_class = argmax(probabilities)
        formatted = ", ".join(f"{name}={probabilities[index]:.3f}" for index, name in enumerate(class_names))
        print(
            f"{inputs} -> predicted={class_names[predicted_class]}, "
            f"target={class_names[target_class]}, probabilities=({formatted})"
        )


if __name__ == "__main__":
    main()

