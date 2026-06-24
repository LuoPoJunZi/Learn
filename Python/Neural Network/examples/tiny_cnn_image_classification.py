"""Tiny CNN-style image classification demo using only the standard library.

This is not a production CNN. It teaches the core idea:
small image -> convolution features -> simple classifier.
"""

from math import exp


IMAGES = [
    ("x", [[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]]),
    ("x", [[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]]),
    ("o", [[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]]),
    ("o", [[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]]),
]

KERNELS = {
    "diag_down": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "diag_up": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    "vertical_edges": [[1, 0, 1], [1, 0, 1], [1, 0, 1]],
    "horizontal_edges": [[1, 1, 1], [0, 0, 0], [1, 1, 1]],
}


def convolve_sum(image, kernel):
    total = 0
    for row in range(3):
        for col in range(3):
            patch_score = 0
            for kr in range(3):
                for kc in range(3):
                    patch_score += image[row + kr][col + kc] * kernel[kr][kc]
            total += patch_score
    return total / 9


def extract_features(image):
    return [convolve_sum(image, kernel) for kernel in KERNELS.values()]


def sigmoid(value):
    return 1 / (1 + exp(-value))


def predict(features, weights, bias):
    score = sum(feature * weight for feature, weight in zip(features, weights)) + bias
    return sigmoid(score)


def train(dataset, epochs=120, learning_rate=0.08):
    weights = [0.0 for _ in KERNELS]
    bias = 0.0

    for _ in range(epochs):
        for label, image in dataset:
            target = 1 if label == "x" else 0
            features = extract_features(image)
            probability = predict(features, weights, bias)
            error = probability - target

            for index, feature in enumerate(features):
                weights[index] -= learning_rate * error * feature
            bias -= learning_rate * error

    return weights, bias


def print_image(image):
    for row in image:
        print("".join("#" if value else "." for value in row))


def main():
    weights, bias = train(IMAGES)
    print("feature order:", ", ".join(KERNELS))
    print("weights:", [round(weight, 3) for weight in weights], "bias:", round(bias, 3))
    print()

    for label, image in IMAGES[::2]:
        features = extract_features(image)
        probability_x = predict(features, weights, bias)
        predicted = "x" if probability_x >= 0.5 else "o"
        print_image(image)
        print("expected:", label, "predicted:", predicted, "probability_x:", round(probability_x, 3))
        print()


if __name__ == "__main__":
    main()
