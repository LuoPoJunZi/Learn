"""Serialize and load a tiny trained model using JSON.

Real frameworks such as PyTorch and TensorFlow have their own model
serialization formats. This pure Python example shows the same core idea:

1. Train a model.
2. Serialize parameters to JSON text.
3. Load parameters from JSON text.
4. Use the loaded model for prediction.
"""

import json


def predict(x, weight, bias):
    return weight * x + bias


def train(samples, learning_rate=0.01, epochs=500):
    weight = 0.0
    bias = 0.0

    for _ in range(epochs):
        weight_gradient = 0.0
        bias_gradient = 0.0

        for x, target in samples:
            error = predict(x, weight, bias) - target
            weight_gradient += 2 * error * x
            bias_gradient += 2 * error

        weight -= learning_rate * weight_gradient / len(samples)
        bias -= learning_rate * bias_gradient / len(samples)

    return {"weight": weight, "bias": bias}


def serialize_model(model):
    return json.dumps(model, indent=2)


def load_model(json_text):
    return json.loads(json_text)


def main():
    samples = [
        (0.0, 1.0),
        (1.0, 3.0),
        (2.0, 5.0),
        (3.0, 7.0),
    ]

    model = train(samples)

    json_text = serialize_model(model)
    loaded_model = load_model(json_text)

    print("Serialized model:")
    print(json_text)
    print(f"Loaded parameters: {loaded_model}")

    for x in [4.0, 5.0]:
        output = predict(x, loaded_model["weight"], loaded_model["bias"])
        print(f"x={x:.1f} -> predicted={output:.3f}")


if __name__ == "__main__":
    main()
