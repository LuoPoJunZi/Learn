"""A simple weighted-sum example for multi-objective optimization.

We want to choose x between 0 and 10.

Objective 1: minimize cost = x^2
Objective 2: minimize risk = (x - 8)^2

The two objectives prefer different x values:

- cost prefers x near 0
- risk prefers x near 8

Weighted sum turns two objectives into one score:

    score = w1 * cost + w2 * risk
"""


def objectives(x):
    cost = x**2
    risk = (x - 8) ** 2
    return cost, risk


def search(weight_cost, weight_risk):
    best_x = None
    best_score = None
    best_values = None

    for index in range(101):
        x = index / 10
        cost, risk = objectives(x)
        score = weight_cost * cost + weight_risk * risk

        if best_score is None or score < best_score:
            best_x = x
            best_score = score
            best_values = (cost, risk)

    return best_x, best_values, best_score


def main():
    weights = [
        (0.9, 0.1),
        (0.7, 0.3),
        (0.5, 0.5),
        (0.3, 0.7),
        (0.1, 0.9),
    ]

    print("weight_cost weight_risk -> best_x, cost, risk, score")
    for weight_cost, weight_risk in weights:
        best_x, (cost, risk), score = search(weight_cost, weight_risk)
        print(
            f"{weight_cost:.1f}         {weight_risk:.1f}         "
            f"-> x={best_x:.1f}, cost={cost:.2f}, risk={risk:.2f}, score={score:.2f}"
        )


if __name__ == "__main__":
    main()

