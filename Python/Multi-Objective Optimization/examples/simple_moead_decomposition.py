"""A tiny MOEA/D-style decomposition example using pure Python.

MOEA/D decomposes a multi-objective problem into many scalar subproblems.
Each subproblem uses a different weight vector.

Problem:

    minimize f1(x) = x^2
    minimize f2(x) = (x - 2)^2

This educational version uses:

- several weight vectors
- Tchebycheff scalarization
- simple neighbor update
- mutation-only search
"""

import random


def objectives(x):
    return x * x, (x - 2) * (x - 2)


def tchebycheff(values, weights, ideal_point):
    scores = []
    for value, weight, ideal in zip(values, weights, ideal_point):
        scores.append(weight * abs(value - ideal))
    return max(scores)


def make_solution(x):
    return {
        "x": x,
        "objectives": objectives(x),
    }


def mutate(x):
    x += random.uniform(-0.15, 0.15)
    return max(-1.0, min(3.0, x))


def main():
    random.seed(11)

    weight_vectors = [
        (0.0, 1.0),
        (0.2, 0.8),
        (0.4, 0.6),
        (0.6, 0.4),
        (0.8, 0.2),
        (1.0, 0.0),
    ]

    population = [make_solution(random.uniform(-1.0, 3.0)) for _ in weight_vectors]
    ideal_point = [0.0, 0.0]

    generations = 80

    for generation in range(generations):
        for index, weights in enumerate(weight_vectors):
            current = population[index]
            child = make_solution(mutate(current["x"]))

            ideal_point[0] = min(ideal_point[0], child["objectives"][0])
            ideal_point[1] = min(ideal_point[1], child["objectives"][1])

            neighbor_indexes = [
                max(0, index - 1),
                index,
                min(len(population) - 1, index + 1),
            ]

            for neighbor_index in neighbor_indexes:
                neighbor_weights = weight_vectors[neighbor_index]
                old_score = tchebycheff(population[neighbor_index]["objectives"], neighbor_weights, ideal_point)
                new_score = tchebycheff(child["objectives"], neighbor_weights, ideal_point)

                if new_score < old_score:
                    population[neighbor_index] = child

        if (generation + 1) % 20 == 0:
            print(f"generation={generation + 1}")

    print("\nFinal decomposed solutions:")
    for weights, solution in zip(weight_vectors, population):
        f1, f2 = solution["objectives"]
        print(f"weights={weights}, x={solution['x']:.3f}, f1={f1:.3f}, f2={f2:.3f}")


if __name__ == "__main__":
    main()

