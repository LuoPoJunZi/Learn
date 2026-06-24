"""Small SPEA2 strength-fitness demo using only the standard library.

SPEA2 assigns each solution a strength based on how many other solutions it
dominates. A solution's raw fitness is influenced by the strengths of solutions
that dominate it.
"""


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def strength_values(points):
    strengths = []
    for point in points:
        strength = sum(1 for other in points if dominates(point, other))
        strengths.append(strength)
    return strengths


def raw_fitness_values(points, strengths):
    raw_fitness = []
    for point in points:
        fitness = sum(
            strengths[index]
            for index, other in enumerate(points)
            if dominates(other, point)
        )
        raw_fitness.append(fitness)
    return raw_fitness


def main():
    points = [
        (1, 9),
        (2, 7),
        (3, 6),
        (4, 5),
        (5, 8),
        (7, 7),
        (8, 3),
        (9, 2),
    ]
    strengths = strength_values(points)
    raw_fitness = raw_fitness_values(points, strengths)

    print("lower objectives are better")
    print("point      strength  raw_fitness")
    print("-" * 34)
    for point, strength, fitness in zip(points, strengths, raw_fitness):
        mark = "front" if fitness == 0 else "dominated"
        print(f"{point!s:<10} {strength:^8} {fitness:^11} {mark}")


if __name__ == "__main__":
    main()
