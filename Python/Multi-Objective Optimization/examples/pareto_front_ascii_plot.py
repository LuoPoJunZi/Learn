"""Pareto front visualization without third-party libraries.

The script prints a small ASCII chart so beginners can see where
non-dominated solutions sit in objective space.
"""


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_front(points):
    front = []
    for candidate in points:
        if not any(dominates(other, candidate) for other in points):
            front.append(candidate)
    return front


def draw(points, front, width=40, height=14):
    max_x = max(x for x, _ in points)
    max_y = max(y for _, y in points)
    front_set = set(front)

    grid = [[" " for _ in range(width + 1)] for _ in range(height + 1)]

    for point in points:
        x, y = point
        col = round(x / max_x * width)
        row = height - round(y / max_y * height)
        grid[row][col] = "*" if point in front_set else "."

    print("Objective 2 (lower is better)")
    for row in grid:
        print("|" + "".join(row))
    print("+" + "-" * (width + 1))
    print(" Objective 1 (lower is better)")
    print("* = Pareto front, . = dominated solution")


def main():
    points = [
        (1, 9), (2, 7), (3, 6), (4, 5), (5, 4),
        (6, 4), (7, 3), (8, 2), (9, 1), (6, 7),
        (4, 8), (8, 6), (3, 9), (9, 5),
    ]
    front = pareto_front(points)
    print("Pareto front:", front)
    draw(points, front)


if __name__ == "__main__":
    main()
