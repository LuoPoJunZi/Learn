"""Find a Pareto front from candidate solutions using pure Python.

In this example every objective is minimized.

A solution A dominates B if:

- A is no worse than B in every objective
- A is better than B in at least one objective
"""


def dominates(a, b):
    a_values = a["objectives"]
    b_values = b["objectives"]

    no_worse = all(a_value <= b_value for a_value, b_value in zip(a_values, b_values))
    strictly_better = any(a_value < b_value for a_value, b_value in zip(a_values, b_values))

    return no_worse and strictly_better


def pareto_front(candidates):
    front = []

    for candidate in candidates:
        dominated = False

        for other in candidates:
            if other is candidate:
                continue
            if dominates(other, candidate):
                dominated = True
                break

        if not dominated:
            front.append(candidate)

    return front


def main():
    candidates = [
        {"name": "A", "objectives": (10, 90)},
        {"name": "B", "objectives": (20, 60)},
        {"name": "C", "objectives": (30, 40)},
        {"name": "D", "objectives": (40, 25)},
        {"name": "E", "objectives": (50, 20)},
        {"name": "F", "objectives": (35, 70)},
        {"name": "G", "objectives": (60, 80)},
    ]

    print("All candidates: lower objective values are better")
    for item in candidates:
        cost, time = item["objectives"]
        print(f"{item['name']}: cost={cost}, time={time}")

    print("\nPareto front:")
    for item in pareto_front(candidates):
        cost, time = item["objectives"]
        print(f"{item['name']}: cost={cost}, time={time}")


if __name__ == "__main__":
    main()

