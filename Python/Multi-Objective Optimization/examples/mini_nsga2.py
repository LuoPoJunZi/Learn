"""A tiny NSGA-II style demo using pure Python.

This is intentionally small and educational. It keeps the key ideas:

- population
- objective functions
- non-dominated sorting
- crowding distance
- selection
- crossover and mutation

Problem:

    minimize f1(x) = x^2
    minimize f2(x) = (x - 2)^2

The two objectives prefer different values:

- f1 prefers x near 0
- f2 prefers x near 2
"""

import random


def objectives(x):
    return x * x, (x - 2) * (x - 2)


def dominates(a, b):
    a_values = a["objectives"]
    b_values = b["objectives"]
    no_worse = all(a_value <= b_value for a_value, b_value in zip(a_values, b_values))
    better = any(a_value < b_value for a_value, b_value in zip(a_values, b_values))
    return no_worse and better


def make_individual(x):
    return {
        "x": x,
        "objectives": objectives(x),
        "rank": None,
        "crowding": 0.0,
    }


def non_dominated_sort(population):
    fronts = []
    domination_counts = {}
    dominated_solutions = {}
    first_front = []

    for p_index, p in enumerate(population):
        dominated_solutions[p_index] = []
        domination_counts[p_index] = 0

        for q_index, q in enumerate(population):
            if p_index == q_index:
                continue
            if dominates(p, q):
                dominated_solutions[p_index].append(q_index)
            elif dominates(q, p):
                domination_counts[p_index] += 1

        if domination_counts[p_index] == 0:
            p["rank"] = 0
            first_front.append(p_index)

    fronts.append(first_front)
    current_rank = 0

    while fronts[current_rank]:
        next_front = []
        for p_index in fronts[current_rank]:
            for q_index in dominated_solutions[p_index]:
                domination_counts[q_index] -= 1
                if domination_counts[q_index] == 0:
                    population[q_index]["rank"] = current_rank + 1
                    next_front.append(q_index)
        current_rank += 1
        fronts.append(next_front)

    return fronts[:-1]


def assign_crowding_distance(population, front):
    if not front:
        return

    for index in front:
        population[index]["crowding"] = 0.0

    objective_count = len(population[front[0]]["objectives"])

    for objective_index in range(objective_count):
        front.sort(key=lambda index: population[index]["objectives"][objective_index])

        population[front[0]]["crowding"] = float("inf")
        population[front[-1]]["crowding"] = float("inf")

        min_value = population[front[0]]["objectives"][objective_index]
        max_value = population[front[-1]]["objectives"][objective_index]

        if max_value == min_value:
            continue

        for position in range(1, len(front) - 1):
            previous_value = population[front[position - 1]]["objectives"][objective_index]
            next_value = population[front[position + 1]]["objectives"][objective_index]
            population[front[position]]["crowding"] += (next_value - previous_value) / (max_value - min_value)


def sort_population(population):
    fronts = non_dominated_sort(population)
    for front in fronts:
        assign_crowding_distance(population, front)

    return sorted(population, key=lambda item: (item["rank"], -item["crowding"]))


def tournament_select(population):
    a, b = random.sample(population, 2)
    if a["rank"] < b["rank"]:
        return a
    if b["rank"] < a["rank"]:
        return b
    return a if a["crowding"] > b["crowding"] else b


def crossover(parent_a, parent_b):
    alpha = random.random()
    x = alpha * parent_a["x"] + (1 - alpha) * parent_b["x"]
    return x


def mutate(x, mutation_rate=0.2):
    if random.random() < mutation_rate:
        x += random.uniform(-0.25, 0.25)
    return max(-1.0, min(3.0, x))


def main():
    random.seed(7)

    population_size = 24
    generations = 30

    population = [make_individual(random.uniform(-1.0, 3.0)) for _ in range(population_size)]
    population = sort_population(population)

    for generation in range(generations):
        children = []

        while len(children) < population_size:
            parent_a = tournament_select(population)
            parent_b = tournament_select(population)
            child_x = mutate(crossover(parent_a, parent_b))
            children.append(make_individual(child_x))

        population = sort_population(population + children)[:population_size]

        if (generation + 1) % 10 == 0:
            first_front_count = sum(1 for item in population if item["rank"] == 0)
            print(f"generation={generation + 1}, first_front_count={first_front_count}")

    front = [item for item in population if item["rank"] == 0]
    front = sorted(front, key=lambda item: item["x"])

    print("\nApproximate Pareto front:")
    seen = set()
    for item in front:
        rounded_x = round(item["x"], 3)
        if rounded_x in seen:
            continue
        seen.add(rounded_x)
        f1, f2 = item["objectives"]
        print(f"x={item['x']:.3f}, f1={f1:.3f}, f2={f2:.3f}")


if __name__ == "__main__":
    main()
