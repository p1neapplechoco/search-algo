from src.search_algo.genetic import Genetic
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt


KNAPSACK_DATA_FOLDER = "data/knapsack/"
MAX_POPULATION = 500
NUM_OF_GENERATIONS = 100
ELITE_RATE = 0.1
MUTATION_RATE = 0.05
NUM_CHILDREN = 3


def get_problem_infos(PROBLEM: int) -> Tuple[int, List[int], List[int], List[int]]:
    problem = f"p{PROBLEM:02d}"

    with open(KNAPSACK_DATA_FOLDER + problem + "_c.txt") as f:
        capacity = int(f.read())

    with open(KNAPSACK_DATA_FOLDER + problem + "_p.txt") as f:
        items = [int(p) for p in f.readlines()]

    with open(KNAPSACK_DATA_FOLDER + problem + "_w.txt") as f:
        weights = [int(w) for w in f.readlines()]

    with open(KNAPSACK_DATA_FOLDER + problem + "_s.txt") as f:
        solution = [int(s) for s in f.readlines()]

    return capacity, items, weights, solution


def relativity_to_solution(ans, sol, items):
    ans_profit = 0
    sol_profit = 0

    for i in range(len(items)):
        if ans[i]:
            ans_profit += items[i]
        if sol[i]:
            sol_profit += items[i]

    return ans_profit / sol_profit


def objective_function(ans, items):
    total_profit = 0
    for i in range(len(items)):
        if ans[i]:
            total_profit += items[i]
    return total_profit


def knapsack_ga(PROBLEM: int, visualize: bool = False):
    capacity, items, weights, solution = get_problem_infos(PROBLEM)
    num_items = len(items)

    def fitness_function(ans):
        total_weight = 0
        total_profit = 0
        for i in range(num_items):
            if ans[i]:
                total_weight += weights[i]
                total_profit += items[i]
        # Penalty for exceeding capacity - return 0 instead of negative
        if total_weight > capacity:
            return 0
        return total_profit

    ga = Genetic(
        population_size=MAX_POPULATION,
        mutation_rate=MUTATION_RATE,
        elite_rate=ELITE_RATE,
        num_children=NUM_CHILDREN,
    )

    ga.set_fitness_function(fitness_function)

    population = np.random.randint(2, size=(MAX_POPULATION, num_items))
    best_fitness_progress = []
    avg_fitness_progress = []

    for generation in tqdm(
        range(NUM_OF_GENERATIONS), desc="Genetic Algorithm Progress"
    ):
        population = ga.create_next_generation(population)
        fitness_scores = np.array([fitness_function(ind) for ind in population])
        best_fitness = np.max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        best_fitness_progress.append(best_fitness)
        avg_fitness_progress.append(avg_fitness)

    best_index = np.argmax(fitness_scores)
    best_solution = population[best_index]
    best_profit = fitness_function(best_solution)
    relativity = relativity_to_solution(best_solution, solution, items)

    optimal_profit = objective_function(solution, items)

    print(f"\nProblem {PROBLEM}: {num_items} items, capacity {capacity}")
    print(f"GA solution profit: {best_profit}")
    print(f"Optimal solution profit: {optimal_profit}")
    print(f"Accuracy: {relativity * 100:.2f}%")
    print(f"Items selected: {np.sum(best_solution)}/{num_items}")

    if visualize:
        plt.figure(figsize=(10, 6))
        generations = range(len(best_fitness_progress))
        plt.plot(
            generations, best_fitness_progress, "g-", linewidth=2, label="Best Fitness"
        )
        plt.plot(
            generations,
            avg_fitness_progress,
            "orange",
            linestyle="--",
            linewidth=1.5,
            label="Average Fitness",
        )
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Fitness Value", fontsize=12)
        plt.title(
            f"Genetic Algorithm - Problem {PROBLEM} Convergence",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Set visualize=True to see the convergence plot
    knapsack_ga(PROBLEM=7, visualize=True)
