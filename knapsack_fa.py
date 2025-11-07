from src.swarm_algo.firefly import Firefly
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt

KNAPSACK_DATA_FOLDER = "data/knapsack/"
MAX_POPULATION = 100
NUM_OF_GENERATIONS = 100


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


def knapsack_firefly(PROBLEM: int, visualize: bool = False):
    capacity, items, weights, solution = get_problem_infos(PROBLEM)
    num_items = len(items)

    def fitness_function(ans):
        total_weight = 0
        total_profit = 0
        for i in range(num_items):
            if ans[i]:
                total_weight += weights[i]
                total_profit += items[i]

        if total_weight > capacity:
            penalty = (total_weight - capacity) * max(items)
            return max(0, total_profit - penalty)

        return total_profit

    firefly = Firefly(
        ndim=num_items,
        num_fireflies=MAX_POPULATION,
        beta=1.0,
        gamma=0.01,  # Lower gamma for slower intensity decay
        alpha=0.5,  # Higher alpha for more exploration
        problem_type="binary",  # Knapsack is a binary problem
    )

    firefly.set_objective_function(fitness_function)

    answer, profit, history = firefly.run(NUM_OF_GENERATIONS, visualize=visualize)

    optimal_profit = objective_function(solution, items)

    print(f"Problem {PROBLEM}: {num_items} items, capacity {capacity}")
    print(f"Firefly solution profit: {profit}")
    print(f"Optimal solution profit: {optimal_profit}")
    print(f"Accuracy: {relativity_to_solution(answer, solution, items) * 100:.2f}%")
    print(f"Items selected: {np.sum(answer)}/{num_items}")


if __name__ == "__main__":
    # Set visualize=True to see the convergence plot
    knapsack_firefly(PROBLEM=7, visualize=True)
