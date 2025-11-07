from swarm_algo.abc import ABC
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt

KNAPSACK_DATA_FOLDER = "data/knapsack/"
MAX_POPULATION = 100
NUM_OF_GENERATIONS = 500


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


def knapsack_abc(PROBLEM: int):
    capacity, items, weights, solution = get_problem_infos(PROBLEM)
    num_items = len(items)

    def fitness_function(ans):
        # Convert continuous values to binary
        binary_ans = (ans > 0.5).astype(int)
        
        total_weight = 0
        total_profit = 0
        for i in range(num_items):
            if binary_ans[i]:
                total_weight += weights[i]
                total_profit += items[i]

        # Penalty for exceeding capacity
        if total_weight > capacity:
            # Penalize based on how much we exceed
            penalty = (total_weight - capacity) * max(items)
            return max(0, total_profit - penalty)

        return total_profit

    abc = ABC(
        dimension=num_items,
        sn=MAX_POPULATION,
        mcn=NUM_OF_GENERATIONS,
        limit=100,
        lb=0.0,
        ub=1.0,
        objective_function=fitness_function,
    )

    # abc.set_objective_function(fitness_function)

    answer, fitness = abc.run()
    
    # Convert to binary solution
    binary_answer = (answer > 0.5).astype(int)
    profit = objective_function(binary_answer, items)

    optimal_profit = objective_function(solution, items)

    print(f"Problem {PROBLEM}: {num_items} items, capacity {capacity}")
    print(f"ABC solution profit: {profit}")
    print(f"Optimal solution profit: {optimal_profit}")
    print(f"Accuracy: {relativity_to_solution(binary_answer, solution, items) * 100:.2f}%")
    print(f"Items selected: {np.sum(binary_answer)}/{num_items}")


if __name__ == "__main__":
    knapsack_abc(PROBLEM=18)
