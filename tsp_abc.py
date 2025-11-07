from src.swarm_algo.abc import ABC
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt

TSP_DATA_FOLDER = "data/tsp/"
NUM_FOOD_SOURCES = 100  # Number of employed bees (food sources)
NUM_GENERATIONS = 500  # Maximum cycle number
LIMIT = 200  # Abandonment limit for scout bees


def get_problem_infos(PROBLEM: int) -> Tuple[np.ndarray, List[int]]:
    """
    Load TSP problem data
    Returns:
        - distance_matrix: 2D array of distances between cities
        - solution: optimal solution path
    """
    problem = f"p{PROBLEM:02d}"

    # Read distance matrix
    with open(TSP_DATA_FOLDER + problem + "_d.txt") as f:
        lines = f.readlines()
        distance_matrix = []
        for line in lines:
            row = [float(x) for x in line.split()]
            distance_matrix.append(row)
        distance_matrix = np.array(distance_matrix)

    # Read solution path
    with open(TSP_DATA_FOLDER + problem + "_s.txt") as f:
        solution = [int(s.strip()) - 1 for s in f.readlines()]  # Convert to 0-indexed

    return distance_matrix, solution


def calculate_path_distance(path: List[int], distance_matrix: np.ndarray) -> float:
    """Calculate total distance of a path"""
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i], path[i + 1]]
    return total_distance


def relativity_to_solution(ans_distance: float, sol_distance: float) -> float:
    """Calculate how close the answer is to the optimal solution"""
    return sol_distance / ans_distance


def continuous_to_permutation(continuous_vector: np.ndarray) -> List[int]:
    """
    Convert continuous vector to permutation using random key encoding.
    The city with smallest value goes first, etc.
    """
    return np.argsort(continuous_vector).tolist()


def tsp_abc(PROBLEM: int, visualize: bool = False):
    """
    Solve TSP problem using Artificial Bee Colony Algorithm
    """
    distance_matrix, solution = get_problem_infos(PROBLEM)
    num_cities = len(distance_matrix)

    def fitness_function(continuous_vector):
        """
        Fitness function for TSP.
        Convert continuous vector to permutation and calculate tour distance.
        ABC's calc_fitness will convert this to fitness (lower distance = higher fitness).
        """
        path = continuous_to_permutation(continuous_vector)

        # Calculate tour distance (including return to start)
        tour_distance = 0
        for i in range(num_cities):
            from_city = path[i]
            to_city = path[(i + 1) % num_cities]
            tour_distance += distance_matrix[from_city, to_city]

        # Return distance directly (ABC's calc_fitness will handle conversion)
        return tour_distance

    # Initialize ABC Algorithm
    abc = ABC(
        dimension=num_cities,
        sn=NUM_FOOD_SOURCES,
        mcn=NUM_GENERATIONS,
        limit=LIMIT,
        lb=0.0,
        ub=1.0,
    )

    abc.set_objective_function(fitness_function)

    print(f"\nSolving TSP Problem {PROBLEM} with {num_cities} cities...")
    print(
        f"Parameters: food_sources={NUM_FOOD_SOURCES}, max_cycles={NUM_GENERATIONS}, limit={LIMIT}"
    )

    # Track progress manually for custom progress bar
    best_solution = None
    best_fitness = -np.inf
    best_fitness_history = []
    avg_fitness_history = []

    # Initialize fitness values
    for i in range(abc.sn):
        abc.fitness[i] = abc.calc_fitness(abc.solutions[i])

    cyc = 1
    pbar = tqdm(total=NUM_GENERATIONS, desc="ABC Algorithm Progress")

    while cyc < abc.mcn:
        # Employed bees phase
        for i in range(abc.sn):
            new_solution = abc.local_search(i)
            new_fit = abc.calc_fitness(new_solution)

            if new_fit > abc.fitness[i]:
                abc.solutions[i] = new_solution
                abc.fitness[i] = new_fit
                abc.trial[i] = 0
            else:
                abc.trial[i] += 1

        abc.calc_probabilities()

        # Onlooker bees phase
        i = 0
        t = 0
        while t < abc.sn:
            if np.random.rand(1) < abc.probs[i]:
                t += 1
                new_solution = abc.local_search(i)
                new_fit = abc.calc_fitness(new_solution)

                if new_fit > abc.fitness[i]:
                    abc.solutions[i] = new_solution
                    abc.fitness[i] = new_fit
                    abc.trial[i] = 0
                else:
                    abc.trial[i] += 1

            i = (i + 1) % abc.sn

        # Update best solution
        best_idx = np.argmax(abc.fitness)
        if abc.fitness[best_idx] > best_fitness:
            best_fitness = abc.fitness[best_idx]
            best_solution = np.copy(abc.solutions[best_idx])

        # Scout bee phase
        si = np.argmax(abc.trial)
        if abc.trial[si] > abc.limit:
            abc.solutions[si] = abc.randomize_solution()
            abc.fitness[si] = abc.calc_fitness(abc.solutions[si])
            abc.trial[si] = 0

        # Track progress
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(np.mean(abc.fitness))

        cyc += 1
        pbar.update(1)

    pbar.close()

    # Calculate optimal solution distance
    optimal_distance = calculate_path_distance(solution, distance_matrix)

    # Results
    print(f"\nProblem {PROBLEM}: {num_cities} cities")
    if best_solution is not None:
        # Convert best solution to path
        best_path = continuous_to_permutation(best_solution)
        # Calculate actual distance from the path
        best_distance = 0
        for i in range(num_cities):
            from_city = best_path[i]
            to_city = best_path[(i + 1) % num_cities]
            best_distance += distance_matrix[from_city, to_city]

        print(f"ABC solution distance: {best_distance:.2f}")
        print(f"Optimal solution distance: {optimal_distance:.2f}")
        print(
            f"Accuracy: {relativity_to_solution(best_distance, optimal_distance) * 100:.2f}%"
        )
        print(
            f"Best path found: {[x + 1 for x in best_path]}"
        )  # Convert back to 1-indexed
        print(f"Optimal path: {[x + 1 for x in solution]}")
    else:
        print("No valid solution found!")

    if visualize:
        plt.figure(figsize=(10, 6))
        generations_range = range(len(best_fitness_history))

        # Convert fitness back to distance
        # fitness = 1/(1+distance), so distance = (1/fitness) - 1
        best_distance_history = [(1.0 / f) - 1.0 for f in best_fitness_history]
        avg_distance_history = [(1.0 / f) - 1.0 for f in avg_fitness_history]

        plt.plot(
            generations_range,
            best_distance_history,
            "b-",
            linewidth=2,
            label="Best Distance",
        )
        plt.plot(
            generations_range,
            avg_distance_history,
            "r--",
            linewidth=1.5,
            label="Average Distance",
        )
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Distance", fontsize=12)
        plt.title(
            f"ABC Algorithm - Problem {PROBLEM} Convergence",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Set visualize=True to see the convergence plot
    tsp_abc(PROBLEM=4, visualize=True)
