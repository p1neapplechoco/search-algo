from src.swarm_algo.firefly import Firefly
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt

TSP_DATA_FOLDER = "data/tsp/"
NUM_FIREFLIES = 100
NUM_GENERATIONS = 1000
BETA = 1.0  # attractiveness coefficient
GAMMA = 0.01  # light absorption coefficient
ALPHA = 0.5  # randomization parameter


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


def tsp_firefly(PROBLEM: int, visualize: bool = False):
    """
    Solve TSP problem using Firefly Algorithm
    """
    distance_matrix, solution = get_problem_infos(PROBLEM)
    num_cities = len(distance_matrix)

    def fitness_function(continuous_vector):
        """
        Fitness function for TSP.
        Convert continuous vector to permutation and calculate tour distance.
        We want to minimize distance, so return negative distance as fitness.
        """
        path = continuous_to_permutation(continuous_vector)

        # Calculate tour distance (including return to start)
        tour_distance = 0
        for i in range(num_cities):
            from_city = path[i]
            to_city = path[(i + 1) % num_cities]
            tour_distance += distance_matrix[from_city, to_city]

        # Return negative distance (we maximize fitness, but minimize distance)
        return -tour_distance

    # Initialize Firefly Algorithm
    firefly = Firefly(
        ndim=num_cities,
        num_fireflies=NUM_FIREFLIES,
        beta=BETA,
        gamma=GAMMA,
        alpha=ALPHA,
        problem_type="continuous",  # Use continuous for permutation encoding
    )

    firefly.set_objective_function(fitness_function)

    print(f"\nSolving TSP Problem {PROBLEM} with {num_cities} cities...")
    print(
        f"Parameters: fireflies={NUM_FIREFLIES}, generations={NUM_GENERATIONS}, β={BETA}, γ={GAMMA}, α={ALPHA}"
    )

    # Track progress manually for custom progress bar
    best_position = None
    best_fitness = -np.inf
    best_fitness_history = []
    avg_fitness_history = []

    for generation in tqdm(range(NUM_GENERATIONS), desc="Firefly Algorithm Progress"):
        for i in range(NUM_FIREFLIES):
            firefly.intensities[i] = fitness_function(firefly.positions[i])

            if firefly.intensities[i] > best_fitness:
                best_fitness = firefly.intensities[i]
                best_position = firefly.positions[i].copy()

        firefly.update_positions()

        # Track progress
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(np.mean(firefly.intensities))

    # Calculate optimal solution distance
    optimal_distance = calculate_path_distance(solution, distance_matrix)

    # Results
    print(f"\nProblem {PROBLEM}: {num_cities} cities")
    if best_position is not None:
        # Convert best position to path
        best_path = continuous_to_permutation(best_position)
        best_distance = -best_fitness  # Convert back to positive distance

        print(f"Firefly solution distance: {best_distance:.2f}")
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

        # Convert fitness back to distance (negate values)
        best_distance_history = [-f for f in best_fitness_history]
        avg_distance_history = [-f for f in avg_fitness_history]

        plt.plot(
            generations_range,
            best_distance_history,
            "g-",
            linewidth=2,
            label="Best Distance",
        )
        plt.plot(
            generations_range,
            avg_distance_history,
            "orange",
            linestyle="--",
            linewidth=1.5,
            label="Average Distance",
        )
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Distance", fontsize=12)
        plt.title(
            f"Firefly Algorithm - Problem {PROBLEM} Convergence",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Set visualize=True to see the convergence plot
    tsp_firefly(PROBLEM=1, visualize=True)
