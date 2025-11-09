from src.swarm_algo.aco import AntColonyOptimizer as ACO
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
import matplotlib.pyplot as plt

TSP_DATA_FOLDER = "data/tsp/"
NUM_ANTS = 50
NUM_ITERATIONS = 10
ALPHA = 1.0  # pheromone importance
BETA = 5.0  # distance importance
RHO = 0.5  # evaporation rate
Q = 100  # pheromone deposit factor


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
        # Convert to 0-indexed
        solution = [int(s.strip()) - 1 for s in f.readlines()]

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


def tsp_aco(PROBLEM: int, visualize: bool = False):
    """
    Solve TSP problem using Ant Colony Optimization
    """
    distance_matrix, solution = get_problem_infos(PROBLEM)
    num_cities = len(distance_matrix)

    # Create city coordinates (for visualization if needed)
    # Since we only have distance matrix, we can treat cities as indices
    colony = np.array([[i, 0] for i in range(num_cities)])  # Dummy coordinates

    # Initialize ACO with distance matrix directly
    aco = ACO(
        colony=colony,
        num_ant=NUM_ANTS,
        iter=NUM_ITERATIONS,
        alpha=ALPHA,
        beta=BETA,
        rho=RHO,
        Q=Q,
    )

    # Replace the distance matrix with the actual one from file
    aco.distance_matrix = distance_matrix
    aco.zeta = np.where(distance_matrix > 0, 1.0 / distance_matrix, 0)

    # # Track progress for visualization
    # best_fitness_progress = []
    # avg_fitness_progress = []

    # print(f"\nSolving TSP Problem {PROBLEM} with {num_cities} cities...")
    # print(
    #     f"Parameters: ants={NUM_ANTS}, iterations={NUM_ITERATIONS}, α={ALPHA}, β={BETA}, ρ={RHO}, Q={Q}"
    # )

    # # Run ACO with progress tracking
    # best_path = None
    # best_fitness = float("inf")

    # for iteration in tqdm(range(NUM_ITERATIONS), desc="ACO Progress"):
    #     all_paths = []
    #     all_ant_paths = []
    #     all_fitness = []

    #     for ant in range(NUM_ANTS):
    #         cur_idx = np.random.randint(0, num_cities)
    #         visitted = {cur_idx}
    #         path = [cur_idx]

    #         while len(visitted) < num_cities:
    #             cur_idx = aco.RWS(cur_idx, visitted)
    #             if cur_idx is None:
    #                 break
    #             path.append(cur_idx)
    #             visitted.add(cur_idx)

    #         if len(path) == num_cities:
    #             ant_path = aco.path_to_ant_path(path)
    #             fit = aco.fitness(ant_path)

    #             all_paths.append(path)
    #             all_ant_paths.append(ant_path)
    #             all_fitness.append(fit)

    #             if fit < best_fitness:
    #                 best_fitness = fit
    #                 best_path = path.copy()

    #     if all_ant_paths:
    #         aco.update_pheromone(all_ant_paths)

    #         # Track progress
    #         best_fitness_progress.append(best_fitness)
    #         if all_fitness:
    #             avg_fitness_progress.append(np.mean(all_fitness))
    #         else:
    #             avg_fitness_progress.append(best_fitness)

    best_path, best_fitness = aco.run()
    # # Calculate optimal solution distance
    optimal_distance = calculate_path_distance(solution, distance_matrix)

    # Results
    print(f"\nProblem {PROBLEM}: {num_cities} cities")
    if best_path is not None:
        print(f"ACO solution distance: {best_fitness:.2f}")
        print(f"Optimal solution distance: {optimal_distance:.2f}")
        print(
            f"Accuracy: {relativity_to_solution(best_fitness, optimal_distance) * 100:.2f}%"
        )
        print(
            f"Best path found: {[x + 1 for x in best_path]}"
        )  # Convert back to 1-indexed
        print(f"Optimal path: {[x + 1 for x in solution]}")
    else:
        print("No valid solution found!")

    # if visualize:
    #     plt.figure(figsize=(10, 6))
    #     iterations_range = range(len(best_fitness_progress))
    #     plt.plot(
    #         iterations_range,
    #         best_fitness_progress,
    #         "g-",
    #         linewidth=2,
    #         label="Best Distance",
    #     )
    #     plt.plot(
    #         iterations_range,
    #         avg_fitness_progress,
    #         "orange",
    #         linestyle="--",
    #         linewidth=1.5,
    #         label="Average Distance",
    #     )
    #     plt.xlabel("Iteration", fontsize=12)
    #     plt.ylabel("Distance", fontsize=12)
    #     plt.title(
    #         f"Ant Colony Optimization - Problem {PROBLEM} Convergence",
    #         fontsize=14,
    #         fontweight="bold",
    #     )
    #     plt.legend(loc="best", fontsize=10)
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.show()


if __name__ == "__main__":
    # Set visualize=True to see the convergence plot
    tsp_aco(PROBLEM=4, visualize=True)
