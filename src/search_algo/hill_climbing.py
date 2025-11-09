import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
from copy import deepcopy
import random


class Problem(ABC):
    """Abtract class for optimization problems (only for hill climbing)"""

    @abstractmethod
    def generate_initial_solution(self) -> Any:
        """Generate a random initial solution"""
        pass

    @abstractmethod
    def evaluate(self, solution) -> float:
        """Evaluate with the objective function (lower is better)"""
        pass

    @abstractmethod
    def get_neighbors(self, solution) -> List[Any]:
        """Generate all the neighbors of the solution"""
        pass

    @abstractmethod
    def is_valid(self, solution) -> bool:
        """Check if the solution is valid"""
        pass


class HillClimbing:
    def __init__(self, problem: Problem, max_iters=100, variant="steepest", verbose=True):
        self.problem = problem
        self.max_iters = max_iters
        self.variant = variant
        self.verbose = verbose

    def solve(self):
        current_solution = self.problem.generate_initial_solution()
        current_fitness = self.problem.evaluate(current_solution)

        best_solution = deepcopy(current_solution)
        best_fitness = float('inf')

        all_fitness = [current_fitness]

        for i in range(self.max_iters):
            current_neighbors = self.problem.get_neighbors(current_solution)

            if not current_neighbors:
                if self.verbose:
                    print("No neighbors found")
                break

            improved = False
            if self.variant == "steepest":
                best_fitness_neighbor = current_fitness
                best_solution_neighbor = None

                for neighbor in current_neighbors:
                    neighbor_fitness = self.problem.evaluate(neighbor)

                    if neighbor_fitness < best_fitness_neighbor:
                        best_fitness_neighbor = neighbor_fitness
                        best_solution_neighbor = deepcopy(neighbor)
                        improved = True

                if improved:
                    current_fitness = best_fitness_neighbor
                    current_solution = best_solution_neighbor
            elif self.variant == "first":
                for neighbor in current_neighbors:
                    neighbor_fitness = self.problem.evaluate(neighbor)

                    if neighbor_fitness < current_fitness:
                        current_solution = neighbor
                        current_fitness = neighbor_fitness
                        improved = True
                        break

            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = deepcopy(current_solution)

            all_fitness.append(current_fitness)

            if self.verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}: Current = {current_fitness:.6f}, "
                      f"Best = {best_fitness:.6f}")

            if not improved:
                if self.verbose:
                    print(f"\nConverged at iteration {i + 1}")
                break

        if self.verbose:
            print(f"\nFinal best fitness: {best_fitness:.6f}")
            print("=" * 60)

        return best_solution, best_fitness, all_fitness


class TSPProblem(Problem):
    def __init__(self, distance_matrix: np.ndarray):
        self.distance_matrix = distance_matrix
        self.n_cities = len(self.distance_matrix)

    def generate_initial_solution(self):
        cities = list(range(self.n_cities))
        random.shuffle(cities)
        return cities

    def evaluate(self, solution):
        def path2formatpath(path: List):
            from_cities = np.array(path)
            to_cities = np.array(path[1:] + [path[0]])
            return (from_cities, to_cities)
        format_path = path2formatpath(solution)
        return np.sum(self.distance_matrix[format_path[0], format_path[1]])

    def get_neighbors(self, solution):
        neighbors = []

        for i in range(len(solution) - 1):
            for j in range(i + 1, len(solution)):
                neighbor = solution.copy()
                # Reverse the segment between i and j
                neighbor[i:j+1] = reversed(neighbor[i:j+1])
                neighbors.append(neighbor)

        return neighbors

    def is_valid(self, solution):
        return (len(solution) == self.n_cities and
                set(solution) == set(range(self.n_cities)))




