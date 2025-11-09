import numpy as np
import matplotlib.pyplot as plt
from math import gamma
from typing import Any, Callable, Optional


class DepthFirstSearch:
    def __init__(
        self,
        start_node: Any,
        get_neighbors: Callable[[Any], list[Any]],
    ):
        self.start_node = start_node
        self.get_neighbors = get_neighbors
        self.visited = set()
        self.stack = [start_node]

        self.fitness_function = None
        self.best_node = None
        self.best_fitness = float("-inf")

    def set_fitness_function(
        self,
        fitness_function: Callable[[Any], float],
    ):
        self.fitness_function = fitness_function

    def run(
        self,
        max_iterations: Optional[int] = 10000,
    ):
        iterations = 0

        while self.stack and (max_iterations is None or iterations < max_iterations):
            current_node = self.stack.pop()

            if current_node not in self.visited:
                self.visited.add(current_node)

                # Evaluate fitness
                if self.fitness_function:
                    current_fitness = self.fitness_function(current_node)
                    if current_fitness > self.best_fitness:
                        self.best_fitness = current_fitness
                        self.best_node = current_node

                neighbors = self.get_neighbors(current_node)
                for neighbor in neighbors:
                    if neighbor not in self.visited:
                        self.stack.append(neighbor)
            iterations += 1

        return self.best_node, self.best_fitness
