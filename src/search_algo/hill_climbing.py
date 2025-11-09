import numpy as np
from typing import Any, Callable, Optional


class HillClimbing:
    def __init__(self, max_iterations=1000):
        self.max_iterations = max_iterations
        self.fitness_function = None
        self.get_neighbor = None
        self.randomize_solution = None

    def set_fitness_function(self, func: Callable[[Any], float]):
        self.fitness_function = func

    def set_neighbor_function(self, func: Callable[[Any], Any]):
        self.get_neighbor = func

    def set_randomize_function(self, func: Callable[[], Any]):
        self.randomize_solution = func

    def run(self):
        current = self.randomize_solution()
        current_value = self.fitness_function(current)

        for _ in range(self.max_iterations):
            neighbor = self.get_neighbor(current)
            neighbor_value = self.fitness_function(neighbor)

            if neighbor_value < current_value:  # minimize
                current, current_value = neighbor, neighbor_value

        return current, current_value
