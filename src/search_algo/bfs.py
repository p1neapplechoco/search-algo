import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from collections import deque
from typing import Any, Callable, List, Set, Tuple, Optional


class BreadthFirstSearch:
    def __init__(
        self,
        start_state: Any,
        get_neighbors: Callable[[Any], List[Any]],
    ):
        self.start_state = start_state
        self.get_neighbors = get_neighbors
        self.visited: Set[Any] = set()
        self.queue: deque = deque([start_state])

        self.fitness_function: Optional[Callable[[Any], float]] = None
        self.best_state: Optional[Any] = None
        self.best_fitness: float = float("-inf")

    def set_fitness_function(
        self,
        fitness_function: Callable[[Any], float],
    ):
        self.fitness_function = fitness_function

    def run(
        self,
        max_iterations: Optional[int] = 10000,
        visualize: bool = False,
    ):
        iterations = 0

        while self.queue and (max_iterations is None or iterations < max_iterations):
            current_state = self.queue.popleft()

            if current_state not in self.visited:
                self.visited.add(current_state)

                # Evaluate fitness
                if self.fitness_function:
                    current_fitness = self.fitness_function(current_state)
                    if current_fitness > self.best_fitness:
                        self.best_fitness = current_fitness
                        self.best_state = current_state

                neighbors = self.get_neighbors(current_state)
                for neighbor in neighbors:
                    if neighbor not in self.visited:
                        self.queue.append(neighbor)
            iterations += 1

        return self.best_state, self.best_fitness
