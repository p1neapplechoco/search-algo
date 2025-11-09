import numpy as np
import matplotlib.pyplot as plt
from math import gamma


class CuckooSearch:
    def __init__(
        self,
        ndim,
        n_cuckoos,
        pa,
        beta,
    ):
        """
        Initialize the Cuckoo Search algorithm parameters.

        Args:
            ndim (int): Number of dimensions,
            n_cuckoos (int): Number of cuckoos,
            pa (float): switching probability,
            beta (float): Levy flight parameter,
        """
        self.ndim = ndim
        self.n_cuckoos = n_cuckoos
        self.pa = pa
        self.beta = beta

        self.positions = np.random.rand(n_cuckoos, ndim)
        self.fitness = np.zeros(n_cuckoos)

        self.objective_function = None

    def set_objective_function(self, objective_function):
        """
        Set the objective function for evaluating cuckoo fitness.
        Args:
            objective_function (callable): The objective function to be minimized.
        """
        self.objective_function = objective_function

    def evaluate_fitness(self):
        """
        Evaluate the fitness of each cuckoo based on the objective function.
        """
        if self.objective_function is None:
            raise ValueError(
                "Objective function not set. Please set it using set_objective_function()."
            )

        for i in range(self.n_cuckoos):
            self.fitness[i] = self.objective_function(self.positions[i])

    def __levy_flight__(self, step_size=0.01):
        """
        Perform Levy flight to generate new solutions.
        Args:
            step_size (float): Step size for the Levy flight.
        Returns:
            np.ndarray: New positions after Levy flight.
        """
        sigma_u = (
            gamma(1 + self.beta)
            * np.sin(np.pi * self.beta / 2)
            / (gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))
        ) ** (1 / self.beta)
        sigma_v = 1

        u = np.random.normal(0, sigma_u, size=(self.n_cuckoos, self.ndim))
        v = np.random.normal(0, sigma_v, size=(self.n_cuckoos, self.ndim))
        step = u / (np.abs(v) ** (1 / self.beta))
        return step_size * step

    def update_positions(self):
        """
        Update the positions of the cuckoos using Levy flights and random replacement.
        Uses greedy selection: only accept new positions if they're better.
        """
        if self.objective_function is None:
            raise ValueError("Objective function not set")

        # Generate new positions via Levy flight
        new_positions = self.positions + self.__levy_flight__()
        new_positions = np.clip(new_positions, 0.0, 1.0)

        # Greedy selection: only accept if better
        for i in range(self.n_cuckoos):
            new_fitness = self.objective_function(new_positions[i])
            if new_fitness < self.fitness[i]:
                self.positions[i] = new_positions[i]
                self.fitness[i] = new_fitness

        # Abandon worst nests with probability pa
        # Sort by fitness to find worst nests
        worst_indices = np.argsort(self.fitness)[-int(self.pa * self.n_cuckoos) :]

        for i in worst_indices:
            # Build new nest by random walk
            j = np.random.randint(self.n_cuckoos)
            k = np.random.randint(self.n_cuckoos)
            new_pos = self.positions[j] + np.random.rand() * (
                self.positions[j] - self.positions[k]
            )
            new_pos = np.clip(new_pos, 0.0, 1.0)

            # Greedy selection
            new_fitness = self.objective_function(new_pos)
            if new_fitness < self.fitness[i]:
                self.positions[i] = new_pos
                self.fitness[i] = new_fitness

    def run(self, max_generations, visualize=False):
        """
        Run the Cuckoo Search algorithm for a specified number of generations.
        Args:
            max_generations (int): Number of generations to run the algorithm.
            visualize (bool): Whether to visualize the optimization process.
        Returns:
            tuple: Best position, its fitness, and history of best fitness values.
        """
        if self.objective_function is None:
            raise ValueError(
                "Objective function not set. Please set it using set_objective_function()."
            )

        best_position = None
        best_fitness = np.inf
        best_fitness_history = []

        for generation in range(max_generations):
            self.evaluate_fitness()

            current_best_idx = np.argmin(self.fitness)
            current_best_fitness = self.fitness[current_best_idx]
            current_best_position = self.positions[current_best_idx]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_position = current_best_position

            best_fitness_history.append(best_fitness)

            self.update_positions()

            if visualize and best_position is not None:
                plt.clf()
                plt.title(f"Cuckoo Search - Generation {generation+1}")
                plt.scatter(
                    self.positions[:, 0],
                    self.positions[:, 1],
                    c="blue",
                    label="Cuckoos",
                )
                plt.scatter(
                    best_position[0],
                    best_position[1],
                    c="red",
                    label="Best Cuckoo",
                )
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.legend()
                plt.pause(0.1)

        return best_position, best_fitness, best_fitness_history
