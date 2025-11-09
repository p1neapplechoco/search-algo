import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from problem.problem import Problem
from typing import Optional


class Firefly:
    def __init__(
        self,
        ndim,
        num_fireflies,
        beta=1.0,
        gamma=1.0,
        alpha=0.2,
        problem_type="continuous",
    ):
        """
        Initialize the Firefly algorithm parameters.

        Args:
            ndim (int): Number of dimensions.
            num_fireflies (int): Number of fireflies.
            beta (float): Attractiveness coefficient.
            gamma (float): Light absorption coefficient.
            alpha (float): Randomization parameter.
            problem_type (str): Type of problem - 'continuous' or 'binary'.
                - 'continuous': For continuous optimization problems
                - 'binary': For discrete/binary problems (e.g., knapsack)
        """

        self.ndim = ndim
        self.num_fireflies = num_fireflies
        self.beta = beta  # attractiveness coefficient
        self.gamma = gamma  # light absorption coefficient
        self.alpha = alpha  # randomization parameter
        self.problem_type = problem_type.lower()

        if self.problem_type not in ["continuous", "binary"]:
            raise ValueError("problem_type must be 'continuous' or 'binary'")

        self.positions = np.random.rand(num_fireflies, ndim)
        self.intensities = np.zeros(num_fireflies)

        self.objective_function = None

    def __distance__(self, firefly_1, firefly_2):
        """
        Calculate the Cartesian distance between two fireflies.
        Args:
            firefly_1 (np.ndarray): Pos>ition of the first firefly.
            firefly_2 (np.ndarray): Position of the second firefly.
        Returns:
            float: Cartesian distance between the two fireflies.
        """

        return np.linalg.norm(firefly_1 - firefly_2)

    def set_objective_function(self, objective_function):
        """
        Set the objective function for evaluating firefly intensities.
        Args:
            objective_function (callable): The objective function to evaluate firefly intensities.
        """

        self.objective_function = objective_function

    def update_positions(self):
        """
        Update the positions of the fireflies based on their intensities.
        .. Math::
            x_i^{(t+1)} = x_i^{(t)} + \\beta_0 \\exp({-\\gamma r_{ij}^2}) (x_j^{(t)} - x_i^{(t)}) + \\alpha \\epsilon_i^{(t)},
        Args:
            objective_function (callable): The objective function to evaluate firefly intensities.
        """

        sorted_indices = np.argsort(-self.intensities)
        self.positions = self.positions[sorted_indices]
        self.intensities = self.intensities[sorted_indices]

        for i in range(self.num_fireflies):
            for j in range(self.num_fireflies):
                if self.intensities[j] > self.intensities[i]:
                    r_ij = self.__distance__(self.positions[i], self.positions[j])
                    beta_ij = self.beta * np.exp(-self.gamma * r_ij**2)
                    random_step = self.alpha * (random.rand(self.ndim) - 0.5)
                    self.positions[i] = self.positions[i] + (
                        beta_ij * (self.positions[j] - self.positions[i]) + random_step
                    )
                    self.positions[i] = np.clip(self.positions[i], 0.0, 1.0)

    def run(self, max_generations, visualize=False):
        """
        Run the Firefly algorithm for a specified number of generations.
        Args:
            max_generations (int): Number of generations to run the algorithm.
            visualize (bool): Whether to visualize the optimization process.
        Returns:
            tuple: Best position, its intensity, and history of best intensities.
        """

        if self.objective_function is None:
            raise ValueError(
                "Objective function not set. Please set it using set_objective_function()."
            )

        best_position = None
        best_intensity = -np.inf
        best_intensity_history = []
        avg_intensity_history = []

        for generation in range(max_generations):
            for i in range(self.num_fireflies):
                # Convert position based on problem type
                if self.problem_type == "binary":
                    # For binary problems: convert continuous to binary
                    evaluated_position = (self.positions[i] > 0.5).astype(int)
                else:
                    # For continuous problems: use position as-is
                    evaluated_position = self.positions[i]

                self.intensities[i] = self.objective_function(evaluated_position)

                if self.intensities[i] > best_intensity:
                    best_intensity = self.intensities[i]
                    best_position = evaluated_position.copy()

            self.update_positions()

            # Track progress
            best_intensity_history.append(best_intensity)
            avg_intensity_history.append(np.mean(self.intensities))

        # Visualize if requested
        if visualize:
            self._visualize_progress(best_intensity_history, avg_intensity_history)

        return best_position, best_intensity, best_intensity_history

    def _visualize_progress(self, best_history, avg_history):
        """
        Visualize the optimization progress.

        Args:
            best_history (list): History of best fitness values.
            avg_history (list): History of average fitness values.
        """
        plt.figure(figsize=(10, 6))
        generations = range(len(best_history))

        plt.plot(generations, best_history, "b-", linewidth=2, label="Best Fitness")
        plt.plot(
            generations, avg_history, "r--", linewidth=1.5, label="Average Fitness"
        )

        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Fitness Value", fontsize=12)
        plt.title(
            "Firefly Algorithm - Convergence Progress", fontsize=14, fontweight="bold"
        )
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
