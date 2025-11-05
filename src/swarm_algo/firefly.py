import numpy as np
from numpy import random
import matplotlib.pyplot as plt


class Firefly:
    def __init__(
        self,
        ndim,
        num_fireflies,
        beta=1.0,
        gamma=1.0,
        alpha=0.2,
    ):
        """
        Initialize the Firefly algorithm parameters.

        Args:
            ndim (int): Number of dimensions.
            num_fireflies (int): Number of fireflies.

            beta (float): Attractiveness coefficient.
            gamma (float): Light absorption coefficient.
            alpha (float): Randomization parameter.
        """

        self.ndim = ndim
        self.num_fireflies = num_fireflies
        self.beta = beta  # attractiveness coefficient
        self.gamma = gamma  # light absorption coefficient
        self.alpha = alpha  # randomization parameter
        self.positions = np.random.rand(num_fireflies, ndim)
        self.intensities = np.zeros(num_fireflies)

        self.objective_function = None

    def __distance__(self, firefly_1, firefly_2):
        """
        Calculate the Cartesian distance between two fireflies.
        Args:
            firefly_1 (np.ndarray): Position of the first firefly.
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
            tuple: Best position and its intensity found during the run.
        """

        if self.objective_function is None:
            raise ValueError(
                "Objective function not set. Please set it using set_objective_function()."
            )

        best_position = None
        best_intensity = -np.inf

        for generation in range(max_generations):
            for i in range(self.num_fireflies):
                binary_position = (self.positions[i] > 0.5).astype(int)
                self.intensities[i] = self.objective_function(binary_position)

                if self.intensities[i] > best_intensity:
                    best_intensity = self.intensities[i]
                    best_position = binary_position.copy()

            self.update_positions()

        return best_position, best_intensity
