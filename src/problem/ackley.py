from problem.problem import Problem
from typing import Tuple, List
import numpy as np


class AckleyFunction(Problem):
    def __init__(self, PROBLEM_FOLDER: str, PROBLEM: int):
        super().__init__()
        self.dimension, self.input_vector, self.initial_value = self.load_problem_infos(
            PROBLEM_FOLDER=PROBLEM_FOLDER, PROBLEM=PROBLEM
        )

    def load_problem_infos(self, **kwargs) -> Tuple:
        """
        Read Ackley test case
        Returns: dimension, input_vector, initial_value
        """

        PROBLEM_FOLDER = kwargs.get("PROBLEM_FOLDER")
        problem = kwargs.get("PROBLEM")

        if PROBLEM_FOLDER is None or problem is None:
            raise ValueError("PROBLEM_FOLDER and PROBLEM must be provided in kwargs")

        test_number = problem

        filename = PROBLEM_FOLDER + f"/test_{test_number:02d}.txt"

        with open(filename, "r") as f:
            lines = f.readlines()

        dimension = None
        initial_value = None
        input_vector = []

        reading_vector = False
        for line in lines:
            line = line.strip()
            if line.startswith("# Dimension:"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("# Initial value:"):
                # Extract value from "# Initial value: f(x) = 12345.67"
                initial_value = float(line.split("=")[1].strip())
            elif line.startswith("# Input vector:"):
                reading_vector = True
            elif reading_vector and line and not line.startswith("#"):
                input_vector.append(float(line))

        return dimension, np.array(input_vector), initial_value

    def calculate_fitness(self, answer) -> float:
        """
        Ackley function - a widely used benchmark for optimization algorithms

        f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c * x_i))) + a + exp(1)

        Global minimum at f(0,0,...,0) = 0

        Args:
            answer: Input vector
        Returns:
            Function value
        """
        a = 20
        b = 0.2
        c = 2 * np.pi

        x = answer
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(c * x))

        term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        result = term1 + term2 + a + np.e

        return result

    def relativity_to_solution(self, answer) -> float:
        return super().relativity_to_solution(answer)
