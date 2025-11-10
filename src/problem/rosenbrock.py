from typing import Tuple
from problem.problem import Problem
import numpy as np


class RosenbrockFunction(Problem):
    def __init__(self, PROBLEM_FOLDER, PROBLEM) -> None:
        super().__init__()

        self.dimension, self.input_vector, self.initial_value = self.load_problem_infos(
            PROBLEM_FOLDER=PROBLEM_FOLDER, PROBLEM=PROBLEM
        )

    def load_problem_infos(self, **kwargs) -> Tuple:
        """
        Read Rosenbrock test case
        Returns: dimension, input_vector, initial_value
        """
        PROBLEM_FOLDER = kwargs.get("PROBLEM_FOLDER")
        problem = kwargs.get("PROBLEM")

        if PROBLEM_FOLDER is None or problem is None:
            raise ValueError("PROBLEM_FOLDER and PROBLEM must be provided")

        test_number = problem

        filename = PROBLEM_FOLDER + f"test_{test_number:02d}.txt"

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
        Rosenbrock function (Banana function)
        f(x) = sum([100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2])
        Global minimum at f(1,1,...,1) = 0
        """
        n = len(answer)
        total = 0
        for i in range(n - 1):
            total += 100 * (answer[i + 1] - answer[i] ** 2) ** 2 + (1 - answer[i]) ** 2
        return total

    def relativity_to_solution(self, answer) -> float:
        """
        Calculate how close the answer is to the optimal solution.
        The optimal value is 0 (at the point [1,1,...,1]).

        Args:
            answer: np.ndarray - the vector to evaluate
        Returns:
            relativity: float - ratio of optimal to answer (higher is better)
        """
        answer_value = self.calculate_fitness(answer)
        optimal_value = 0.0  # Global minimum of Rosenbrock function

        if answer_value == 0:
            return float("inf")  # Perfect solution

        relativity = optimal_value / answer_value
        return relativity
