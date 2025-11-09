from problem.problem import Problem
from typing import Tuple, List
import numpy as np


class AckleyFunction(Problem):
    """
    Ackley Function optimization problem.
    
    The Ackley function is a widely used benchmark for optimization algorithms.
    f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c * x_i))) + a + exp(1)
    
    Global minimum at f(0,0,...,0) = 0
    """
    
    def __init__(self, PROBLEM_FOLDER: str, PROBLEM: int):
        super().__init__()
        
        self.dimension, self.input_vector, self.initial_value = (
            self.load_problem_infos(PROBLEM_FOLDER=PROBLEM_FOLDER, PROBLEM=PROBLEM)
        )
        
        # Ackley function constants
        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi
        
    def load_problem_infos(self, **kwargs) -> Tuple[int, np.ndarray, float]:
        """
        kwargs:
            PROBLEM_FOLDER (str): folder path containing problem data files
            PROBLEM (int): problem number

        behavior:
            Reads PROBLEM_FOLDER/test_{PROBLEM:02d}.txt for test case data
            
        Returns:
            dimension: int - dimensionality of the problem
            input_vector: np.ndarray - initial input vector
            initial_value: float - initial function value
        """
        PROBLEM_FOLDER = kwargs.get("PROBLEM_FOLDER")
        PROBLEM = kwargs.get("PROBLEM")
        
        if PROBLEM_FOLDER is None or PROBLEM is None:
            raise ValueError("PROBLEM_FOLDER and PROBLEM cannot be None")
        
        filename = f"ackley/test_{PROBLEM:02d}.txt"
        
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
    
    def get_problem_infos(self, **kwargs) -> Tuple[int, np.ndarray, float]:
        """
        Get problem information.
        
        Returns:
            dimension: int
            input_vector: np.ndarray
            initial_value: float
        """
        return self.dimension, self.input_vector, self.initial_value
    
    def relativity_to_solution(self, **kwargs) -> float:
        """
        Calculate how close the answer is to the optimal solution.
        The optimal value is 0 (at the origin).
        
        kwargs:
            answer: np.ndarray - the vector to evaluate
            
        Returns:
            relativity: float - ratio of optimal to answer (higher is better)
        """
        answer = kwargs.get("answer", None)
        
        if answer is None:
            raise ValueError("answer must be provided in kwargs")
        
        answer_value = self.calculate_fitness(answer)
        optimal_value = 0.0  # Global minimum of Ackley function
        
        # Avoid division by zero
        if answer_value == 0:
            return 1.0
        
        # Return inverse ratio (closer to 0 is better)
        # Add small epsilon to avoid division by zero
        return optimal_value / (answer_value + 1e-10)
    
    def calculate_fitness(self, answer: np.ndarray) -> float:
        """
        Calculate the Ackley function value for a given vector.
        
        Args:
            answer: np.ndarray - input vector
            
        Returns:
            fitness: float - Ackley function value (lower is better)
        """
        n = len(answer)
        sum_sq = np.sum(answer**2)
        sum_cos = np.sum(np.cos(self.c * answer))
        
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        result = term1 + term2 + self.a + np.e
        
        return result
