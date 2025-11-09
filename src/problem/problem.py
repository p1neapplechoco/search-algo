import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional


class Problem(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_problem_infos(self, **kwargs) -> Tuple:
        pass

    @abstractmethod
    def relativity_to_solution(self, answer):
        pass

    @abstractmethod
    def calculate_fitness(self, answer):
        pass
