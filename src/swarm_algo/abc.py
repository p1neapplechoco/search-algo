import numpy as np
from numpy import random
import matplotlib.pyplot as plt


class ArtificialBeeColony:
    def __init__(
        self,
        dimension,
        sn,
        mcn,
        limit,
        lb=-5.12,
        ub=5.12,
    ):
        """
        Initialize the Artificial Bee Colony algorithm parameters.

        Args:
            dimension (int): Number of dimensions.
            sn (int): Number of food sources (employed bees).
            mcn (int): Maximum cycle number (iterations).
            limit (int): Abandonment limit for scout bees.
            lb (float): Lower bound of the search space.
            ub (float): Upper bound of the search space.
        """

        self.dimension = dimension
        self.sn = sn
        self.mcn = mcn
        self.limit = limit
        self.lb = lb
        self.ub = ub

        self.solutions = np.array([self.randomize_solution() for _ in range(sn)])
        self.fitness = np.zeros(self.sn)
        self.probs = np.zeros(self.sn)
        self.trial = np.zeros(self.sn)

        self.objective_function = None

    def randomize_solution(self):
        """
        Generate a random solution within the search space bounds.

        Returns:
            np.ndarray: Random solution vector.
        """

        return self.lb + np.random.rand(self.dimension) * (self.ub - self.lb)

    def set_objective_function(self, objective_function):
        """
        Set the objective function for evaluating solutions.

        Args:
            objective_function (callable): The objective function to minimize.
        """

        self.objective_function = objective_function

    def calc_fitness(self, solution):
        """
        Calculate the fitness value of a solution.

        Args:
            solution (np.ndarray): Solution vector to evaluate.

        Returns:
            float: Fitness value (higher is better).
        """

        if self.objective_function is None:
            raise ValueError("Objective function not set.")

        value = self.objective_function(solution)
        fitness = 1 / (1 + value) if value >= 0 else 1 + np.abs(value)
        return fitness

    def calc_probabilities(self):
        """
        Calculate selection probabilities for onlooker bees based on fitness values.
        """

        maxfit = np.max(self.fitness)
        self.probs = 0.1 + 0.9 * (self.fitness / maxfit)

    def local_search(self, i):
        """
        Perform local search around a solution.

        Args:
            i (int): Index of the solution to search around.

        Returns:
            np.ndarray: New candidate solution.
        """

        idxs = np.delete(np.arange(self.sn), i)
        k = np.random.choice(idxs)
        j = np.random.randint(self.dimension)
        phi = np.random.uniform(-1, 1)

        new_solution = np.copy(self.solutions[i])
        new_solution[j] = new_solution[j] + phi * (
            new_solution[j] - self.solutions[k, j]
        )
        new_solution[j] = np.clip(new_solution[j], self.lb, self.ub)

        return new_solution

    def run(self, visualize=False):
        """
        Run the Artificial Bee Colony algorithm.

        Args:
            visualize (bool): Whether to visualize the optimization process.

        Returns:
            tuple: Best solution, its fitness, and history of best fitness values.
        """

        if self.objective_function is None:
            raise ValueError(
                "Objective function not set. Please set it using set_objective_function()."
            )

        # Initialize fitness values
        for i in range(self.sn):
            self.fitness[i] = self.calc_fitness(self.solutions[i])

        best_solution = None
        best_fitness = -np.inf
        best_fitness_history = []
        avg_fitness_history = []

        cyc = 1
        while cyc < self.mcn:
            # Employed bees phase
            for i in range(self.sn):
                new_solution = self.local_search(i)
                new_fit = self.calc_fitness(new_solution)

                if new_fit > self.fitness[i]:
                    self.solutions[i] = new_solution
                    self.fitness[i] = new_fit
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1

            self.calc_probabilities()

            # Onlooker bees phase
            i = 0
            t = 0
            while t < self.sn:
                if np.random.rand(1) < self.probs[i]:
                    t += 1
                    new_solution = self.local_search(i)
                    new_fit = self.calc_fitness(new_solution)

                    if new_fit > self.fitness[i]:
                        self.solutions[i] = new_solution
                        self.fitness[i] = new_fit
                        self.trial[i] = 0
                    else:
                        self.trial[i] += 1

                i = (i + 1) % self.sn

            # Update best solution
            best_idx = np.argmax(self.fitness)
            if self.fitness[best_idx] > best_fitness:
                best_fitness = self.fitness[best_idx]
                best_solution = np.copy(self.solutions[best_idx])

            # Scout bee phase
            si = np.argmax(self.trial)
            if self.trial[si] > self.limit:
                self.solutions[si] = self.randomize_solution()
                self.fitness[si] = self.calc_fitness(self.solutions[si])
                self.trial[si] = 0

            # Track progress
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(self.fitness))

            cyc += 1

        # Visualize if requested
        if visualize:
            self._visualize_progress(best_fitness_history, avg_fitness_history)

        return best_solution, best_fitness, best_fitness_history

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
            "ABC Algorithm - Convergence Progress", fontsize=14, fontweight="bold"
        )
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
