import numpy as np
import matplotlib.pyplot as plt


class Genetic:
    def __init__(
        self,
        population_size,
        mutation_rate,
        elite_rate=0.1,
        num_children=2,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        self.num_children = num_children

        self.fitness_function = None

    def set_fitness_function(self, fitness_function):
        self.fitness_function = fitness_function

    def select_parents(self, population, fitness_scores):
        fitness_sum = np.sum(fitness_scores)
        selection_probs = fitness_scores / fitness_sum

        parents_indices = np.random.choice(
            len(population),
            size=2,
            p=selection_probs,
        )

        return population[parents_indices]

    def __mate__(self, parent_1, parent_2):
        crossover_point = np.random.randint(1, len(parent_1) - 1)
        child = np.concatenate((parent_1[:crossover_point], parent_2[crossover_point:]))

        for i in range(len(child)):
            if np.random.rand() < self.mutation_rate:
                child[i] = np.random.rand()

        return child

    def create_next_generation(self, population):
        fitness_scores = np.array([self.fitness_function(ind) for ind in population])
        sorted_indices = np.argsort(-fitness_scores)
        population = population[sorted_indices]
        fitness_scores = fitness_scores[sorted_indices]

        num_elites = int(self.elite_rate * self.population_size)
        next_generation = population[:num_elites].copy()

        while len(next_generation) < self.population_size:
            parent_1, parent_2 = self.select_parents(population, fitness_scores)
            for _ in range(self.num_children):
                child = self.__mate__(parent_1, parent_2)
                next_generation = np.vstack((next_generation, child))
                if len(next_generation) >= self.population_size:
                    break

        return next_generation[: self.population_size]

    def run(self, generations, visualize=False):
        if self.fitness_function is None:
            raise ValueError(
                "Fitness function not set. Please set it using set_fitness_function()."
            )

        population = np.random.rand(self.population_size, 10)

        best_individual = None
        best_fitness = -np.inf
        best_fitness_history = []
        avg_fitness_history = []

        for generation in range(generations):
            fitness_scores = np.array(
                [self.fitness_function(ind) for ind in population]
            )

            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()

            # Track progress
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(fitness_scores))

            population = self.create_next_generation(population)

        # Visualize if requested
        if visualize:
            self._visualize_progress(best_fitness_history, avg_fitness_history)

        return best_individual, best_fitness, best_fitness_history

    def _visualize_progress(self, best_history, avg_history):
        """
        Visualize the optimization progress.

        Args:
            best_history (list): History of best fitness values.
            avg_history (list): History of average fitness values.
        """
        plt.figure(figsize=(10, 6))
        generations = range(len(best_history))

        plt.plot(generations, best_history, "g-", linewidth=2, label="Best Fitness")
        plt.plot(
            generations,
            avg_history,
            "orange",
            linestyle="--",
            linewidth=1.5,
            label="Average Fitness",
        )

        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Fitness Value", fontsize=12)
        plt.title(
            "Genetic Algorithm - Convergence Progress", fontsize=14, fontweight="bold"
        )
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
