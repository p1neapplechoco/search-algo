import numpy as np


class ABC:
    def __init__(
        self,
        dimension,
        sn,
        mcn,
        limit,
        lb=-5.12,
        ub=5.12,
        objective_function=None
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
            objective_function (callable): The objective function to minimize.
        """
        self.dimension = dimension
        self.sn = sn
        self.mcn = mcn
        self.limit = limit
        self.lb = lb
        self.ub = ub

        self.objective_function = objective_function
        
        self.solutions = np.array([self.randomize_solution() for _ in range(sn)])
        self.fitness = np.zeros(self.sn)
        self.probs = np.zeros(self.sn)
        

    def set_objective_function(self, objective_function):
        """
        Set the objective function for evaluating solutions.

        Args:
            objective_function (callable): The objective function to minimize.
        """
        self.objective_function = objective_function

    def randomize_solution(self):
        """
        Generate a random solution within the search space bounds.

        Returns:
            np.ndarray: Random solution vector.
        """
        return self.lb + np.random.rand(self.dimension) * (self.ub - self.lb)

    def calc_fitness(self, solution):
        """
        Calculate the fitness value of a solution.

        Args:
            solution (np.ndarray): Solution vector to evaluate.

        Returns:
            float: Fitness value (higher is better).
        """
        if self.objective_function is None:
            raise ValueError(
                "Objective function not set."
            )

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
        new_solution[j] = new_solution[j] + phi * (new_solution[j] - self.solutions[k, j])
        new_solution[j] = np.clip(new_solution[j], self.lb, self.ub)

        return new_solution

    def run(self):
        """
        Run the Artificial Bee Colony algorithm.

        Returns:
            tuple: Best solution found and its fitness value.
        """
        if self.objective_function is None:
            raise ValueError(
                "Objective function not set. Please set it using set_objective_function()."
            )

        # Initialize fitness values
        for i in range(self.sn):
            self.fitness[i] = self.calc_fitness(self.solutions[i])

        cyc = 1
        trial = np.zeros(self.sn)
        global_best = {
            'solution': None,
            'fitness': -np.inf
        }

        while cyc < self.mcn:
            # Employed bees phase
            for i in range(self.sn):
                new_solution = self.local_search(i)
                new_fit = self.calc_fitness(new_solution)

                if new_fit > self.fitness[i]:
                    self.solutions[i] = new_solution
                    self.fitness[i] = new_fit
                    trial[i] = 0
                else:
                    trial[i] += 1

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
                        trial[i] = 0
                    else:
                        trial[i] += 1

                i = (i + 1) % self.sn

            best_idx = np.argmax(self.fitness)
            if self.fitness[best_idx] > global_best['fitness']:
                global_best['solution'] = np.copy(self.solutions[best_idx])
                global_best['fitness'] = self.fitness[best_idx]

            # Scout bee phase
            si = np.argmax(trial)
            if trial[si] > self.limit:
                self.solutions[si] = self.randomize_solution()
                self.fitness[si] = self.calc_fitness(self.solutions[si])
                trial[si] = 0

            cyc += 1

        return global_best['solution'], global_best['fitness']
