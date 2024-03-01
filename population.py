"""
Daniel Tregea
A class to represent a poopulation of individuals in a genetic algorithm
"""
import random
from individual import Individual
import individual as individual

"""
Generate a population of random individuals
"""
def generate_random():
    return Population([individual.generate_random() for x in range(Population.size)])


class Population:
    size = 0

    def __init__(self, individuals):
        self.individuals = individuals
        self.ratios = []
        self.lowest_cost1 = -1
        self.lowest_cost2 = -1
        self.fitnesses = []

    """
    Determine if the population has an individual that reached termination criteria
    """
    def reached_termination(self):
        return self.individuals[self.lowest_cost1].cost() <= 0.1

    """
    Calculate the highest fit indivudal, and fitness ratios
    """
    def evaluate(self):
        sum = 0
        self.lowest_cost1 = 0
        for i in range(len(self.individuals)):
            fitness = 1 / (self.individuals[i].cost() + 1)
            sum += fitness
            self.fitnesses.append(fitness)
            if fitness > self.fitnesses[self.lowest_cost2] and not fitness > self.fitnesses[self.lowest_cost1]:
                self.lowest_cost2 = i
            if fitness > self.fitnesses[self.lowest_cost1]:
                self.lowest_cost2 = self.lowest_cost1
                self.lowest_cost1 = i
        for i in range(len(self.individuals)):
            self.ratios.append(self.fitnesses[i] / sum)

    """
    Return a list of parents for a next generation
    """
    def select_parents(self):
        return random.choices(self.individuals, self.ratios, k=Population.size - 1)

    """
    Generate a new population from crossing over parents
    """
    def crossover(self):
        new_individuals = []
        parents = self.select_parents()
        for i in range(0, len(parents), 2):
            gamma_crossover = random.randint(1, Individual.binary_len)
            beta_crossover = random.randint(1, Individual.binary_len)
            parent1 = parents[i]
            parent2 = parents[i + 1]

            child1 = Individual([], [])
            child2 = Individual([], [])

            for j in range(len(parent1.betas)):
                child1.gammas.append(parent1.gammas[j][0:gamma_crossover] + parent2.gammas[j][gamma_crossover:])
                child2.gammas.append(parent2.gammas[j][0:gamma_crossover] + parent1.gammas[j][gamma_crossover:])

                child1.betas.append(parent1.betas[j][0:beta_crossover] + parent2.betas[j][beta_crossover:])
                child2.betas.append(parent2.betas[j][0:beta_crossover] + parent1.betas[j][beta_crossover:])
                child1.mutate()
                child2.mutate()
            new_individuals.append(child1)
            new_individuals.append(child2)

        return new_individuals

    """
    Get the most fit individual in the population
    """
    def best(self):
        return self.individuals[self.lowest_cost1]

    """
    Get the second most fit individual in the population
    """
    def second_best(self):
        return self.individuals[self.lowest_cost2]
