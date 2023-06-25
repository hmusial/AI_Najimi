from deap import base, algorithms
from deap import creator
from deap import tools

import random
import matplotlib.pyplot as plt
import numpy as np


# Task's value
ONE_MAX_LENGTH = 100

# Algoritm's values
POPULATION_SIZE = 1000
P_CROSSOVER = 0.1
P_MUTATION = 0.1
MAX_GENERATIONS = 50

RANDOM_SEED = 10
random.seed(RANDOM_SEED)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def oneMaxFitness(individual):
    return sum(individual),

toolbox = base.Toolbox()

toolbox.register("zeroOrOne", random.randint, 0, 1)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)

def select(population):
    return population

toolbox.register("evaluate", oneMaxFitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("values", np.array)

population, logbook = algorithms.eaSimple(population, toolbox,
                                        cxpb=P_CROSSOVER,
                                        mutpb=P_MUTATION,
                                        ngen=MAX_GENERATIONS,
                                        stats=stats,
                                        verbose=False)

maxFitnessValues, meanFitnessValues, minFitnessValues, vals = logbook.select("max", "avg", "min", "values")

    
plt.plot(maxFitnessValues, color='red', label='Maximum')
plt.plot(meanFitnessValues, color='black', label='Average')
plt.plot(minFitnessValues, color='green', label='Minimum')
plt.xlabel('Generation')
plt.ylabel('Max/Average/Min')
plt.title('Values of each generations')
plt.legend(loc="lower right")
plt.show()