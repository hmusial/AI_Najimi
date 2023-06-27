from deap import base, algorithms
from deap import creator
from deap import tools

from graph_show import show_graph

import random
import matplotlib.pyplot as plt
import numpy as np

inf = 100
D = ((0, 3, 1, 3, inf, inf),
     (3, 0, 4, inf, inf, inf),
     (1, 4, 0, inf, 7, 5),
     (3, inf, inf, 0, inf, 2),
     (inf, inf, 7, inf, 0, 4),
     (inf, inf, 5, 2, 4, 0))

start = 0
end = 5
LENGTH_D = len(D)
LENGTH_CHROM = len(D)*len(D[0])
POPULATION_SIZE = 1000
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 30
HALL_OF_FAME_SIZE = 1

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("randomOrder", random.sample, range(LENGTH_D), LENGTH_D)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randomOrder, LENGTH_D)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)


def dikstryFitness(individual):
    s = 0
    for n, path in enumerate(individual):
        path = path[:path.index(n)+1]

        si = start
        for j in path:
            s += D[si][j]
            si = j

    return s,

def cxOrdered(ind1, ind2):
    for p1, p2 in zip(ind1, ind2):
        tools.cxOrdered(p1, p2)

    return ind1, ind2

def mutShuffleIndexes(individual, indpb):
    for ind in individual:
        tools.mutShuffleIndexes(ind, indpb)

    return individual,


toolbox.register("evaluate", dikstryFitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", cxOrdered)
toolbox.register("mutate", mutShuffleIndexes, indpb=1.0/LENGTH_CHROM/10)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("max", np.max)

population, logbook = algorithms.eaSimple(population, toolbox,
                                        cxpb=P_CROSSOVER/LENGTH_D,
                                        mutpb=P_MUTATION/LENGTH_D,
                                        ngen=MAX_GENERATIONS,
                                        halloffame=hof,
                                        stats=stats,
                                        verbose=True)

minFitnessValues, meanFitnessValues, maxFitnessValues = logbook.select("min", "avg", "max")

plt.plot(maxFitnessValues, color='red', label='Maximum')
plt.plot(meanFitnessValues, color='green', label='Average')
plt.plot(minFitnessValues, color='blue', label='Minimum')
plt.xlabel('Generation')
plt.ylabel('Max/Average/Min')
plt.title('Values of each generations')
plt.legend(loc="lower right")
fig, ax = plt.subplots()
best = hof.items[0]
show_graph(ax, best, start, end)
plt.show()
