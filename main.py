import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from deap import base, creator, tools, algorithms
from statistics import mean

# Deap's parameters
POPULATION_SIZE = 1000
NUM_GENERATIONS = 75
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.001

#Task's parameters
NUM_POINTS = 10

point_coordinates = [
    (1, 2), (2, 6), (6, 6), (6, 2), (3, 14),
    (10, 10), (15, 5), (15, 17), (17, 15), (19, 19)
]

connections = {
    0: [1, 2],
    1: [0, 2, 4],
    2: [0, 1, 3],
    3: [2],
    4: [1, 5, 7],
    5: [2, 4, 6, 7],
    6: [5, 9],
    7: [4, 5, 8, 9],
    8: [7],
    9: [6, 7]
}

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attribute", random.sample, range(NUM_POINTS), NUM_POINTS)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_fitness(individual):
    distance = 0
    for i in range(NUM_POINTS - 1):
        point1_index = individual[i]
        point2_index = individual[i + 1]
        x1, y1 = point_coordinates[point1_index]
        x2, y2 = point_coordinates[point2_index]
        distance += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if point2_index not in connections[point1_index]:
            distance += 100

    return distance,


toolbox.register("evaluate", evaluate_fitness)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("max", np.max)

hof = tools.HallOfFame(1)
population = toolbox.population(n=POPULATION_SIZE)
logbook = tools.Logbook()
logbook.header = ["gen", "nevals"] + stats.fields

for generation in range(NUM_GENERATIONS):
    fitness_values = map(toolbox.evaluate, population)

    for individual, fitness_value in zip(population, fitness_values):
        individual.fitness.values = fitness_value

    offspring = toolbox.select(population, len(population))
    offspring = algorithms.varAnd(offspring, toolbox, cxpb=CROSSOVER_PROB, mutpb=MUTATION_PROB)
    fitness_values = map(toolbox.evaluate, offspring)

    for child, fitness_value in zip(offspring, fitness_values):
        child.fitness.values = fitness_value

    population[:] = offspring
    hof.update(population)
    record = stats.compile(population)
    logbook.record(gen=generation, nevals=len(population), **record)

best_individual = hof[0]
best_fitness = evaluate_fitness(best_individual)[0]
print("Best Connections:", best_individual)
print("Best Fintess:", best_fitness)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
min_fitness_values = logbook.select("min")
avg_fitness_values = logbook.select("avg")
max_fitness_values = logbook.select("max")
print(f"Min: {round(min(min_fitness_values))} Avg: {round(mean(avg_fitness_values))} Max: {round(max(max_fitness_values))} ")

# Pops' stats, graph 1
ax1.plot(min_fitness_values, color="blue", label="Min")
ax1.plot(avg_fitness_values, color="green", label="Avg")
ax1.plot(max_fitness_values, color="red", label="Max")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness")
ax1.legend(loc="best")
ax1.set_title("Evolution of Fitness")
ax1.grid(True)

# Create the graph for points and connections between them
G = nx.Graph()
G.add_nodes_from(range(NUM_POINTS))
for point, connected_points in connections.items():
    for connected_point in connected_points:
        G.add_edge(point, connected_point)

# Best individual's route
G_best = nx.Graph()
G_best.add_nodes_from(range(NUM_POINTS))
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        G_best.add_edge(i, j)

# Connections between the points, graph 2
pos = {i: point_coordinates[i] for i in range(NUM_POINTS)}
nx.draw_networkx_nodes(G_best, pos=pos, ax=ax2)
nx.draw_networkx_labels(G_best, pos=pos, font_color="black", ax=ax2)
edges = G.edges
nx.draw_networkx_edges(G, pos=pos, edgelist=edges, edge_color="black", ax=ax2)

# Shortest path in red + connections between the points, graph 3
pos = {i: point_coordinates[i] for i in range(NUM_POINTS)}
nx.draw_networkx_nodes(G_best, pos=pos, ax=ax3)
nx.draw_networkx_labels(G_best, pos=pos, font_color="black", ax=ax3)
edges = G.edges
nx.draw_networkx_edges(G, pos=pos, edgelist=edges, edge_color="black", ax=ax3)
shortest_path = best_individual
shortest_path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
nx.draw_networkx_edges(G_best, pos=pos, edgelist=shortest_path_edges, edge_color="red", ax=ax3)

ax2.set_title("Connections sheme")
ax3.set_title("Best Route")
plt.tight_layout()
plt.show()
