import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from deap import base, creator, tools, algorithms

# Problem Constants
NUM_CITIES = 10

# Genetic Algorithm Constants
POPULATION_SIZE = 1000
NUM_GENERATIONS = 10
CROSSOVER_PROB = 0.5
MUTATION_PROB = 0.1

city_coordinates = [
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

# Create the graph
G = nx.Graph()

# Add nodes (cities) to the graph
G.add_nodes_from(range(NUM_CITIES))

for city, connected_cities in connections.items():
    for connected_city in connected_cities:
        G.add_edge(city, connected_city)

# Create the Fitness and Individual classes
creator.create("FitnessMin", base.Fitness, weights=(-0.1,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize the toolbox
toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attribute", random.sample, range(NUM_CITIES), NUM_CITIES)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_fitness(individual):
    # Calculate the total distance between cities
    distance = 0
    for i in range(NUM_CITIES - 1):
        city1_index = individual[i]
        city2_index = individual[i + 1]
        x1, y1 = city_coordinates[city1_index]
        x2, y2 = city_coordinates[city2_index]
        distance += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # Check if the connection exists between the current cities
        if city2_index not in connections[city1_index]:
            # Add a penalty for not following the predefined connection
            distance += 999999999999999

    return distance,


# Register the evaluation function
toolbox.register("evaluate", evaluate_fitness)

# Genetic operators
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create the statistics and Hall of Fame objects
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("max", np.max)

hof = tools.HallOfFame(1)

# Create the initial population
population = toolbox.population(n=POPULATION_SIZE)

# Create the logbook to store the evolution statistics
logbook = tools.Logbook()
logbook.header = ["gen", "nevals"] + stats.fields

# Perform the evolution
for generation in range(NUM_GENERATIONS):
    # Evaluate fitness for all individuals in the population
    fitness_values = map(toolbox.evaluate, population)
    for individual, fitness_value in zip(population, fitness_values):
        individual.fitness.values = fitness_value
    
    # Perform genetic operations
    offspring = toolbox.select(population, len(population))
    offspring = algorithms.varAnd(offspring, toolbox, cxpb=CROSSOVER_PROB, mutpb=MUTATION_PROB)
    
    # Update fitness values for the offspring
    fitness_values = map(toolbox.evaluate, offspring)
    for child, fitness_value in zip(offspring, fitness_values):
        child.fitness.values = fitness_value
    
    # Replace the population with the offspring
    population[:] = offspring
    
    # Update the Hall of Fame
    hof.update(population)
    
    # Gather statistics
    record = stats.compile(population)
    logbook.record(gen=generation, nevals=len(population), **record)

# Print the best individual and its fitness value
best_individual = hof[0]
best_fitness = evaluate_fitness(best_individual)[0]
print("Best Individual:", best_individual)
print("Best Fitness:", best_fitness)

# Plot the evolution statistics and the best individual's route
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Evolution statistics
min_fitness_values = logbook.select("min")
avg_fitness_values = logbook.select("avg")
max_fitness_values = logbook.select("max")

ax1.plot(min_fitness_values, color="blue", label="Min")
ax1.plot(avg_fitness_values, color="green", label="Avg")
ax1.plot(max_fitness_values, color="red", label="Max")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness")
ax1.legend(loc="best")
ax1.set_title("Evolution of Fitness")
ax1.grid(True)

# Best individual's route
G_best = nx.Graph()
G_best.add_nodes_from(range(NUM_CITIES))
for i in range(NUM_CITIES):
    for j in range(i + 1, NUM_CITIES):
        G_best.add_edge(i, j)

# Draw all nodes with their numbers
pos = {i: city_coordinates[i] for i in range(NUM_CITIES)}
nx.draw_networkx_nodes(G_best, pos=pos, ax=ax2)
nx.draw_networkx_labels(G_best, pos=pos, font_color="black", ax=ax2)

# Draw the connections between cities
edges = G.edges
nx.draw_networkx_edges(G, pos=pos, edgelist=edges, edge_color="black", ax=ax2)

# Draw all nodes with their numbers
pos = {i: city_coordinates[i] for i in range(NUM_CITIES)}
nx.draw_networkx_nodes(G_best, pos=pos, ax=ax3)
nx.draw_networkx_labels(G_best, pos=pos, font_color="black", ax=ax3)
edges = G.edges
nx.draw_networkx_edges(G, pos=pos, edgelist=edges, edge_color="black", ax=ax3)

# Draw the shortest path in red

shortest_path = best_individual
shortest_path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
print(f'shpe: {shortest_path_edges}')
nx.draw_networkx_edges(G_best, pos=pos, edgelist=shortest_path_edges, edge_color="red", ax=ax3)

ax2.set_title("Connection sheme")
ax3.set_title("Best Individual's Route")

plt.tight_layout()
plt.show()
