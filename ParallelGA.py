import array
import random
import json
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import multiprocessing
import matplotlib.pyplot as plt

IND_SIZE = NUMBER_OF_CITIES =100

distances = np.zeros((NUMBER_OF_CITIES, NUMBER_OF_CITIES))
for city in range(NUMBER_OF_CITIES):
    cities = [ i for i in range(NUMBER_OF_CITIES) if not i == city ]
    for to_city in cities:
        distances[to_city][city] = \
            distances[city][to_city] = random.randint(50, 2000)
        
def EVALUATE(individual):
    summation = 0
    start = individual[0]
    for i in range(1, len(individual)):
        end = individual[i]
        summation += distances[start][end]
        start = end
    return [summation]

def main():

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", EVALUATE)

    #マルチプロセスの定義
    #################################################
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    #################################################
    
    random.seed(169)

    pop = toolbox.population(n=300)

    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, stats = algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 100, stats=stats, halloffame=hof, verbose=True)


#生成器を作成(mainの外で行わなければマルチプロセスの場合エラーが起きる)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, typecode='i', fitness=creator.FitnessMin)

if __name__ == '__main__':
    main()

