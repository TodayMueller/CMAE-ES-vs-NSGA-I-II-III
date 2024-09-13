import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Определение задачи оптимизации
def rosenbrock(x):
    return sum(100.0 * (x[i+1] - x[i]**2.0)**2.0 + (1 - x[i])**2.0 for i in range(len(x)-1)),

# Создание классов для DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

# Определение инструментов DEAP
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -5, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", rosenbrock)

# Кастомный метод сравнения для HallOfFame
def custom_similar(ind1, ind2):
    return np.array_equal(ind1, ind2)

# Основная функция для выполнения оптимизации
def main():
    np.random.seed(42)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1, similar=custom_similar)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=300, lambda_=300, cxpb=0.7, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)
    
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()
    print("Best individual is ", hof[0], hof[0].fitness.values)
    
    # Построение графика конвергенции
    gen = log.select("gen")
    avg = log.select("avg")
    min_ = log.select("min")
    max_ = log.select("max")
    
    plt.figure(figsize=(10, 6))
    plt.plot(gen, avg, label="Average Fitness")
    plt.plot(gen, min_, label="Minimum Fitness")
    plt.plot(gen, max_, label="Maximum Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Convergence of the Rosenbrock Function using CMA-ES")
    plt.legend()
    plt.grid()
    plt.savefig('rosenbrock_convergence.png')
    plt.show()
