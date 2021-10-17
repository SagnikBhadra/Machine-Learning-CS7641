import mlrose_hiive
import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
from randomized_algorithms import randomized_hill_climb, simulated_annealing, genetic_algorithm, mimic

if __name__ == "__main__":
    np.random.seed(42)
    knapsack_len=150
    weights = np.random.randint(1, 500, knapsack_len)
    values = np.random.randint(1, 500, knapsack_len)
    max_weight_pct = 0.5
    problem = mlrose_hiive.KnapsackOpt(weights=weights, values=values, max_weight_pct=max_weight_pct)

    
    randomized_hill_climb(problem)
    simulated_annealing(problem, temperature_list = [50, 100, 250, 500, 1000])
    genetic_algorithm(problem, population_sizes = [50, 100], mutation_rates = [0.1, 0.25, 0.5])
    mimic(problem, population_sizes = [500, 1000, 5000], keep_percent_list = [0.2, 0.25, 0.3])