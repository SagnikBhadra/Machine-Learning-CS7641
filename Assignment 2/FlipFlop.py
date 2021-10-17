import mlrose_hiive
import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
from randomized_algorithms import randomized_hill_climb, simulated_annealing, genetic_algorithm, mimic

if __name__ == "__main__":
    np.random.seed(42)
    problem = mlrose_hiive.FlipFlopOpt(length=400)
    
    randomized_hill_climb(problem)
    simulated_annealing(problem, temperature_list = [25, 50, 100, 500])
    genetic_algorithm(problem, population_sizes = [20,50,100], mutation_rates = [0.1, 0.25, 0.5])
    mimic(problem, population_sizes = [20,50,100], keep_percent_list = [0.25, 0.5, 0.75])
    
    