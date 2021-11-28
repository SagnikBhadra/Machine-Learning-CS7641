import gym
import numpy as np
import random
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv

from util import provide_scores, adjust_data_structure, show_decisions, tsting

def exploration_exploitation_choice(game, change_per_iteration, algo_data_structure, x):
    tmp = random.uniform(0,1)
    if tmp > change_per_iteration:
        current_decision = algo_data_structure[x, :]
        y = np.random.choice(np.where(current_decision == current_decision.max())[0])
    else:
        y = game.action_space.sample()

    return y

def update_q_table(algo_data_structure, x, y, learning_rate, score, gamma, x_prime, complete):
    if not complete:
        algo_data_structure[x, y] = algo_data_structure[x, y] + learning_rate*(score + gamma*np.max(algo_data_structure[x_prime, :]) - algo_data_structure[x, y])
    else:
        algo_data_structure[x, y] = algo_data_structure[x,y] + learning_rate*(score - algo_data_structure[x,y])

    return algo_data_structure

def execute_algorithm(game, gamma=0.9, complete_iterations=1e5, learning_rate=0.1, epsilon_decrease=None, smallest_possible_change=0.01):
    
    begin = timer()
    
    x_len = game.observation_space.n
    y_len = game.action_space.n
    
    algo_data_structure = np.zeros((x_len, y_len))
    change_per_iteration = 1.0
    
    if not epsilon_decrease:
        epsilon_decrease = 1./complete_iterations
    
    scores = []
    for iteration_i in range(int(complete_iterations)):
        
        x = game.reset()
        cumulative_score = 0
        complete = False
        while True:
            y = exploration_exploitation_choice(game, change_per_iteration, algo_data_structure, x)
 
            x_prime, score, complete, knowledge = game.step(y)
            cumulative_score += score
            algo_data_structure = update_q_table(algo_data_structure, x, y, learning_rate, score, gamma, x_prime, complete)

            x = x_prime
            if complete:
                break
                
        scores.append(cumulative_score)
        change_per_iteration = max(1.0 -  epsilon_decrease * iteration_i, smallest_possible_change) 
    
    finish = timer()
    complexity_length = timedelta(seconds=finish-begin)
    print("Solved in: {} complete_iterations and {} seconds".format(complete_iterations, complexity_length))

    return np.argmax(algo_data_structure, axis=1), complete_iterations, complexity_length, algo_data_structure, scores

def run_algo_helper2(algorithm_matrix, index, value, gamma_instance, change_instance, learning_rate, speed_of_decrease_instance):
    algorithm_matrix[gamma_instance][change_instance][learning_rate][speed_of_decrease_instance][index] = value

    return algorithm_matrix


def run_algo_helper(algorithm_matrix, gamma_instance, change_instance, learning_rate, speed_of_decrease_instance, smallest_possible_change):
    algorithm_matrix[gamma_instance][change_instance][learning_rate][speed_of_decrease_instance] = {}

    algorithm_decision, total_num_of_iterations, total_num_of_seconds_taken, algo_data_structure, scores = execute_algorithm(game, gamma_instance, change_instance, learning_rate, speed_of_decrease_instance, smallest_possible_change)
    algorithm_average_score, algorithm_average_episodes, tmp1, tmp2 = tsting(game, algorithm_decision)

    algorithm_matrix = run_algo_helper2(algorithm_matrix, "mean_reward", algorithm_average_score, gamma_instance, change_instance, learning_rate, speed_of_decrease_instance)
    algorithm_matrix = run_algo_helper2(algorithm_matrix, "mean_eps", algorithm_average_episodes, gamma_instance, change_instance, learning_rate, speed_of_decrease_instance)
    algorithm_matrix = run_algo_helper2(algorithm_matrix, "q-table", algo_data_structure, gamma_instance, change_instance, learning_rate, speed_of_decrease_instance)
    algorithm_matrix = run_algo_helper2(algorithm_matrix, "scores", scores, gamma_instance, change_instance, learning_rate, speed_of_decrease_instance)
    algorithm_matrix = run_algo_helper2(algorithm_matrix, "iteration", total_num_of_iterations, gamma_instance, change_instance, learning_rate, speed_of_decrease_instance)
    algorithm_matrix = run_algo_helper2(algorithm_matrix, "complexity_length", total_num_of_seconds_taken, gamma_instance, change_instance, learning_rate, speed_of_decrease_instance)
    algorithm_matrix = run_algo_helper2(algorithm_matrix, "policy", algorithm_decision, gamma_instance, change_instance, learning_rate, speed_of_decrease_instance)

    return algorithm_matrix

def run_algorithm_iteration(game, gamma=[0.9], complete_iterations=[1e5], learning_rates=[0.1], speed_of_decrease=[0.01], mute=False):
    
    smallest_possible_change = 0.01
    
    algorithm_matrix = {}
    for gamma_instance in gamma:
        algorithm_matrix[gamma_instance] = {}
        for change_instance in complete_iterations:
            algorithm_matrix[gamma_instance][change_instance] = {}
            for learning_rate in learning_rates:
                algorithm_matrix[gamma_instance][change_instance][learning_rate] = {}
                for speed_of_decrease_instance in speed_of_decrease:
                    algorithm_matrix = run_algo_helper(algorithm_matrix, gamma_instance, change_instance, learning_rate, speed_of_decrease_instance, smallest_possible_change)

    return algorithm_matrix


def data_structure(matrix):
    matrix_instance = pd.DataFrame(columns=["Discount Rate", "Training Episodes", "Learning Rate", "Decay Rate", "Reward", "Time Spent"])

    for gamma_instance in matrix:
        for change_instance in matrix[gamma_instance]:
            for learning_rate in matrix[gamma_instance][change_instance]:
                for speed_of_decrease_instance in matrix[gamma_instance][change_instance][learning_rate]:
                    sco = matrix[gamma_instance][change_instance][learning_rate][speed_of_decrease_instance]["mean_reward"]
                    complexity_length = matrix[gamma_instance][change_instance][learning_rate][speed_of_decrease_instance]["complexity_length"].total_seconds()
                    cdr = {"Discount Rate": gamma_instance, "Training Episodes": change_instance, "Learning Rate":learning_rate, "Decay Rate":speed_of_decrease_instance, "Reward": sco, "Time Spent": complexity_length}
                    matrix_instance = matrix_instance.append(cdr, ignore_index=True)
    return matrix_instance

def calculate_average(data, index):
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[index:] - cumsum[:-index]) / float(index)


def plot_seabron_graph(index0, index1, index2, index3, index):
    data = algorithm_matrix[index0][index1][index2][index3]['scores']
    trailing_average = calculate_average(data, index)
    xes = [i+index for i in list(range(len(trailing_average)))]
    sns.lineplot(np.log10(xes), trailing_average)
    plt.show()



if __name__ == "__main__":

    np.random.seed(42)

    game = gym.make('FrozenLake-v1')    
    complete_iterations = [1e4, 1e5, 1e6]
    speed_of_decrease = [1e-6]

    four = generate_random_map(4)
    sixteen = generate_random_map(16)
    index = 1000
    originial_static_grid = {
        "4x4": four,
        "16x16": sixteen
    }
    
    print("4x4")

    algorithm_matrix = run_algorithm_iteration(game, gamma=[0.75, 0.9, 0.99, 0.9999], complete_iterations=complete_iterations, learning_rates=[0.01, 0.1], speed_of_decrease=speed_of_decrease)
    
    
    decisions = algorithm_matrix[0.99][int(1e6)][0.1][1e-06]['policy']
    show_decisions(4, decisions, originial_static_grid, "4x4")
    
    speed_of_decrease = [1e-3, 1e-5]
    algorithm_matrix = run_algorithm_iteration(game, gamma= [0.9999], complete_iterations=complete_iterations, learning_rates=[0.1, 0.01], speed_of_decrease=speed_of_decrease)
    
    plot_seabron_graph(0.9999, int(1e6), 0.1, 1e-03, index)
    plot_seabron_graph(0.9999, int(1e6), 0.01, 1e-03, index)
    plot_seabron_graph(0.9999, int(1e6), 0.01, 1e-05, index)


    algo_results = data_structure(algorithm_matrix)
    pl = sns.lineplot(x="Training Episodes", y="Reward", data=algo_results)
    plt.show()
    
    display(algo_results)
    
    decisions = algorithm_matrix[0.9999][int(1e6)][0.01][1e-03]['policy']
    show_decisions(4, decisions, originial_static_grid, "4x4")
    
    
    print("16x16")
    
    game = FrozenLakeEnv(desc=originial_static_grid["16x16"])
    speed_of_decrease = [1e-3, 1e-5]
    algo_results = run_algorithm_iteration(game, gamma= [0.9999], complete_iterations=complete_iterations, learning_rates=[0.1, 0.01], speed_of_decrease=speed_of_decrease)
    
    decisions = algo_results[0.9999][int(1e6)][0.1][1e-05]['policy']
    show_decisions(16, decisions, originial_static_grid, "16x16")
    
    print((algo_results[0.9999][int(1e6)][0.1][1e-05]['q-table'] > 0).any())
    
    display(data_structure(algo_results))

    
