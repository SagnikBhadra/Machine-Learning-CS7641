from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
from hiive.mdptoolbox.example import forest
import gym
import numpy as np
import sys
import os
from numpy.random import choice
import pandas as pd
import seaborn as sns

from util import provide_scores, adjust_data_structure, show_decisions, tsting

# Action = i
# State = j

def iteration_over_episode(probability_matrix, reward_matrix, policy, j, episodes, cumulative_reward, iterations_per_state=1000, gamma=0.9):
    iterative_reward = 0
    for tmp in range(iterations_per_state):
        r = 0
        discount = 1
        while True:
            # take step
            i = policy[j]
            # get next step using probability_matrix
            chance = probability_matrix[i][j]
            a = list(range(len(probability_matrix[i][j])))
            s_prime =  choice(a, 1, p=chance)[0]
            # get the score
            r_delta = reward_matrix[j][i] * discount
            discount *= gamma
            r += r_delta
            if s_prime == 0:
                break
        iterative_reward += r

    return iterative_reward

def iteration_over_state(probability_matrix, reward_matrix, policy, total_states, episodes, cumulative_reward, iterations_per_state, gamma):
    for j in range(total_states):
        iterative_reward = 0

        iterative_reward = iteration_over_episode(probability_matrix, reward_matrix, policy, j, episodes, cumulative_reward, iterations_per_state, gamma)
        
        cumulative_reward += iterative_reward

    return cumulative_reward


def testing(probability_matrix, reward_matrix, policy, iterations_per_state=1000, gamma=0.9):
    total_states = probability_matrix.shape[-1]
    episodes = total_states * iterations_per_state

    cumulative_reward = 0

    cumulative_reward = iteration_over_state(probability_matrix, reward_matrix, policy, total_states, episodes, cumulative_reward, iterations_per_state, gamma)
    
    return cumulative_reward / episodes

def value_iteration(probability_matrix, reward_matrix, epsilon, gamma=0.9):
    value_iteration_data_frame = pd.DataFrame(columns=["Epsilon", "Policy", "Iteration", "Time", "Reward", "Value Function"])
    for i in epsilon:
        value_iteration = ValueIteration(probability_matrix, reward_matrix, gamma=gamma, epsilon=i, max_iter=int(1e15))
        value_iteration.run()
        r = testing(probability_matrix, reward_matrix, value_iteration.policy)
        value_iteration_data_frame.loc[len(value_iteration_data_frame)] = [float(i), value_iteration.policy, value_iteration.iter, value_iteration.time, r, value_iteration.V]
    return value_iteration_data_frame

def Q_learning(probability_matrix, reward_matrix, gamma=0.9, learning_rate_decay=[0.99], learning_rate_cut_off=[0.001], epsilon=[1.0], epsilon_decay=[0.99], episodes=[1000000]):
    Q_learning_data_frame = pd.DataFrame(columns=["Iterations", "Alpha Decay", "Alpha Min", "Epsilon", "Epsilon Decay", "Reward", "Time", "Policy", "Value Function", "Training Rewards"])
    
    total = 0
    for i in episodes:
        for j in epsilon:
            for k in epsilon_decay:
                for learning_rate_d in learning_rate_decay:
                    for learning_rate_m in learning_rate_cut_off:
                        algo = QLearning(probability_matrix, reward_matrix, gamma, alpha_decay=learning_rate_d, alpha_min=learning_rate_m, epsilon=j, epsilon_decay=k, n_iter=i)
                        algo.run()
                        score = testing(probability_matrix, reward_matrix, algo.policy)
                        total += 1
                        print("{}: {}".format(total, score))
                        scores = [tmp['Reward'] for tmp in algo.run_stats]
                        
                        Q_learning_data_frame.loc[len(Q_learning_data_frame)] = [i, learning_rate_d, learning_rate_m, j, k, score, algo.time, algo.policy, algo.V, scores]

    return Q_learning_data_frame

def run_policy_iteration(probability_matrix, reward_matrix):
    print("Policy Iteration")

    policy_iteration = PolicyIteration(probability_matrix, reward_matrix, gamma=0.9, max_iter=1e6)
    policy_iteration.run()
    policy_iteration_policy = policy_iteration.policy
    policy_iteration_score = testing(probability_matrix, reward_matrix, policy_iteration_policy)
    print(policy_iteration.iter, policy_iteration.time, policy_iteration_score)
    
    display(policy_iteration_policy)


def run_forest_management(probability_matrix, reward_matrix):
    value_iteration_data_frame = value_iteration(probability_matrix, reward_matrix, epsilon=[1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15])
    display(value_iteration_data_frame)
    
    run_policy_iteration(probability_matrix, reward_matrix)
    
    
    print("Q_learning")
    
    learning_rate_decay = [0.99, 0.999]
    learning_rate_cut_off =[0.001, 0.0001]
    i = [10.0, 1.0]
    k = [0.99, 0.999]
    episodes = [1000000, 10000000]
    Q_learning_data_frame = Q_learning(probability_matrix, reward_matrix, gamma=0.9, learning_rate_decay=learning_rate_decay, learning_rate_cut_off=learning_rate_cut_off, epsilon=i, epsilon_decay=k, episodes=episodes)
    
    
    
    testing(probability_matrix,reward_matrix, Q_learning_data_frame.Policy[18])
    
    display(Q_learning_data_frame)
    
    display(Q_learning_data_frame.groupby("Iterations").mean())
    
    display(Q_learning_data_frame.groupby("Epsilon Decay").mean())


if __name__ == "__main__":
    np.random.seed(44)
    
    print("20 States\n\n\n")
    
    probability_matrix, reward_matrix = forest(S=20, r1=10, r2=6, p=0.1)

    run_forest_management(probability_matrix, reward_matrix)
    
    print("500 States\n\n\n")
    
    probability_matrix, reward_matrix = forest(S=500, r1=100, r2= 15, p=0.01)
    
    run_forest_management(probability_matrix, reward_matrix)
    