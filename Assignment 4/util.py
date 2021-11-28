import gym
import numpy as np
import random
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv

BEGIN = "S"
FINISH = "G"
CRACK = "H"
ICE = "F"
LEFT = '\u2190'
RIGHT = '\u2192'
UP = '\u2191'
DOWN = '\u2193'

TEXT_ALIGN = "center"

def provide_scores(grid, length):
    new_grid = np.zeros((length,length))
    for value1, i in enumerate(grid):
        for value2, j in enumerate(i):
            if j == BEGIN or j == ICE:
                new_grid[value1, value2] = 0
            elif j == CRACK:
                new_grid[value1, value2] = -1
            elif j == FINISH:
                new_grid[value1, value2] = 1

    return new_grid


def adjust_data_structure(decision, length):
    #size = int(np.sqrt(len(decision)))
    new_decision = decision.reshape((length, length))
    return new_decision

def testing_helper(game, algo_array, x):
    iteration_score = 0
    complete = False
    iter = 0
    while not complete and iter < 1000:
        x, score, complete, tmp = game.step(int(algo_array[x]))
        iteration_score += score
        iter += 1

    return iteration_score, iter

def testing2(scores, array_of_iterations, game, algo_array, iterations):
    for value1 in range(iterations):
        x = game.reset()
        
        iteration_score, iter = testing_helper(game, algo_array, x)

        scores.append(iteration_score)
        array_of_iterations.append(iter)

    return scores, array_of_iterations

def tsting(game, algo_array, iterations=1000):
    scores = []
    array_of_iterations = []
    
    scores, array_of_iterations = testing2(scores, array_of_iterations, game, algo_array, iterations)
    
    average_score = sum(scores) / len(scores)
    average_episodes = sum(array_of_iterations)/len(array_of_iterations)
    return average_score, average_episodes, scores, array_of_iterations


def show_decisions(length, decision, originial_static_grid, grid_title):
    data = provide_scores(originial_static_grid[grid_title], length)
    new_decision = adjust_data_structure(np.asarray(decision), length)
    plt.imshow(data, interpolation="nearest")

    for value1 in range(new_decision[0].size):
        for value2 in range(new_decision[0].size):
            arrow = LEFT
            if new_decision[value1, value2] == 1:
                arrow = DOWN
            elif new_decision[value1, value2] == 2:
                arrow = RIGHT
            elif new_decision[value1, value2] == 3:
                arrow = UP
            tmp = plt.text(value2, value1, arrow, ha=TEXT_ALIGN, va=TEXT_ALIGN, color="w")
    plt.show()