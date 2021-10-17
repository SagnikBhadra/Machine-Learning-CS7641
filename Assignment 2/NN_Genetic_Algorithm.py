import six
import sys
sys.modules['sklearn.externals.six'] = six
import sklearn
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
pd.options.display.max_columns = None
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('figure', figsize=[10,5])
from sklearn.metrics import f1_score, accuracy_score, classification_report
import mlrose
from functools import partial
import timeit
from util import evaluate_data, graph_LC, graph_C, normalize_data, create_stratified_data, prepare_data, preprocess_ufc_data

def testing(neural_network_genetic_algorithm, input_test_set, output_test_set, input_train_set, output_train_set):
    predictions = neural_network_genetic_algorithm.predict(input_test_set)
    print(classification_report(pd.get_dummies(output_test_set.values.ravel()).values, predictions))
    
    predictions_train = neural_network_genetic_algorithm.predict(input_train_set)
    print(classification_report(pd.get_dummies(output_train_set.values.ravel()).values, predictions_train))

def wine_dataset():
    target_feature = "class"
    mean = "weighted"
    weighting = "f1_weighted"
    hyperparameter = "hidden_layer_sizes"
    cross_validation_splitter = 10
    parallel = -1
    max_iters = 10000
    max_attempts = 100
    pop_size=200
    mutation_prob=0.25

    np.random.seed(42)
    wine_dataset = pd.read_csv("wine_dataset.csv")


    target = wine_dataset[target_feature]
    wine_dataset = normalize_data(wine_dataset)
    wine_dataset[target_feature] = target
    
    in_sample, out_of_sample = create_stratified_data(target_feature, wine_dataset)

    output_train_set, input_train_set, output_test_set, input_test_set = prepare_data(in_sample, out_of_sample, target_feature)
    
    
    print('Genetic Algorithm')
    start = timeit.default_timer()
    
    neural_network_genetic_algorithm1 = mlrose.NeuralNetwork(hidden_nodes = [1700], 
                                     algorithm = 'genetic_alg', 
                                     max_iters = max_iters,
                                     learning_rate = 0.1,
                                     early_stopping = True,
                                     max_attempts = max_attempts,
                                     random_state = 42,
                                     pop_size=pop_size,
                                     mutation_prob=mutation_prob)
    neural_network_genetic_algorithm1.fit(input_train_set, pd.get_dummies(output_train_set.values.ravel()).values)   
    
    stop = timeit.default_timer()
    print('Randimized Hill Climb Time: ', stop - start)  
    

    testing(neural_network_genetic_algorithm1, input_test_set, output_test_set, input_train_set, output_train_set)
    
    
    neural_network_genetic_algorithm2 = mlrose.NeuralNetwork(hidden_nodes = [1700], 
                                     algorithm = 'genetic_alg', 
                                     max_iters = max_iters,
                                     learning_rate = 0.01,
                                     early_stopping = True,
                                     max_attempts = max_attempts,
                                     random_state = 42,
                                     pop_size=pop_size,
                                     mutation_prob=mutation_prob)

    neural_network_genetic_algorithm2.fit(input_train_set, pd.get_dummies(output_train_set.values.ravel()).values)
    
    
    testing(neural_network_genetic_algorithm2, input_test_set, output_test_set, input_train_set, output_train_set)
    
    

if __name__ == "__main__":
    wine_dataset()