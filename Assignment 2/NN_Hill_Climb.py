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
import mlrose_hiive
from functools import partial
import timeit
from util import evaluate_data, graph_LC, graph_C, normalize_data, create_stratified_data, prepare_data, preprocess_ufc_data

def testing(output, input_test_set, output_test_set, input_train_set, output_train_set):
    predictions = output[3].predict(input_test_set)
    print(classification_report(pd.get_dummies(output_test_set.values.ravel()).values, predictions))
    
    train_predictions = output[3].predict(input_train_set)
    print(classification_report(pd.get_dummies(output_train_set.values.ravel()).values, train_predictions))

def wine_dataset():
    target_feature = "class"
    mean = "weighted"
    weighting = "f1_weighted"
    hyperparameter = "hidden_layer_sizes"
    cross_validation_splitter = 10
    parallel = -1
    scoring = partial(f1_score, average=mean)

    np.random.seed(42)
    wine_dataset = pd.read_csv("wine_dataset.csv")


    target = wine_dataset[target_feature]
    wine_dataset = normalize_data(wine_dataset)
    wine_dataset[target_feature] = target
    
    in_sample, out_of_sample = create_stratified_data(target_feature, wine_dataset)

    output_train_set, input_train_set, output_test_set, input_test_set = prepare_data(in_sample, out_of_sample, target_feature)
    
    
    hyperparameter_tuning = ({
      'learning_rate': [0.01, 0.001],
      'restarts': [25, 50],
      'activation': [mlrose_hiive.neural.activation.relu]
    })
    
    print('Randimized Hill Climb')
    start = timeit.default_timer()
    neural_network_randomized_hill_climb = mlrose_hiive.NNGSRunner(input_train_set, pd.get_dummies(output_train_set.values.ravel()).values, 
                                     input_test_set, pd.get_dummies(output_test_set.values.ravel()).values, 
                                     "NeuralNetworkRHC",
                                     output_directory="./",
                                     seed=10, iteration_list=[10000], 
                                     algorithm=mlrose_hiive.random_hill_climb,
                                     hidden_layer_sizes=[[60,60]],
                                     grid_search_parameters=hyperparameter_tuning,
                                     grid_search_scorer_method=scoring,
                                     n_jobs=-2, cv=5)
    output = neural_network_randomized_hill_climb.run()
    
    stop = timeit.default_timer()
    print('Randimized Hill Climb Time: ', stop - start)  
    
    testing(output, input_test_set, output_test_set, input_train_set, output_train_set)
    
    
if __name__ == "__main__":
    wine_dataset()