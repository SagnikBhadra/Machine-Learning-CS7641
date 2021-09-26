import numpy as np
import sklearn
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
pd.options.display.max_columns = None
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('figure', figsize=[10,5])
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

def evaluate_data_helper3(input_data_i, output_data_i, input_j, output_j):
    in_sample_input_split = input_data_i.iloc[input_j]
    in_sample_output_split = output_data_i.iloc[input_j]
    out_of_sample_input_split = input_data_i.iloc[output_j]
    out_of_sample_output_split = output_data_i.iloc[output_j]

    return in_sample_input_split, in_sample_output_split, out_of_sample_input_split, out_of_sample_output_split

def evaluate_data_helper2(learner, evaluater, average, input_data_i, output_data_i, skfolds):
    aevaluater = []
    evaluater_in_sample = []

    for input_j, output_j in skfolds.split(input_data_i, output_data_i):
        in_sample_input_split, in_sample_output_split, out_of_sample_input_split, out_of_sample_output_split = evaluate_data_helper3(input_data_i, output_data_i, input_j, output_j)

        learner.fit(in_sample_input_split.values, in_sample_output_split.values.ravel())
        in_sample_split_y_hat = learner.predict(in_sample_input_split)
        out_of_sample_split_y_hat = learner.predict(out_of_sample_input_split)
        aevaluater.append(evaluater(out_of_sample_output_split.values, out_of_sample_split_y_hat, average=average))
        evaluater_in_sample.append(evaluater(in_sample_output_split.values, in_sample_split_y_hat, average=average))

    return aevaluater, evaluater_in_sample
    

def evaluate_data_helper(beginning, input_data, output_data, learner, evaluater, average, fold, in_sample_evaluater, out_of_sample_evaluater, hyperparameter_values):
    for i in range(beginning, len(input_data), 100):
        input_data_i = input_data[:i]
        output_data_i = output_data[:i]
        skfolds = StratifiedKFold(n_splits=fold)
        
        aevaluater, evaluater_in_sample = evaluate_data_helper2(learner, evaluater, average, input_data_i, output_data_i, skfolds)

        out_of_sample_result = sum(aevaluater)/len(aevaluater)
        in_sample_result = sum(evaluater_in_sample)/len(evaluater_in_sample)
        out_of_sample_evaluater.append(out_of_sample_result)
        in_sample_evaluater.append(in_sample_result)
        hyperparameter_values.append(i)

    return in_sample_evaluater, out_of_sample_evaluater, hyperparameter_values


def evaluate_data(input_data, output_data, learner, evaluater, average=None, fold=5, beginning=100):
    in_sample_evaluater = []
    out_of_sample_evaluater = []
    hyperparameter_values = []

    in_sample_evaluater, out_of_sample_evaluater, hyperparameter_values = evaluate_data_helper(beginning, input_data, output_data, learner, evaluater, average, fold, in_sample_evaluater, out_of_sample_evaluater, hyperparameter_values)
    
    return in_sample_evaluater, out_of_sample_evaluater, hyperparameter_values


def graph_LC(in_sample_data, out_of_sample_data, hyperparameter_values, title, inverse_x=False):
    plt.plot(hyperparameter_values, in_sample_data , "r-+", linewidth=2, label="in-sample")
    plt.plot(hyperparameter_values, out_of_sample_data, "b-+", linewidth=2, label="out-of-sample")
    plt.title(title)
    plt.legend()
    plt.xlabel('Instances')
    if inverse_x:
        plt.gca().invert_xaxis()
    plt.savefig(title)
    plt.show()
        

def graph_C(in_sample_data, out_of_sample_data, hyperparameter_values, title, inverse_x=False):
    plt.plot(hyperparameter_values, in_sample_data , "r-+", linewidth=2, label="in-sample")
    plt.plot(hyperparameter_values, out_of_sample_data, "b-+", linewidth=2, label="out-of-sample")
    plt.title(title)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig(title)
    plt.show()
    
    
def normalize_data(dataset):
    data = dataset.values 
    normalize_foactor = preprocessing.MinMaxScaler()
    normalized_data = normalize_foactor.fit_transform(data)
    dataset = pd.DataFrame(normalized_data, columns=dataset.columns)
    return dataset

def create_stratified_data(target_feature, dataset):
    divide = StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=30)
    for in_sample_j, out_of_sample_j in divide.split(dataset, dataset[target_feature]):
        in_sample = dataset.loc[in_sample_j]
        out_of_sample = dataset.loc[out_of_sample_j]
        
    return in_sample, out_of_sample

def prepare_data(in_sample, out_of_sample, target_feature):
    output_train_set = in_sample[[target_feature]]
    input_train_set = in_sample.drop(target_feature, axis=1)
    output_test_set = out_of_sample[[target_feature]]
    input_test_set = out_of_sample.drop(target_feature, axis=1)
    
    return output_train_set, input_train_set, output_test_set, input_test_set


def preprocess_ufc_data(ufc_dataset):
    target_feature = "Winner"
    fav = "Red"
    not_fav = "Blue"
    drop_features = "no_of_rounds"
    drop_features_B = "B_draw"
    drop_features_R = "R_draw"

    ufc_dataset.loc[ufc_dataset[target_feature] == fav, target_feature] = 1
    ufc_dataset.loc[ufc_dataset[target_feature] == not_fav, target_feature] = 0
    ufc_dataset.drop([drop_features], axis=1 ,inplace=True)
    ufc_dataset.drop([drop_features_B], axis=1 ,inplace=True)
    ufc_dataset.drop([drop_features_R], axis=1 ,inplace=True)

    return ufc_dataset