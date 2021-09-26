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
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_curve, precision_score, accuracy_score, average_precision_score, recall_score, f1_score, classification_report, confusion_matrix
from util import evaluate_data, graph_LC, graph_C, normalize_data, create_stratified_data, prepare_data, preprocess_ufc_data

def define_KNN(input_train_set, output_train_set):
    KNN_Learner = KNeighborsClassifier()
    KNN_Learner.fit(input_train_set, output_train_set.values.ravel())
    
    print(KNN_Learner)
    return KNN_Learner

def KNN_predictions(KNN_Learner, input_train_set, output_train_set):
    y_hat = KNN_Learner.predict(input_train_set)
    percent_match = accuracy_score(output_train_set, y_hat)
    print(classification_report(output_train_set, y_hat))
    return percent_match


def compute_score(input_train_set, output_train_set, KNN_Learner, mean, title):
    result = evaluate_data(input_train_set, output_train_set, KNN_Learner, f1_score, average=mean)
    graph_LC(result[0], result[1], result[2], title=title)
    
    
def run_validation_curve(KNN_Learner, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, reverse, cross_validation_splitter, title):
    result_train, result_test = validation_curve(
        KNN_Learner, input_train_set, output_train_set.values.ravel(), param_name=hyperparameter, param_range=hyperparameter_values,
        scoring=weighting, verbose=1, cv=cross_validation_splitter, n_jobs=-1
    )
    print(result_train.mean(axis=1), result_test.mean(axis=1))
    graph_C(result_train.mean(axis=1), result_test.mean(axis=1), hyperparameter_values, title=title, inverse_x=reverse)
    
    return result_train, result_test

def wine_dataset():
    target_feature = "class"
    mean = "weighted"
    weighting = "f1_weighted"
    hyperparameter = "n_neighbors"
    cross_validation_splitter = 10
    n_jobs = -1
    reverse = True
    
    
    np.random.seed(42)
    wine_dataset = pd.read_csv("wine_dataset.csv")
    
    target = wine_dataset[target_feature]
    wine_dataset = normalize_data(wine_dataset)
    wine_dataset[target_feature] = target
    
    in_sample, out_of_sample = create_stratified_data(target_feature, wine_dataset)

    output_train_set, input_train_set, output_test_set, input_test_set = prepare_data(in_sample, out_of_sample, target_feature)
    
    KNN_Learner = define_KNN(input_train_set, output_train_set)
    
    percent_match = KNN_predictions(KNN_Learner, input_train_set, output_train_set)
    
    result = cross_val_score(KNN_Learner, input_train_set, output_train_set.values.ravel(), scoring=weighting, cv=cross_validation_splitter)
    print(result.mean())
    
    compute_score(input_train_set, output_train_set, KNN_Learner, mean, "F1_Score on Initial KNN Learner - Wine")
    
    KNN_Learner = KNeighborsClassifier(n_neighbors=100, weights='uniform')
    
    hyperparameter_values = [24,26,28,30,32,34,36,38,40]
    result_train, result_test = run_validation_curve(KNN_Learner, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, reverse, cross_validation_splitter, "F1_Score on KNN Learner for optimizing uniform weight - Wine")
    
    
    KNN_Learner = KNeighborsClassifier(n_neighbors=100, weights='distance')

    hyperparameter_values = [24,26,28,30,32,34,36,38,40]
    result_train, result_test = run_validation_curve(KNN_Learner, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, reverse, cross_validation_splitter, "F1_Score on KNN Learner for optimizing distance weight - Wine")
    
    optimal_KNN_Learner = KNeighborsClassifier(n_neighbors=28, weights='uniform')
    
    compute_score(input_train_set, output_train_set, optimal_KNN_Learner, mean, "F1_Score for optimal KNN Learner - Wine")

    optimal_KNN_Learner.fit(input_train_set, output_train_set.values.ravel())
    test_y_hat = optimal_KNN_Learner.predict(input_test_set)
    print(classification_report(output_test_set, test_y_hat,digits=5))
    print(confusion_matrix(output_test_set, test_y_hat))
    

def ufc_dataset():
    target_feature = "Winner"
    mean = "macro"
    weighting = "f1_macro"
    hyperparameter = "n_neighbors"
    cross_validation_splitter = 10
    n_jobs = -1
    reverse = True
    
    np.random.seed(42)
    ufc_dataset = pd.read_csv("ufc_data.csv")
    ufc_dataset = preprocess_ufc_data(ufc_dataset)
    ufc_dataset = normalize_data(ufc_dataset)
    
    in_sample, out_of_sample = create_stratified_data(target_feature, ufc_dataset)

    output_train_set, input_train_set, output_test_set, input_test_set = prepare_data(in_sample, out_of_sample, target_feature)
    
    KNN_Learner = define_KNN(input_train_set, output_train_set)
    
    percent_match = KNN_predictions(KNN_Learner, input_train_set, output_train_set)
    
    scores = cross_val_score(KNN_Learner, input_train_set, output_train_set.values.ravel(), scoring=weighting, cv=cross_validation_splitter)
    print(scores.mean())
    
    compute_score(input_train_set, output_train_set, KNN_Learner, mean, "F1_Score on Initial KNN Learner - UFC")
    
    KNN_Learner = KNeighborsClassifier(n_neighbors=100, weights='uniform')

    hyperparameter_values = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    result_train, result_test = run_validation_curve(KNN_Learner, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, reverse, cross_validation_splitter, "F1_Score on KNN Learner for optimizing uniform weight - UFC")
    
    
    KNN_Learner = KNeighborsClassifier(n_neighbors=100, weights='distance')

    hyperparameter_values = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    result_train, result_test = run_validation_curve(KNN_Learner, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, reverse, cross_validation_splitter, "F1_Score on KNN Learner for optimizing distance weight - UFC")

    optimal_KNN_Learner = KNeighborsClassifier(n_neighbors=2, weights='distance')

    compute_score(input_train_set, output_train_set, optimal_KNN_Learner, mean, "F1_Score for optimal KNN Learner - UFC")

    optimal_KNN_Learner.fit(input_train_set, output_train_set.values.ravel())
    test_y_hat = optimal_KNN_Learner.predict(input_test_set)
    print(classification_report(output_test_set, test_y_hat,digits=5))
    print(confusion_matrix(output_test_set, test_y_hat))
    
    
if __name__ == "__main__":
    wine_dataset()
    ufc_dataset()