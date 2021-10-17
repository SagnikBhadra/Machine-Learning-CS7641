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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_curve, precision_score, accuracy_score, average_precision_score, recall_score, f1_score, classification_report, confusion_matrix
from util import evaluate_data, graph_LC, graph_C, normalize_data, create_stratified_data, prepare_data, preprocess_ufc_data

def define_SVM_Classifier(input_train_set, output_train_set, gamma, class_weight):
    SVM_Learner = SVC(gamma=gamma, class_weight = class_weight)
    SVM_Learner.fit(input_train_set, output_train_set.values.ravel())
    
    print(SVM_Learner)
    return SVM_Learner

def svm_predictions(SVM_Learner, input_train_set, output_train_set, mean):
    y_hat = SVM_Learner.predict(input_train_set)
    percent_match = accuracy_score(output_train_set, y_hat)
    print(percent_match)
    print(classification_report(output_train_set, y_hat))
    print(f1_score(output_train_set, y_hat, average=mean))


def compute_score(input_train_set, output_train_set, SVM_Learner, mean, title):
    result = evaluate_data(input_train_set, output_train_set, SVM_Learner, f1_score, average=mean)
    graph_LC(result[0], result[1], result[2], title=title)
    

def run_gridsearch(values, SVM_Learner, input_train_set, output_train_set, cross_validation_splitter, weighting, parallel):
    grid_suggestion = GridSearchCV(SVM_Learner, values, cv=cross_validation_splitter, scoring=weighting, n_jobs = parallel)
    grid_suggestion.fit(input_train_set, output_train_set.values.ravel())
    print(grid_suggestion.best_estimator_, grid_suggestion.best_score_)
    
    return grid_suggestion
    
    
def run_validation_curve(SVM_Learner, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, reverse, cross_validation_splitter, title):
    result_train, result_test = validation_curve(
        SVM_Learner, input_train_set, output_train_set.values.ravel(), param_name=hyperparameter, param_range=hyperparameter_values,
        scoring=weighting, verbose=1, cv=cross_validation_splitter, n_jobs=-1
    )
    print(result_train.mean(axis=1), result_test.mean(axis=1))
    graph_C(result_train.mean(axis=1), result_test.mean(axis=1), hyperparameter_values, title=title, inverse_x=reverse)
    
    return result_train, result_test

def wine_dataset():
    target_feature = "class"
    alpha = 0.005
    criterion = "entropy"
    mean = "weighted"
    weighting = "f1_weighted"
    hyperparameter = "C"
    cross_validation_splitter = 10
    parallel = -1
    gamma = 'auto'
    class_weight = None
    reverse = False
    
    
    np.random.seed(42)
    wine_dataset = pd.read_csv("wine_dataset.csv")
    
    target = wine_dataset[target_feature]
    wine_dataset = normalize_data(wine_dataset)
    wine_dataset[target_feature] = target
    
    in_sample, out_of_sample = create_stratified_data(target_feature, wine_dataset)
    
    output_train_set, input_train_set, output_test_set, input_test_set = prepare_data(in_sample, out_of_sample, target_feature)
    
    SVM_Learner = define_SVM_Classifier(input_train_set, output_train_set, gamma, class_weight)
    
    svm_predictions(SVM_Learner, input_train_set, output_train_set, mean)
    
    result = cross_val_score(SVM_Learner, input_train_set, output_train_set.values.ravel(), scoring=weighting, cv=cross_validation_splitter)
    print(result.mean())
    
    compute_score(input_train_set, output_train_set, SVM_Learner, mean, "F1_Score on Initial SVM Learner - Wine")
    
    size = input_train_set.shape[1]
    values = {
        'kernel': ['linear', 'poly', 'rbf', 'poly'],
        'degree': [2, 3, 4, 5],
        'gamma': [1/size,2/size,5/size, 6/size, 7/size, 8/size, 9/size]
    }

    SVM_Learner = SVC()
    
    
    grid_suggestion = run_gridsearch(values, SVM_Learner, input_train_set, output_train_set, cross_validation_splitter, weighting, parallel)

    print(grid_suggestion.best_estimator_, grid_suggestion.best_score_)

    compute_score(input_train_set, output_train_set, grid_suggestion.best_estimator_, mean, "F1_Score on GridSearch optimized SVM Learner - Wine")
    
    SVM_Learner = SVC(degree=2, gamma=0.6923)

    hyperparameter_values = np.linspace(1, 50, 49)
    weighting='accuracy'
    result_train, result_test = run_validation_curve(SVM_Learner, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, reverse, cross_validation_splitter, "F1_Score on SVM Learner for optimizing C - Wine")
    
    j = result_test.mean(axis=1).argmax()
    linear_C = hyperparameter_values[j]
    print(linear_C)
    
    
    optimal_SVM = SVC(degree=2, gamma=0.6923, C = 3)

    compute_score(input_train_set, output_train_set, optimal_SVM, mean, "F1_Score for optimal SVM Learner - Wine")
    
    optimal_SVM.fit(input_train_set, output_train_set.values.ravel())
    test_y_hat = optimal_SVM.predict(input_test_set)
    print(classification_report(output_test_set, test_y_hat, digits=5))
    print(confusion_matrix(output_test_set, test_y_hat))
    
    
def ufc_dataset():
    target_feature = "Winner"
    mean = "macro"
    weighting = "f1_macro"
    hyperparameter = "C"
    cross_validation_splitter = 10
    parallel = -1
    gamma = 'auto'
    class_weight = 'balanced'
    reverse = False
    
    
    np.random.seed(42)
    ufc_dataset = pd.read_csv("ufc_data.csv")
    ufc_dataset = preprocess_ufc_data(ufc_dataset)
    
    ufc_dataset = normalize_data(ufc_dataset)
    
    in_sample, out_of_sample = create_stratified_data(target_feature, ufc_dataset)
    
    output_train_set, input_train_set, output_test_set, input_test_set = prepare_data(in_sample, out_of_sample, target_feature)
    
    SVM_Learner = define_SVM_Classifier(input_train_set, output_train_set, gamma, class_weight)
    
    svm_predictions(SVM_Learner, input_train_set, output_train_set, mean)
    
    scores = cross_val_score(SVM_Learner, input_train_set, output_train_set.values.ravel(), scoring=weighting, cv=cross_validation_splitter)
    print(scores.mean())
    
    compute_score(input_train_set, output_train_set, SVM_Learner, mean, "F1_Score on Initial SVM Learner - UFC")
    
    size = input_train_set.shape[1]
    values = {
        'gamma': [0.25/size, 0.5/size, 0.75/size]
    }

    SVM_Learner_Linear = SVC(class_weight="balanced",kernel='linear')
    
    SVM_Learner_Sigmoid = SVC(class_weight="balanced",kernel='sigmoid')

    grid_suggestion = run_gridsearch(values, SVM_Learner_Linear, input_train_set, output_train_set, cross_validation_splitter, weighting, parallel)

    print(grid_suggestion.best_estimator_, grid_suggestion.best_score_)

    compute_score(input_train_set, output_train_set, grid_suggestion.best_estimator_, mean, "F1_Score on GridSearch optimized linear SVM Learner - UFC")
    
    grid_suggestion = run_gridsearch(values, SVM_Learner_Sigmoid, input_train_set, output_train_set, cross_validation_splitter, weighting, parallel)

    print(grid_suggestion.best_estimator_, grid_suggestion.best_score_)

    compute_score(input_train_set, output_train_set, grid_suggestion.best_estimator_, mean, "F1_Score on GridSearch optimized sigmoid SVM Learner - UFC")
    
    
    SVM_Learner_Linear = SVC(class_weight="balanced",kernel='linear', gamma = 0.0016)
    
    SVM_Learner_Sigmoid = SVC(class_weight="balanced",kernel='sigmoid', gamma = 0.0032)
    
    hyperparameter_values = np.linspace(1, 50, 49)
    
    cross_validation_splitter = 5
    result_train, result_test = run_validation_curve(SVM_Learner_Linear, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, reverse, cross_validation_splitter, "F1_Score on SVM Learner for optimizing linear kernel - UFC")
    
    j = result_test.mean(axis=1).argmax()
    linear_C = hyperparameter_values[j]
    print(linear_C)
    
    result_train, result_test = run_validation_curve(SVM_Learner_Sigmoid, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, reverse, cross_validation_splitter, "F1_Score on SVM Learner for optimizing sigmoid kernel - UFC")
    
    j = result_test.mean(axis=1).argmax()
    sigmoid_C = hyperparameter_values[j]
    print(sigmoid_C)
    
    
    cross_validation_splitter = 10

    optimal_SVM_linear = SVC(C=5.5, class_weight='balanced', gamma=0.0016, kernel='linear')

    optimal_SVM_sig = SVC(C=33, class_weight='balanced', gamma=0.0032, kernel='sigmoid')
    
    
    compute_score(input_train_set, output_train_set, optimal_SVM_linear, mean, "F1_Score for optimal linear SVM Learner - UFC")
    compute_score(input_train_set, output_train_set, optimal_SVM_sig, mean, "F1_Score for optimal sigmoid SVM Learner - UFC")
    
    optimal_SVM_linear.fit(input_train_set, output_train_set.values.ravel())
    test_y_hat = optimal_SVM_linear.predict(input_test_set)
    print(classification_report(output_test_set, test_y_hat, digits=5))
    print(confusion_matrix(output_test_set, test_y_hat))
    
    optimal_SVM_sig.fit(input_train_set, output_train_set.values.ravel())
    test_y_hat = optimal_SVM_sig.predict(input_test_set)
    print(classification_report(output_test_set, test_y_hat, digits=5))
    print(confusion_matrix(output_test_set, test_y_hat))
    
    
if __name__ == "__main__":
    wine_dataset()
    ufc_dataset()