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

def define_Boosting_Classifier(input_train_set, output_train_set):
    AdaBoost_Learner = AdaBoostClassifier()
    AdaBoost_Learner.fit(input_train_set, output_train_set.values.ravel())
    
    print(AdaBoost_Learner)
    return AdaBoost_Learner

def boosting_predictions(AdaBoost_Learner, input_train_set, output_train_set):
    y_hat = AdaBoost_Learner.predict(input_train_set)
    print(classification_report(output_train_set, y_hat))


def compute_score(input_train_set, output_train_set, AdaBoost_Learner, mean, title):
    result = evaluate_data(input_train_set, output_train_set, AdaBoost_Learner, f1_score, average=mean)
    graph_LC(result[0], result[1], result[2], title=title)
    

def run_gridsearch(values, adaBoost, input_train_set, output_train_set, cross_validation_splitter, weighting, parallel):
    grid_suggestion = GridSearchCV(adaBoost, values, cv=cross_validation_splitter, scoring=weighting, n_jobs = parallel)
    grid_suggestion.fit(input_train_set, output_train_set.values.ravel())
    print(grid_suggestion.best_estimator_, grid_suggestion.best_score_)
    
    return grid_suggestion
    
    
def run_validation_curve(AdaBoost_Learner, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, inverse_x, cross_validation_splitter, title):
    result_train, result_test = validation_curve(
        AdaBoost_Learner, input_train_set, output_train_set.values.ravel(), param_name=hyperparameter, param_range=hyperparameter_values,
        scoring=weighting, verbose=1, cv=cross_validation_splitter, n_jobs=-1
    )
    print(result_train.mean(axis=1), result_test.mean(axis=1))
    graph_C(result_train.mean(axis=1), result_test.mean(axis=1), hyperparameter_values, title=title, inverse_x=inverse_x)
    
    return result_train, result_test

def wine_dataset():
    target_feature = "class"
    mean = "weighted"
    weighting = "f1_weighted"
    hyperparameter = "n_estimators"
    cross_validation_splitter = 10
    parallel = -1
    inverse_x = False
    
    
    np.random.seed(42)
    wine_dataset = pd.read_csv("wine_dataset.csv")
    
    target = wine_dataset["class"]
    wine_dataset = normalize_data(wine_dataset)
    wine_dataset["class"] = target
    
    in_sample, out_of_sample = create_stratified_data(target_feature, wine_dataset)
    
    output_train_set, input_train_set, output_test_set, input_test_set = prepare_data(in_sample, out_of_sample, target_feature)
    
    AdaBoost_Learner = define_Boosting_Classifier(input_train_set, output_train_set)
    
    boosting_predictions(AdaBoost_Learner, input_train_set, output_train_set)
    
    result = cross_val_score(AdaBoost_Learner, input_train_set, output_train_set.values.ravel(), scoring=weighting, cv=cross_validation_splitter)
    print(result.mean())
    
    
    compute_score(input_train_set, output_train_set, AdaBoost_Learner, mean, "F1_Score on Initial AdaBoosting Learner - Wine")

    values = {
        'base_estimator__criterion': ['gini', 'entropy'],
        "base_estimator__splitter" : ["best", "random"],
        'algorithm': ['SAMME', 'SAMME.R'],
        'learning_rate': [0.3, 0.4, 0.5, 0.8, 1.0, 1.2]
    }
    DT_Learner = DecisionTreeClassifier(ccp_alpha = 0.0011)
    adaBoost = AdaBoostClassifier(DT_Learner)
    
    
    grid_suggestion = run_gridsearch(values, adaBoost, input_train_set, output_train_set, cross_validation_splitter, weighting, parallel)

    print(grid_suggestion.best_estimator_, grid_suggestion.best_score_)

    compute_score(input_train_set, output_train_set, grid_suggestion.best_estimator_, mean, "F1_Score on GridSearch optimized AdaBoost Learner - Wine")
    
    
    boosted_tree = AdaBoostClassifier(algorithm='SAMME', base_estimator=DecisionTreeClassifier(ccp_alpha=0.0011,splitter='random'), learning_rate=0.4)

    
    hyperparameter_values = np.linspace(10, 310, 30).astype(int)
    result_train, result_test = run_validation_curve(boosted_tree, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, inverse_x, cross_validation_splitter, "F1_Score on AdaBoost Learner for optimizing n_estimators - Wine")
    
    j = result_test.mean(axis=1).argmax()
    optimal_estimator = hyperparameter_values[j]
    print(optimal_estimator)
    
    hyperparameter = 'base_estimator__ccp_alpha'
    hyperparameter_values = np.linspace(0.0001, 0.002, 100)
    result_train, result_test = run_validation_curve(boosted_tree, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, inverse_x, cross_validation_splitter, "F1_Score on AdaBoost Learner for optimizing ccp_alpha - Wine")
    
    j = result_test.mean(axis=1).argmax()
    optimal_complexity = hyperparameter_values[j]
    print(optimal_complexity)
    
    
    optimal_Boosting_Learner = AdaBoostClassifier(algorithm='SAMME', base_estimator=DecisionTreeClassifier(ccp_alpha=0.0004,splitter='random'),learning_rate=0.4, n_estimators=134)
    
    compute_score(input_train_set, output_train_set, optimal_Boosting_Learner, mean, "F1_Score on optimal AdaBoost Learner - Wine")
    
    optimal_Boosting_Learner.fit(input_train_set, output_train_set.values.ravel())
    test_y_hat = optimal_Boosting_Learner.predict(input_test_set)
    print(classification_report(output_test_set, test_y_hat, digits=5))
    print(confusion_matrix(output_test_set, test_y_hat))
    

def ufc_dataset():
    target_feature = "Winner"
    alpha = 0.005
    criterion = "entropy"
    mean = "macro"
    weighting = "f1_macro"
    hyperparameter = "n_estimators"
    cross_validation_splitter = 10
    parallel = -1
    inverse_x = False
    
    
    np.random.seed(42)
    ufc_dataset = pd.read_csv("ufc_data.csv")
    ufc_dataset = preprocess_ufc_data(ufc_dataset)
    
    ufc_dataset = normalize_data(ufc_dataset)
    
    in_sample, out_of_sample = create_stratified_data(target_feature, ufc_dataset)
    
    train_set = in_sample
    test_set = out_of_sample
    
    output_train_set, input_train_set, output_test_set, input_test_set = prepare_data(train_set, test_set, target_feature)
    
    AdaBoost_Learner = define_Boosting_Classifier(input_train_set, output_train_set)
    
    boosting_predictions(AdaBoost_Learner, input_train_set, output_train_set)
    
    scores = cross_val_score(AdaBoost_Learner, input_train_set, output_train_set.values.ravel(), scoring=weighting, cv=cross_validation_splitter)
    print(scores.mean())
    
    compute_score(input_train_set, output_train_set, AdaBoost_Learner, mean, "F1_Score on Initial AdaBoosting Learner - UFC")
    
    values = {
        'base_estimator__criterion': ['gini', 'entropy'],
        "base_estimator__splitter" : ["best", "random"],
        'algorithm': ['SAMME', 'SAMME.R'],
        'learning_rate': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }
    DT_Learner = DecisionTreeClassifier(ccp_alpha = 0.0031, class_weight="balanced")
    adaBoost = AdaBoostClassifier(DT_Learner)
    
    grid_suggestion = run_gridsearch(values, adaBoost, input_train_set, output_train_set, cross_validation_splitter, weighting, parallel)

    print(grid_suggestion.best_estimator_, grid_suggestion.best_score_)

    compute_score(input_train_set, output_train_set, grid_suggestion.best_estimator_, mean, "F1_Score on GridSearch optimized AdaBoost Learner - UFC")
    
    
    DT_Learner = DecisionTreeClassifier(splitter="random", class_weight='balanced')
    adaBoost = AdaBoostClassifier(algorithm='SAMME', base_estimator=DT_Learner, learning_rate=0.9)
    
    hyperparameter_values = np.linspace(10, 210, 20).astype(int)
    result_train, result_test = run_validation_curve(adaBoost, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, inverse_x, cross_validation_splitter, "F1_Score on AdaBoost Learner for optimizing n_estimators - UFC")

    j = result_test.mean(axis=1).argmax()
    optimal_estimator = hyperparameter_values[j]
    print(optimal_estimator)
    
    hyperparameter_values = np.linspace(0.001, 0.0032, 31)
    hyperparameter="base_estimator__ccp_alpha"
    result_train, result_test = run_validation_curve(adaBoost, input_train_set, output_train_set, hyperparameter_values, weighting, hyperparameter, inverse_x, cross_validation_splitter, "F1_Score on AdaBoost Learner for optimizing ccp_alpha - UFC")
    
    j = result_test.mean(axis=1).argmax()
    optimal_complexity = hyperparameter_values[j]
    print(optimal_complexity)
    
    optimal_Boosting_Learner = AdaBoostClassifier(algorithm='SAMME', base_estimator=DecisionTreeClassifier(ccp_alpha=0.0031, class_weight='balanced',splitter='best'), learning_rate=0.9, n_estimators=optimal_estimator)
    
    
    compute_score(input_train_set, output_train_set, optimal_Boosting_Learner, mean, "F1_Score on optimal AdaBoost Learner - UFC")

    
    optimal_Boosting_Learner.fit(input_train_set, output_train_set.values.ravel())
    test_y_hat = optimal_Boosting_Learner.predict(input_test_set)
    print(classification_report(output_test_set, test_y_hat, digits=5))
    print(confusion_matrix(output_test_set, test_y_hat))
    
    
if __name__ == "__main__":
    wine_dataset()
    ufc_dataset()