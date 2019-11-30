# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:08:49 2019

@author: jeffnerd
"""

from sklearn.datasets import load_iris, make_classification
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
X, y = make_classification(n_features=20, n_samples=10000, n_classes=3,
                           n_informative=3)
#put this data as sine
#1.use cross val score
#2,use grid search cv
#3.plot the weights
#4 plot the training and test data
#5 plot the error bar
#6 plot the test data and predict data sets
iris = load_iris()
print(f"The data points shape is {X.shape}")
print(f"The feature point shape is {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = .25, 
                                                    random_state = 42)
# cross model
model = cross_val_score(SVC(), X, y, cv = 10, scoring = 'precision_micro')
print('The score is %s (+- %s)' %(model.mean(), model.std()))

#Grid search cv
paragrid =  [{'C':[0.2,0.1,0.4], 'gamma' :[0.3, 0.5,0.7], 
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']},
             {'C':[0.3,0.9,0.4], 'gamma' : [0.5,0.6,0.9], 
              'kernel': ['rbf','linear', 'sigmoid']}]
estimator = GridSearchCV(SVC(), param_grid = paragrid, cv =5, 
                         scoring='accuracy')
estimator.fit(X_train,y_train)

print(estimator.best_estimator_)
print(estimator.best_params_)
print(estimator.best_score_)
mean_score = estimator.cv_results_['mean_test_score']
std_score = estimator.cv_results_['std_test_score']
param = estimator.cv_results_['params']

for mean, std, param in zip(mean_score, std_score, param):
    print('The mean score %s ( std +- %s) for %s' %(mean, std, param))
    
    





