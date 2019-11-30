# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:23:54 2017

@author: jeffnerd
"""

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

#create an estimator for Kneughbors classifirer
knn = KNeighborsClassifier(n_neighbors = 10)
##define the parameter values that should be searched
k_range = list(range(1,31))
print(k_range)

#creating a param_grid. param_grid is a dictonary like with key being the parameter
# and value being the list of values that should be searched for that parameter
param_grid = dict(n_neighbors = k_range)


#instantiate the gridsearch cv
grid = GridSearchCV(knn, param_grid, cv=10, scoring = 'accuracy')

#fit the grid with data
grid.fit(X,y)

#View the complete results
grid.grid_scores_
#grid scores results
[mean: 0.96000, std: 0.05333, params: {'n_neighbors': 1},
 mean: 0.95333, std: 0.05207, params: {'n_neighbors': 2},
 mean: 0.96667, std: 0.04472, params: {'n_neighbors': 3},
 mean: 0.96667, std: 0.04472, params: {'n_neighbors': 4},
 mean: 0.96667, std: 0.04472, params: {'n_neighbors': 5},
 mean: 0.96667, std: 0.04472, params: {'n_neighbors': 6},
 mean: 0.96667, std: 0.04472, params: {'n_neighbors': 7},
 mean: 0.96667, std: 0.04472, params: {'n_neighbors': 8},
 mean: 0.97333, std: 0.03266, params: {'n_neighbors': 9},
 mean: 0.96667, std: 0.04472, params: {'n_neighbors': 10},
 mean: 0.96667, std: 0.04472, params: {'n_neighbors': 11},
 mean: 0.97333, std: 0.03266, params: {'n_neighbors': 12},
 mean: 0.98000, std: 0.03055, params: {'n_neighbors': 13},
 mean: 0.97333, std: 0.04422, params: {'n_neighbors': 14},
 mean: 0.97333, std: 0.03266, params: {'n_neighbors': 15},
 mean: 0.97333, std: 0.03266, params: {'n_neighbors': 16},
 mean: 0.97333, std: 0.03266, params: {'n_neighbors': 17},
 mean: 0.98000, std: 0.03055, params: {'n_neighbors': 18},
 mean: 0.97333, std: 0.03266, params: {'n_neighbors': 19},
 mean: 0.98000, std: 0.03055, params: {'n_neighbors': 20},
 mean: 0.96667, std: 0.03333, params: {'n_neighbors': 21},
 mean: 0.96667, std: 0.03333, params: {'n_neighbors': 22},
 mean: 0.97333, std: 0.03266, params: {'n_neighbors': 23},
 mean: 0.96000, std: 0.04422, params: {'n_neighbors': 24},
 mean: 0.96667, std: 0.03333, params: {'n_neighbors': 25},
 mean: 0.96000, std: 0.04422, params: {'n_neighbors': 26},
 mean: 0.96667, std: 0.04472, params: {'n_neighbors': 27},
 mean: 0.95333, std: 0.04269, params: {'n_neighbors': 28},
 mean: 0.95333, std: 0.04269, params: {'n_neighbors': 29},
 mean: 0.95333, std: 0.04269, params: {'n_neighbors': 30}]

##Examinig the individual tuples incase you need to evaluate them in feature
print(grid.grid_scores_[0].parameters)
print(grid.grid_scores_[0].cv_validation_scores)
print(grid.grid_scores_[0].mean_validation_score)
#the results is below
{'n_neighbors': 1}
[ 1.          0.93333333  1.          0.93333333  0.86666667  1.
  0.86666667  1.          1.          1.        ]
0.96

#creating mean scores only
grid_mean_score = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_score)


#plotting the results
plt.plot(k_range,grid_mean_score)
plt.xlabel('Value of K for knn')
plt.ylabel('cross validated accuracy')

#choosing the best model form the above
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

#results from the above
0.98
{'n_neighbors': 13}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=13, p=2,
           weights='uniform')


#Searching for multiple parameters#Exhaustive grid search
k_range = list(range(1,31))
weight_options = ['uniform','distance']

#create a parm_grid: map the parameter names to the values that shoud be searched
param_grid = dict(n_neighbors = k_range,weights = weight_options)
print(param_grid)

#instantiate and fit the grid
grid =GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
grid.fit(X,y)

#printing the entire results
print(grid.grid_scores_)

#Finding the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

#output
0.98
{'n_neighbors': 13, 'weights': 'uniform'}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=13, p=2,
           weights='uniform')
#Once you find the best optional parameters you can use them to make predictions
#train our model using all the data and best oknown parameters
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=13, p=2,
           weights='uniform')
knn.fit(X,y)

#predict our data
knn.predict([3,5,4,2])
#Gridsearch automaticall refers to the best model using sll the data
grid.predict([3,5,4,2])

#randomized serach to do our predictions
param_dist = dict(n_neighbors = k_range, weights = weight_options)
print(param_dist)

#n_iter contains number of random distributions tha it will try
#random_state is put for purposes of reproducability
rand = RandomizedSearchCV(knn, param_dist, cv = 10, scoring= 'accuracy',n_iter = 10,
                          random_state = 5)
rand.fit(X,y)
rand.grid_scores_

[mean: 0.97333, std: 0.03266, params: {'weights': 'distance', 'n_neighbors': 16},
 mean: 0.96667, std: 0.03333, params: {'weights': 'uniform', 'n_neighbors': 22},
 mean: 0.98000, std: 0.03055, params: {'weights': 'uniform', 'n_neighbors': 18},
 mean: 0.96667, std: 0.04472, params: {'weights': 'uniform', 'n_neighbors': 27},
 mean: 0.95333, std: 0.04269, params: {'weights': 'uniform', 'n_neighbors': 29},
 mean: 0.97333, std: 0.03266, params: {'weights': 'distance', 'n_neighbors': 10},
 mean: 0.96667, std: 0.04472, params: {'weights': 'distance', 'n_neighbors': 22},
 mean: 0.97333, std: 0.04422, params: {'weights': 'uniform', 'n_neighbors': 14},
 mean: 0.97333, std: 0.04422, params: {'weights': 'distance', 'n_neighbors': 12},
 mean: 0.97333, std: 0.03266, params: {'weights': 'uniform', 'n_neighbors': 15}]

#Examine the best model
print(rand.best_score_)
print(rand.best_params_)
print(rand.best_estimator_)

#output for the above results
0.98
{'weights': 'uniform', 'n_neighbors': 18}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=18, p=2,
           weights='uniform')


