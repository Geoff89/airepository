# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:41:20 2018

@author: jeffnerd
"""

print(__doc__)

import numpy as np
from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

#get some data
digits = load_digits()
X, y = digits.data, digits.target

#building a classifier
clf = RandomForestClassifier(n_estimators=20)

#utility function toreport best scores
def report(results, n_top = 3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
             "max_features": sp_randint(1, 11),
             "min_samples_split": sp_randint(2, 11),
             "min_samples_leaf": sp_randint(1,11),
             "bootstrap": [True, False],
             "criterion": ["gini", "entropy"]}
#run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, 
                                   n_iter= n_iter_search)

start = time()
random_search.fit(X,y)
print("RandomizedSearchCV took %0.2f seconds for %d candidates parameter settings"
     %((time() - start), n_iter_search))
report(random_search.cv_results_)

#for grid search cv
#specify parameters and distributions to sample from
param_grid = {"max_depth": [3, None],
             "max_features": [2,3,10],
             "min_samples_split": [2,3,10],
             "min_samples_leaf": [2,3,10],
             "bootstrap": [True, False],
             "criterion": ["gini", "entropy"]}
#run gridsearch cv

grid_search = GridSearchCV(clf,param_grid=param_grid)

start = time()
grid_search.fit(X,y)
print("GridSearchCV took %0.2f seconds for %d candidates parameter settings"
     %(time() - start, len(grid_search.cv_results_)))
report(grid_search.cv_results_)