# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 07:38:46 2017

@author: jeffnerd
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
cancer = pd.read_csv(r'C:\Users\jeffnerd\Desktop\wisc_bc_data.CSV')
y = cancer.diagnosis
y = cancer['diagnosis']
cancer.loc[:,['radius_mean','perimeter_mean']]
X = cancer.loc[:,['radius_mean','texture_mean','perimeter_mean','area_mean',
'smoothness_mean','compactness_mean','concavity_mean','concave points_mean',
'symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se',
'smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se',
'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst',
'smoothness_worst','compactness_worst','concavity_worst','concave points_worst',
'symmetry_worst','fractal_dimension_worst']]
#accessing features part 2
X = cancer[['radius_mean','texture_mean','perimeter_mean','area_mean'\
'smoothness_mean','compactness_mean','concavity_mean','concave points_mean'\
'symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se'\
'smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se'\
'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst'\
'smoothness_worst','compactness_worst','concavity_worst','concave points_worst'\
'symmetry_worst','fractal_dimension_worst']]
#preprocess this feature data with scale
x_scale = scale(X)
x_scale.mean(axis = 0)
x_scale.std(axis = 0)
#preprocess this feature with standardscaler
x_train = StandardScaler().fit(x)
x_train_transformed = x_train.fit_transform(x_train)
x_train_transformed.mean_
x_train_transformed.scale_
#preprocessing with MinMaxScaler
x_train = MinMaxScaler().fit(X)
x_train_transformed = X_train.fit_transform(X)
#preprocessing with MaxAbsScaler
x_train = MaxAbsScaler().fit(X)
x_train_transform = x_train.fit_transform(X)

#train test split this data
X_train,X_test,y_train,y_test = train_test_split(x_train_transformed,y)
#To see their shape and how the have been split
X_train.shape,X_test.shape,y_train.shape, y_test.shape

#Choose the right nearest neighbor ie K-nn, best K
#create an estimator for Kneughbors classifirer
knn = KNeighborsClassifier(n_neighbors = 10)
##define the parameter values that should be searched
k_range = list(range(1,31))
print(k_range)

weight_options = ['uniform','distance']
#create a parm_grid: map the parameter names to the values that shoud be searched
param_grid = dict(n_neighbors = k_range,weights = weight_options)
print(param_grid)

#instantiate and fit the grid
grid =GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
#Fit the gridsearch cv to find the optimal parameters
grid.fit(X_train,y_train)
grid.grid_scores_
#Find the best parameters for our models
grid.best_params_
#output is
{'n_neighbors': 10, 'weights': 'distance'}
grid.best_score_
#output is
0.96948356807511737
grid.best_estimator_
#output is
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=10, p=2,
           weights='distance')

#The next step now to train our model
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=10, p=2,
           weights='distance')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
#Defining the accuracy of the model
y_true = y_test
print(accuracy_score(y_true,y_pred))


#randomized serach to do our predictions
param_dist = dict(n_neighbors = k_range, weights = weight_options)
print(param_dist)
#n_iter contains number of random distributions tha it will try
#random_state is put for purposes of reproducability
rand = RandomizedSearchCV(knn, param_dist, cv = 10, scoring= 'accuracy',n_iter = 10,
                          random_state = 5)


rand.fit(X_train,y_train)
rand.cv_results_
#Examine the best model
print(rand.best_score_)
print(rand.best_params_)
print(rand.best_estimator_)