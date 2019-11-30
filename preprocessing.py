# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 09:40:15 2017

@author: jeffnerd
"""
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler#for scaling data with many outliers
from sklearn.preprocessing import robust_scale#for sclaing data with many oukiers
from sklearn.preprocessing import MaxAbsScaler#for sparse data
from sklearn.preprocessing import maxabs_scale
from sklearn.decomposition import PCA#sometimes when scaling and centering features is not enough
from sklearn.decomposition import RandomizedPCA#use when centering and scaling is not enough
#set whiten = True t0 remove linear correlation between variables
from sklearn.neighbors import Kneighbors
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
#standardization of datasets is a requirement for features to be representative
#features should be centred around mean 0 and unit variance(standard deviation =1)
x = np.array([[1.,-1.,3.],
              [2.,0.,0.],
              [0.,1.,-1.]])
x_scaled = preprocessing.scale(x)
x_scaled
#scaled data has mean 0 and unit variance
x_scaled.mean(axis = 0)
x_scaled.std(axis = 0)
#The preprocessing module further provides a standardscaler that implements Transformer API
#to compute mean and standard deviation on training set
scaler = preprocessing.StandardScaler().fit(x)
scaler
scaler.mean_
scaler.var_
scaler.std_
scaler.scale_

scaler.transform(x)

#scaling features to a range
#An alternative standardization is scaling features to lie between minimum and 
#maximum values often between o and 1
x_train =np.array([[1.,-1.,3.],
                   [2.,0.,0.],
                   [0.,1.,-1.]])
min_max_scaler = preprocessing.MinMaxScaler().fit(x_train)
x_train_minmax = min_max_scaler.transform(x_train)
x_train_minmax1 = min_max_scaler.fit_transform(x_train)
#The same instance can be applied to a new test data set that has not been seen during fit call
x_test = np.array([-3.,-1.,4.])
x_test_minmax = min_max_scaler.transform(x_test)
min_max_scaler.min_
min_max_scaler.scale_
min_max_scaler.

#Aother one is MaxAbsScaler scales training data to range[-1,1]
max_abs_scaler = preprocessing.MaxAbsScaler().fit(x_train)
x_train_maxabs =max_abs_scaler.transform(x_train)
#applying it to test data set
x_test = np.array([[-3.,-1.,4.]])
x_test_maxabs = max_abs_scaler.transform(x_train)
max_abs_scaler.scale_

#Normalization
#MaxAbsScaler() and MaxAbs_scaler are designed to scaling sparse data
#scaling data with many otuliers.use robust_scaler and RobustScaler 






