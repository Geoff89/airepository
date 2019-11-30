# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 22:22:46 2018

@author: jeffnerd
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import datasets
from sklearn import metrics

iris = datasets.load_iris()
iris.data.shape,iris.target.shape

x_train, x_test, y_train, y_test = train_test_split(
                    iris.data, iris.target, test_size=0.4, random_state=0)

x_train.shape, y_train.shape
x_test.shape, y_test.shape
clf = svm.SVC(kernel='linear', C=1).fit(x_train,y_train)
clf.score(x_test, y_test)
#altenatively you can predict
y_pred = clf.predict(x_test)
metrics.accuracy_score(y_test,y_pred)

#preprocessing
#label encoder