# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:43:06 2017

@author: jeffnerd
"""

#cross val scores and predict
from sklearn.datasets import load_iris
from sklearn.datasets import digits
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = load_iris()
X = iris.data
y = iris.target
print(Y.shape)
print(x.shape)
print(type(Y))
print(type(x))

X_train,X_test,y_train,y_test = train_test_split(X,y)

#instantiate the estimate
clf = SVC()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
#testing the accuracy
metrics.accuracy_score(y_test,y_pred)

#Kneighbors
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)
# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]
# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)

knn.predict(iris_X_test)
iris_y_test

#casting
from sklearn import random_projection
rng = np.random.RandomState(0)
x =rng.rand(10,2000)
x = np.array(x, dtype = 'float32')
x.dtype
#transforming the data from float32 to float64
transformer = random_projection.GaussianRandomProjection()
x_new = transformer.fit_transform(x)
x_new.dtype

#Refitting and updating parameters
from sklearn.svm import SVC
rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)
clf = SVC()
clf.set_params(kernel='linear').fit(X, y)
clf.predict(X_test)
clf.set_params(kernel='rbf').fit(X, y)
clf.predict(X_test)