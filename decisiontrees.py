# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 11:40:16 2017

@author: jeffnerd
"""

from sklearn.datasets import load_iris
import graphviz
from sklearn import tree
from sklearn import metrics

X = [[0,0],[1,1]]
y = [0,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
y_pred = clf.predict([[2.,2.]])
y_true = [1,0]
metrics.accuracy_score([2.,1.],y_pred)

#altrnatively we can predict the propability of the training samples in same class
clf.predict_proba([[2.,2.]])
clf.predict(iris.data[:1, :])
clf.predict_proba(iris.data[:1, :])

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

#Plotting this using graphviz we can export the tree after trainig our model
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names = iris.feature_names,
                                class_names = iris.target_names,
                                filled = True,rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)


