# -*- coding: utf-8 -*-
"""
Created on Sat May 20 12:06:54 2017

@author: jeffnerd
"""

import numpy as np
#checking accuracy of models
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

#accuracy score
y_pred = [0,1,2,3]
y_true = [0,1,2,3]
print(accuracy_score(y_true,y_pred))

#multilabel classification
accuracy_score(np.array([[0,1],[1,1]]), np.ones((2,2)))
#0.5 is the accuracy for the above
from sklearn.metrics import cohen_kappa_score
#cohen kappa score computes cohen kappa statistic.Kappa score is a number betwen
# -1 and 1.scores  above .8 are generally considered good agreements and 0 or
#lower are considered not good agreements.used for binary ad multiclass and not
# multilabels
y_true = [2,0,2,2,0,1]
y_pred = [0,0,2,2,0,2]
cohen_kappa_score(y_true,y_pred)

from sklearn.metrics import confusion_matrix
#confusion matrix
y_true = [0,0,0,1,1,1,1,1]
y_pred = [0,1,0,1,0,1,0,1]
tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()
#output is (2, 1, 2, 3)
#classification report
from sklearn.metrics import classification_report
y_true = [0,1,2,2,0]
y_pred = [0,0,2,1,0]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))

#hamming loss
from sklearn.metrics import hamming_loss
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
hamming_loss(y_true,y_pred)
#in multilabel case with binary label indicators
hamming_loss(np.array([[0,1],[1,1]]), np.zeros((2,2)))

#jaccard similarilty coefficient score--intersection and union of predoxted values
from sklearn.metrics import jaccard_similarity_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
jaccard_similarity_score(y_true,y_pred)
jaccard_similarity_score(y_true,y_pred, normalize=False)
#in case of multilabel with binary label indicators
jaccard_similarity_score(np.array([[0,1],[1,1]]), np.ones((2,2)))
#some examples in binary classification
from sklearn import metrics
y_pred = [0,1,0,1]
y_true = [0,1,0,1]
metrics.precision_score(y_true, y_pred)
metrics.recall_score(y_true,y_pred)
metrics.f1_score(y_true,y_pred)
metrics.fbeta_score(y_true,y_pred,beta=0.5)
metrics.fbeta_score(y_true,y_pred,beta=1)
metrics.fbeta_score(y_true,y_pred,beta=2)
metrics.precision_recall_fscore_support(y_true,y_pred,beta=0.5)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score 
y_true = np.array([0,0,1,1])
y_scores = np.array([0.1,0.4,0.35,0.8])
precision,recall,threshold = precision_recall_curve(y_true,y_scores)
precision#the output will be an array
recall
threshold
average_precision_score(y_true,y_scores)
#for  multiclass and multilabel of each class
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
metrics.precision_score(y_true,y_pred,average='macro')
metrics.recall_score(y_true,y_pred, average='micro')
metrics.f1_score(y_true,y_pred,average='weighted')
metrics.fbeta_score(y_true,y_pred,average='macro',beta=0.5)
metrics.precision_recall_fscore_support(y_true,y_pred,beta=0.5, average=None)
#for multiclass classification with negatve class it is possioble to exclude
#some labels
metrics.recall_score(y_true,y_pred, labels=[1,2], average='micro')#excluding 0,no labels were crrectly recalled
#similarly labels not in data may be accounted by macro-averagong
metrics.precision_score(y_true,y_pred,labels=[0,1,2,3], average='macro')
#hinge loss
from sklearn import svm
from sklearn.metrics import hinge_loss
X = [[0],[1]]
y = [-1,1]
est = svm.LinearSVC(random_state=0)
est.fit(X,y)
pred_decision =  est.decision_function([[-2],[3],[0.5]])
pred_decision
hinge_loss([-1,1,1],pred_decision)
#for mutilabel functions
X = np.array([[0], [1], [2], [3]])
Y = np.array([0, 1, 2, 3])
labels = np.array([0, 1, 2, 3])
est = svm.LinearSVC()
est.fit(X, Y)
pred_decision = est.decision_function([[-1], [2], [3]])
y_true = [0, 2, 3]
hinge_loss(y_true, pred_decision, labels)
#log loss
from sklearn.metrics import log_loss
y_true = [0,0,1,1]
y_pred = [[.9,.1],[.8,.2],[.3,.7],[.01,.99]]
log_loss(y_true,y_pred)
#mathews correation coefficient
from sklearn.metrics import matthews_corrcoef
y_true = [+1, +1, +1, -1]
y_pred = [+1, -1, +1, +1]
matthews_corrcoef(y_true,y_pred)
#roc curve
from sklearn.metrics import roc_curve
import numpy as np
y = np.array([1,1,2,2])
scores = np.array([0.1,0.4,0.35,0.8])
fpr,tpr,thresholds = roc_curve(y,scores,pos_label=2)
fpr
tpr
thresholds
#roc_auc_score
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y_true,y_scores)
#zero one loss both in  binary and multilabel instances
from sklearn.metrics import zero_one_loss
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
zero_one_loss(y_true,y_pred)
zero_one_loss(y_true,y_pred,normalize=False)
#for multilabel case with binary indicators
zero_one_loss(np.array([[0,1],[1,1]]), np.ones((2,2)))
zero_one_loss(np.array([[0,1],[1,1]]), np.ones((2,2)), normalize=False)
#brier_scoring _loss
from sklearn.metrics import brier_score_loss
y_true = np.array([0,1,1,0])
y_true_categorical = np.array(["spam","ham","ham","spam"])
y_prob = np.array([0.1, 0.9, 0.8, 0.4])
y_pred = np.array([0, 1, 1, 0])
brier_score_loss(y_true,y_prob)
brier_score_loss(y_true,1-y_prob,pos_label=0)
brier_score_loss(y_true_categorical,y_prob,pos_label="ham")
brier_score_loss(y_true,y_prob > 0.5)
#multilabel metrics
from sklearn.metrics import coverage_error
y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
coverage_error(y_true,y_score)
#label ranking average precision
from sklearn.metrics import label_ranking_average_precision_score
y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
label_ranking_average_precision_score(y_true,y_score)
#ranking loss
from sklearn.metrics import label_ranking_loss
y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
label_ranking_loss(y_true,y_score)
# With the following prediction, we have perfect and minimal loss
y_score = np.array([[1.0, 0.1, 0.2], [0.1, 0.2, 0.9]])
label_ranking_loss(y_true, y_score)
